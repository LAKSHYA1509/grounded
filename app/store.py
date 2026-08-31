"""
The vector store: Qdrant.

WHAT A VECTOR STORE ACTUALLY DOES
---------------------------------
It holds (vector, text, metadata) rows and answers one question fast:
"which stored vectors are closest to this query vector?"

"Closest" here is cosine similarity — the angle between two vectors. Angle
rather than distance, because we care about direction (meaning) and not
magnitude (roughly, length of text).

WHY QDRANT AND NOT AN EMBEDDED STORE
------------------------------------
This started on Chroma, which runs in-process and persists to a local
directory. That was the right call while the project ran only on a laptop,
and the wrong call the moment it was containerised. Two reasons, and the
second one was the surprise:

  1. A container filesystem is ephemeral. Every redeploy would wipe the
     index, so the service would come back up knowing nothing. Storage that
     outlives the process has to live outside the process.

  2. Chroma dragged in ~235 MB of dependencies we never used — a Kubernetes
     client, an ONNX inference runtime, gRPC, OpenTelemetry — because it can
     also embed locally and run distributed. We use it to store vectors and
     do a similarity search. Moving to a thin HTTP client cut the install
     from 422 MB to 187 MB.

That is the general lesson worth keeping: an embedded database is a
convenience for one process on one disk. As soon as there is more than one
of either, it becomes a liability.

WHY NOT JUST POSTGRES
---------------------
You can — pgvector does exactly this, and for a modest corpus it is often
the right call, because it is one less system to run. A dedicated store
earns its place at scale, or when you want heavier metadata filtering.

Brute-force comparison against every stored vector is O(n) and fine for
thousands of chunks. Beyond that you need an approximate nearest-neighbour
index — HNSW is the usual one — which trades a little recall for a lot of
speed. Qdrant does that for us.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings
from app.llm import get_embeddings

_store: Optional[QdrantVectorStore] = None


def _make_client() -> QdrantClient:
    """
    Three modes, chosen by whether QDRANT_URL is set:

      - set        -> a real Qdrant server (Qdrant Cloud, or your own)
      - not set    -> in-memory, no server, no setup

    The in-memory mode is not a toy. It means `pytest` runs with no
    infrastructure and `uvicorn` works on a fresh clone with nothing but an
    OpenAI key. Anything that makes the tests need a running server is
    something that will eventually stop the tests being run at all.
    """
    if settings.qdrant_url:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=30,
        )
    return QdrantClient(location=":memory:")


def _ensure_collection(client: QdrantClient) -> None:
    """
    Create the collection if it isn't there.

    NOTE ON THE VECTOR SIZE: it must match the embedding model exactly.
    text-embedding-3-small produces 1536 dimensions. Qdrant will reject
    vectors of any other length, which is a good thing — it turns a silent
    "why are my results garbage" into a loud error at insert time.

    This is also why changing EMBEDDING_MODEL means re-indexing everything:
    a different model produces vectors in a different space, so similarity
    between old and new vectors is meaningless even when the dimensions
    happen to match.
    """
    if client.collection_exists(settings.collection):
        return

    client.create_collection(
        collection_name=settings.collection,
        vectors_config=VectorParams(
            size=settings.embedding_dim,
            distance=Distance.COSINE,
        ),
    )


def get_store() -> QdrantVectorStore:
    """Lazily open the store. Built once and reused."""
    global _store
    if _store is None:
        client = _make_client()
        _ensure_collection(client)
        _store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection,
            embedding=get_embeddings(),
        )
    return _store


def reset_store() -> None:
    """Drop the cached store. Used by tests to get a clean instance."""
    global _store
    _store = None


def add_documents(docs: List[Document]) -> int:
    """
    Embed and index.

    This is the expensive part of ingestion — an embedding API call per batch
    of chunks — which is exactly why ingestion should be asynchronous in a
    real system rather than blocking an HTTP request. Noted as a known
    limitation in the README rather than pretended away.
    """
    if not docs:
        return 0
    get_store().add_documents(docs)
    return len(docs)


def search(query: str, k: int = 4) -> List[Document]:
    """
    Embed the query with the SAME model used at ingestion, then find the k
    nearest chunks.

    A NOTE ON MULTI-TENANCY, because it is the most important thing here:

    If this service were multi-tenant, the tenant filter would have to be
    applied *inside* this search call, not to its results — Qdrant takes a
    `filter` argument for exactly this:

        get_store().similarity_search(query, k=k, filter=<tenant filter>)

    Filtering afterwards is a real bug, not a style preference. Retrieve the
    global top-4 and then drop other tenants' chunks and you are left with
    fewer than 4 — sometimes zero — and the user gets a worse answer for no
    visible reason. Worse, one mistake in that post-filter leaks another
    tenant's data.

    Push the filter into the query. Always.
    """
    return get_store().similarity_search(query, k=k)


def count() -> int:
    """How many chunks are indexed. Used by the health check."""
    try:
        client = _make_client()
        if not client.collection_exists(settings.collection):
            return 0
        return client.count(settings.collection, exact=True).count
    except Exception:
        return -1
