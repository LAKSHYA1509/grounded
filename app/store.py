"""
The vector store.

WHAT A VECTOR STORE ACTUALLY DOES
---------------------------------
It holds (vector, text, metadata) rows and answers one question fast:
"which stored vectors are closest to this query vector?"

"Closest" is usually cosine similarity — the angle between two vectors.
Angle rather than distance, because we care about direction (meaning) and
not magnitude (roughly, length of text).

WHY NOT JUST POSTGRES
---------------------
You can — pgvector does exactly this, and for a modest corpus it's often the
right call, because it's one less system to run. A dedicated store earns its
place at scale, or when you need heavy metadata filtering.

Brute-force comparison against every stored vector is O(n) and fine for
thousands of chunks. Beyond that you need an approximate nearest-neighbour
index (HNSW is the common one), which trades a little recall for a lot of
speed. Chroma handles that for us.

WHY CHROMA HERE
---------------
It runs in-process and persists to a local directory — no server to start.
For a project whose point is understanding the *graph*, spending your setup
budget on a database server would be a bad trade.
"""

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.llm import get_embeddings

_store: Optional[Chroma] = None


def get_store() -> Chroma:
    """Lazily open the store. Persists to disk, so it survives restarts."""
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=settings.collection,
            embedding_function=get_embeddings(),
            persist_directory=settings.persist_dir,
        )
    return _store


def add_documents(docs: List[Document]) -> int:
    """
    Embed and index. This is the expensive part of ingestion — one embedding
    API call per batch of chunks — which is exactly why ingestion should be
    asynchronous in a real system rather than blocking an HTTP request.
    (See the "Known limitations" section of the README.)
    """
    if not docs:
        return 0
    get_store().add_documents(docs)
    return len(docs)


def search(query: str, k: int = 4) -> List[Document]:
    """
    Embed the query with the SAME model used at ingestion, then find the k
    nearest chunks.

    A NOTE ON MULTI-TENANCY, because it's the most important thing here:

    If this service were multi-tenant, the tenant filter would have to be
    applied *inside* this search call, not to its results:

        get_store().similarity_search(query, k=k, filter={"tenant": tid})

    Filtering afterwards is a real bug, not a style preference. Retrieve the
    global top-4 and then drop other tenants' chunks, and you're left with
    fewer than 4 — sometimes zero — and the user sees a worse answer for no
    visible reason. Worse, one mistake in that post-filter leaks another
    tenant's data.

    Push the filter into the query. Always.
    """
    return get_store().similarity_search(query, k=k)


def count() -> int:
    """How many chunks are indexed. Useful for a health check."""
    try:
        return get_store()._collection.count()
    except Exception:
        return -1
