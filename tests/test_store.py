"""
Tests for the vector store, against a real in-memory Qdrant.

WHY THIS CAN BE TESTED AT ALL
-----------------------------
Two things make it possible, and both were design decisions rather than
happy accidents:

  1. Qdrant's client runs in-memory. So this is not a mock pretending to be
     a vector store — it is the real client, the real collection creation,
     the real similarity search. Only the process boundary is missing.

  2. Embeddings come from one function, `get_embeddings()`, so a test can
     substitute a deterministic fake and never touch the network. If model
     access were scattered through the codebase there would be nothing to
     substitute, and this file could not exist.

That is the practical argument for putting I/O behind a single function:
not purity, but that it gives you exactly one seam to test through.
"""

from dataclasses import replace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app import store
from app.config import settings

FAKE_DIM = 32


@pytest.fixture(autouse=True)
def in_memory_store(monkeypatch):
    """
    Point the store at an in-memory Qdrant with fake embeddings.

    Deterministic embeddings matter: the same text must always produce the
    same vector, or search results would change between runs and these tests
    would flake.

    `Settings` is frozen, so we swap in a modified copy rather than mutating
    it — the immutability that makes settings safe to trust in real code
    stays intact.
    """
    monkeypatch.setattr(
        store, "settings",
        replace(settings, qdrant_url="", collection="test_grounded"),
    )
    monkeypatch.setattr(
        store, "get_embeddings", lambda: DeterministicFakeEmbedding(size=FAKE_DIM)
    )
    # store asks llm.embedding_dimension() rather than reading config, so
    # that is the seam to substitute here.
    monkeypatch.setattr(store, "embedding_dimension", lambda: FAKE_DIM)
    store.reset_store()
    yield
    store.reset_store()


def test_empty_add_is_a_noop():
    """Adding nothing should not create a collection or raise."""
    assert store.add_documents([]) == 0


def test_documents_are_indexed_and_retrievable():
    docs = [
        Document(page_content="Refunds are processed within five working days.",
                 metadata={"source": "policy.md", "chunk_index": 0}),
        Document(page_content="Payouts above 5000 rupees require manual review.",
                 metadata={"source": "policy.md", "chunk_index": 1}),
    ]
    assert store.add_documents(docs) == 2

    results = store.search("refund timing", k=2)
    assert len(results) == 2
    assert all(isinstance(d, Document) for d in results)


def test_metadata_survives_the_round_trip():
    """
    The whole point of returning sources is that an answer can cite them.
    If metadata is lost on the way into the store, citation is impossible
    and the service's core promise quietly breaks.
    """
    store.add_documents([
        Document(page_content="Ledgers are append-only.",
                 metadata={"source": "handbook.md", "chunk_index": 7}),
    ])
    got = store.search("ledgers", k=1)[0]
    assert got.metadata["source"] == "handbook.md"
    assert got.metadata["chunk_index"] == 7


def test_k_limits_the_number_of_results():
    store.add_documents([
        Document(page_content=f"Document number {i}.",
                 metadata={"source": "many.md", "chunk_index": i})
        for i in range(10)
    ])
    assert len(store.search("document", k=3)) == 3


def test_collection_is_created_with_the_configured_dimension():
    """
    Vector size must match the embedding model or Qdrant rejects every
    insert. Getting this wrong is a loud failure rather than a silent one,
    which is exactly what you want — but only if it's actually wired up.
    """
    store.add_documents([
        Document(page_content="anything", metadata={"source": "s", "chunk_index": 0})
    ])
    client = store.get_store().client
    info = client.get_collection("test_grounded")
    assert info.config.params.vectors.size == FAKE_DIM
