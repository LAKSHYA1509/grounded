"""
Tests for chunking.

WHY THESE TESTS AND NOT OTHERS
------------------------------
Chunking is the one part of this system that is pure logic — no network, no
model, no API key. So it's the part that can be tested properly and quickly,
and therefore the part that should be.

That's a general rule worth stating in an interview: push the logic you want
to test away from the parts you can't control. The graph is hard to test
because it calls a model; the splitter isn't, because it doesn't.
"""

from app.chunking import chunk_text


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_one_chunk():
    docs = chunk_text("A single short sentence.", source="test.md")
    assert len(docs) == 1
    assert docs[0].page_content == "A single short sentence."


def test_metadata_is_attached():
    """Source and index must survive chunking — they're how answers cite."""
    docs = chunk_text("Some text about ledgers.", source="handbook.md")
    assert docs[0].metadata["source"] == "handbook.md"
    assert docs[0].metadata["chunk_index"] == 0


def test_long_text_splits_into_multiple_chunks():
    # Well past the default 800-char chunk size.
    text = ("Paragraph about payouts. " * 20 + "\n\n") * 5
    docs = chunk_text(text, source="long.md")
    assert len(docs) > 1


def test_chunk_indices_are_sequential():
    text = ("Paragraph about payouts. " * 20 + "\n\n") * 5
    docs = chunk_text(text, source="long.md")
    indices = [d.metadata["chunk_index"] for d in docs]
    assert indices == list(range(len(docs)))


def test_chunks_overlap():
    """
    The property that matters: consecutive chunks share some text, so an idea
    spanning a boundary survives intact in at least one of them.

    This test is the reason CHUNK_OVERLAP exists. If someone sets it to 0 to
    'save tokens', this goes red and tells them what they broke.
    """
    text = " ".join(f"word{i}" for i in range(600))
    docs = chunk_text(text, source="overlap.md")
    assert len(docs) > 1

    first_tail = docs[0].page_content.split()[-5:]
    second = docs[1].page_content
    assert any(w in second for w in first_tail)
