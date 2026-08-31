"""
Splitting documents into chunks.

WHY CHUNK AT ALL
----------------
Two reasons, and it's worth being able to state both:

  1. You cannot fit a 200-page document into a prompt. There's a context
     limit, and even where there isn't, it costs money per token.

  2. More subtly: retrieval is only useful if it's *precise*. If your whole
     document is one chunk, "retrieve the relevant part" returns everything,
     which is the same as retrieving nothing useful. Chunking is what makes
     "relevant part" a meaningful idea.

WHY CHUNKING IS THE MOST COMMON RAG FAILURE
-------------------------------------------
Because a bad split destroys information that no later stage can recover.
If the answer to a question spans a boundary and neither chunk contains it
whole, then no amount of clever retrieval or a better model will help — the
information isn't in any single chunk to be found.

That's why overlap exists, and why the splitter below tries paragraph breaks
before sentence breaks before arbitrary character positions: split at the
least damaging place available.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def make_splitter() -> RecursiveCharacterTextSplitter:
    """
    "Recursive" means: try the separators in order, and only fall through to
    a more damaging one if the chunk is still too big.

    ["\\n\\n", "\\n", ". ", " ", ""] reads as:
        prefer paragraph breaks,
        then line breaks,
        then sentence ends,
        then word gaps,
        and only as a last resort cut mid-word.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_text(text: str, source: str = "untitled") -> List[Document]:
    """
    Turn raw text into Documents, each carrying metadata.

    The metadata is not decoration. `source` is what lets an answer say
    "according to handbook.md" instead of just asserting something, and it's
    what a tenant filter would key off in a multi-tenant system.
    """
    if not text or not text.strip():
        return []

    splitter = make_splitter()
    chunks = splitter.split_text(text)

    return [
        Document(
            page_content=chunk,
            metadata={"source": source, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]
