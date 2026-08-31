"""
Request and response shapes.

WHY PYDANTIC MODELS INSTEAD OF PLAIN DICTS
------------------------------------------
Three things you get for one declaration:

  1. Validation at the boundary. Bad input is rejected before it reaches
     your logic, with a useful error message, automatically.
  2. A typed contract. The shape is written down once, and both the code and
     the docs read from it.
  3. Free OpenAPI docs at /docs — FastAPI generates them from these classes.

This is the same argument as any typed contract: the schema is the
documentation, and it can't drift from the implementation because it *is*
the implementation.
"""

from typing import List

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str = Field(..., description="Raw document text to index.")
    source: str = Field("untitled", description="A label, e.g. a filename.")


class IngestResponse(BaseModel):
    source: str
    chunks_added: int


class AskRequest(BaseModel):
    question: str
    # thread_id is what the checkpointer keys on. Same id = same conversation
    # and shared memory; different id = a clean slate.
    thread_id: str = Field("default", description="Conversation thread id.")


class SourceChunk(BaseModel):
    text: str
    source: str
    chunk_index: int


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]

    # These two fields exist so the caller can see how the graph behaved,
    # not just what it concluded. retrieval_attempts > 1 means the cycle
    # fired — the first retrieval was graded insufficient and we went back.
    #
    # Exposing this is an observability decision: a system whose reasoning
    # you can't inspect is a system you can't debug in production.
    retrieval_attempts: int
    context_grade: str
