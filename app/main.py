"""
The HTTP layer.

DESIGN NOTE
-----------
Notice how thin this file is. Routes parse input, call one function, and
shape the response. All the actual thinking lives in graph.py and store.py.

That separation is deliberate and worth being able to defend: the graph
should be testable without starting a web server, and the web server should
be replaceable (with a CLI, a queue consumer, a Lambda handler) without
touching the graph. A route handler that contains business logic is a route
handler you can only test through HTTP.
"""

from fastapi import FastAPI, HTTPException

from app.chunking import chunk_text
from app.graph import ask
from app.models import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    SourceChunk,
)
from app.store import add_documents, count

app = FastAPI(
    title="grounded",
    description="A RAG service that won't answer without sources.",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Liveness plus one useful fact: is anything actually indexed?"""
    return {"status": "ok", "chunks_indexed": count()}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """
    Chunk, embed, index.

    This is synchronous, which is fine for a document you paste in and wrong
    for a 300-page PDF — embedding is slow, and an HTTP request shouldn't
    wait on it. The real shape is: accept, enqueue, return a job id, and let
    a worker do the embedding. Called out in the README as a known limitation
    rather than pretended away.
    """
    docs = chunk_text(req.text, source=req.source)
    if not docs:
        raise HTTPException(status_code=400, detail="No text to ingest.")

    added = add_documents(docs)
    return IngestResponse(source=req.source, chunks_added=added)


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """
    Run the graph.

    The response includes the source chunks, not just the answer. That is a
    deliberate product decision: an answer you cannot verify is an answer you
    cannot trust. Returning sources is what lets a caller check whether the
    model actually used the documents or wandered off.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    result = ask(req.question, thread_id=req.thread_id)

    return AskResponse(
        question=req.question,
        answer=result["answer"],
        sources=[
            SourceChunk(
                text=d.page_content,
                source=d.metadata.get("source", "unknown"),
                chunk_index=d.metadata.get("chunk_index", -1),
            )
            for d in result.get("documents", [])
        ],
        retrieval_attempts=result.get("attempts", 0),
        context_grade=result.get("grade", "unknown"),
    )
