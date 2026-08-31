"""
Gradio front-end, for the Hugging Face Spaces deployment.

WHY THIS FILE EXISTS
--------------------
The real interface of this project is the FastAPI service in `app/`. This is
a thin UI over the same code, and it exists for one reason: Hugging Face
Docker Spaces became a paid feature, and Gradio Spaces are the remaining
free, no-credit-card way to put a running instance behind a public URL.

So this is a deployment constraint, not a design preference. Worth being
straight about that rather than pretending a UI was always the plan.

WHAT IT IS AND ISN'T
--------------------
It imports `app.graph`, `app.chunking` and `app.store` — the identical code
paths the API uses. The graph, the retrieve-grade-regenerate cycle, the
Qdrant store and the prompts are all shared. Nothing is reimplemented here,
and if this file were deleted the service would be unaffected.

That's the payoff from keeping the route handlers thin. Because no business
logic ever lived in `app/main.py`, a completely different front-end is about
sixty lines. Had the logic sat in the HTTP handlers, this would have meant
either duplicating it or refactoring under time pressure.

WHY THERE IS AN AUTH GATE HERE TOO
----------------------------------
A public Space holds a model provider key, and every question spends quota.
Gradio takes a simple username/password pair, which is the equivalent of the
X-API-Key header on the API. Set SPACE_USER and SPACE_PASS on the Space and
it locks; leave them unset and it's open, which is fine locally.
"""

import os

import gradio as gr

from app.chunking import chunk_text
from app.graph import ask
from app.store import add_documents, count

EXAMPLE_DOC = """Refunds are processed within 5 working days of approval.
Payouts above INR 5,000 require manual review by an operations lead.
Refunds are issued to the original payment method only.
A payout that fails three times is moved to a manual reconciliation queue."""


def do_ingest(text: str, source: str):
    """Chunk, embed and index a document."""
    if not text or not text.strip():
        return "Nothing to ingest — paste some text first."
    try:
        docs = chunk_text(text, source=source or "pasted")
        added = add_documents(docs)
        return f"Indexed {added} chunk(s) from '{source or 'pasted'}'. Total in store: {count()}."
    except Exception as e:
        # Show the real error. A demo that fails silently is worse than one
        # that fails loudly — you cannot debug what you cannot see.
        return f"Ingestion failed: {type(e).__name__}: {e}"


def do_ask(question: str):
    """
    Run the graph and show not just the answer, but how it was reached.

    The retrieval attempts and the grade are surfaced deliberately. The whole
    point of this project is that the system checks its own retrieval, and a
    UI that hid that would be hiding the only interesting part.
    """
    if not question or not question.strip():
        return "Ask something first.", "", ""
    try:
        result = ask(question)
    except Exception as e:
        return f"Failed: {type(e).__name__}: {e}", "", ""

    answer = result.get("answer", "")
    attempts = result.get("attempts", 0)
    grade = result.get("grade", "unknown")

    trace = f"**Retrieval attempts:** {attempts}  \n**Context graded:** {grade}"
    if attempts > 1:
        trace += "\n\nThe first retrieval was graded insufficient, so the graph looped back and retrieved again with a wider `k`. That backward edge is why this is a graph and not a chain."

    sources = "\n\n---\n\n".join(
        f"**[{i + 1}] {d.metadata.get('source', 'unknown')}** "
        f"(chunk {d.metadata.get('chunk_index', '?')})\n\n{d.page_content}"
        for i, d in enumerate(result.get("documents", []))
    ) or "_No sources retrieved._"

    return answer, trace, sources


with gr.Blocks(title="grounded") as demo:
    gr.Markdown(
        """
        # grounded

        A document Q&A service that **refuses to answer without sources** — and checks
        its own retrieval before it answers.

        Most RAG retrieves once and hopes. This grades the retrieved context first, and
        if it isn't good enough, loops back and retrieves again with a wider net.
        That backward edge is why it's built as a graph rather than a chain.

        `FastAPI` · `LangGraph` · `Qdrant` · `Gemini` —
        [source on GitHub](https://github.com/LAKSHYA1509/grounded)
        """
    )

    with gr.Tab("1. Add a document"):
        gr.Markdown("Paste any text. It gets split into overlapping chunks, embedded, and indexed.")
        doc_text = gr.Textbox(label="Document text", lines=10, value=EXAMPLE_DOC)
        doc_source = gr.Textbox(label="Source label", value="policy.md")
        ingest_btn = gr.Button("Index it", variant="primary")
        ingest_out = gr.Markdown()
        ingest_btn.click(do_ingest, [doc_text, doc_source], ingest_out)

    with gr.Tab("2. Ask about it"):
        question = gr.Textbox(
            label="Question",
            placeholder="How long do refunds take?",
        )
        ask_btn = gr.Button("Ask", variant="primary")
        answer_out = gr.Markdown(label="Answer")
        gr.Markdown("### How it got there")
        trace_out = gr.Markdown()
        gr.Markdown("### Sources it used")
        sources_out = gr.Markdown()
        ask_btn.click(do_ask, question, [answer_out, trace_out, sources_out])

        gr.Examples(
            examples=[
                "How long do refunds take?",
                "What happens if a payout fails repeatedly?",
                "What is the company holiday policy?",  # deliberately unanswerable
            ],
            inputs=question,
        )
        gr.Markdown(
            "_The third example is deliberately not in the document. "
            "A grounded system should say it doesn't know rather than invent an answer._"
        )


if __name__ == "__main__":
    user, password = os.getenv("SPACE_USER"), os.getenv("SPACE_PASS")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        auth=(user, password) if user and password else None,
    )
