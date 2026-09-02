"""
The LangGraph. This is the heart of the project — read this file first.

WHAT THIS IS
------------
A graph with three nodes and one cycle:

        START
          |
          v
     [ retrieve ]  <-------------+
          |                      |
          v                      | "not good enough,
      [ grade ]                  |  try again wider"
          |                      |
          +----------------------+
          |
          | "good enough" (or out of attempts)
          v
     [ generate ]
          |
          v
         END

WHY A GRAPH AND NOT A CHAIN
---------------------------
A chain is a straight line: A -> B -> C. It cannot go backwards.

This workflow needs to go backwards. After retrieving documents we *check*
whether they actually contain enough to answer the question. If they don't,
we want to retrieve again with a wider net. That is a loop, and a straight
line fundamentally cannot express a loop.

That is the entire reason LangGraph exists. If you only remember one thing
from this file, remember that sentence — it is the answer to "why LangGraph
instead of a chain?"

WHY BOTHER GRADING AT ALL
-------------------------
Naive RAG retrieves once and hopes. If the retrieval was bad, the model
either invents an answer (a hallucination) or gives up. Grading makes the
failure *visible to the system* so it can do something about it, instead of
passing the problem to the user.

This is the same instinct as separating generation from verification: don't
trust a single pass.
"""

from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.llm import get_chat_model
from app.store import search

# How many times we are willing to go back and retrieve again.
# Without a cap, a bad question could loop forever. Every cycle in a graph
# needs a termination guarantee — this is ours.
MAX_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# 1. THE STATE
# ---------------------------------------------------------------------------
# Every node receives the state and returns a *partial* update to it.
# LangGraph merges that update in for you.
#
# Think of it as the shared scratchpad for one run of the graph.
class RAGState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    grade: str        # "sufficient" | "insufficient"
    attempts: int
    k: int            # how many chunks to retrieve; widens on retry


# ---------------------------------------------------------------------------
# 2. THE NODES
# ---------------------------------------------------------------------------
# A node is just a function: state in, partial state out. That's the whole
# contract. There is nothing magic about them.


def retrieve(state: RAGState) -> dict:
    """Find the chunks most similar to the question."""
    k = state.get("k") or settings.retrieval_k
    docs = search(state["question"], k=k)
    return {
        "documents": docs,
        "attempts": state.get("attempts", 0) + 1,
    }


def grade(state: RAGState) -> dict:
    """
    Ask the model whether the retrieved chunks are enough to answer.

    Note what we are NOT doing: we are not asking it to answer. We are asking
    a narrow yes/no question, because narrow questions are far more reliable
    than open ones. A grader with one job is a better grader.
    """
    if not state["documents"]:
        # Nothing retrieved at all. No need to spend a model call on this.
        return {"grade": "insufficient"}

    context = _format_docs(state["documents"])
    model = get_chat_model()

    prompt = (
        "You are grading whether some retrieved context is sufficient to "
        "answer a question. Do NOT answer the question.\n\n"
        f"QUESTION:\n{state['question']}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Reply with exactly one word: SUFFICIENT if the context contains "
        "enough information to answer the question, or INSUFFICIENT if it "
        "does not."
    )

    reply = _text(model.invoke(prompt)).strip().upper()
    verdict = "sufficient" if "SUFFICIENT" in reply and "INSUFFICIENT" not in reply else "insufficient"
    return {"grade": verdict}


def generate(state: RAGState) -> dict:
    """
    Answer the question using only the retrieved context.

    The two instructions in this prompt are doing all the work:
      1. "only the context"  -> stops it answering from training data
      2. "say you don't know" -> gives it a legal way out, so it doesn't
                                 have to invent something to satisfy us
    Without (2), a model asked to always answer will always answer.
    """
    if not state["documents"]:
        return {
            "answer": "I don't know — nothing in the indexed documents is "
                      "relevant to that question."
        }

    context = _format_docs(state["documents"])
    model = get_chat_model()

    prompt = (
        "Answer the question using ONLY the context below. If the context "
        "does not contain the answer, say \"I don't know\" and stop — do not "
        "use outside knowledge. Cite the source numbers you used, like [1].\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{state['question']}\n\n"
        "ANSWER:"
    )

    return {"answer": _text(model.invoke(prompt)).strip()}


def _text(message) -> str:
    """
    Get plain text out of a model reply.

    WHY THIS IS NOT JUST `.content`
    -------------------------------
    `.content` used to always be a string. On newer multimodal models it can
    be a *list of content blocks* instead - each block a dict like
    {"type": "text", "text": "..."} - because a reply may carry reasoning,
    images or tool calls alongside the text.

    So `.content.strip()` works on one provider and throws
    AttributeError: 'list' object has no attribute 'strip' on another. That
    is exactly the kind of breakage a provider-agnostic layer is supposed to
    absorb, so it gets absorbed here once rather than at every call site.
    """
    c = message.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for block in c:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(c)


def _format_docs(docs: List[Document]) -> str:
    """Number the chunks so the model can cite them."""
    return "\n\n".join(
        f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs)
    )


# ---------------------------------------------------------------------------
# 3. THE ROUTER  (this is what makes it a graph)
# ---------------------------------------------------------------------------
# A conditional edge is a function that returns the *name of the next node*.
# That's it. This one closes the loop.
def route_after_grade(state: RAGState) -> Literal["retrieve", "generate"]:
    if state["grade"] == "sufficient":
        return "generate"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        # Out of retries. Generate anyway — the prompt in generate() will
        # make it say "I don't know" rather than invent an answer.
        return "generate"
    return "retrieve"


def widen(state: RAGState) -> dict:
    """
    Retrying with the same k would retrieve the same chunks and grade the
    same way — an infinite loop that also achieves nothing. So each retry
    casts a wider net.
    """
    return {"k": (state.get("k") or settings.retrieval_k) * 2}


# ---------------------------------------------------------------------------
# 4. WIRING IT UP
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(RAGState)

    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("widen", widen)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")

    # The conditional edge. The dict maps the router's return value to a node.
    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {"retrieve": "widen", "generate": "generate"},
    )
    g.add_edge("widen", "retrieve")     # <-- THE CYCLE
    g.add_edge("generate", END)

    # The checkpointer saves state after every node, keyed by thread_id.
    # That gives you: multi-turn memory, crash recovery, and the ability to
    # pause and resume a run.
    #
    # InMemorySaver dies with the process. For anything real you'd use the
    # SQLite or Postgres saver — same interface, different backend.
    return g.compile(checkpointer=InMemorySaver())


# Build once at import. Compiling on every request would be wasteful.
graph = build_graph()


def ask(question: str, thread_id: str = "default") -> dict:
    """Run the graph for one question."""
    result = graph.invoke(
        {
            "question": question,
            "documents": [],
            "answer": "",
            "grade": "",
            "attempts": 0,
            "k": settings.retrieval_k,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    return result
