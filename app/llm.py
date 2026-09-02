"""
Model access, in one place.

WHY A WRAPPER INSTEAD OF CALLING A VENDOR SDK DIRECTLY
------------------------------------------------------
Because every other file then depends on *this* interface rather than on a
particular vendor. Change CHAT_MODEL in .env from google_genai:... to
openai:... and nothing else in the codebase changes.

That's dependency inversion — the same OOP principle from Slow Path ch. 16,
applied here. Depend on the abstraction, not the vendor.

This is not a hypothetical benefit. The project started on OpenAI and moved
to Google Gemini because Gemini's free tier needs no credit card. That
migration was two lines of .env and one dependency — no application code
changed at all. That is the entire argument for the indirection, and it
earned itself in about ten minutes.

It's also the honest answer to "how would you handle switching providers?"
You don't switch. You configure.
"""

from functools import lru_cache
from typing import List

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings

from app.config import settings


class LocalEmbeddings(Embeddings):
    """
    Embeddings that run on this machine, with no API key and no network call.

    WHY THIS EXISTS
    ---------------
    Every hosted embedding API is a dependency you carry into a live demo: a
    key that can be wrong, a rate limit that can be hit, a network that can
    drop. For embeddings specifically that dependency buys very little -
    the model is small, the work is cheap, and it runs fine on a CPU.

    fastembed runs a quantised ONNX model locally. First use downloads
    ~130 MB once and caches it; after that it is offline and instant.

    Implementing LangChain's Embeddings interface here rather than pulling in
    langchain_community is deliberate: the interface is two methods, and the
    package is large. Depend on the interface, not the package.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self._model.embed([text]))).tolist()

# NOTE ON VERSION DRIFT
# ---------------------
# LangChain's import paths move between major versions. If either factory
# fails to import, it has been relocated — check the current docs. What they
# do is simple: take a "provider:model" string and return an object with a
# standard interface. Nothing here depends on where they live, only on that.


@lru_cache(maxsize=1)
def get_chat_model():
    """
    Cached because constructing a client on every call is wasteful — it sets
    up HTTP connection pooling that we want to reuse.

    temperature=0 asks for the least random output the model will give. That
    matters most for the grader: a step whose whole job is to answer
    SUFFICIENT or INSUFFICIENT should not give different verdicts on
    identical input. Determinism is not guaranteed even at 0, but variance
    in a control-flow decision is worse than variance in prose.
    """
    return init_chat_model(settings.chat_model, temperature=0)


@lru_cache(maxsize=1)
def get_embeddings():
    """
    The embedding model turns text into a vector.

    IMPORTANT: the same model must be used for ingestion and for queries.
    Two different embedding models produce vectors in different spaces, so
    similarity between them is meaningless — even when the dimensions happen
    to match, which makes it a silent failure rather than a loud one.

    So changing EMBEDDING_MODEL means re-indexing everything. That's a real
    operational gotcha and a good thing to mention out loud.

    A "local:" prefix runs the model on this machine instead of calling an
    API - see LocalEmbeddings for why that is often the better trade.
    """
    spec = settings.embedding_model
    if spec.startswith("local:"):
        return LocalEmbeddings(spec.split(":", 1)[1] or "BAAI/bge-small-en-v1.5")
    return init_embeddings(spec)


@lru_cache(maxsize=1)
def embedding_dimension() -> int:
    """
    How many numbers are in one vector from this model.

    WHY THIS IS MEASURED RATHER THAN CONFIGURED
    -------------------------------------------
    Qdrant needs the vector size up front to create a collection, and it
    rejects anything of a different length. So a wrong number here means
    every insert fails.

    The obvious approach is to put the number in config. The problem is that
    config can then disagree with reality — someone swaps the embedding model
    and forgets the dimension, and the failure appears later, somewhere else,
    as a confusing rejection from the database.

    So we ask the model instead. One short call at startup, and the two can
    never drift apart. EMBEDDING_DIM is still available as an override for
    anyone who wants to skip the call and knows their number.

    The general principle: when a value can be derived from the system
    itself, deriving it beats configuring it. Configuration is for choices;
    this is a fact.
    """
    if settings.embedding_dim:
        return settings.embedding_dim
    return len(get_embeddings().embed_query("dimension probe"))
