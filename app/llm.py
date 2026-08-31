"""
Model access, in one place.

WHY A WRAPPER INSTEAD OF CALLING THE SDK DIRECTLY
-------------------------------------------------
Because every other file then depends on *this* interface rather than on
OpenAI specifically. Change CHAT_MODEL in .env from openai:... to
anthropic:... and nothing else in the codebase changes.

That's dependency inversion — the same OOP principle from Slow Path ch. 16,
applied here. Depend on the abstraction, not the vendor.

It's also the honest answer to "how would you handle switching providers?"
in an interview: you don't switch, you configure.
"""

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

from app.config import settings

# NOTE ON VERSION DRIFT
# ---------------------
# LangChain's import paths move between major versions. If `init_chat_model`
# fails to import, it has been relocated — check the current docs. What it
# does is simple: takes a "provider:model" string and returns a chat model
# object with a standard .invoke() method. Nothing here depends on where it
# lives, only on that behaviour.


@lru_cache(maxsize=1)
def get_chat_model():
    """
    Cached because constructing a client on every call is wasteful — it sets
    up HTTP connection pooling that we want to reuse.
    """
    return init_chat_model(settings.chat_model, temperature=0)


@lru_cache(maxsize=1)
def get_embeddings():
    """
    The embedding model turns text into a vector.

    IMPORTANT: the same model must be used for ingestion and for queries.
    Two different embedding models produce vectors in different spaces, so
    similarity between them is meaningless. If you ever change
    EMBEDDING_MODEL, you have to re-index everything.

    That's a real operational gotcha and a good thing to mention out loud.
    """
    return OpenAIEmbeddings(model=settings.embedding_model)
