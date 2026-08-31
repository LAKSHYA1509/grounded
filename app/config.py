"""
Settings, read from environment variables.

WHY THIS FILE EXISTS
--------------------
Nothing in the codebase should hardcode a model name, a chunk size, or an
API key. Two reasons:

  1. Secrets in git are permanent. Even if you delete them in a later commit,
     they stay in the history forever.
  2. The values you want to tune are exactly the ones you want to change
     without editing code — chunk size, k, model, which Qdrant to talk to.

The pattern: one place reads the environment, everything else imports from
here. If you ever need to know what this service can be configured with,
this file is the complete answer.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # reads .env into the environment, if the file exists


@dataclass(frozen=True)
class Settings:
    # --- provider ---------------------------------------------------------
    # provider:model strings. Swapping these is the whole point of using a
    # standard interface instead of a vendor SDK directly.
    #
    #   google_genai:gemini-2.5-flash-lite   <- default; free tier, no card
    #   openai:gpt-4o-mini
    #   anthropic:claude-haiku-4-5-20251001
    #
    # Google is the default because its free tier needs no credit card and
    # does not expire. Nothing in the application code knows or cares.
    chat_model: str = os.getenv("CHAT_MODEL", "google_genai:gemini-2.5-flash-lite")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "google_genai:gemini-embedding-001"
    )

    # Vector size. Leave at 0 to measure it from the model at startup, which
    # is the safe default - see llm.embedding_dimension() for why measuring
    # beats configuring here. Set it only to skip that one startup call.
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "0"))

    # --- chunking ---------------------------------------------------------
    # 800 characters is a reasonable default: big enough to hold a complete
    # thought, small enough that retrieval stays precise.
    #
    # The overlap matters more than people expect. Without it, a sentence
    # that straddles a boundary is cut in half and neither chunk contains
    # the whole idea. With overlap, at least one chunk keeps it intact.
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # --- retrieval --------------------------------------------------------
    # How many chunks to pull. Too few and you miss the answer; too many and
    # the real answer gets diluted by irrelevant text.
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))

    # --- vector store -----------------------------------------------------
    # Leave QDRANT_URL empty for in-memory mode: no server, no setup. That's
    # what makes `pytest` and a fresh clone work with zero infrastructure.
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    collection: str = os.getenv("COLLECTION", "grounded")

    # --- access control ---------------------------------------------------
    # If set, every request must carry it as the X-API-Key header.
    #
    # This exists because the deployed instance is public and holds a model
    # provider key. Without a gate, anyone who finds the URL can spend your
    # quota on /ask. An unauthenticated endpoint that calls a metered API is
    # not an open demo, it's an open wallet.
    #
    # Empty means open — which is fine locally and NOT fine in production.
    # The /health endpoint stays open either way so a platform can probe it.
    api_key: str = os.getenv("API_KEY", "")

    # --- server -----------------------------------------------------------
    # Read from the environment so the same image runs anywhere: Hugging Face
    # Spaces probes 7860, Render and Cloud Run inject their own $PORT.
    port: int = int(os.getenv("PORT", "7860"))


settings = Settings()
