"""
Settings, read from environment variables.

WHY THIS FILE EXISTS
--------------------
Nothing in the codebase should hardcode a model name, a chunk size, or an
API key. Two reasons:

  1. Secrets in git are permanent. Even if you delete them in a later commit,
     they stay in the history forever.
  2. The values you want to tune are exactly the ones you want to change
     without editing code — chunk size, k, model.

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
    # A provider:model string. Swapping this is the whole point of using a
    # standard interface instead of a vendor SDK directly.
    #   "openai:gpt-4o-mini"
    #   "anthropic:claude-haiku-4-5-20251001"
    #   "google_genai:gemini-2.0-flash"
    chat_model: str = os.getenv("CHAT_MODEL", "openai:gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

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

    # --- storage ----------------------------------------------------------
    persist_dir: str = os.getenv("PERSIST_DIR", "./.chroma")
    collection: str = os.getenv("COLLECTION", "grounded")


settings = Settings()
