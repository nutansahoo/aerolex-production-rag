"""Central configuration for the AeroLex RAG pipeline.

Everything that a workshop participant might want to tweak lives here:
model names, retrieval sizes, chunking budgets, and cache locations.
Keeping these in one place means the notebook stays clean and readable.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Pre-computed artifacts (so the live notebook never re-parses or re-embeds).
CHUNKS_CACHE = CACHE_DIR / "chunks.json"
EMBEDDINGS_CACHE = CACHE_DIR / "embeddings.npz"

# SQLite stores for the stateful features.
PROFILE_DB = CACHE_DIR / "user_profiles.sqlite3"   # long-term memory (feature #9)
FEEDBACK_DB = CACHE_DIR / "feedback.sqlite3"        # FLAIR feedback (feature #10)

# --------------------------------------------------------------------------- #
# Gemini models  (names change often — verify with scripts/list_models.py)
# --------------------------------------------------------------------------- #
EMBED_MODEL = "gemini-embedding-001"
# gemini-embedding-001 supports Matryoshka dims (768 / 1536 / 3072).
# 768 keeps the in-memory matrix small and cosine fast for a workshop.
EMBED_DIM = 768
GEN_MODEL = "gemini-flash-latest"

# Cross-encoder re-ranker runs locally (Gemini has no cross-encoder).
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --------------------------------------------------------------------------- #
# Retrieval hyper-parameters
# --------------------------------------------------------------------------- #
BM25_TOP_K = 20        # candidates from sparse search
DENSE_TOP_K = 20       # candidates from dense search
RRF_K = 60             # reciprocal rank fusion constant (standard default)
RRF_TOP_K = 20         # size of the fused candidate pool handed to the re-ranker
RERANK_TOP_K = 5       # final chunks kept as grounding context

# --------------------------------------------------------------------------- #
# Hierarchical chunking budgets (words, not exact model tokens)
# --------------------------------------------------------------------------- #
CHUNK_MAX_WORDS = 320
CHUNK_OVERLAP_WORDS = 40

REFERENCE_EXPANSION_ENABLED = True
REFERENCE_MAX_HOPS = 1
REFERENCE_MAX_INCOMING = 2
REFERENCE_MAX_ADDED_CHUNKS = 20
REFERENCE_MAX_CONTEXT_WORDS = 6000


def load_api_key() -> str | None:
    """Load GEMINI_API_KEY from the environment (and a local .env if present).

    Returns None when no key is configured so callers can fall back to an
    offline/templated mode instead of crashing mid-workshop.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass  # python-dotenv is optional; real env vars still work.
    return os.getenv("GEMINI_API_KEY") or None
