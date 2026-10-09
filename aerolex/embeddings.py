"""Feature #3 — Dense Semantic Search embeddings (Gemini).

Wraps ``gemini-embedding-001`` with batching, retries, and asymmetric task
types (documents vs. queries). Vectors are returned as float32 NumPy arrays and
left un-normalized — the vector store normalizes them (important because
reduced Matryoshka dimensions are not unit-length by default).
"""
from __future__ import annotations

import re
import time

import numpy as np

from . import config


class GeminiEmbedder:
    def __init__(self, api_key: str | None = None,
                 model: str = config.EMBED_MODEL, dim: int = config.EMBED_DIM):
        from google import genai

        key = api_key or config.load_api_key()
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set — required for embeddings. "
                "Copy .env.example to .env and add your key (https://ai.dev)."
            )
        self.client = genai.Client(api_key=key)
        self.model = model
        self.dim = dim

    def _embed(self, texts: list[str], task_type: str,
               batch_size: int = 50, max_retries: int = 8) -> np.ndarray:
        from google.genai import types

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            for attempt in range(max_retries):
                try:
                    resp = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=self.dim,
                        ),
                    )
                    vectors.extend(e.values for e in resp.embeddings)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    self._backoff(e, attempt)
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def _backoff(err: Exception, attempt: int) -> None:
        """Sleep appropriately: honor the 429 retry delay, short backoff for 503."""
        msg = repr(err)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"(\d+)s'", msg)
            delay = float(m.group(1)) if m else 55.0
            time.sleep(min(delay + 3.0, 90.0))   # free-tier embeddings: ~100/min
        else:  # 503 UNAVAILABLE or transient network
            time.sleep(2 * (attempt + 1))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed corpus chunks (asymmetric: RETRIEVAL_DOCUMENT)."""
        return self._embed(list(texts), "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single search query (asymmetric: RETRIEVAL_QUERY)."""
        return self._embed([text], "RETRIEVAL_QUERY")[0]
