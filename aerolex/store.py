"""Feature #4 — In-Memory Serverless Retrieval (pure NumPy).

No external vector database: the entire corpus lives in a single ``(N, D)``
float32 matrix. Cosine similarity is just a normalized dot product, which keeps
the retrieval mathematically transparent — exactly what we want to *teach*.
"""
from __future__ import annotations

import numpy as np

from . import config


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (so dot product == cosine similarity)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class InMemoryVectorStore:
    """A tiny, transparent vector store backed by one NumPy matrix."""

    def __init__(self, ids: list[str], matrix: np.ndarray):
        if len(ids) != len(matrix):
            raise ValueError("ids and matrix must have the same length")
        self.ids = list(ids)
        self.matrix = l2_normalize(matrix)  # store normalized once

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query_vec: np.ndarray, top_k: int = config.DENSE_TOP_K
               ) -> list[tuple[str, float]]:
        """Return the ``top_k`` (chunk_id, cosine_score) pairs, best first."""
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        norm = np.linalg.norm(q)
        if norm:
            q = q / norm
        scores = self.matrix @ q  # (N,) cosine similarities
        top_k = min(top_k, len(scores))
        # argpartition for O(N) top-k, then sort just the winners.
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.ids[i], float(scores[i])) for i in idx]

    def save(self, path=None) -> None:
        path = path or config.EMBEDDINGS_CACHE
        np.savez_compressed(path, ids=np.array(self.ids, dtype=object), matrix=self.matrix)

    @classmethod
    def load(cls, path=None) -> "InMemoryVectorStore":
        path = path or config.EMBEDDINGS_CACHE
        data = np.load(path, allow_pickle=True)
        return cls(list(data["ids"]), data["matrix"])
