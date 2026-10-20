"""Feature #4 — In-Memory Serverless Retrieval (pure NumPy).

No external vector database: the entire corpus lives in a single ``(N, D)``
float32 matrix. Cosine similarity is just a normalized dot product, which keeps
the retrieval mathematically transparent — exactly what we want to *teach*.
"""
from __future__ import annotations

import numpy as np

from . import config
from .chunking import Chunk, fingerprint_chunks


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (so dot product == cosine similarity)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class InMemoryVectorStore:
    """A tiny, transparent vector store backed by one NumPy matrix."""

    def __init__(self, ids: list[str], matrix: np.ndarray, *,
                 corpus_fingerprint: str | None = None):
        if len(ids) != len(matrix):
            raise ValueError("ids and matrix must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("Vector store contains duplicate IDs")
        self.ids = list(ids)
        self.matrix = l2_normalize(matrix)  # store normalized once
        self.corpus_fingerprint = corpus_fingerprint

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], matrix: np.ndarray) -> "InMemoryVectorStore":
        return cls([c.chunk_id for c in chunks], matrix,
                   corpus_fingerprint=fingerprint_chunks(chunks))

    def validate_chunks(self, chunks: list[Chunk]) -> None:
        if (self.ids != [c.chunk_id for c in chunks]
                or self.corpus_fingerprint != fingerprint_chunks(chunks)):
            raise ValueError(
                "Embeddings do not match the current chunks (or lack corpus metadata). "
                "Rebuild embeddings with InMemoryVectorStore.from_chunks(chunks, matrix).")

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
        if top_k <= 0:
            return []
        # argpartition for O(N) top-k, then sort just the winners.
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.ids[i], float(scores[i])) for i in idx]

    def save(self, path=None) -> None:
        path = path or config.EMBEDDINGS_CACHE
        if self.corpus_fingerprint is None:
            raise ValueError("Use from_chunks() before saving a corpus-validated vector cache")
        np.savez_compressed(path, ids=np.array(self.ids, dtype=str), matrix=self.matrix,
                            corpus_fingerprint=np.array(self.corpus_fingerprint))

    @classmethod
    def load(cls, path=None) -> "InMemoryVectorStore":
        path = path or config.EMBEDDINGS_CACHE
        with np.load(path, allow_pickle=False) as data:
            if "corpus_fingerprint" not in data:
                raise ValueError("Legacy vector cache lacks corpus metadata; rebuild embeddings")
            return cls(list(data["ids"]), data["matrix"],
                       corpus_fingerprint=str(data["corpus_fingerprint"].item()))
