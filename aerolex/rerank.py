"""Feature #6 — Two-Stage Cross-Encoder Re-Ranking.

First-stage retrieval (BM25 + dense + RRF) is fast but coarse: it scores query
and document *independently*. A cross-encoder instead reads the query and a
candidate *together*, so it models fine-grained relevance and suppresses
distractor chunks. We only run it over the small fused candidate pool.
"""
from __future__ import annotations

from . import config


class CrossEncoderReranker:
    def __init__(self, model_name: str = config.RERANK_MODEL):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[str, str]],
               top_k: int = config.RERANK_TOP_K) -> list[tuple[str, float]]:
        """Re-score ``(chunk_id, text)`` candidates; return top-k (chunk_id, score).

        The cross-encoder jointly encodes (query, text) pairs — deep
        cross-attention — which is far more precise than cosine on the pool.
        """
        if not candidates:
            return []
        pairs = [(query, text) for _, text in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip((cid for cid, _ in candidates), scores),
            key=lambda cs: -float(cs[1]),
        )
        return [(cid, float(score)) for cid, score in ranked[:top_k]]
