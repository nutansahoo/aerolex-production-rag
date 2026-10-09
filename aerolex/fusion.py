"""Feature #5 — Reciprocal Rank Fusion (RRF).

BM25 scores and cosine scores live on incomparable scales, so we fuse by *rank*
instead of by raw score. Each list contributes ``1 / (k + rank)`` to an item's
fused score; summing across lists rewards documents that rank well in *either*
retriever. ``k = 60`` is the standard constant from the original RRF paper.
"""
from __future__ import annotations

from . import config


def _as_id_ranking(ranked) -> list[str]:
    """Accept either a list of ids or a list of (id, score) pairs."""
    out = []
    for item in ranked:
        out.append(item[0] if isinstance(item, (tuple, list)) else item)
    return out


def reciprocal_rank_fusion(
    ranked_lists,
    k: int = config.RRF_K,
    top_k: int | None = config.RRF_TOP_K,
) -> list[tuple[str, float]]:
    """Fuse several ranked lists into one (id, fused_score) list, best first.

    Args:
        ranked_lists: iterable of ranked results; each result is either a list
            of ids (best first) or a list of (id, score) pairs.
        k: RRF constant (dampens the influence of top ranks).
        top_k: truncate the fused list to this many items (None = keep all).
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, _id in enumerate(_as_id_ranking(ranked)):
            fused[_id] = fused.get(_id, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return ordered[:top_k] if top_k else ordered
