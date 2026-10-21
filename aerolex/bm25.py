"""Feature #2 — Lexical Keyword Search (BM25).

Dense search blurs exact tokens; statute codes must match *exactly*. BM25 is a
sparse bag-of-words scorer that nails references like ``§ 260.6``, ``261/2004``,
or ``14 CFR``. The custom tokenizer deliberately preserves those tokens.
"""
from __future__ import annotations

import re

from . import config

# Keep words, the section sign, and dotted/slashed numeric refs ("260.6",
# "261/2004", "14") as single tokens so exact statute matching works.
_TOKEN_RE = re.compile(r"[a-z]+|\u00A7|\d+(?:[./-]\d+)*")

def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Thin wrapper over ``rank_bm25.BM25Okapi`` with a statute-aware tokenizer."""

    def __init__(self, ids: list[str], texts: list[str]):
        from rank_bm25 import BM25Okapi

        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        self.ids = list(ids)
        self._tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokens)

    def search(self, query_tokens: list[str], top_k: int = config.BM25_TOP_K
               ) -> list[tuple[str, float]]:
        """Return the ``top_k`` (chunk_id, bm25_score) pairs, best first."""
        scores = self._bm25.get_scores(query_tokens)
        top_k = min(top_k, len(scores))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.ids[i], float(scores[i])) for i in order]
