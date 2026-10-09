"""Feature #10 — Adaptive Feedback Learning (FLAIR push-pull).

We log feedback triplets ``(query, chunk_id, signal)`` where ``signal`` is +1
(helpful) or -1 (unhelpful). At retrieval time a chunk's score is nudged by its
accumulated votes — but each past vote is *weighted by how similar its query is
to the current query*. That makes feedback generalize to unseen-but-similar
queries (the "push-pull"), with no model fine-tuning.
"""
from __future__ import annotations

import sqlite3
import time

import numpy as np

from . import config


def _vec_to_blob(vec) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class FeedbackStore:
    """Persisted feedback triplets + a query-aware push-pull re-scorer."""

    def __init__(self, db_path=None):
        self.db_path = str(db_path or config.FEEDBACK_DB)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                   query      TEXT NOT NULL,
                   query_vec  BLOB NOT NULL,
                   chunk_id   TEXT NOT NULL,
                   signal     INTEGER NOT NULL,
                   created_at REAL NOT NULL
               )"""
        )
        self._conn.commit()

    def record(self, query: str, query_vec, chunk_id: str, signal: int) -> None:
        if signal not in (-1, 1):
            raise ValueError("signal must be +1 (helpful) or -1 (unhelpful)")
        self._conn.execute(
            "INSERT INTO feedback VALUES (?,?,?,?,?)",
            (query, _vec_to_blob(query_vec), chunk_id, int(signal), time.time()),
        )
        self._conn.commit()

    def _rows(self):
        return self._conn.execute(
            "SELECT query_vec, chunk_id, signal FROM feedback"
        ).fetchall()

    def vote_scores(self, query_vec, sim_threshold: float = 0.5) -> dict[str, float]:
        """Aggregate query-weighted votes per chunk for the *current* query.

        Each stored vote contributes ``cos(current_query, past_query) * signal``
        but only when that cosine clears ``sim_threshold`` (so unrelated
        feedback is ignored).
        """
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        qn = np.linalg.norm(q) or 1.0
        q = q / qn
        votes: dict[str, float] = {}
        for blob, chunk_id, signal in self._rows():
            pv = _blob_to_vec(blob)
            pn = np.linalg.norm(pv) or 1.0
            sim = float(q @ (pv / pn))
            if sim >= sim_threshold:
                votes[chunk_id] = votes.get(chunk_id, 0.0) + sim * signal
        return votes

    def apply(
        self,
        ranked: list[tuple[str, float]],
        query_vec,
        alpha: float = 0.1,
        prune_at: float = -0.5,
        sim_threshold: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Re-score a ranked ``(chunk_id, score)`` list with push-pull votes.

        Chunks whose aggregated vote drops below ``prune_at`` are removed;
        the rest are nudged by ``alpha * vote`` and re-sorted.
        """
        votes = self.vote_scores(query_vec, sim_threshold)
        adjusted = []
        for chunk_id, score in ranked:
            v = votes.get(chunk_id, 0.0)
            if v <= prune_at:
                continue  # downvoted for this kind of query -> prune
            adjusted.append((chunk_id, score + alpha * v))
        adjusted.sort(key=lambda kv: -kv[1])
        return adjusted

    def close(self) -> None:
        self._conn.close()
