"""Feature #9 — Cross-Session Long-Term Persistence.

A tiny SQLite-backed profile store. Facts the user states in one session
("I'm flying Delta from JFK", "I have EU passenger rights") survive process
restarts and can be injected into future prompts as durable context.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from . import config


@dataclass
class Fact:
    key: str
    value: str
    updated_at: float


class ProfileStore:
    """Per-user key/value facts persisted in SQLite."""

    def __init__(self, db_path=None):
        self.db_path = str(db_path or config.PROFILE_DB)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS facts (
                   user_id    TEXT NOT NULL,
                   key        TEXT NOT NULL,
                   value      TEXT NOT NULL,
                   updated_at REAL NOT NULL,
                   PRIMARY KEY (user_id, key)
               )"""
        )
        self._conn.commit()

    def remember(self, user_id: str, key: str, value: str) -> None:
        self._conn.execute(
            """INSERT INTO facts (user_id, key, value, updated_at) VALUES (?,?,?,?)
               ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value,
                                                       updated_at=excluded.updated_at""",
            (user_id, key, value, time.time()),
        )
        self._conn.commit()

    def get(self, user_id: str, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM facts WHERE user_id=? AND key=?", (user_id, key)
        ).fetchone()
        return row[0] if row else None

    def all_facts(self, user_id: str) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT key, value, updated_at FROM facts WHERE user_id=? ORDER BY updated_at",
            (user_id,),
        ).fetchall()
        return [Fact(*r) for r in rows]

    def forget(self, user_id: str, key: str) -> None:
        self._conn.execute("DELETE FROM facts WHERE user_id=? AND key=?", (user_id, key))
        self._conn.commit()

    def context_block(self, user_id: str) -> str:
        """Render stored facts as a prompt-injectable block (empty if none)."""
        facts = self.all_facts(user_id)
        if not facts:
            return ""
        lines = "\n".join(f"- {f.key}: {f.value}" for f in facts)
        return f"Known facts about the user:\n{lines}"

    def close(self) -> None:
        self._conn.close()
