"""Feature #8 — Conversational Memory via Query Reformulation.

Follow-ups like "what about delays?" or "and in the US?" are meaningless to a
retriever on their own. Before searching, we *condense* the chat history + the
new message into a single standalone query. A heuristic fallback keeps things
working offline.
"""
from __future__ import annotations

from . import config
from ._gemini import generate_text

_CONDENSE_SYSTEM = (
    "Rewrite the user's follow-up into a single standalone search query that "
    "makes sense without the chat history. Resolve pronouns and ellipsis using "
    "the history. Return ONLY the rewritten query, no preamble."
)

# Cheap signal that a message depends on prior turns.
_FOLLOWUP_HINTS = (
    "it", "that", "this", "they", "them", "those", "and ", "what about",
    "how about", "also", "then", "instead", "same",
)


class ConversationMemory:
    def __init__(self, api_key: str | None = None, model: str = config.GEN_MODEL):
        self.history: list[tuple[str, str]] = []  # (role, text)
        self.model = model
        self.client = None
        key = api_key or config.load_api_key()
        if key:
            from google import genai

            self.client = genai.Client(api_key=key)

    def add_user(self, text: str) -> None:
        self.history.append(("user", text))

    def add_assistant(self, text: str) -> None:
        self.history.append(("assistant", text))

    def _looks_like_followup(self, text: str) -> bool:
        low = " " + text.lower().strip()
        return len(text.split()) <= 6 or any(h in low for h in _FOLLOWUP_HINTS)

    def condense(self, follow_up: str) -> str:
        """Return a standalone query for ``follow_up`` given the history."""
        if not self.history or not self._looks_like_followup(follow_up):
            return follow_up
        if self.client is None:
            return self._heuristic(follow_up)

        convo = "\n".join(f"{r}: {t}" for r, t in self.history[-6:])
        prompt = (_CONDENSE_SYSTEM + "\n\n"
                  + f"Chat history:\n{convo}\n\nFollow-up: {follow_up}\n\nStandalone query:")
        try:
            return generate_text(self.client, self.model, prompt).strip() or follow_up
        except Exception:
            return self._heuristic(follow_up)

    def _heuristic(self, follow_up: str) -> str:
        """Offline fallback: stitch the last user turn onto the follow-up."""
        last_user = next((t for r, t in reversed(self.history) if r == "user"), "")
        return f"{last_user} — {follow_up}".strip(" —") if last_user else follow_up
