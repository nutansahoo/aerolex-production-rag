"""Feature #7 — Grounded answer generation with inline citations.

The model is told to answer strictly from numbered context passages and to tag
each claim with ``[[n]]`` pointing at the passage it came from. A regex
post-processor then resolves those tags to full legal citations and drops any
hallucinated numbers. If no API key is present we fall back to an extractive
answer so the retrieval stack still demos offline.
"""
from __future__ import annotations

import re

from . import config
from ._gemini import generate_text

CITATION_RE = re.compile(r"\[\[(\d+)\]\]")

_SYSTEM = (
    "You are AeroLex, an assistant answering questions about airline passenger "
    "rights using ONLY the provided context passages. Every factual sentence "
    "must end with a citation tag like [[1]] or [[2]] referring to the passage "
    "you used. If the context does not answer the question, say so plainly. "
    "Never invent regulations, article numbers, or passage numbers."
)


def format_context(chunks) -> tuple[str, dict[int, str]]:
    """Number the final chunks and return (context_text, {n: citation})."""
    lines, index = [], {}
    for i, c in enumerate(chunks, start=1):
        index[i] = c.citation
        lines.append(f"[{i}] ({c.citation})\n{c.text}")
    return "\n\n".join(lines), index


def build_prompt(query: str, context_text: str, profile_block: str = "") -> str:
    parts = []
    if profile_block:
        parts.append(profile_block)
    parts.append("Context passages:\n" + context_text)
    parts.append(f"Question: {query}")
    parts.append("Answer (cite every claim with [[n]]):")
    return "\n\n".join(parts)


def resolve_citations(answer: str, index: dict[int, str]) -> tuple[str, list[str]]:
    """Validate ``[[n]]`` tags against the context; return (answer, references).

    Tags pointing outside the context are stripped (anti-hallucination).
    """
    used: list[int] = []

    def _check(m: re.Match) -> str:
        n = int(m.group(1))
        if n in index:
            if n not in used:
                used.append(n)
            return m.group(0)
        return ""  # drop citation to a non-existent passage

    cleaned = CITATION_RE.sub(_check, answer)
    references = [f"[[{n}]] {index[n]}" for n in sorted(used)]
    return cleaned, references


class Generator:
    def __init__(self, api_key: str | None = None, model: str = config.GEN_MODEL):
        self.model = model
        self.client = None
        key = api_key or config.load_api_key()
        if key:
            from google import genai

            self.client = genai.Client(api_key=key)

    def generate(self, query: str, chunks, profile_block: str = "") -> dict:
        context_text, index = format_context(chunks)
        if self.client is None:
            return self._offline(query, chunks, index)

        prompt = _SYSTEM + "\n\n" + build_prompt(query, context_text, profile_block)
        text = generate_text(self.client, self.model, prompt)
        answer, references = resolve_citations(text, index)
        return {"answer": answer, "references": references,
                "used_passages": [index[n] for n in range(1, len(index) + 1)],
                "offline": False}

    def _offline(self, query: str, chunks, index: dict[int, str]) -> dict:
        """Extractive fallback when no API key is configured."""
        lines = [f"(Offline mode — no GEMINI_API_KEY. Showing the most relevant "
                 f"passages for: {query!r})", ""]
        for i, c in enumerate(chunks, start=1):
            snippet = c.text.strip().replace("\n", " ")
            lines.append(f"[[{i}]] {c.citation}: {snippet[:280]}…")
        references = [f"[[{n}]] {index[n]}" for n in index]
        return {"answer": "\n".join(lines), "references": references,
                "used_passages": list(index.values()), "offline": True}
