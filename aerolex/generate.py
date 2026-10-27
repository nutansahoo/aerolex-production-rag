"""Feature #7 — Grounded answer generation as plain text.

The model is told to answer from context passages and cite their chunk IDs.
Answers are returned unchanged, without citation resolution or validation.
Without an API key, return the supplied context with an offline-mode notice.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from . import config
from ._gemini import generate_text
from .chunking import Chunk

CITATION_RE = re.compile(r"\[\[(\d+)\]\]")

DEFAULT_SYSTEM_PROMPT = (
    "You are AeroLex, an assistant answering questions about airline passenger "
    "rights using ONLY the provided context passages. Every factual sentence "
    "must end with a citation tag like [[chunk_id]] referring to the passage "
    "you used. If the context does not answer the question, say so plainly. "
    "Never invent regulations, article numbers, or passage numbers."
)


def format_context(chunks: Iterable[Chunk]) -> str:
    """Format the final chunks with their IDs and citations as context text."""
    lines = []
    for c in chunks:
        lines.append(f"[{c.chunk_id}] ({c.citation})\n{c.text}")
    return "\n\n".join(lines)


def build_prompt(
    query: str,
    context_text: str,
    profile_block: str = "",
    *,
    user_prompt_template: str | None = None,
) -> str:
    """Build the user prompt, optionally formatting a caller-supplied template."""
    if user_prompt_template is not None:
        try:
            return user_prompt_template.format(
                query=query, context_text=context_text, profile_block=profile_block
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                "Invalid user_prompt_template: use {query}, {context_text}, and/or "
                "{profile_block}; escape literal braces as {{ and }}. "
                f"Formatting failed: {exc}"
            ) from exc

    parts = []
    if profile_block:
        parts.append(profile_block)
    parts.append("Context passages:\n" + context_text)
    parts.append(f"Question: {query}")
    parts.append("Answer (cite every claim with [[chunk_id]]):")
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

    def generate(
        self,
        query: str,
        chunks: Iterable[Chunk] | str,
        profile_block: str = "",
        *,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
    ) -> str:
        """Return plain text from chunk objects or a prebuilt context string.

        ``user_prompt_template`` receives ``query``,
        ``context_text``, and ``profile_block``. ``None`` uses the default prompt.
        No citation post-processing is applied to the generated answer.
        """
        context_text = chunks if isinstance(chunks, str) else format_context(chunks)
        user_prompt = build_prompt(
            query, context_text, profile_block,
            user_prompt_template=user_prompt_template,
        )

        if self.client is None:
            return self._offline(query, context_text)

        system = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        prompt = system + "\n\n" + user_prompt
        text = generate_text(self.client, self.model, prompt)
        return text

    def _offline(self, query: str, context_text: str) -> str:
        """Extractive fallback when no API key is configured."""
        return (
            f"(Offline mode — no GEMINI_API_KEY. Showing the most relevant "
            f"passages for: {query!r})\n\n{context_text}"
        )
