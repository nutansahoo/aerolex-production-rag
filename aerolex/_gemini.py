"""Internal helper: Gemini text generation via the Interactions API.

google-genai 2.x exposes ``client.interactions.create(model=..., input=...)``
(OpenAI-Responses-style). It accepts only ``model`` and ``input`` as direct
fields, so we fold any system prompt into ``input``. Transient 503 "high
demand" responses (common on the free tier) are retried with backoff.
"""
from __future__ import annotations

import time


def generate_text(client, model: str, prompt: str, max_retries: int = 4) -> str:
    """Return the model's text for ``prompt`` (empty string on empty output)."""
    last_err = None
    for attempt in range(max_retries):
        try:
            result = client.interactions.create(model=model, input=prompt)
            return result.output_text or ""
        except Exception as e:  # retry only transient high-demand errors
            last_err = e
            msg = repr(e)
            if "503" in msg or "UNAVAILABLE" in msg or "429" in msg:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise last_err
