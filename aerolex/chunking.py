"""Feature #1 — Hierarchical Document Preprocessing & Chunking.

Legal/regulatory text has an inherent hierarchy (Article / Section / Rule ->
paragraph). Splitting on arbitrary character windows shreds that structure and
destroys citations. Instead we split on the *logical boundaries* of each
document family and attach rich metadata to every chunk so answers can cite
"EU Regulation 261/2004 — Article 7" rather than "chunk 42".

Top-level boundaries by family:
    eu    -> ``Article N``
    cfr   -> ``§ 260.N``
    delta -> ``RULE N:``

Oversized sections are sub-split on paragraph boundaries with a small overlap,
while still carrying their section label so retrieval stays grounded.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator

from . import config
from .loaders import Document

# --------------------------------------------------------------------------- #
# Boundary patterns (robust to the "§" rendering as U+FFFD in some extractors)
# --------------------------------------------------------------------------- #
TOP_LEVEL_PATTERNS: dict[str, re.Pattern] = {
    "eu": re.compile(r"(?m)^[ \t]*Article\s+(\d+)\b"),
    "cfr": re.compile(r"(?m)^[ \t]*(?:[\u00A7\uFFFD]|Sec\.|Section)\s*260\.(\d+)\b"),
    "delta": re.compile(r"(?m)^[ \t]*RULE\s+(\d+)\s*:"),
}

_SECONDARY_SPLIT = re.compile(r"\n\s*\n")  # paragraph breaks


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    section_label: str      # "Article 7" | "§ 260.6" | "RULE 24"
    heading: str            # full heading line (label + title when available)
    part_index: int         # 0-based part within an oversized section
    part_count: int
    text: str               # clean body text (for display + citation)
    char_start: int         # offset within the cleaned document text

    @property
    def citation(self) -> str:
        label = f"{self.doc_title} — {self.section_label}"
        if self.part_count > 1:
            label += f" (part {self.part_index + 1}/{self.part_count})"
        return label

    @property
    def searchable_text(self) -> str:
        """What BM25 + the embedder index: label + heading prepended so exact
        statute references (e.g. '§ 260.6', 'Article 7') are matchable."""
        return f"{self.section_label} — {self.heading}\n{self.text}"


def _section_label(doc_type: str, number: str) -> str:
    return {
        "eu": f"Article {number}",
        "cfr": f"§ 260.{number}",
        "delta": f"RULE {number}",
    }[doc_type]


def _derive_heading(section_text: str, doc_type: str, label: str) -> tuple[str, str]:
    """Return (heading, body) for a raw section block."""
    lines = section_text.splitlines()
    boundary = lines[0].strip() if lines else label
    rest = lines[1:]
    heading = boundary
    # EU article headings are bare ("Article 7"); the title sits on the next line.
    if doc_type == "eu":
        for i, ln in enumerate(rest):
            if ln.strip():
                cand = ln.strip()
                if len(cand) <= 80 and not re.match(r"^\(?\d", cand):
                    heading = f"{label} — {cand}"
                    rest = rest[i + 1:]
                break
    body = "\n".join(rest).strip()
    return heading, body


def _hard_wrap(text: str, max_chars: int, overlap: int) -> list[str]:
    """Last-resort window split for a single paragraph that exceeds the budget."""
    out, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        out.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [c for c in out if c]


def _split_oversized(body: str, max_chars: int, overlap: int) -> list[str]:
    """Pack paragraph units into chunks <= max_chars with char overlap."""
    if len(body) <= max_chars:
        return [body]
    units: list[str] = []
    for para in _SECONDARY_SPLIT.split(body):
        para = para.strip()
        if not para:
            continue
        units.extend(_hard_wrap(para, max_chars, overlap) if len(para) > max_chars else [para])

    chunks: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > max_chars:
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + "\n" + u).strip()
        else:
            cur = (cur + "\n" + u).strip() if cur else u
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def iter_document_sections(doc: Document) -> Iterator[tuple[str, str, str, int]]:
    """Yield (label, heading, body, section offset) before any body splitting."""
    text = doc.text
    # The CFR repeats every "§ 260.x" in a table of contents; skip to the body.
    if doc.doc_type == "cfr":
        idx = text.find("Authority:")
        if idx != -1:
            text = text[idx:]

    pattern = TOP_LEVEL_PATTERNS[doc.doc_type]
    matches = list(pattern.finditer(text))

    # Build (label, section_text, char_start) triples, including an optional
    # leading preamble (e.g. EU recitals) before the first top-level boundary.
    sections: list[tuple[str, str, int]] = []
    if matches and matches[0].start() > config.CHUNK_MIN_CHARS:
        sections.append(("Preamble", text[: matches[0].start()], 0))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((_section_label(doc.doc_type, m.group(1)), text[start:end], start))

    for label, sec_text, char_start in sections:
        if label == "Preamble":
            heading, body = "Preamble (Recitals)", sec_text.strip()
        else:
            heading, body = _derive_heading(sec_text, doc.doc_type, label)
        if len(body) < config.CHUNK_MIN_CHARS:
            continue  # skip stubs / leftover TOC fragments
        yield label, heading, body, char_start


def chunk_document(doc: Document) -> list[Chunk]:
    """Split one document into hierarchical, metadata-rich chunks."""
    chunks: list[Chunk] = []
    for label, heading, body, char_start in iter_document_sections(doc):
        parts = _split_oversized(body, config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP_CHARS)
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        for pi, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{slug}::{pi}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    doc_type=doc.doc_type,
                    section_label=label,
                    heading=heading,
                    part_index=pi,
                    part_count=len(parts),
                    text=part,
                    char_start=char_start,
                )
            )
    return chunks


def chunk_corpus(docs: Iterable[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks


# --------------------------------------------------------------------------- #
# (De)serialization — used by the cache builder so the notebook loads instantly
# --------------------------------------------------------------------------- #
def save_chunks(chunks: list[Chunk], path=None) -> None:
    path = path or config.CHUNKS_CACHE
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)


def load_chunks(path=None) -> list[Chunk]:
    path = path or config.CHUNKS_CACHE
    with open(path, "r", encoding="utf-8") as f:
        return [Chunk(**d) for d in json.load(f)]
