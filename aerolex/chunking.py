"""Structure-first, word-bounded chunks with explicit legal-reference links."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

from . import config
from .loaders import Document
from .references import ReferenceMention, extract_mentions

TOP_LEVEL_PATTERNS: dict[str, re.Pattern] = {
    "eu": re.compile(r"(?m)^[ \t]*Article\s+(\d+)\b"),
    "cfr": re.compile(r"(?m)^[ \t]*(?:[\u00A7\uFFFD]|Sec\.|Section)\s*260\.(\d+)\b"),
    "delta": re.compile(r"(?m)^[ \t]*RULE\s+(\d+)\s*:"),
}
_WORDS = re.compile(r"\S+")
_BLOCK = re.compile(r"\n[ \t]*\n|\n(?=[ \t]*(?:\d+\.\s|\([a-z0-9ivx]+\)\s|[A-Z]\.\s))")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(])")
_CACHE_VERSION = 2


class ChunkCacheError(ValueError):
    """A known-stale chunk artifact that must be rebuilt."""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    section_label: str
    heading: str
    part_index: int
    part_count: int
    text: str
    char_start: int
    char_end: int = 0
    references: list[str] = field(default_factory=list)
    reference_mentions: list[ReferenceMention] = field(default_factory=list)

    @property
    def section_id(self) -> tuple[str, str]:
        return self.doc_id, self.section_label

    @property
    def citation(self) -> str:
        label = f"{self.doc_title} — {self.section_label}"
        if self.part_count > 1:
            label += f" (part {self.part_index + 1}/{self.part_count})"
        return label

    @property
    def searchable_text(self) -> str:
        return f"{self.section_label} — {self.heading}\n{self.text}"


def _section_label(doc_type: str, number: str) -> str:
    return {"eu": f"Article {number}", "cfr": f"§ 260.{number}",
            "delta": f"RULE {number}"}[doc_type]


def _body_span(text: str, start: int, end: int, doc_type: str,
               label: str) -> tuple[str, int, int]:
    lines = list(re.finditer(r"[^\n]+", text[start:end]))
    heading = lines[0].group().strip() if lines else label
    body_start = start + lines[0].end() if lines else end
    if doc_type == "eu" and len(lines) > 1:
        title = lines[1].group().strip()
        if len(title) <= 80 and not re.match(r"^\(?\d", title):
            heading = f"{label} — {title}"
            body_start = start + lines[1].end()
    while body_start < end and text[body_start].isspace():
        body_start += 1
    while end > body_start and text[end - 1].isspace():
        end -= 1
    return heading, body_start, end


def _split_spans(body: str, max_words: int, overlap_words: int) -> list[tuple[int, int]]:
    """Return contiguous source spans; preferred boundaries never override size."""
    if (any(not isinstance(n, int) or isinstance(n, bool) for n in (max_words, overlap_words))
            or max_words <= 0 or not 0 <= overlap_words < max_words):
        raise ValueError("Require max_words > 0 and 0 <= overlap_words < max_words")
    words = list(_WORDS.finditer(body))
    if not words:
        return []
    word_starts = {m.start(): i for i, m in enumerate(words)}

    def boundaries(pattern):
        result = set()
        for match in pattern.finditer(body):
            pos = match.end()
            while pos < len(body) and body[pos].isspace():
                pos += 1
            if pos in word_starts:
                result.add(word_starts[pos])
        return result

    blocks, sentences = boundaries(_BLOCK), boundaries(_SENTENCE)
    # Do not separate a reference keyword/section sign from its number.
    protected = {
        i + 1 for i, word in enumerate(words[:-1])
        if re.fullmatch(r"(?i)(?:articles?|rules?|sections?|§|�)", word.group())
        and re.match(r"\d", words[i + 1].group())
    }
    spans, start, previous_end = [], 0, 0
    while start < len(words):
        limit = min(start + max_words, len(words))
        if limit == len(words):
            end = limit
        else:
            minimum = max(start + 1, previous_end + 1)
            preferred = [i for i in blocks if minimum <= i <= limit and i not in protected]
            if not preferred:
                preferred = [i for i in sentences if minimum <= i <= limit and i not in protected]
            end = max(preferred) if preferred else limit
            if end in protected and end - 1 >= minimum:
                end -= 1
        spans.append((words[start].start(), words[end - 1].end()))
        if end == len(words):
            break
        overlap_start = max(start + 1, end - overlap_words)
        complete_units = [i for i in blocks | sentences if overlap_start <= i < end]
        start, previous_end = min(complete_units, default=overlap_start), end
    return spans


def chunk_document(doc: Document, *, max_words: int | None = None,
                   overlap_words: int | None = None) -> list[Chunk]:
    max_words = config.CHUNK_MAX_WORDS if max_words is None else max_words
    overlap_words = config.CHUNK_OVERLAP_WORDS if overlap_words is None else overlap_words
    # Validate even an empty document.
    _split_spans("", max_words, overlap_words)
    text, base = doc.text, 0
    if doc.doc_type == "cfr":
        authority = text.find("Authority:")
        if authority >= 0:
            base = authority
    matches = list(TOP_LEVEL_PATTERNS[doc.doc_type].finditer(text, base))
    if not matches:
        raise ValueError(f"{doc.doc_id}: no recognized section headings")
    numbers = [m.group(1) for m in matches]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"{doc.doc_id}: duplicate section headings; check TOC cleaning")
    sections = []
    if text[base:matches[0].start()].strip():
        sections.append(("Preamble", base, matches[0].start()))
    for i, match in enumerate(matches):
        sections.append((_section_label(doc.doc_type, match.group(1)), match.start(),
                         matches[i + 1].start() if i + 1 < len(matches) else len(text)))

    chunks = []
    for label, start, end in sections:
        if label == "Preamble":
            heading = "Preamble (Recitals)" if doc.doc_type == "eu" else "Preamble"
            while start < end and text[start].isspace():
                start += 1
        else:
            heading, start, end = _body_span(text, start, end, doc.doc_type, label)
        body = text[start:end]
        spans = _split_spans(body, max_words, overlap_words)
        mentions = extract_mentions(body, doc.doc_id, doc.doc_type, label, offset=start)
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        for part_index, (a, b) in enumerate(spans):
            part = body[a:b]
            digest = hashlib.sha256(f"{label} — {heading}\n{part}".encode()).hexdigest()[:12]
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}::{slug}::{part_index}::{digest}",
                doc_id=doc.doc_id, doc_title=doc.title, doc_type=doc.doc_type,
                section_label=label, heading=heading, part_index=part_index,
                part_count=len(spans), text=part, char_start=start + a, char_end=start + b,
                reference_mentions=[ReferenceMention(**asdict(m)) for m in mentions
                                    if m.start < start + b and m.end > start + a],
            ))
    if not chunks:
        raise ValueError(f"{doc.doc_id}: recognized headings have no substantive text")
    _link_references(chunks)
    return chunks


def _link_references(chunks: list[Chunk]) -> None:
    sections: dict[tuple[str, str], list[Chunk]] = {}
    for chunk in chunks:
        sections.setdefault(chunk.section_id, []).append(chunk)
    for chunk in chunks:
        chunk.references = []
        for mention in chunk.reference_mentions:
            if mention.status not in {"pending", "resolved", "missing"}:
                continue
            target = (mention.target_doc_id, mention.section_label)
            members = sorted(sections.get(target, []), key=lambda c: c.part_index)
            mention.chunk_ids = [c.chunk_id for c in members]
            mention.status = "resolved" if members else "missing"
            mention.reason = "" if members else "Target section is not in this corpus"
            if target != chunk.section_id:
                chunk.references.extend(mention.chunk_ids)
        chunk.references = list(dict.fromkeys(chunk.references))


def chunk_corpus(docs: Iterable[Document], **kwargs) -> list[Chunk]:
    chunks = [chunk for doc in docs for chunk in chunk_document(doc, **kwargs)]
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("Corpus contains duplicate chunk IDs")
    _link_references(chunks)
    return chunks


def fingerprint_chunks(chunks: Iterable[Chunk]) -> str:
    payload = [(c.chunk_id, c.searchable_text) for c in chunks]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def _source_fingerprint(documents: Iterable[Document]) -> str:
    payload = [(d.doc_id, d.title, d.doc_type, d.text) for d in documents]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def _policy(max_words=None, overlap_words=None) -> dict:
    return {"max_words": config.CHUNK_MAX_WORDS if max_words is None else max_words,
            "overlap_words": config.CHUNK_OVERLAP_WORDS if overlap_words is None else overlap_words}


def save_chunks(chunks: list[Chunk], path=None, *, documents: Iterable[Document] | None = None,
                max_words: int | None = None, overlap_words: int | None = None) -> None:
    """Persist chunks; supply documents to enable source validation on reload."""
    path = path or config.CHUNKS_CACHE
    payload = {"schema_version": _CACHE_VERSION, "policy": _policy(max_words, overlap_words),
               "source_fingerprint": _source_fingerprint(documents) if documents is not None else None,
               "corpus_fingerprint": fingerprint_chunks(chunks),
               "chunks": [asdict(c) for c in chunks]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_chunks(path=None, *, documents: Iterable[Document] | None = None,
                max_words: int | None = None, overlap_words: int | None = None) -> list[Chunk]:
    path = path or config.CHUNKS_CACHE
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if (not isinstance(payload, dict) or payload.get("schema_version") != _CACHE_VERSION
            or payload.get("policy") != _policy(max_words, overlap_words)):
        raise ChunkCacheError("Chunk cache schema/policy changed; rebuild with chunk_corpus()")
    if documents is not None and payload["source_fingerprint"] != _source_fingerprint(documents):
        raise ChunkCacheError("Source documents changed; rebuild with chunk_corpus()")
    chunks = [Chunk(**{**row, "reference_mentions": [
        ReferenceMention(**m) for m in row["reference_mentions"]]}) for row in payload["chunks"]]
    if fingerprint_chunks(chunks) != payload["corpus_fingerprint"]:
        raise ValueError("Chunk cache content fingerprint is corrupt; rebuild the artifact")
    ids = {c.chunk_id for c in chunks}
    if len(ids) != len(chunks) or any(cid not in ids for c in chunks for cid in c.references):
        raise ValueError("Chunk cache contains duplicate IDs or dangling references")
    return chunks
