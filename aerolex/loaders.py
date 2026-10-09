"""PDF loading + noise cleaning for the AeroLex corpus.

Prefers PyMuPDF (``fitz``) for layout-faithful extraction (e.g. the ``§``
section sign in the CFR), and transparently falls back to ``pypdf`` so the
code still runs in a minimal environment.

Each source PDF is mapped to a stable ``doc_id`` and a ``doc_type`` that drives
the hierarchical chunker (``eu`` / ``cfr`` / ``delta``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import config


# --------------------------------------------------------------------------- #
# Corpus registry: filename substring -> metadata
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    title: str
    doc_type: str          # "eu" | "cfr" | "delta"
    match: str             # case-insensitive substring of the filename


DOCUMENTS: list[DocMeta] = [
    DocMeta("eu261", "EU Regulation (EC) No 261/2004", "eu", "261"),
    DocMeta("cfr260", "US DOT 14 CFR Part 260", "cfr", "14 CFR Part 260"),
    DocMeta("delta_domestic", "Delta Domestic General Rules Tariff (Contract of Carriage)",
            "delta", "delta-coc-18"),
    DocMeta("delta_intl", "Delta International General Rules Tariff (Contract of Carriage)",
            "delta", "delta-coc-international"),
]


@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str
    text: str
    source_path: str


# --------------------------------------------------------------------------- #
# Raw text extraction
# --------------------------------------------------------------------------- #
def extract_raw_text(path: str | Path) -> str:
    """Extract text from a PDF, preferring PyMuPDF and falling back to pypdf."""
    path = Path(path)
    # Prefer the modern `pymupdf` import; fall back to legacy `fitz`, then pypdf.
    mod = None
    try:
        import pymupdf as mod
    except ImportError:
        try:
            import fitz as mod  # older PyMuPDF
        except ImportError:
            mod = None
    if mod is not None:
        with mod.open(path) as doc:
            return "\f".join(page.get_text("text") for page in doc)

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\f".join((page.extract_text() or "") for page in reader.pages)


# --------------------------------------------------------------------------- #
# Noise cleaning
# --------------------------------------------------------------------------- #
# Running headers/footers and table-of-contents artifacts, per document family.
_NOISE_PATTERNS: dict[str, list[re.Pattern]] = {
    "common": [
        re.compile(r"^\s*\.{4,}.*$", re.M),              # dotted TOC leaders
        re.compile(r"\.{4,}\s*\d+\s*$", re.M),           # "... 12" TOC tails
        re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.M, ),
        re.compile(r"^\s*EN\s*$", re.M),                 # stray language markers
    ],
    "eu": [
        # "17.2.2004 L 46/1 Official Journal of the European Union EN"
        re.compile(r"^.*Official Journal of the European Union.*$", re.M),
        re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{4}\s*L?\s*\d+/\d+.*$", re.M),
    ],
    "cfr": [
        re.compile(r"^.*\(enhanced display\).*$", re.M),
        re.compile(r"^\s*14 CFR Part 260 \(up to date.*$", re.M),
        re.compile(r"^.*Refunds for Airline Fare and Ancillary Service Fees\s+14 CFR.*$", re.M),
    ],
    "delta": [
        re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.M),
    ],
}


def clean_text(text: str, doc_type: str) -> str:
    """Strip page headers/footers, TOC leaders, and form-feeds."""
    text = text.replace("\f", "\n")
    for pat in _NOISE_PATTERNS["common"] + _NOISE_PATTERNS.get(doc_type, []):
        text = pat.sub("", text)
    # Collapse runs of blank lines and trailing spaces.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _match_meta(filename: str) -> DocMeta | None:
    low = filename.lower()
    for meta in DOCUMENTS:
        if meta.match.lower() in low:
            return meta
    return None


def load_documents(data_dir: str | Path | None = None) -> list[Document]:
    """Load, identify, and clean every known PDF in ``data_dir``."""
    data_dir = Path(data_dir or config.DATA_DIR)
    docs: list[Document] = []
    for pdf in sorted(data_dir.glob("*.pdf")):
        meta = _match_meta(pdf.name)
        if meta is None:
            continue  # ignore unknown files rather than guessing their structure
        raw = extract_raw_text(pdf)
        docs.append(
            Document(
                doc_id=meta.doc_id,
                title=meta.title,
                doc_type=meta.doc_type,
                text=clean_text(raw, meta.doc_type),
                source_path=str(pdf),
            )
        )
    return docs
