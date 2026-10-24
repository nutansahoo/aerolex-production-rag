"""LangChain recursive character splitting within the existing legal sections."""
from __future__ import annotations

import re
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .chunking import Chunk, iter_document_sections
from .loaders import Document


def chunk_corpus_recursive(docs: Iterable[Document]) -> list[Chunk]:
    """Return citation-preserving Chunks with strategy-specific IDs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_MAX_CHARS,
        chunk_overlap=config.CHUNK_OVERLAP_CHARS,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        is_separator_regex=False,
    )
    chunks: list[Chunk] = []
    for doc in docs:
        for label, heading, body, char_start in iter_document_sections(doc):
            parts = splitter.split_text(body)
            slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
            for pi, part in enumerate(parts):
                chunks.append(
                    Chunk(
                        chunk_id=f"recursive::{doc.doc_id}::{slug}::{pi}",
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
