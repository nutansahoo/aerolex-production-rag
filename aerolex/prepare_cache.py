"""Explicit one-time preparation of the workshop's chunk and embedding caches."""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable

import numpy as np

from . import config
from .chunking import Chunk, chunk_corpus, load_chunks, save_chunks
from .embeddings import GeminiEmbedder
from .loaders import DOCUMENTS, Document, load_documents
from .store import InMemoryVectorStore


def prepare_cache(
    *,
    chunker: Callable[[Iterable[Document]], list[Chunk]] | None = None,
    chunks_path: str | Path | None = None,
    embeddings_path: str | Path | None = None,
) -> None:
    """Build a matching cache pair; omitted arguments retain the baseline."""
    chunker = chunker if chunker is not None else chunk_corpus
    chunks_path = Path(chunks_path) if chunks_path is not None else config.CHUNKS_CACHE
    embeddings_path = (
        Path(embeddings_path) if embeddings_path is not None else config.EMBEDDINGS_CACHE
    )
    for path in (chunks_path, embeddings_path):
        if path.exists():
            raise FileExistsError(
                f"Cache already exists: {path}. No embeddings were requested. "
                "Move both cache files aside deliberately before rebuilding."
            )

    docs = load_documents()
    expected_docs = {meta.doc_id for meta in DOCUMENTS}
    if len(docs) != len(expected_docs) or {d.doc_id for d in docs} != expected_docs:
        raise ValueError(f"Expected all four workshop PDFs in {config.DATA_DIR}.")

    chunks = chunker(docs)
    ids = [chunk.chunk_id for chunk in chunks]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Expected nonempty chunks with unique IDs.")

    print(
        f"Loaded {len(docs)} PDFs; embedding {len(chunks)} chunks with "
        f"{config.EMBED_MODEL} ({config.EMBED_DIM} dimensions).\n"
        "Sending corpus text to Gemini; this consumes API quota. "
        "Batching and retries may take several minutes.",
        flush=True,
    )
    embedder = GeminiEmbedder()
    try:
        vectors = embedder.embed_documents([c.searchable_text for c in chunks])
    finally:
        embedder.client.close()

    expected_shape = (len(chunks), config.EMBED_DIM)
    if vectors.shape != expected_shape:
        raise ValueError(f"Expected embedding shape {expected_shape}; got {vectors.shape}.")
    if not np.isfinite(vectors).all() or not (np.linalg.norm(vectors, axis=1) > 0).all():
        raise ValueError("Embeddings contain non-finite values or zero vectors.")

    store = InMemoryVectorStore(ids, vectors)
    with TemporaryDirectory(prefix="prepare-", dir=config.CACHE_DIR) as directory:
        staged_chunks = Path(directory) / "chunks.json"
        staged_embeddings = Path(directory) / "embeddings.npz"
        save_chunks(chunks, staged_chunks)
        store.save(staged_embeddings)

        loaded_chunks = load_chunks(staged_chunks)
        loaded_store = InMemoryVectorStore.load(staged_embeddings)
        if ([c.chunk_id for c in loaded_chunks] != ids
                or loaded_store.ids != ids
                or loaded_store.matrix.shape != expected_shape
                or not np.isfinite(loaded_store.matrix).all()):
            raise ValueError("Cache round-trip validation failed.")

        staged_chunks.replace(chunks_path)
        staged_embeddings.replace(embeddings_path)

    print(f"Saved: {chunks_path}")
    print(f"Saved: {embeddings_path}")
    print(f"Cache verified: {expected_shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare_cache()


if __name__ == "__main__":
    main()
