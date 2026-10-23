"""Prepare separate recursive-splitter caches, sending corpus text to Gemini."""
from __future__ import annotations

import argparse

from . import config
from .chunking_recursive import chunk_corpus_recursive
from .prepare_cache import prepare_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare_cache(
        chunker=chunk_corpus_recursive,
        chunks_path=config.RECURSIVE_CHUNKS_CACHE,
        embeddings_path=config.RECURSIVE_EMBEDDINGS_CACHE,
    )


if __name__ == "__main__":
    main()
