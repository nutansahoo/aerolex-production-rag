import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from aerolex import config
from aerolex.chunking import Chunk, load_chunks
from aerolex.prepare_cache import prepare_cache
from aerolex.store import InMemoryVectorStore


class PrepareCacheTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(TemporaryDirectory())
        self.directory = Path(directory)
        self.chunks_path = self.directory / "chunks.json"
        self.embeddings_path = self.directory / "embeddings.npz"
        for name, value in (
            ("CACHE_DIR", self.directory),
            ("CHUNKS_CACHE", self.chunks_path),
            ("EMBEDDINGS_CACHE", self.embeddings_path),
        ):
            self.stack.enter_context(patch.object(config, name, value))
        self.docs = self.stack.enter_context(
            patch("aerolex.prepare_cache.load_documents")
        )
        self.docs.return_value = [
            SimpleNamespace(doc_id=doc_id)
            for doc_id in ("eu261", "cfr260", "delta_domestic", "delta_intl")
        ]
        self.chunks = [
            Chunk(
                chunk_id=f"eu261::article-{number}::0",
                doc_id="eu261",
                doc_title="EU261",
                doc_type="eu",
                section_label=f"Article {number}",
                heading=f"Article {number}",
                part_index=0,
                part_count=1,
                text=f"Test passage {number}",
                char_start=0,
            )
            for number in (6, 7)
        ]
        self.chunker = self.stack.enter_context(
            patch("aerolex.prepare_cache.chunk_corpus", return_value=self.chunks)
        )
        self.embedder = self.stack.enter_context(
            patch("aerolex.prepare_cache.GeminiEmbedder")
        )
        self.embedder.return_value.embed_documents.return_value = np.ones(
            (2, config.EMBED_DIM), dtype=np.float32
        )
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def test_prepares_matching_readable_caches(self):
        prepare_cache()
        ids = [chunk.chunk_id for chunk in self.chunks]
        self.assertEqual([c.chunk_id for c in load_chunks()], ids)
        store = InMemoryVectorStore.load()
        self.assertEqual(store.ids, ids)
        self.assertEqual(store.matrix.shape, (2, config.EMBED_DIM))
        np.testing.assert_allclose(np.linalg.norm(store.matrix, axis=1), 1, atol=1e-6)
        self.embedder.return_value.embed_documents.assert_called_once_with(
            [c.searchable_text for c in self.chunks]
        )
        self.embedder.return_value.client.close.assert_called_once()

    def test_either_existing_cache_prevents_api_calls(self):
        for path in (self.chunks_path, self.embeddings_path):
            with self.subTest(path=path):
                path.write_bytes(b"existing cache")
                with self.assertRaises(FileExistsError):
                    prepare_cache()
                self.assertEqual(path.read_bytes(), b"existing cache")
                self.embedder.assert_not_called()
                self.docs.assert_not_called()
                path.unlink()

    def test_missing_documents_prevent_api_calls(self):
        self.docs.return_value.pop()
        with self.assertRaises(ValueError):
            prepare_cache()
        self.embedder.assert_not_called()

    def test_duplicate_chunk_ids_prevent_api_calls(self):
        self.chunker.return_value = [self.chunks[0], self.chunks[0]]
        with self.assertRaises(ValueError):
            prepare_cache()
        self.embedder.assert_not_called()

    def test_invalid_vectors_are_not_saved(self):
        for vectors in (
            np.ones((1, config.EMBED_DIM)),
            np.zeros((2, config.EMBED_DIM)),
            np.full((2, config.EMBED_DIM), np.nan),
            np.full((2, config.EMBED_DIM), np.inf),
        ):
            with self.subTest(shape=vectors.shape):
                self.embedder.return_value.embed_documents.return_value = vectors
                with self.assertRaises(ValueError):
                    prepare_cache()
                self.assertFalse(self.chunks_path.exists())
                self.assertFalse(self.embeddings_path.exists())

    def test_api_failure_is_visible_and_does_not_publish_caches(self):
        self.embedder.return_value.embed_documents.side_effect = RuntimeError("API failed")
        with self.assertRaisesRegex(RuntimeError, "API failed"):
            prepare_cache()
        self.assertFalse(self.chunks_path.exists())
        self.assertFalse(self.embeddings_path.exists())
        self.embedder.return_value.client.close.assert_called_once()

    def test_round_trip_mismatch_does_not_publish_caches(self):
        with patch(
            "aerolex.prepare_cache.load_chunks",
            return_value=list(reversed(self.chunks)),
        ):
            with self.assertRaisesRegex(ValueError, "round-trip"):
                prepare_cache()
        self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
