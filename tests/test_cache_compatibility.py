import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aerolex import config
from aerolex.chunking import ChunkCacheError, chunk_corpus, load_chunks, save_chunks
from aerolex.store import InMemoryVectorStore
from tests.helpers import corpus, document


class CacheTests(unittest.TestCase):
    def test_chunk_round_trip_and_invalidation(self):
        chunks = corpus()
        docs = [document("source fingerprint fixture")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.json"
            save_chunks(chunks, path, documents=docs)
            loaded = load_chunks(path, documents=docs)
            self.assertEqual([asdict(c) for c in loaded], [asdict(c) for c in chunks])
            with self.assertRaises(ChunkCacheError):
                load_chunks(path, documents=[document("different source")])
            with patch.object(config, "CHUNK_MAX_WORDS", 999):
                with self.assertRaises(ChunkCacheError):
                    load_chunks(path)
            path.write_text(json.dumps([asdict(c) for c in chunks]))
            with self.assertRaises(ChunkCacheError):
                load_chunks(path)
            path.write_text("{malformed")
            with self.assertRaises(json.JSONDecodeError):
                load_chunks(path)

    def test_corrupt_chunk_cache_is_not_a_silent_rebuild(self):
        chunks = corpus()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.json"
            save_chunks(chunks, path, documents=[])
            payload = json.loads(path.read_text())
            payload["chunks"][0]["text"] = "corrupted"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "corrupt"):
                load_chunks(path)

    def test_custom_policy_and_original_save_signature(self):
        docs = [document("Article 1\nScope\n" + "word " * 40)]
        chunks = chunk_corpus(docs, max_words=10, overlap_words=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.json"
            save_chunks(chunks, path, documents=docs, max_words=10, overlap_words=2)
            self.assertEqual(chunks, load_chunks(path, documents=docs, max_words=10, overlap_words=2))
            with self.assertRaises(ChunkCacheError):
                load_chunks(path)
            save_chunks(chunks, path)
            self.assertEqual(chunks, load_chunks(path))
            with self.assertRaises(ChunkCacheError):
                load_chunks(path, documents=docs)

    def test_vector_round_trip_changed_text_ids_and_old_metadata(self):
        chunks = corpus()
        matrix = np.eye(len(chunks), dtype=np.float32)
        store = InMemoryVectorStore.from_chunks(chunks, matrix)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vectors.npz"
            store.save(path)
            loaded = InMemoryVectorStore.load(path)
            loaded.validate_chunks(chunks)
            np.testing.assert_allclose(loaded.matrix, store.matrix)
            self.assertEqual(loaded.search(matrix[0], 1)[0][0], chunks[0].chunk_id)
            for other in (chunks[:-1], list(reversed(chunks)),
                          [replace(chunks[0], text="changed"), *chunks[1:]]):
                with self.assertRaisesRegex(ValueError, "Rebuild embeddings"):
                    loaded.validate_chunks(other)
            # Graph metadata changes do not alter embedded content.
            graph_changed = [replace(c, references=[]) for c in chunks]
            loaded.validate_chunks(graph_changed)
            np.savez(path, ids=np.array(store.ids, dtype=object), matrix=matrix)
            with self.assertRaisesRegex(ValueError, "Legacy"):
                InMemoryVectorStore.load(path)
            with self.assertRaisesRegex(ValueError, "from_chunks"):
                InMemoryVectorStore(store.ids, matrix).save(path)

    def test_invalid_vector_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            InMemoryVectorStore(["same", "same"], np.eye(2))
        with self.assertRaises(ValueError):
            InMemoryVectorStore(["only"], np.eye(2))
