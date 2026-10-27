import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from aerolex import config
from aerolex.chunking import chunk_corpus, save_chunks
from aerolex.chunking_recursive import chunk_corpus_recursive
from aerolex.loaders import Document
from aerolex.pipeline import RagPipeline
from aerolex.store import InMemoryVectorStore


class ChunkingNotebookTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        directory = Path(self.stack.enter_context(TemporaryDirectory()))
        self.stack.enter_context(
            patch.multiple(
                config,
                CACHE_DIR=directory,
                CHUNKS_CACHE=directory / "chunks.json",
                EMBEDDINGS_CACHE=directory / "embeddings.npz",
                RECURSIVE_CHUNKS_CACHE=directory / "chunks_recursive.json",
                RECURSIVE_EMBEDDINGS_CACHE=directory / "embeddings_recursive.npz",
            )
        )
        self.stack.enter_context(patch.object(config, "load_api_key", return_value=None))
        self.docs = [
            Document(
                doc_id="eu261",
                title="EU261",
                doc_type="eu",
                text="Article 7\nCompensation\n" + "1. Refund and compensation. " * 150,
                source_path="fixture.pdf",
            ),
            Document(
                doc_id="cfr260",
                title="CFR260",
                doc_type="cfr",
                text="§ 260.6 Refunds\n" + "Refund for cancelled flights. " * 120,
                source_path="fixture.pdf",
            ),
        ]
        self.stack.enter_context(
            patch("aerolex.loaders.load_documents", return_value=self.docs)
        )
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        notebook = json.loads(
            (Path(__file__).resolve().parents[1] / "notebooks/workshop_solution.ipynb")
            .read_text(encoding="utf-8")
        )
        self.cells = {
            cell["id"]: "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        }
        self.baseline = chunk_corpus(self.docs)
        self.alternate = chunk_corpus_recursive(self.docs)
        for chunks, chunks_path, embeddings_path in (
            (self.baseline, config.CHUNKS_CACHE, config.EMBEDDINGS_CACHE),
            (self.alternate, config.RECURSIVE_CHUNKS_CACHE, config.RECURSIVE_EMBEDDINGS_CACHE),
        ):
            save_chunks(chunks, chunks_path)
            vectors = np.random.default_rng(0).normal(size=(len(chunks), config.EMBED_DIM))
            InMemoryVectorStore([c.chunk_id for c in chunks], vectors).save(embeddings_path)

    def execute(self, cell_id, namespace):
        exec(compile(self.cells[cell_id], f"notebook:{cell_id}", "exec"), namespace)

    def select_strategy(self, strategy):
        namespace = {}
        setup = self.cells["970866dd"].replace(
            'CHUNKING_STRATEGY = "hierarchical"', f"CHUNKING_STRATEGY = {strategy!r}"
        )
        exec(compile(setup, "notebook:setup", "exec"), namespace)
        self.execute("ec6b8035", namespace)
        self.execute("recursive-chunking-selection", namespace)
        return namespace

    def test_both_strategies_feed_notebook_retrieval_and_pipeline(self):
        fusion_cell = next(
            key for key, source in self.cells.items()
            if source.startswith("bm25_ids = ")
        )
        for strategy, expected in (
            ("hierarchical", self.baseline),
            ("recursive", self.alternate),
        ):
            with self.subTest(strategy=strategy):
                namespace = self.select_strategy(strategy)
                self.assertEqual(namespace["chunks"], expected)
                self.assertEqual(namespace["alternate_chunks"], self.alternate)
                self.assertEqual(namespace["hierarchical_chunks"], self.baseline)
                for cell_id in (
                    "c7d10611",  # BM25
                    "eea5a9ac",  # selected store with alignment guards
                    "10ca0b58",  # cosine self-check
                    "041e2750",  # offline Article 7 query vector
                    "348e641e",  # RRF definition and assertion
                    fusion_cell,
                ):
                    self.execute(cell_id, namespace)
                ids = {c.chunk_id for c in expected}
                self.assertEqual(set(namespace["by_id"]), ids)
                self.assertIn(namespace["art7"].chunk_id, namespace["store"].ids)
                self.assertTrue({cid for cid, _ in namespace["dense_hits"]} <= ids)
                self.assertTrue({cid for cid, _ in namespace["fused"]} <= ids)

                embedder = Mock()
                embedder.embed_query.return_value = namespace["qvec"]
                reranker = Mock()
                reranker.rerank.side_effect = lambda query, pairs, top_k: [
                    (cid, float(top_k - i))
                    for i, (cid, _) in enumerate(pairs[:top_k])
                ]
                generator = Mock()
                generator.generate.return_value = {
                    "answer": "Fixture answer [[1]]", "references": [], "offline": True
                }
                pipeline = RagPipeline(
                    chunks=expected,
                    store=namespace["store"],
                    bm25=namespace["bm25"],
                    embedder=embedder,
                    reranker=reranker,
                    generator=generator,
                )
                answer = pipeline.answer("refund")
                self.assertTrue(answer["chunks"])
                self.assertTrue({c.chunk_id for c in answer["chunks"]} <= ids)
                generator.generate.assert_called_once()

    def test_invalid_strategy_fails_in_setup(self):
        with self.assertRaisesRegex(ValueError, "CHUNKING_STRATEGY"):
            self.select_strategy("typo")

    def test_missing_recursive_chunks_are_split_locally(self):
        config.RECURSIVE_CHUNKS_CACHE.unlink()
        namespace = self.select_strategy("recursive")
        self.assertEqual(namespace["alternate_chunks"], self.alternate)
        self.execute("eea5a9ac", namespace)

    def test_missing_baseline_chunks_are_split_locally(self):
        config.CHUNKS_CACHE.unlink()
        namespace = self.select_strategy("hierarchical")
        self.assertEqual(namespace["hierarchical_chunks"], self.baseline)
        self.execute("eea5a9ac", namespace)

    def test_missing_selected_embeddings_name_the_correct_command(self):
        for strategy, path, command in (
            ("hierarchical", config.EMBEDDINGS_CACHE, "aerolex.prepare_cache"),
            ("recursive", config.RECURSIVE_EMBEDDINGS_CACHE, "aerolex.prepare_recursive_cache"),
        ):
            with self.subTest(strategy=strategy):
                namespace = self.select_strategy(strategy)
                path.unlink()
                with self.assertRaisesRegex(FileNotFoundError, command):
                    self.execute("eea5a9ac", namespace)

    def test_wrong_strategy_embeddings_are_rejected(self):
        namespace = self.select_strategy("recursive")
        namespace["embeddings_cache"] = config.EMBEDDINGS_CACHE
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.execute("eea5a9ac", namespace)

    def test_bad_embedding_dimensions_and_duplicate_ids_are_rejected(self):
        namespace = self.select_strategy("recursive")
        ids = [c.chunk_id for c in self.alternate]
        InMemoryVectorStore(ids, np.ones((len(ids), 4))).save(
            config.RECURSIVE_EMBEDDINGS_CACHE
        )
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.execute("eea5a9ac", namespace)
        namespace["chunks"] = [self.alternate[0]] * len(ids)
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.execute("eea5a9ac", namespace)

    def test_repeated_selection_does_not_replace_preserved_baseline(self):
        namespace = self.select_strategy("recursive")
        self.execute("recursive-chunking-selection", namespace)
        self.assertEqual(namespace["hierarchical_chunks"], self.baseline)


if __name__ == "__main__":
    unittest.main()
