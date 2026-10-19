import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aerolex.generate import Generator, build_prompt, format_context, resolve_citations
from aerolex.pipeline import RagPipeline
from aerolex.store import InMemoryVectorStore
from tests.helpers import corpus


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.chunks = corpus()
        self.ids = [c.chunk_id for c in self.chunks]
        self.store = InMemoryVectorStore.from_chunks(self.chunks, np.eye(len(self.chunks)))
        self.bm25 = SimpleNamespace(search=lambda query, top_k: [(self.ids[0], 1.0)])
        self.embedder = SimpleNamespace(embed_query=lambda query: np.eye(len(self.chunks))[0])
        self.reranker = SimpleNamespace(
            rerank=lambda query, candidates, top_k: [(self.ids[0], 1.0)])
        with patch("aerolex.generate.config.load_api_key", return_value=None):
            self.generator = Generator()
        self.pipe = RagPipeline(self.chunks, self.store, self.bm25, self.embedder,
                                self.reranker, self.generator)

    def test_seed_reranking_then_reference_expansion(self):
        baseline = self.pipe.retrieve("cancellation", expand_references=False)
        self.assertEqual(baseline, [self.chunks[0]])
        expanded = self.pipe.retrieve_context("cancellation")
        self.assertTrue({"Article 5", "Article 7", "Article 8", "Article 9"} <=
                        {c.section_label for c in expanded.chunks})
        self.assertEqual(expanded.seed_ids, [self.ids[0]])
        self.assertEqual(len(expanded.chunks), len(set(c.chunk_id for c in expanded.chunks)))
        no_reranker = self.pipe.retrieve_context("cancellation", use_reranker=False)
        self.assertTrue(no_reranker.chunks)

    def test_generation_receives_exact_evidence_and_diagnostics(self):
        output = self.pipe.answer("cancellation")
        self.assertTrue(output["offline"])
        self.assertIn("Article 7", " ".join(output["references"]))
        self.assertEqual(len(output["used_passages"]), len(output["chunks"]))
        self.assertIn("provenance", output["context_expansion"])
        self.assertEqual(output["query"], "cancellation")
        captured = {}

        def generate(query, chunks, **kwargs):
            captured.update(chunks=chunks, **kwargs)
            return {"answer": "Evidence [[2]].", "references": []}

        self.pipe.generator = SimpleNamespace(generate=generate)
        output = self.pipe.answer("cancellation", max_added_chunks=0)
        self.assertEqual(captured["chunks"], output["chunks"])
        self.assertTrue(captured["context_diagnostics"]["diagnostics"])
        self.assertEqual(output["chunks"], [self.chunks[0]])

    def test_real_generation_prompt_and_expanded_citations(self):
        context = self.pipe.retrieve_context("cancellation")
        text, index = format_context(context.chunks)
        prompt = build_prompt("question", text, context_diagnostics=context.metadata())
        self.assertIn("not legal evidence", prompt)
        self.assertIn("Article 7", prompt)
        answer, references = resolve_citations("Amount [[2]]. Invalid [[999]].", index)
        self.assertNotIn("999", answer)
        self.assertEqual(references, [f"[[2]] {context.chunks[1].citation}"])
        self.generator.client = object()
        with patch("aerolex.generate.generate_text", return_value="Amount [[2]].") as call:
            result = self.generator.generate("question", context.chunks,
                                             context_diagnostics=context.metadata())
        self.assertIn("Retrieval notes", call.call_args.args[2])
        self.assertEqual(result["references"], references)

    def test_profile_memory_feedback_and_stale_store(self):
        events = []
        memory = SimpleNamespace(condense=lambda query: "standalone",
                                 add_user=lambda text: events.append(text),
                                 add_assistant=lambda text: events.append(text))
        profile = SimpleNamespace(context_block=lambda user: "Profile facts")
        feedback = SimpleNamespace(apply=lambda ranked, vector: ranked)
        result = self.pipe.answer("follow up", memory=memory, profile=profile, feedback=feedback)
        self.assertEqual(result["standalone_query"], "standalone")
        self.assertEqual(events[0], "follow up")
        self.assertEqual(len(events), 2)
        stale = InMemoryVectorStore(self.ids, self.store.matrix)
        with self.assertRaisesRegex(ValueError, "Rebuild"):
            RagPipeline(self.chunks, stale, self.bm25, self.embedder)
