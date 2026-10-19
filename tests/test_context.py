import unittest

from aerolex.context import ContextBudgetError, ReferenceExpander
from tests.helpers import chunk, corpus


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.chunks = corpus()
        self.expander = ReferenceExpander(self.chunks)
        self.article = {c.section_label: c for c in self.chunks}

    def test_outgoing_incoming_and_provenance(self):
        result = self.expander.expand([self.article["Article 5"]], max_incoming=0)
        self.assertTrue({"Article 5", "Article 7", "Article 8", "Article 9"} <=
                        {c.section_label for c in result.chunks})
        incoming = self.expander.expand([self.article["Article 7"]])
        self.assertIn(self.article["Article 5"], incoming.chunks)
        self.assertTrue(any(e.direction == "incoming" for e in incoming.provenance))
        self.assertEqual(result.metadata()["evidence_words"],
                         sum(len(c.text.split()) for c in result.chunks))

    def test_cycles_depth_and_multiple_seeds(self):
        a, b, c = chunk("A"), chunk("B"), chunk("C")
        a.references = [b.chunk_id]
        b.references = [a.chunk_id, c.chunk_id]
        expander = ReferenceExpander([a, b, c])
        one = expander.expand([a], max_incoming=0)
        self.assertEqual(one.chunks, [a, b])
        self.assertIn("hop_limit", [d.reason for d in one.diagnostics])
        two = expander.expand([a], max_hops=2, max_incoming=0)
        self.assertEqual(two.chunks, [a, b, c])
        self.assertEqual(expander.expand([a, b, a]).seed_ids, [a.chunk_id, b.chunk_id])
        self.assertEqual(expander.expand([]).chunks, [])

    def test_whole_sections_added_atomically(self):
        a = chunk("A")
        b0, b1 = chunk("B", part=0, parts=2), chunk("B", part=1, parts=2)
        a.references = [b0.chunk_id, b1.chunk_id]
        expander = ReferenceExpander([a, b0, b1])
        limited = expander.expand([a], max_added_chunks=1)
        self.assertEqual(limited.chunks, [a])
        self.assertEqual(limited.diagnostics[0].reason, "added_chunk_limit")
        self.assertEqual(expander.expand([b1], max_hops=0).chunks, [b0, b1])
        with self.assertRaises(ContextBudgetError):
            expander.expand([b1], max_added_chunks=0)
        with self.assertRaises(ContextBudgetError):
            expander.expand([b1], max_context_words=1)

    def test_word_and_incoming_limits_ordering(self):
        a, b, c = chunk("A"), chunk("B", "two words"), chunk("C")
        b.references, c.references = [a.chunk_id], [a.chunk_id]
        expander = ReferenceExpander([a, b, c])
        result = expander.expand([a], max_incoming=1, candidate_ranks={c.chunk_id: 0})
        self.assertEqual(result.chunks, [a, c])
        self.assertIn("incoming_limit", [d.reason for d in result.diagnostics])
        result = expander.expand([a], max_context_words=2)
        self.assertNotIn(b, result.chunks)
        self.assertIn("word_limit", [d.reason for d in result.diagnostics])
        self.assertLessEqual(result.metadata()["evidence_words"], 2)

    def test_validation_and_unresolved_diagnostics(self):
        a = chunk("A")
        a.references = ["absent"]
        with self.assertRaisesRegex(ValueError, "Dangling"):
            ReferenceExpander([a])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            ReferenceExpander([chunk("B", part=0, parts=2)])
        for options in ({"max_hops": -1}, {"max_incoming": 1.5},
                        {"max_context_words": 0}, {"max_added_chunks": True}):
            with self.assertRaises(ValueError):
                self.expander.expand([], **options)
        with self.assertRaisesRegex(ValueError, "Seed"):
            self.expander.expand([chunk("absent")])
