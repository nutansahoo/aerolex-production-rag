import unittest

from aerolex import config
from aerolex.bm25 import BM25Index
from aerolex.chunking import chunk_corpus
from aerolex.context import ReferenceExpander
from aerolex.loaders import load_documents


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = load_documents()
        cls.chunks = chunk_corpus(cls.documents)
        cls.by_id = {c.chunk_id: c for c in cls.chunks}
        cls.expander = ReferenceExpander(cls.chunks)

    def test_budget_ids_and_source_spans(self):
        self.assertEqual(len(self.documents), 4)
        docs = {d.doc_id: d for d in self.documents}
        self.assertEqual(len(self.by_id), len(self.chunks))
        for c in self.chunks:
            self.assertLessEqual(len(c.text.split()), config.CHUNK_MAX_WORDS)
            self.assertEqual(c.text, docs[c.doc_id].text[c.char_start:c.char_end])
            self.assertTrue(all(cid in self.by_id for cid in c.references))
        sections = {c.section_id for c in self.chunks}
        self.assertIn(("eu261", "Article 18"), sections)
        self.assertIn(("cfr260", "§ 260.9"), sections)
        self.assertFalse(any("...." in c.heading for c in self.chunks))
        notification = " ".join(c.text for c in self.chunks if c.section_id == ("cfr260", "§ 260.9"))
        contract = " ".join(c.text for c in self.chunks if c.section_id == ("cfr260", "§ 260.11"))
        self.assertIn("notify affected consumers", notification)
        self.assertNotIn("notify affected consumers", contract)

    def test_article5_and_article7(self):
        for label, required in [
            ("Article 5", {"Article 5", "Article 7", "Article 8", "Article 9"}),
            ("Article 7", {"Article 5", "Article 7", "Article 8"}),
        ]:
            with self.subTest(label=label):
                seed = next(c for c in self.chunks if c.section_id == ("eu261", label))
                result = self.expander.expand([seed])
                self.assertTrue(required <= {c.section_label for c in result.chunks})
                selected = {c.section_id for c in result.chunks}
                self.assertEqual({c.chunk_id for c in result.chunks},
                                 {c.chunk_id for c in self.chunks if c.section_id in selected})
                self.assertLessEqual(result.metadata()["evidence_words"], 6000)
                self.assertLessEqual(len(result.chunks) - 1, 20)

    def test_no_external_or_cross_contract_mislinks(self):
        mentions = [m for c in self.chunks if c.section_id == ("eu261", "Article 2")
                    for m in c.reference_mentions if m.section_label == "Article 2"]
        self.assertTrue(mentions)
        self.assertTrue(all(m.status == "external" and not m.chunk_ids for m in mentions))
        for c in self.chunks:
            if c.doc_type == "delta":
                for cid in c.references:
                    self.assertEqual(self.by_id[cid].doc_id, c.doc_id)

    def test_seed_controlled_evidence_coverage(self):
        cases = [
            ("cancellation compensation", "eu261", "Article 5", {"Article 5", "Article 7"}),
            ("rerouting reduction", "eu261", "Article 7", {"Article 7", "Article 8"}),
            ("reimbursement means", "eu261", "Article 8", {"Article 7", "Article 8"}),
            ("care after cancellation", "eu261", "Article 5", {"Article 5", "Article 9"}),
            ("refund conditions", "cfr260", "§ 260.6", {"§ 260.6", "§ 260.8"}),
        ]
        for question, doc_id, section, required in cases:
            with self.subTest(question=question):
                seed = next(c for c in self.chunks if c.section_id == (doc_id, section))
                expanded = self.expander.expand([seed])
                actual = {c.section_label for c in expanded.chunks if c.doc_id == doc_id}
                self.assertTrue(required <= actual)
                self.assertGreater(len(required & actual), len(required & {section}))

    def test_real_bm25_seeds_separately_from_graph_fixtures(self):
        index = BM25Index([c.chunk_id for c in self.chunks], [c.searchable_text for c in self.chunks])
        for query, required in [
            ("EU261 Article 5 cancellation right to compensation",
             {("eu261", "Article 5"), ("eu261", "Article 7")}),
            ("14 CFR 260.6 refund alternative compensation offer",
             {("cfr260", "§ 260.6"), ("cfr260", "§ 260.8")}),
        ]:
            with self.subTest(query=query):
                hit = index.search(query, top_k=1)[0][0]
                seed = self.by_id[hit]
                result = self.expander.expand([seed])
                self.assertTrue(required <= {c.section_id for c in result.chunks})
                self.assertGreater(len(result.chunks), 1)
