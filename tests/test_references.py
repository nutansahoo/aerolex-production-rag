import unittest

from aerolex.chunking import chunk_corpus, chunk_document
from aerolex.references import extract_mentions
from tests.helpers import corpus, document


class ReferenceTests(unittest.TestCase):
    def test_eu_lists_and_inherited_locators(self):
        mentions = extract_mentions(
            "See Articles 7, 8 and 9. Article 9(1)(a) and 9(2).", "eu261", "eu", "Article 5")
        self.assertEqual([(m.section_label, m.locator) for m in mentions], [
            ("Article 7", ""), ("Article 8", ""), ("Article 9", ""),
            ("Article 9", "(1)(a)"), ("Article 9", "(2)")])

    def test_wrapped_cfr_and_delta(self):
        mentions = extract_mentions("See §\n260.6(a)(2) and 260.8.", "cfr260", "cfr", "§ 260.2")
        self.assertEqual([(m.section_label, m.locator) for m in mentions],
                         [("§ 260.6", "(a)(2)"), ("§ 260.8", "")])
        mentions = extract_mentions("Rules 23(A) and 23(B); Rule 20(c)(1) and (4).",
                                    "delta_intl", "delta", "RULE 1")
        self.assertEqual([(m.section_label, m.locator) for m in mentions], [
            ("RULE 23", "(A)"), ("RULE 23", "(B)"),
            ("RULE 20", "(c)(1)"), ("RULE 20", "(c)(4)")])

    def test_external_and_explicit_cross_document(self):
        cases = [
            ("Article 2, point 2, of Council Directive 90/314/EEC", "external", None),
            ("Article 17 of the Montreal Convention", "external", None),
            ("Rule 2 of the Canadian General Rules Tariff", "external", None),
            ("Treaty establishing the Community, in particular Article 80(2) thereof",
             "external", None),
            ("Article 7 of Regulation (EC) No 261/2004", "pending", "eu261"),
        ]
        for text, status, target in cases:
            with self.subTest(text=text):
                m = extract_mentions(text, "delta_intl", "delta", "RULE 1")[0]
                self.assertEqual(m.status, status)
                self.assertEqual(m.target_doc_id, target)
        docs = [document("Article 7\nAmounts\nEUR 600."),
                document("RULE 1: Terms\nArticle 7 of Regulation (EC) No 261/2004.",
                         "delta_intl", "delta")]
        chunks = chunk_corpus(docs)
        self.assertEqual(chunks[1].references, [chunks[0].chunk_id])

    def test_document_scope_missing_and_same_section(self):
        docs = [document("RULE 1: Scope\nSee Rule 2.\nRULE 2: Terms\nDomestic.",
                         "delta_domestic", "delta"),
                document("RULE 1: Scope\nSee Rule 2.\nRULE 2: Terms\nInternational.",
                         "delta_intl", "delta")]
        chunks = chunk_corpus(docs)
        by_id = {c.chunk_id: c for c in chunks}
        for c in chunks:
            self.assertTrue(all(by_id[cid].doc_id == c.doc_id for cid in c.references))
        c = chunk_document(document("Article 5\nScope\nParagraphs 1 and 2; Article 99."))[0]
        self.assertEqual([m.locator for m in c.reference_mentions if m.status == "resolved"],
                         ["1", "2"])
        self.assertFalse(c.references)
        self.assertEqual(c.reference_mentions[-1].status, "missing")

    def test_all_parts_and_boundary_spanning_citation(self):
        doc = document("Article 5\nCancellation\none two three four See Article 7.\n"
                       "Article 7\nAmounts\n" + "word " * 22)
        chunks = chunk_document(doc, max_words=6, overlap_words=1)
        sources = [c for c in chunks if c.section_label == "Article 5"]
        targets = [c.chunk_id for c in chunks if c.section_label == "Article 7"]
        self.assertEqual(set(cid for c in sources for cid in c.references), set(targets))
        self.assertTrue(any(m.text == "Article 7" for c in sources for m in c.reference_mentions))

    def test_all_resolved_references_exist(self):
        chunks = corpus()
        ids = {c.chunk_id for c in chunks}
        self.assertTrue(all(cid in ids for c in chunks for cid in c.references))
