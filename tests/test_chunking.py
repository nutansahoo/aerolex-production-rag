import unittest

from aerolex.chunking import _split_spans, chunk_corpus, chunk_document
from aerolex.loaders import clean_text
from tests.helpers import document


class ChunkingTests(unittest.TestCase):
    def test_boundaries_spans_and_words(self):
        text = ("Article 1\nScope\n1. " + "alpha " * 70 + "\n\n2. " + "beta " * 70
                + "\nArticle 2\nDefinitions\n1. " + "gamma " * 60)
        doc = document(text)
        chunks = chunk_document(doc, max_words=24, overlap_words=5)
        self.assertGreater(len(chunks), 5)
        for c in chunks:
            self.assertLessEqual(len(c.text.split()), 24)
            self.assertEqual(c.text, text[c.char_start:c.char_end])
            self.assertNotIn("Article 2", c.text)
            self.assertNotIn("alp", c.text.split())
            self.assertTrue(c.text)
        self.assertEqual([c.chunk_id for c in chunks], [
            c.chunk_id for c in chunk_document(doc, max_words=24, overlap_words=5)])
        for a, b in zip(chunks, chunks[1:]):
            if a.section_id == b.section_id:
                self.assertGreater(b.char_start, a.char_start)
                self.assertGreater(b.char_end, a.char_end)
                self.assertLessEqual(len(text[b.char_start:a.char_end].split()), 5)
            else:
                self.assertGreaterEqual(b.char_start, a.char_end)

    def test_block_and_sentence_boundaries(self):
        body = "First short sentence. Second short sentence.\n\nThird short paragraph."
        spans = _split_spans(body, 8, 2)
        self.assertEqual(body[slice(*spans[0])], "First short sentence. Second short sentence.")
        sentence_spans = _split_spans("One two three. Four five six. Seven eight nine.", 7, 0)
        self.assertEqual("One two three. Four five six.",
                         "One two three. Four five six. Seven eight nine."[slice(*sentence_spans[0])])

    def test_progress_with_large_overlap_and_empty_body(self):
        for size in (1, 2, 8):
            spans = _split_spans(" ".join(str(i) for i in range(100)), size, size - 1)
            self.assertLessEqual(len(spans), 100)
            self.assertEqual(spans[-1][1], len(" ".join(str(i) for i in range(100))))
        self.assertEqual(_split_spans("", 3, 0), [])
        for size, overlap in ((0, 0), (2, 2), (2, -1), (2.5, 0), (True, 0)):
            with self.assertRaises(ValueError):
                _split_spans("", size, overlap)

    def test_short_provisions_and_content_identity(self):
        doc = document("Article 18\nRepeal\nRepealed.\nArticle 19\nEntry\nEffective.")
        chunks = chunk_document(doc)
        self.assertEqual([c.section_label for c in chunks], ["Article 18", "Article 19"])
        changed = chunk_document(document(doc.text.replace("Repealed.", "Repealed today.")))
        self.assertNotEqual(chunks[0].chunk_id, changed[0].chunk_id)
        self.assertEqual(chunks[1].chunk_id, changed[1].chunk_id)
        changed_heading = chunk_document(document(doc.text.replace("Repeal", "Revocation", 1)))
        self.assertNotEqual(chunks[0].chunk_id, changed_heading[0].chunk_id)

    def test_cfr_offsets_and_delta_toc(self):
        text = "§ 260.9 Notice.\nAuthority: Congress\n§ 260.9 Notice.\nShort but binding."
        doc = document(text, "cfr260", "cfr")
        chunks = chunk_document(doc)
        section = next(c for c in chunks if c.section_label == "§ 260.9")
        self.assertEqual(text[section.char_start:section.char_end], "Short but binding.")
        cleaned = clean_text("RULE 1: Provisions ........ 2\nRULE 1: Provisions\nBinding.", "delta")
        chunks = chunk_document(document(cleaned, "delta_domestic", "delta"))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Binding.")

    def test_missing_duplicate_headings_and_duplicate_documents(self):
        with self.assertRaisesRegex(ValueError, "no recognized"):
            chunk_document(document("not a recognized regulation"))
        with self.assertRaisesRegex(ValueError, "duplicate section"):
            chunk_document(document("Article 1\nScope\nOne.\nArticle 1\nScope\nTwo."))
        doc = document("Article 1\nScope\nOne.")
        with self.assertRaisesRegex(ValueError, "duplicate chunk"):
            chunk_corpus([doc, doc])
