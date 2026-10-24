from dataclasses import asdict
import unittest
from unittest.mock import patch

from aerolex import config
from aerolex.chunking import Chunk, chunk_corpus, iter_document_sections
from aerolex.chunking_recursive import chunk_corpus_recursive
from aerolex.loaders import Document


def make_document(text, doc_type="eu", doc_id="fixture"):
    return Document(
        doc_id=doc_id,
        title="Fixture law",
        doc_type=doc_type,
        text=text,
        source_path="fixture.pdf",
    )


class SectionRegressionTests(unittest.TestCase):
    def test_baseline_chunk_records(self):
        body = "1. " + "passenger compensation " * 12
        doc = make_document(f"Article 7\nCompensation\n{body}")
        expected = Chunk(
            chunk_id="fixture::article-7::0",
            doc_id="fixture",
            doc_title="Fixture law",
            doc_type="eu",
            section_label="Article 7",
            heading="Article 7 — Compensation",
            part_index=0,
            part_count=1,
            text=body.strip(),
            char_start=0,
        )
        self.assertEqual([asdict(c) for c in chunk_corpus([doc])], [asdict(expected)])

    def test_baseline_custom_packing_is_unchanged(self):
        doc = make_document("Article 7\nCompensation\n" + "a" * 1500 + "\n\n" + "b" * 1600)
        self.assertEqual([len(c.text) for c in chunk_corpus([doc])], [1500, 1751])

    def test_cfr_toc_skip_preserves_existing_section_offsets(self):
        doc = make_document(
            "Section 260.6\nTable of contents\nAuthority:\nSection 260.6\n"
            + "Refund terms " * 20,
            doc_type="cfr",
        )
        sections = list(iter_document_sections(doc))
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0][0], "§ 260.6")
        self.assertEqual(sections[0][3], len("Authority:\n"))

    def test_preamble_and_short_section_filtering_are_shared(self):
        preamble = "Introductory recitals. " * 10
        doc = make_document(
            preamble + "\nArticle 6\nShort heading\nToo short.\n"
            "Article 7\nCompensation\n" + "1. Details. " * 30
        )
        for chunker in (chunk_corpus, chunk_corpus_recursive):
            with self.subTest(chunker=chunker.__name__):
                chunks = chunker([doc])
                self.assertEqual(
                    [c.section_label for c in chunks], ["Preamble", "Article 7"]
                )

    def test_unrecognized_sections_stay_empty(self):
        doc = make_document("Unrecognized headings\n" + "content " * 100)
        self.assertEqual(chunk_corpus([doc]), [])
        self.assertEqual(chunk_corpus_recursive([doc]), [])


class RecursiveChunkingTests(unittest.TestCase):
    def test_all_document_families_preserve_metadata(self):
        for doc_type, boundary, label in (
            ("eu", "Article 7\nCompensation", "Article 7"),
            ("cfr", "§ 260.6 Refunds", "§ 260.6"),
            ("delta", "RULE 24: Refunds", "RULE 24"),
        ):
            with self.subTest(doc_type=doc_type):
                doc = make_document(
                    boundary + "\n" + "1. Passenger refund details. " * 160,
                    doc_type=doc_type,
                )
                baseline = chunk_corpus([doc])
                chunks = chunk_corpus_recursive([doc])
                self.assertGreater(len(chunks), 1)
                self.assertTrue(all(isinstance(c, Chunk) for c in chunks))
                self.assertEqual(
                    {(c.section_label, c.heading, c.char_start) for c in chunks},
                    {(c.section_label, c.heading, c.char_start) for c in baseline},
                )
                self.assertEqual([c.part_index for c in chunks], list(range(len(chunks))))
                for c in chunks:
                    self.assertEqual(c.part_count, len(chunks))
                    self.assertEqual(c.doc_id, doc.doc_id)
                    self.assertEqual(c.doc_title, doc.title)
                    self.assertEqual(c.doc_type, doc_type)
                    self.assertIn(label, c.citation)
                    self.assertIn(label, c.searchable_text)
                    self.assertTrue(c.text)
                    self.assertLessEqual(len(c.text), config.CHUNK_MAX_CHARS)

    def test_separators_and_long_unbroken_text_obey_size_limit(self):
        for body in (
            "\n\n".join(["Paragraph text " * 40] * 9),
            "\n".join(["Line text " * 10] * 45),
            "repeated words " * 400,
            "x" * 4000,
        ):
            with self.subTest(body_prefix=body[:20]):
                doc = make_document("Article 7\nCompensation\n" + body)
                chunks = chunk_corpus_recursive([doc])
                self.assertGreater(len(chunks), 1)
                self.assertTrue(all(0 < len(c.text) <= 1600 for c in chunks))
                self.assertEqual(chunks, chunk_corpus_recursive([doc]))
                self.assertEqual(len({c.chunk_id for c in chunks}), len(chunks))
                self.assertTrue(all(c.chunk_id.startswith("recursive::") for c in chunks))
                self.assertTrue(
                    {c.chunk_id for c in chunks}.isdisjoint(
                        c.chunk_id for c in chunk_corpus([doc])
                    )
                )

    def test_unbroken_text_uses_configured_character_overlap(self):
        doc = make_document("Article 7\nCompensation\n" + "x" * 4000)
        self.assertEqual(
            [len(c.text) for c in chunk_corpus_recursive([doc])],
            [1600, 1600, 1100],
        )

    def test_chunks_never_cross_legal_sections(self):
        doc = make_document(
            "Article 7\nCompensation\n"
            + "ONLY_SEVEN " * 400
            + "\nArticle 8\nRefund\n"
            + "ONLY_EIGHT " * 400
        )
        chunks = chunk_corpus_recursive([doc])
        self.assertEqual({c.section_label for c in chunks}, {"Article 7", "Article 8"})
        for chunk in chunks:
            if chunk.section_label == "Article 7":
                self.assertNotIn("ONLY_EIGHT", chunk.text)
            else:
                self.assertNotIn("ONLY_SEVEN", chunk.text)

    def test_short_final_part_is_not_filtered(self):
        doc = make_document("Article 7\nCompensation\n" + "x" * 1601)
        with patch.object(config, "CHUNK_OVERLAP_CHARS", 0):
            self.assertEqual(
                [len(c.text) for c in chunk_corpus_recursive([doc])], [1600, 1]
            )


if __name__ == "__main__":
    unittest.main()
