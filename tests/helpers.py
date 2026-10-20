from aerolex.chunking import Chunk, chunk_corpus
from aerolex.loaders import Document


def document(text, doc_id="eu261", doc_type="eu"):
    return Document(doc_id, doc_id, doc_type, text, "fixture.pdf")


def corpus():
    return chunk_corpus([document(
        "Article 5\nCancellation\n"
        "1. Compensation under Article 7 unless notice was given two weeks before departure. "
        "Assistance under Articles 8 and 9.\n"
        "Article 7\nCompensation\n1. EUR 600 for eligible flights. Re-routing under Article 8.\n"
        "Article 8\nRe-routing\n1. Reimbursement or re-routing.\n"
        "Article 9\nCare\n1. Meals and accommodation.\n"
        "Article 18\nRepeal\nThe earlier regulation is repealed."
    )])


def chunk(label, text="evidence", *, doc_id="demo", part=0, parts=1):
    return Chunk(f"{doc_id}::{label}::{part}", doc_id, doc_id, "eu",
                 label, label, part, parts, text, 0, len(text))
