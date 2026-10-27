"""Compare BM25, dense, RRF and cross-encoder-reranked RRF with Gemini.

Run from the project root:
    python notebooks\\evaluation.py
    python notebooks\\evaluation.py --limit 1 --output cache\\smoke.csv

The full run makes 49 query-embedding calls and 196 answer-generation calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from google import genai
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from aerolex import config
from aerolex._gemini import generate_text
from aerolex.bm25 import BM25Index, tokenize
from aerolex.chunking import Chunk, load_chunks
from aerolex.embeddings import GeminiEmbedder
from aerolex.fusion import reciprocal_rank_fusion
from aerolex.rerank import CrossEncoderReranker
from aerolex.store import InMemoryVectorStore

GEN_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
DEFAULT_EMBEDDINGS = config.CACHE_DIR / "new_embeddings.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "notebooks" / "evaluation_results.csv"
STRATEGIES = ("lexical", "dense", "rrf", "rrf_reranked")
CSV_COLUMNS = (
    "question_id", "category", "source", "question", "model", "context_top_k",
    "lexical_chunk_ids", "lexical_answer",
    "dense_chunk_ids", "dense_answer",
    "rrf_chunk_ids", "rrf_answer",
    "rrf_reranked_chunk_ids", "rrf_reranked_answer",
)

# Use chunk-id citations to match the context labels.
SYSTEM = (
    "You are AeroLex, an assistant answering questions about airline passenger "
    "rights using ONLY the provided context passages. Every factual sentence "
    "must end with a citation tag like [[chunk_id]] using the exact chunk_id of the passage "
    "you used. If the context does not answer the question, say so plainly. "
    "Never invent regulations, article numbers, or passage numbers."
)


@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    category: str
    source: str
    question: str


QUESTION_GROUPS = (
    (
        "us", "U.S. passenger refund rights",
        "U.S. DOT refund rules - 14 CFR Part 260",
        (
            "My airline cancelled my flight, but says my ticket is nonrefundable. Can I get my money back?",
            "The airline moved my flight several hours later, and I no longer want to travel. Am I entitled to a refund?",
            "My flight now leaves from a different airport. Can I reject the change and get a refund?",
            "The airline added a connection to my nonstop flight. Can I get my money back instead?",
            "The airline cancelled my flight and offered a voucher. Can I insist on a refund to my original payment method?",
            "I paid by credit card. How long does the airline have to refund me?",
            "I paid with a debit card. How long should my refund take?",
            "My checked bag arrived a day late. Can I get my baggage fee back?",
            "My bag is missing. What do I need to report to qualify for a baggage-fee refund?",
            "I paid for Wi-Fi, but it didn't work. Can I get a refund, and do I need to report the problem?",
            "I paid for a particular seat but didn't get it. Can I get the seat fee refunded?",
            "The airline offered me a travel credit instead of money. What must it tell me about the credit's restrictions before I accept?",
        ),
    ),
    (
        "eu", "European passenger rights - not limited to Delta",
        "EU Regulation 261/2004",
        (
            "My flight from an EU airport was overbooked, and I was denied boarding. What compensation and help should I receive?",
            "What is the refund for a cancelled flight in EU region?",
            "The airline wants volunteers to take a later flight from an EU airport. If I volunteer, can I still choose a refund instead?",
            "My flight from an EU airport was cancelled the day before departure. Am I entitled to compensation as well as a refund?",
            "The airline told me about the cancellation of my flight from an EU airport three weeks in advance. What options do I still have?",
            "My flight from an EU airport was cancelled because of severe weather. Does the airline still have to provide meals and a hotel?",
            "My flight from an EU airport is delayed by three hours. Should the airline provide food and drinks?",
            "My flight from an EU airport has been delayed for more than five hours, and I want to abandon the trip. Can I get a refund?",
            "My flight from an EU airport is delayed until tomorrow. Who pays for my hotel and the journey there?",
            "My connecting flight on a trip departing from an EU airport was cancelled, and the trip no longer serves any purpose. Can I get a flight home as well as a refund?",
            "For my flight from an EU airport, the airline offered to fly me to a different airport nearby. Who pays for the transfer to my original airport?",
            "On my flight from an EU airport, I was moved from business class to economy. How much of my ticket price should I get back?",
            "For my flight from an EU airport, the airline offered compensation in vouchers. Can I ask for money instead?",
            "I booked a flight from an EU airport with frequent-flyer miles. Do I still have the same passenger rights?",
            "For my flight from an EU airport, the airline says I must sign away my rights before receiving help. Do I have to agree?",
            "The airline hasn't explained my rights after cancelling my flight from an EU airport. What information should it give me?",
            "The airline refuses to honour my rights for a flight from an EU airport. Where can I complain?",
        ),
    ),
    (
        "domestic", "Delta domestic flights",
        "Delta Domestic Conditions of Carriage",
        (
            "Delta bumped me from an oversold domestic flight. How much compensation could I receive?",
            "My Delta flight was cancelled, and the next available flight is tomorrow. What help will Delta provide?",
            "My suitcase was damaged on a Delta flight. What should I do next, and what can I claim?",
            "Delta lost my checked bag. What are the limits on what it will pay?",
            "My wheelchair was damaged during my flight. What assistance and compensation can I request?",
            "I'm travelling with a trained service dog. What do I need to arrange before my Delta flight?",
            "I need assistance getting through the airport and boarding. What help can Delta provide?",
            "My child needs to fly alone. What restrictions and arrangements apply?",
            "I missed my flight. Will Delta cancel the remaining flights on my booking?",
            "I arrived at the gate after boarding closed. What happens to my reservation?",
        ),
    ),
    (
        "international", "Delta international flights",
        "Delta International Conditions of Carriage",
        (
            "My bag was damaged on an international Delta flight. How quickly do I need to report it?",
            "My luggage was delayed on an international trip. What expenses can I claim, and what records should I keep?",
            "Delta lost my suitcase on an international flight. How much compensation can I claim?",
            "Delta changed my international flight, and I can no longer make the trip. What refund or rebooking options do I have?",
            "I bought my ticket from Delta, but another airline operated the flight. Which airline's rules apply to my baggage problem?",
            "I was denied boarding on an international Delta flight. What compensation rules apply to my journey?",
        ),
    ),
    (
        "multi", "Realistic questions requiring more than one document",
        "Multiple documents",
        (
            "My Delta flight from Paris to New York was cancelled, and I don't want the replacement flight. Can I get a refund, compensation, and reimbursement for my hotel?",
            "Delta cancelled my U.S. domestic flight and offered only an eCredit. Can I request my money back under U.S. refund rules?",
            "My suitcase was damaged and delivered late after a Delta international flight to the U.S. Can I claim for the damage and get my baggage fee refunded?",
            "I was denied boarding on a Delta flight leaving an EU airport. What does Delta offer, and what additional rights do I have under European rules?",
        ),
    ),
)

QUESTIONS = tuple(
    EvaluationQuestion(f"{prefix}-{number:02d}", category, source, question)
    for prefix, category, source, questions in QUESTION_GROUPS
    for number, question in enumerate(questions, start=1)
)


def keyword_tokens(query: str) -> list[str]:
    """Match the workshop's stopword-filtered query tokenizer."""
    return [token for token in tokenize(query) if token not in ENGLISH_STOP_WORDS]


def retrieve_variants(
    query: str,
    bm25: BM25Index,
    store: InMemoryVectorStore,
    embedder: GeminiEmbedder,
    context_top_k: int,
    *,
    by_id: Mapping[str, Chunk],
    reranker: CrossEncoderReranker,
) -> dict[str, list[str]]:
    """Rerank the full RRF pool before truncating the fourth arm's context."""
    if not 1 <= context_top_k <= min(
        config.BM25_TOP_K, config.DENSE_TOP_K, config.RRF_TOP_K
    ):
        raise ValueError("context_top_k must be positive and fit all candidate pools.")
    lexical = bm25.search(keyword_tokens(query), top_k=config.BM25_TOP_K)
    query_vec = embedder.embed_query(query)
    dense = store.search(query_vec, top_k=config.DENSE_TOP_K)
    fused = reciprocal_rank_fusion(
        [lexical, dense], k=config.RRF_K, top_k=config.RRF_TOP_K
    )
    variants = {}
    for strategy, ranking in zip(STRATEGIES[:3], (lexical, dense, fused)):
        ids = [cid for cid, _ in ranking[:context_top_k]]
        if len(ids) != context_top_k or len(set(ids)) != len(ids):
            raise RuntimeError(
                f"{strategy} must return {context_top_k} distinct chunks for {query!r}."
            )
        variants[strategy] = ids
    candidates = [(cid, by_id[cid].searchable_text) for cid, _ in fused]
    reranked = reranker.rerank(query, candidates, top_k=context_top_k)
    ids = [cid for cid, _ in reranked]
    if (
        len(ids) != context_top_k
        or len(set(ids)) != len(ids)
        or not set(ids).issubset(cid for cid, _ in fused)
    ):
        raise RuntimeError(
            f"rrf_reranked must return {context_top_k} distinct RRF candidate ids "
            f"for {query!r}."
        )
    variants["rrf_reranked"] = ids
    return variants


def generate_answer(
    client: genai.Client, query: str, chunks: Sequence[Chunk]
) -> str:
    """Use an independent Interactions API request with only this arm's chunks."""
    if not chunks:
        raise ValueError("Answer generation requires at least one context chunk.")
    context = "\n------------\n".join(
        f"[chunk_id={chunk.chunk_id}]({chunk.citation})\n{chunk.text}"
        for chunk in chunks
    )
    prompt = (
        f"{SYSTEM}\n\n"
        f"Context passages:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite every claim with [[chunk_id]]):"
    )
    answer = generate_text(client, GEN_MODEL, prompt)
    if not answer.strip():
        raise RuntimeError(f"Gemini returned an empty answer for {query!r}.")
    return answer


def run_evaluation(
    questions: Sequence[EvaluationQuestion],
    *,
    chunks: Sequence[Chunk],
    bm25: BM25Index,
    store: InMemoryVectorStore,
    embedder: GeminiEmbedder,
    client: genai.Client,
    reranker: CrossEncoderReranker,
    output_path: Path = DEFAULT_OUTPUT,
    context_top_k: int = 10,
    overwrite: bool = False,
) -> Path:
    """Write one row per question, flushing every completed four-arm result."""
    if not questions:
        raise ValueError("At least one evaluation question is required.")
    if not 1 <= context_top_k <= min(
        config.BM25_TOP_K, config.DENSE_TOP_K, config.RRF_TOP_K
    ):
        raise ValueError("context_top_k must be positive and fit all candidate pools.")
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if not by_id or len(by_id) != len(chunks):
        raise ValueError("The chunk corpus must be nonempty with unique chunk ids.")
    if len(store.ids) != len(by_id) or set(store.ids) != set(by_id):
        raise ValueError("Vector-store ids do not match the chunk corpus.")
    if len(bm25.ids) != len(by_id) or set(bm25.ids) != set(by_id):
        raise ValueError("BM25 ids do not match the chunk corpus.")
    if len(by_id) < context_top_k:
        raise ValueError("The corpus is smaller than context_top_k.")
    if len({q.question_id for q in questions}) != len(questions):
        raise ValueError("Evaluation question ids must be unique.")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(
        "w" if overwrite else "x", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        handle.flush()
        for number, question in enumerate(questions, start=1):
            print(
                f"[{number}/{len(questions)}] {question.question_id}: "
                f"{question.question}",
                flush=True,
            )
            variants = retrieve_variants(
                question.question, bm25, store, embedder, context_top_k,
                by_id=by_id, reranker=reranker,
            )
            row: dict[str, str | int] = {
                "question_id": question.question_id,
                "category": question.category,
                "source": question.source,
                "question": question.question,
                "model": GEN_MODEL,
                "context_top_k": context_top_k,
            }
            for strategy in STRATEGIES:
                ids = variants[strategy]
                print(f"  {strategy}: generating from {len(ids)} chunks", flush=True)
                row[f"{strategy}_chunk_ids"] = json.dumps(ids, ensure_ascii=False)
                row[f"{strategy}_answer"] = generate_answer(
                    client, question.question, [by_id[cid] for cid in ids]
                )
            writer.writerow(row)
            handle.flush()
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--context-top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output CSV."
    )
    args = parser.parse_args(argv)
    if args.limit is not None and not 1 <= args.limit <= len(QUESTIONS):
        parser.error(f"--limit must be between 1 and {len(QUESTIONS)}.")
    if not 1 <= args.context_top_k <= min(
        config.BM25_TOP_K, config.DENSE_TOP_K, config.RRF_TOP_K
    ):
        parser.error("--context-top-k must be positive and fit all candidate pools.")
    if args.output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {args.output}. Use --overwrite to replace it.")

    key = config.load_api_key()
    if not key:
        parser.error("GEMINI_API_KEY is required; configure it in the environment or .env.")
    chunks = load_chunks()
    store = InMemoryVectorStore.load(args.embeddings)
    if store.matrix.shape[1] != EMBED_DIM or not np.isfinite(store.matrix).all():
        parser.error(f"Expected finite {EMBED_DIM}-dimensional Gemini document vectors.")
    bm25 = BM25Index(
        [chunk.chunk_id for chunk in chunks],
        [chunk.searchable_text for chunk in chunks],
    )
    embedder = GeminiEmbedder(api_key=key, model=EMBED_MODEL, dim=EMBED_DIM)
    print(f"Loading cross-encoder: {config.RERANK_MODEL}", flush=True)
    reranker = CrossEncoderReranker()
    questions = QUESTIONS if args.limit is None else QUESTIONS[:args.limit]
    with genai.Client(api_key=key) as client:
        output = run_evaluation(
            questions,
            chunks=chunks,
            bm25=bm25,
            store=store,
            embedder=embedder,
            client=client,
            reranker=reranker,
            output_path=args.output,
            context_top_k=args.context_top_k,
            overwrite=args.overwrite,
        )
    print(
        f"Saved {len(questions)} questions and "
        f"{len(questions) * len(STRATEGIES)} answers to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())