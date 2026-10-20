# aerolex-production-rag
# ✈️ AeroLex: Production-Grade RAG Pipeline.  A reference implementation of a production-grade, multi-stage Retrieval-Augmented Generation (RAG) system built from first principles in Python—demonstrated over complex statutory aviation regulations (EU261, US DOT 14 CFR 260) and carrier contracts of carriage.

## Feature #1: reference-aware evidence

The notebooks teach two distinct steps: **structure-first chunking** at ingestion
and **reference expansion** after retrieval. Better chunk boundaries alone do not
fetch conditions documented in another article.

```text
PDF -> clean -> sections -> word-bounded chunks -> explicit citation links
query -> BM25 + dense -> fusion -> feedback/rerank -> seeds
      -> complete seed sections -> bounded linked sections -> cited answer
```

The legal parser is in `aerolex/references.py`; the reusable graph traversal is in
`aerolex/context.py`. No framework, graph database, semantic-splitter model, or new
runtime dependency is introduced. EU Article 5 points to Articles 7–9; incoming
links let an Article 7 seed discover Article 5's cancellation conditions.

### Offline workshop start

Use the existing `rag` conda environment and launch Jupyter from this repository.
Feature #1 needs the existing NumPy dependency and a supported PDF reader.
The pypdf fallback needs a version supporting `extraction_mode="layout"` (pypdf
4 or later); CFR uses layout-aware extraction to keep indented clauses under
their actual section headings.

```bash
conda activate rag
# If the existing workshop dependencies are not already installed:
python -m pip install numpy "pypdf>=4"
```

Open `notebooks/workshop.ipynb` for the two small reference-ID/expansion exercises,
or `notebooks/workshop_solution.ipynb` for completed functions. Setup and Feature
#1 run without an API key or vector cache. Legal parsing stays in backend Python;
the notebook workflow is deliberately small:

```python
from aerolex.loaders import load_documents
from aerolex.chunking import chunk_corpus
from aerolex.context import ReferenceExpander

docs = load_documents()
chunks = chunk_corpus(docs)
expander = ReferenceExpander(chunks)
seed = next(c for c in chunks if c.section_id == ("eu261", "Article 7"))
context = expander.expand([seed])
print([c.citation for c in context.chunks])
print(context.metadata())  # discovery paths, evidence size, omissions
```

The examples compare seed-controlled **evidence coverage**, not search accuracy
or answer correctness. Later stages additionally use the repository's existing
`rank-bm25`, `google-genai`, `sentence-transformers`, and optionally
`python-dotenv`; the exercise notebook also uses scikit-learn. Gemini embeddings
and live generation need `GEMINI_API_KEY`. Local reranking needs its downloaded
cross-encoder model. A complete fresh-environment setup can use:

```bash
python -m pip install numpy "pypdf>=4" rank-bm25 google-genai \
  sentence-transformers python-dotenv scikit-learn jupyterlab ipykernel
python -m ipykernel install --user --name rag --display-name "Python (rag)"
python -m jupyterlab
```

Choose the `Python (rag)` kernel. These are existing workshop dependencies, not
additional requirements of reference expansion. Do not run embedding cells
without a key or assume the entire notebook is an offline application.

### Configuration and evidence limits

Settings live in `aerolex/config.py`; `ReferenceExpander.expand()` also accepts
per-call overrides.

| Setting | Default | Meaning |
|---|---:|---|
| `CHUNK_MAX_WORDS` | 320 | Maximum whitespace-delimited words per chunk |
| `CHUNK_OVERLAP_WORDS` | 40 | Maximum overlap, included within the chunk budget |
| `REFERENCE_MAX_HOPS` | 1 | Reference traversals; section completion costs no hop |
| `REFERENCE_MAX_INCOMING` | 2 | New incoming source sections per frontier section |
| `REFERENCE_MAX_ADDED_CHUNKS` | 20 | All non-seed additions, including seed-section siblings |
| `REFERENCE_MAX_CONTEXT_WORDS` | 6000 | Sum of evidence text words, including overlap |

Chunking prefers paragraphs/numbered blocks, then sentence boundaries, then word
windows, never crossing sections or cutting inside a word. It preserves short
provisions, cleaned-source character spans, citation labels, and deterministic
content-sensitive IDs. Word budgets are not exact model-token limits.

Expansion follows outgoing links before incoming links, deduplicates results,
and terminates on cycles. Incoming candidates use existing retrieval ranks when
provided, otherwise source order. A linked section is added only if **all** parts
fit; otherwise diagnostics identify the omission. If complete seed sections
alone cannot fit, `ContextBudgetError` asks you to reduce seeds or raise limits.
Nothing silently slices away an exception. Try:

```python
limited = expander.expand([seed], max_added_chunks=0)
two_hops = expander.expand([seed], max_hops=2)
print(limited.metadata()["diagnostics"])
```

In the assembled pipeline, expansion is enabled by default:

```python
context = pipe.retrieve_context(question)  # detailed evidence and trace
chunks = pipe.retrieve(question)           # same evidence, list return shape
baseline = pipe.retrieve(question, expand_references=False)  # original top-k
answer = pipe.answer(question, expand_references=True)
```

Reranking selects seeds; expanded evidence is **not** reranked back down to top-k.
Generation numbers all final chunks independently and receives explicit
missing-evidence notes. `answer["references"]` still means answer citations;
`answer["context_expansion"]` contains retrieval provenance and diagnostics.
The offline generator remains extractive and returns the same diagnostics.

### Reference metadata and supported grammar

Each chunk's JSON includes the following fields (IDs are illustrative):

```json
{
  "chunk_id": "eu261::article-5::0::<content-digest>",
  "section_label": "Article 5",
  "references": ["eu261::article-9::0::<content-digest>"],
  "reference_mentions": [{
    "text": "Article 9(1)(b)",
    "start": 123,
    "end": 138,
    "target_doc_id": "eu261",
    "section_label": "Article 9",
    "locator": "(1)(b)",
    "status": "resolved",
    "reason": "",
    "chunk_ids": ["eu261::article-9::0::<content-digest>"]
  }]
}
```

`start`/`end` and chunk `char_start`/`char_end` are offsets in the **cleaned
document**, not PDF bytes or page coordinates. References are extracted before
subdivision and assigned to intersecting source spans, preserving mentions that
straddle chunk boundaries.

Supported patterns include `Article 7`, `Articles 8 and 9`,
`Article 9(1)(a) and 9(2)`, `§ 260.6(a)(2)`, line-wrapped section references,
`Rule 6(C)`, `Rules 23(A) and 23(B)`, `Rule 20(c)(1) and (4)`, and same-section
paragraph mentions. Subclause locators are retained but are **not independently
indexed or verified**: a resolved locator fetches its whole parent section.

Unqualified citations stay within their document and citation family.
Explicit registered instrument names can link across the corpus. External,
ambiguous, unsupported, and missing targets remain visible in mention metadata
and expansion diagnostics. Article 2 of Directive 90/314/EEC must not become
EU261 Article 2; domestic and international Delta rule numbers are distinct.
This is a tested corpus adapter, not a universal legal citation parser.

For another domain, provide chunks with section identity and explicit target IDs,
replace the citation adapter, and reuse `ReferenceExpander` unchanged.

### Cache regeneration

The chunk JSON now has a versioned envelope with source/policy fingerprints and
the chunk records. Old schema/policy/source caches raise `ChunkCacheError`.
Malformed files remain errors rather than being silently replaced. Both
notebooks rebuild known-stale chunk caches explicitly.

```python
from aerolex.chunking import save_chunks, load_chunks

chunks = chunk_corpus(docs)
save_chunks(chunks, documents=docs)
chunks = load_chunks(documents=docs)
```

If using nondefault `max_words`/`overlap_words` arguments to `chunk_corpus`,
pass the same values to `save_chunks` and `load_chunks`.
Supply `documents` when saving/loading to validate source freshness.

Rebuild embeddings once after rechunking; never relabel an old matrix with new IDs:

```python
from aerolex.embeddings import GeminiEmbedder
from aerolex.store import InMemoryVectorStore

matrix = GeminiEmbedder().embed_documents([c.searchable_text for c in chunks])
store = InMemoryVectorStore.from_chunks(chunks, matrix)
store.save()
store.validate_chunks(chunks)
```

NPZ artifacts record the ordered IDs/indexed-text fingerprint and reject legacy
or mismatched vectors. Graph metadata alone does not require re-embedding.
The exercise notebook uses `cache/new_embeddings.npz`; the solution uses
`cache/embeddings.npz`. Generated cache artifacts are ignored by git. Profile
and feedback databases are not deleted; changed text/heading produces new chunk
IDs so old feedback cannot silently apply to replaced content.

### Checks and limitations

```bash
conda run -n rag python -m unittest discover -s tests -v
conda run -n rag python -m compileall -q aerolex
```

Tests use the standard library, existing dependencies, deterministic model
substitutes, temporary caches, and the four real PDFs. They cover bounds,
scoping, graph traversal, cache invalidation, expanded citations, and offline
notebook integration without API calls. Real-retrieval coverage is evaluated
separately from forced-seed graph coverage.

This is a small explicit-reference graph, **not full GraphRAG or a legal reasoning
engine**. Incoming mentions are discovery hints, not proof of applicability.
One hop cannot guarantee all conditions are present; implicit dependencies,
missing user facts, external sources, and later amendments/judgments remain
limitations. In particular, the EU261 PDF alone does not establish every later
delay-compensation ruling. Ask precise conditional questions and inspect
omission diagnostics before treating an amount table as an entitlement.

Exact subclause indexing, external-source crawling, temporal law resolution,
multilingual parsing, and embedding-based semantic chunking are intentionally
deferred. Neither word-based splitting nor graph expansion is a universal
accuracy improvement; measure evidence recall, context cost, and answer quality.

Design references: [Microsoft chunking guidance](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents),
[document-layout chunking](https://learn.microsoft.com/en-us/azure/search/search-how-to-semantic-chunking),
[LlamaIndex recursive retrieval](https://developers.llamaindex.ai/python/framework/integrations/retrievers/recursive_retriever_nodes/),
and [node parser distinctions](https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/modules/).
