"""End-to-end RAG pipeline composing all stages.

    query
      -> (memory) condense follow-up into a standalone query   [#8]
      -> dense semantic search (Gemini embeddings + NumPy)      [#3,#4]
      -> lexical BM25 search                                    [#2]
      -> reciprocal rank fusion                                 [#5]
      -> (feedback) FLAIR push-pull re-scoring                  [#10]
      -> cross-encoder re-ranking                               [#6]
      -> (profile) inject long-term facts                       [#9]
      -> grounded generation with inline citations              [#7]

Every stage is optional so the notebook can switch features on one at a time.
"""
from __future__ import annotations

from . import config
from .fusion import reciprocal_rank_fusion


class RagPipeline:
    def __init__(self, chunks, store, bm25, embedder, reranker=None, generator=None):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker
        self.generator = generator

    def retrieve(self, query: str, feedback=None, use_reranker: bool = True):
        """Hybrid retrieve -> fuse -> (feedback) -> (re-rank) -> final chunks."""
        qvec = self.embedder.embed_query(query)
        dense = self.store.search(qvec, top_k=config.DENSE_TOP_K)
        sparse = self.bm25.search(query, top_k=config.BM25_TOP_K)
        fused = reciprocal_rank_fusion([sparse, dense], top_k=config.RRF_TOP_K)

        if feedback is not None:
            fused = feedback.apply(fused, qvec)

        if use_reranker and self.reranker is not None:
            cands = [(cid, self.chunks[cid].searchable_text)
                     for cid, _ in fused if cid in self.chunks]
            reranked = self.reranker.rerank(query, cands, top_k=config.RERANK_TOP_K)
            final_ids = [cid for cid, _ in reranked]
        else:
            final_ids = [cid for cid, _ in fused[:config.RERANK_TOP_K]]

        return [self.chunks[cid] for cid in final_ids if cid in self.chunks]

    def answer(self, query: str, memory=None, profile=None,
               user_id: str = "default", feedback=None, *,
               system_prompt: str | None = None,
               user_prompt_template: str | None = None) -> dict:
        """Return answer text and query/chunk metadata, without resolved references."""
        standalone = memory.condense(query) if memory is not None else query
        chunks = self.retrieve(standalone, feedback=feedback)

        profile_block = profile.context_block(user_id) if profile is not None else ""
        if self.generator is None:
            from .generate import Generator
            self.generator = Generator()
        generation_options = {"profile_block": profile_block}
        if system_prompt is not None:
            generation_options["system_prompt"] = system_prompt
        if user_prompt_template is not None:
            generation_options["user_prompt_template"] = user_prompt_template
        answer = self.generator.generate(standalone, chunks, **generation_options)
        if not isinstance(answer, str):
            raise TypeError("generator.generate() must return a string")
        result = {
            "answer": answer,
            "query": query,
            "standalone_query": standalone,
            "chunks": chunks,
        }

        if memory is not None:
            memory.add_user(query)
            memory.add_assistant(answer)
        return result
