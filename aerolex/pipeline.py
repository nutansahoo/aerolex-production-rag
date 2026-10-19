"""End-to-end RAG pipeline composing all stages.

    query
      -> (memory) condense follow-up into a standalone query   [#8]
      -> dense semantic search (Gemini embeddings + NumPy)      [#3,#4]
      -> lexical BM25 search                                    [#2]
      -> reciprocal rank fusion                                 [#5]
      -> (feedback) FLAIR push-pull re-scoring                  [#10]
      -> cross-encoder re-ranking                               [#6]
      -> bounded reference expansion                            [#1]
      -> (profile) inject long-term facts                       [#9]
      -> grounded generation with inline citations              [#7]

Every stage is optional so the notebook can switch features on one at a time.
"""
from __future__ import annotations

from . import config
from .fusion import reciprocal_rank_fusion
from .context import ExpandedContext, ReferenceExpander


class RagPipeline:
    def __init__(self, chunks, store, bm25, embedder, reranker=None, generator=None):
        chunks = list(chunks)
        store.validate_chunks(chunks)
        self.chunks = {c.chunk_id: c for c in chunks}
        self.expander = ReferenceExpander(chunks)
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker
        self.generator = generator

    def retrieve_context(self, query: str, feedback=None, use_reranker: bool = True,
                         expand_references: bool = config.REFERENCE_EXPANSION_ENABLED,
                         **expansion_options) -> ExpandedContext:
        """Select relevance seeds, then assemble linked evidence without re-pruning."""
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

        seeds = [self.chunks[cid] for cid in final_ids if cid in self.chunks]
        if not expand_references:
            return ExpandedContext(seeds, [c.chunk_id for c in seeds])
        return self.expander.expand(
            seeds, candidate_ranks={cid: i for i, (cid, _) in enumerate(fused)},
            **expansion_options)

    def retrieve(self, query: str, feedback=None, use_reranker: bool = True,
                 expand_references: bool = config.REFERENCE_EXPANSION_ENABLED,
                 **expansion_options):
        """Return evidence chunks; disable expansion for the original top-k baseline."""
        return self.retrieve_context(
            query, feedback, use_reranker, expand_references, **expansion_options).chunks

    def answer(self, query: str, memory=None, profile=None,
               user_id: str = "default", feedback=None,
               expand_references: bool = config.REFERENCE_EXPANSION_ENABLED,
               **expansion_options) -> dict:
        """Full pipeline: optional memory/profile/feedback + grounded answer."""
        standalone = memory.condense(query) if memory is not None else query
        context = self.retrieve_context(standalone, feedback=feedback,
                                        expand_references=expand_references,
                                        **expansion_options)
        chunks = context.chunks

        profile_block = profile.context_block(user_id) if profile is not None else ""
        if self.generator is None:
            from .generate import Generator
            self.generator = Generator()
        result = self.generator.generate(standalone, chunks, profile_block=profile_block,
                                         context_diagnostics=context.metadata())
        result.update(query=query, standalone_query=standalone, chunks=chunks)
        result["context_expansion"] = context.metadata()

        if memory is not None:
            memory.add_user(query)
            memory.add_assistant(result["answer"])
        return result
