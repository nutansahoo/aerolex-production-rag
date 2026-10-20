"""Reusable evidence assembly over explicit links, without domain-specific grammar."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

from . import config
from .chunking import Chunk

SectionId = tuple[str, str]


class ContextBudgetError(ValueError):
    """Complete seed sections exceed the configured evidence budget."""


@dataclass(frozen=True)
class ContextEdge:
    source_chunk_id: str
    target_chunk_id: str
    source_section: SectionId
    target_section: SectionId
    direction: str
    depth: int


@dataclass(frozen=True)
class ContextDiagnostic:
    reason: str
    source_section: SectionId | None
    target_section: SectionId | None
    detail: str


@dataclass
class ExpandedContext:
    chunks: list[Chunk]
    seed_ids: list[str]
    provenance: list[ContextEdge] = field(default_factory=list)
    diagnostics: list[ContextDiagnostic] = field(default_factory=list)
    max_hops: int = 0

    def metadata(self) -> dict:
        return {"seed_ids": self.seed_ids, "max_hops": self.max_hops,
                "evidence_words": sum(len(c.text.split()) for c in self.chunks),
                "provenance": [asdict(e) for e in self.provenance],
                "diagnostics": [asdict(d) for d in self.diagnostics]}


class ReferenceExpander:
    def __init__(self, chunks: list[Chunk]):
        self.by_id = {c.chunk_id: c for c in chunks}
        if len(self.by_id) != len(chunks):
            raise ValueError("Reference graph contains duplicate chunk IDs")
        self.sections: dict[SectionId, list[Chunk]] = {}
        self.outgoing: dict[SectionId, dict[SectionId, list[tuple[str, str]]]] = {}
        self.incoming: dict[SectionId, dict[SectionId, list[tuple[str, str]]]] = {}
        for c in chunks:
            self.sections.setdefault(c.section_id, []).append(c)
        for members in self.sections.values():
            members.sort(key=lambda c: c.part_index)
            if ([c.part_index for c in members] != list(range(members[0].part_count))
                    or any(c.part_count != len(members) for c in members)):
                raise ValueError(f"Incomplete section in corpus: {members[0].section_id}")
        for c in chunks:
            for cid in c.references:
                if cid not in self.by_id:
                    raise ValueError(f"Dangling reference from {c.chunk_id}: {cid}")
                target = self.by_id[cid].section_id
                if target == c.section_id:
                    continue
                edge = (c.chunk_id, cid)
                self.outgoing.setdefault(c.section_id, {}).setdefault(target, []).append(edge)
                self.incoming.setdefault(target, {}).setdefault(c.section_id, []).append(edge)
        self.order = {section: i for i, section in enumerate(self.sections)}

    def expand(
        self, seeds: list[Chunk], *,
        max_hops: int = config.REFERENCE_MAX_HOPS,
        max_incoming: int = config.REFERENCE_MAX_INCOMING,
        max_added_chunks: int = config.REFERENCE_MAX_ADDED_CHUNKS,
        max_context_words: int = config.REFERENCE_MAX_CONTEXT_WORDS,
        candidate_ranks: Mapping[str, int] | None = None,
    ) -> ExpandedContext:
        if (any(not isinstance(n, int) or isinstance(n, bool) or n < 0
                for n in (max_hops, max_incoming, max_added_chunks))
                or not isinstance(max_context_words, int) or isinstance(max_context_words, bool)
                or max_context_words <= 0):
            raise ValueError("Expansion limits must be nonnegative integers; word budget must be positive")
        seed_ids = list(dict.fromkeys(c.chunk_id for c in seeds))
        for cid in seed_ids:
            if cid not in self.by_id:
                raise ValueError(f"Seed is not in this corpus: {cid}")
        root_sections = list(dict.fromkeys(self.by_id[cid].section_id for cid in seed_ids))
        result = ExpandedContext([], seed_ids, max_hops=max_hops)
        included: set[SectionId] = set()
        word_count = 0
        edge_keys = set()
        diagnostic_keys = set()

        def diagnostic(reason, source, target, detail):
            key = reason, source, target, detail
            if key not in diagnostic_keys:
                diagnostic_keys.add(key)
                result.diagnostics.append(ContextDiagnostic(reason, source, target, detail))

        def add_section(section, source=None, *, seed=False):
            nonlocal word_count
            if section in included:
                return False
            members = self.sections[section]
            words = sum(len(c.text.split()) for c in members)
            additions = sum(c.chunk_id not in seed_ids for c in result.chunks + members)
            reason = ("added_chunk_limit" if additions > max_added_chunks else
                      "word_limit" if word_count + words > max_context_words else "")
            if reason:
                if seed:
                    raise ContextBudgetError(
                        f"Complete seed section {section} exceeds {reason}. "
                        "Use fewer seeds, increase the budget, or explicitly disable expansion.")
                diagnostic(reason, source, section, "Whole section omitted; no partial section supplied")
                return False
            included.add(section)
            result.chunks.extend(members)
            word_count += words
            return True

        def provenance(source, target, pairs, direction, depth):
            for a, b in pairs:
                key = a, b, direction, depth
                if key not in edge_keys:
                    edge_keys.add(key)
                    result.provenance.append(ContextEdge(a, b, source, target, direction, depth))

        for section in root_sections:
            add_section(section, seed=True)
        frontier = root_sections
        ranks = candidate_ranks or {}

        def incoming_rank(section):
            rank = min((ranks[c.chunk_id] for c in self.sections[section] if c.chunk_id in ranks),
                       default=float("inf"))
            return rank, self.order[section]

        for depth in range(1, max_hops + 1):
            next_frontier = []
            for direction, adjacency in (("outgoing", self.outgoing), ("incoming", self.incoming)):
                for section in frontier:
                    neighbors = adjacency.get(section, {})
                    targets = list(neighbors)
                    if direction == "incoming":
                        targets.sort(key=incoming_rank)
                    incoming_used = 0
                    for target in targets:
                        if direction == "incoming" and target not in included:
                            if incoming_used >= max_incoming:
                                diagnostic("incoming_limit", section, target,
                                           "Incoming source section not followed")
                                continue
                            incoming_used += 1
                        added = add_section(target, section)
                        if target in included:
                            source, destination = ((section, target) if direction == "outgoing"
                                                   else (target, section))
                            provenance(source, destination, neighbors[target], direction, depth)
                        if added:
                            next_frontier.append(target)
            frontier = next_frontier
            if not frontier:
                break

        for section in frontier:
            for adjacency in (self.outgoing, self.incoming):
                for target in adjacency.get(section, {}):
                    if target not in included:
                        diagnostic("hop_limit", section, target, "Outside configured traversal depth")
        for c in result.chunks:
            for mention in c.reference_mentions:
                if mention.status not in {"resolved", "pending"}:
                    target = ((mention.target_doc_id, mention.section_label)
                              if mention.target_doc_id else None)
                    diagnostic(mention.status, c.section_id, target,
                               f"{mention.text}: {mention.reason}")
        return result
