"""Citation grammar for this corpus; graph traversal lives in context.py.

Resolve sections, not exact subclauses. Unknown instruments and missing targets
remain visible rather than being guessed from a coincidentally matching number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Resolution = Literal["pending", "resolved", "external", "ambiguous", "unsupported", "missing"]
_LOCATOR = r"(?:\s*\([a-z0-9]+\))*"
_NUMBER = r"\d+(?:\.\d+)?"
_CITATION = re.compile(
    rf"(?P<kind>\bArticles?|\bRules?|\bSections?|[§�]{{1,2}})\s+"
    rf"(?P<number>{_NUMBER})(?P<locator>{_LOCATOR})", re.I)
_NEXT = re.compile(
    rf"\s*(?:,\s*(?:(?:and|or)\s+)?|(?:and|or)\s+)"
    rf"(?:(?P<number>{_NUMBER})(?P<locator>{_LOCATOR})|"
    rf"(?P<inherited>(?:\s*\([a-z0-9]+\))+))", re.I)
_PARAGRAPH = re.compile(
    rf"\bparagraphs?\s+(?P<number>\d+|\([a-z0-9]+\))(?P<locator>{_LOCATOR})", re.I)
_INSTRUMENT = re.compile(
    r"(?:Council\s+)?Directive\s+[\d/]+(?:/[A-Z]+)?"
    r"|(?:the\s+)?(?:Warsaw|Montreal)\s+Convention"
    r"|(?:the\s+)?Treaty\b"
    r"|Regulation\s*(?:\([A-Z]+\)\s*)?(?:No\s*)?[\d/]+"
    r"|14\s*C\.?F\.?R\.?\s*(?:Part\s*)?\d+"
    r"|(?:Delta(?:['’]s)?\s+)?(?:Domestic|International|Canadian)\s+General\s+Rules\s+Tariff", re.I)


@dataclass
class ReferenceMention:
    text: str
    start: int
    end: int
    target_doc_id: str | None
    section_label: str
    locator: str
    status: Resolution
    reason: str = ""
    chunk_ids: list[str] = field(default_factory=list)


def _instrument_target(instrument: str) -> str | None:
    compact = re.sub(r"\s+", " ", instrument).lower()
    if "regulation" in compact and re.search(r"\b261/2004\b", compact):
        return "eu261"
    if re.search(r"14\s*c\.?f\.?r\.?\s*(?:part\s*)?260\b", compact):
        return "cfr260"
    if "delta" in compact and "domestic" in compact:
        return "delta_domestic"
    if "delta" in compact and "international" in compact:
        return "delta_intl"
    return None


def _scope(text: str, start: int, end: int, doc_id: str) -> tuple[str | None, Resolution, str]:
    # Look ahead through wrapped qualifiers, stopping before another citation.
    tail = re.split(r"[.;](?:\s|$)", text[end:end + 240], maxsplit=1)[0]
    following = _CITATION.search(tail)
    if following:
        tail = tail[:following.start()]
    instruments = [m.group() for m in _INSTRUMENT.finditer(tail)]
    # Also support an instrument immediately preceding "Article/Rule/Section".
    prefix = text[max(0, start - 120):start]
    prior = list(_INSTRUMENT.finditer(prefix))
    if prior and re.fullmatch(r"[\s,:\-]*(?:under\s+)?", prefix[prior[-1].end():], re.I):
        instruments.insert(0, prior[-1].group())
    elif prior and re.match(r"\s+thereof\b", tail, re.I):
        instruments.insert(0, prior[-1].group())
    if not instruments:
        return doc_id, "pending", ""
    targets = {_instrument_target(instrument) for instrument in instruments}
    if len(targets) > 1:
        return None, "ambiguous", "Multiple instrument qualifiers; not guessed"
    target = targets.pop()
    if target is None:
        return None, "external", "Instrument is outside the registered corpus"
    return target, "pending", ""


def _label(kind: str, number: str) -> tuple[str, str]:
    if kind.lower().startswith("article"):
        return f"Article {number}", "eu"
    if kind.lower().startswith("rule"):
        return f"RULE {number}", "delta"
    return f"§ {number}", "cfr"


def extract_mentions(text: str, doc_id: str, doc_type: str, section_label: str,
                     *, offset: int = 0) -> list[ReferenceMention]:
    mentions = []
    for match in _CITATION.finditer(text):
        items = [(match.group("number"), match.group("locator"))]
        end = match.end()
        while continuation := _NEXT.match(text, end):
            number = continuation.group("number")
            locator = continuation.group("locator") or ""
            if number is None:
                number = items[-1][0]
                parents = re.findall(r"\([a-z0-9]+\)", items[-1][1], re.I)
                locator = "".join(parents[:-1]) + continuation.group("inherited")
            items.append((number, locator))
            end = continuation.end()
        target, status, reason = _scope(text, match.start(), end, doc_id)
        for number, locator in items:
            label, family = _label(match.group("kind"), number)
            item_status, item_reason = status, reason
            if status == "pending":
                target_family = {"eu261": "eu", "cfr260": "cfr",
                                 "delta_domestic": "delta", "delta_intl": "delta"}.get(target, doc_type)
                if family != target_family or (family == "cfr" and not number.startswith("260.")):
                    item_status, item_reason = "unsupported", "Citation family/part is not indexed here"
            mentions.append(ReferenceMention(
                text=text[match.start():end], start=offset + match.start(), end=offset + end,
                target_doc_id=target, section_label=label,
                locator=re.sub(r"\s+", "", locator), status=item_status, reason=item_reason))
    for match in _PARAGRAPH.finditer(text):
        end = match.end()
        locators = [match.group("number") + match.group("locator")]
        while continuation := _NEXT.match(text, end):
            locators.append(continuation.group("inherited") or
                            continuation.group("number") + (continuation.group("locator") or ""))
            end = continuation.end()
        suffix = text[end:end + 40]
        if re.match(r"\s+of\s+Article\b", suffix, re.I):
            continue  # The explicit article mention handles the target section.
        for locator in locators:
            mentions.append(ReferenceMention(
                text=text[match.start():end], start=offset + match.start(), end=offset + end,
                target_doc_id=doc_id, section_label=section_label,
                locator=re.sub(r"\s+", "", locator), status="pending",
                reason="Same-section paragraph; supplied by section completion"))
    return sorted(mentions, key=lambda m: m.start)
