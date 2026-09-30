"""MCP server: FDA drug-safety tools for any MCP client (Claude Desktop, Claude Code, Cursor, LangGraph...).

drug-safety-mcp                      # stdio (default, for desktop clients)
drug-safety-mcp --http --port 8000   # streamable HTTP at /mcp
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .openfda import NotFound, OpenFDAClient, clean_name

DISCLAIMER = (
    "Data comes from openFDA (drug labels, FAERS spontaneous reports, enforcement reports). "
    "FAERS counts are reports, not incidence rates, and do not prove causation. "
    "Not medical advice - consult the full prescribing information and a clinician."
)

mcp = FastMCP(
    "drug-safety",
    instructions=(
        "Tools for looking up FDA drug safety data. Always cite which tool/section a fact came from, "
        "and remind users that FAERS counts are not incidence rates. " + DISCLAIMER
    ),
)

_client: OpenFDAClient | None = None


def get_client() -> OpenFDAClient:
    global _client
    if _client is None:
        if os.getenv("DRUG_SAFETY_OFFLINE") == "1":
            from .offline import offline_transport

            _client = OpenFDAClient(transport=offline_transport())
        else:
            _client = OpenFDAClient()
    return _client


def set_client(client: OpenFDAClient | None) -> None:
    """Dependency injection hook (tests, custom base URLs, API keys)."""
    global _client
    _client = client


LABEL_SECTIONS = {
    "boxed_warning": "boxed_warning",
    "indications": "indications_and_usage",
    "contraindications": "contraindications",
    "warnings": "warnings_and_cautions",
    "adverse_reactions": "adverse_reactions",
    "drug_interactions": "drug_interactions",
}


def _section(label: dict, key: str, max_chars: int) -> str | None:
    raw = label.get(key) or (label.get("warnings") if key == "warnings_and_cautions" else None)
    if not raw:
        return None
    text = re.sub(r"\s+", " ", " ".join(raw)).strip()
    return text if len(text) <= max_chars else text[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def _names(label: dict) -> dict:
    ofda = label.get("openfda", {})
    return {
        "generic_name": (ofda.get("generic_name") or ["unknown"])[0].lower(),
        "brand_names": sorted({b for b in ofda.get("brand_name", [])})[:5],
        "manufacturer": (ofda.get("manufacturer_name") or [None])[0],
    }


# ---------------------------------------------------------------- output models (structured tool output)


class LabelSummary(BaseModel):
    found: bool
    query: str
    generic_name: str | None = None
    brand_names: list[str] = Field(default_factory=list)
    manufacturer: str | None = None
    has_boxed_warning: bool = False
    sections: dict[str, str] = Field(default_factory=dict)
    effective_time: str | None = None
    source: str = "openFDA /drug/label"


class ReactionCount(BaseModel):
    reaction: str
    reports: int
    share_of_top: float


class AdverseEventSummary(BaseModel):
    found: bool
    query: str
    total_reports: int = 0
    serious_only: bool = False
    top_reactions: list[ReactionCount] = Field(default_factory=list)
    caveat: str = "FAERS counts are spontaneous reports, not incidence; they do not establish causation."
    source: str = "openFDA /drug/event (FAERS)"


class Recall(BaseModel):
    recall_number: str | None
    classification: str | None
    status: str | None
    report_date: str | None
    reason: str | None
    product: str | None


class RecallSummary(BaseModel):
    found: bool
    query: str
    recalls: list[Recall] = Field(default_factory=list)
    source: str = "openFDA /drug/enforcement"


class InteractionEvidence(BaseModel):
    label_of: str
    section: str
    excerpt: str


class InteractionCheck(BaseModel):
    drug_a: str
    drug_b: str
    mentioned_in_labels: bool
    evidence: list[InteractionEvidence] = Field(default_factory=list)
    note: str = (
        "Absence of a label mention does not mean the combination is safe; labels often name drug classes "
        "(e.g. 'NSAIDs') rather than individual drugs."
    )


# ---------------------------------------------------------------- tools


@mcp.tool(
    title="Get FDA drug label",
    description=(
        "Summarise the FDA structured product label for a drug (generic or brand name): boxed warning, "
        "indications, contraindications, warnings, adverse reactions, drug interactions."
    ),
)
async def get_drug_label(
    drug: str,
    sections: list[str] | None = None,
    max_chars_per_section: int = 1200,
) -> LabelSummary:
    wanted = sections or list(LABEL_SECTIONS)
    bad = set(wanted) - set(LABEL_SECTIONS)
    if bad:
        raise ValueError(f"unknown sections {sorted(bad)}; choose from {sorted(LABEL_SECTIONS)}")
    try:
        label = await get_client().label(drug)
    except NotFound:
        return LabelSummary(found=False, query=drug)
    out = {s: txt for s in wanted if (txt := _section(label, LABEL_SECTIONS[s], max_chars_per_section))}
    return LabelSummary(
        found=True,
        query=drug,
        **_names(label),
        has_boxed_warning=bool(label.get("boxed_warning")),
        sections=out,
        effective_time=label.get("effective_time"),
    )


@mcp.tool(
    title="Top reported adverse events (FAERS)",
    description="Most frequently reported adverse reactions for a drug in FAERS, optionally serious reports only.",
)
async def top_adverse_events(drug: str, limit: int = 10, serious_only: bool = False) -> AdverseEventSummary:
    limit = max(1, min(limit, 50))
    client = get_client()
    try:
        counts, total = await asyncio.gather(
            client.adverse_event_counts(drug, limit, serious_only), client.adverse_event_total(drug)
        )
    except NotFound:
        return AdverseEventSummary(found=False, query=drug, serious_only=serious_only)
    top_sum = sum(c["count"] for c in counts) or 1
    return AdverseEventSummary(
        found=True,
        query=drug,
        total_reports=total,
        serious_only=serious_only,
        top_reactions=[
            ReactionCount(reaction=c["term"].lower(), reports=c["count"], share_of_top=round(c["count"] / top_sum, 3))
            for c in counts
        ],
    )


@mcp.tool(title="Recent FDA recalls", description="Most recent FDA enforcement (recall) reports for a drug.")
async def recent_recalls(drug: str, limit: int = 5) -> RecallSummary:
    try:
        rows = await get_client().recalls(drug, max(1, min(limit, 20)))
    except NotFound:
        return RecallSummary(found=False, query=drug)
    return RecallSummary(
        found=True,
        query=drug,
        recalls=[
            Recall(
                recall_number=r.get("recall_number"),
                classification=r.get("classification"),
                status=r.get("status"),
                report_date=r.get("report_date"),
                reason=(r.get("reason_for_recall") or "")[:400] or None,
                product=(r.get("product_description") or "")[:200] or None,
            )
            for r in rows
        ],
    )


def _excerpts(text: str, terms: set[str], width: int = 220) -> list[str]:
    out = []
    low = text.lower()
    for term in terms:
        for m in re.finditer(rf"\b{re.escape(term)}\b", low):
            start = max(0, m.start() - width // 2)
            out.append(("…" if start else "") + text[start : m.end() + width // 2].strip() + "…")
            break
    return out


@mcp.tool(
    title="Check label-documented interaction",
    description=(
        "Check whether either drug's FDA label mentions the other (by generic or brand name) in its drug "
        "interactions, warnings or contraindications sections, and return the relevant excerpts."
    ),
)
async def check_interaction(drug_a: str, drug_b: str) -> InteractionCheck:
    a, b = clean_name(drug_a), clean_name(drug_b)
    client = get_client()
    results = await asyncio.gather(client.label(a), client.label(b), return_exceptions=True)
    labels: dict[str, dict] = {}
    for name, res in zip((a, b), results, strict=True):
        if isinstance(res, NotFound):
            continue
        if isinstance(res, BaseException):
            raise res
        labels[name] = res

    def terms_for(name: str) -> set[str]:
        lab = labels.get(name)
        names = {name}
        if lab:
            meta = _names(lab)
            names |= {meta["generic_name"], *(n.lower() for n in meta["brand_names"])}
        return {n for n in names if n and n != "unknown"}

    evidence: list[InteractionEvidence] = []
    for owner, other in ((a, b), (b, a)):
        lab = labels.get(owner)
        if not lab:
            continue
        for sec in ("drug_interactions", "contraindications", "warnings_and_cautions", "boxed_warning"):
            text = _section(lab, sec, 100_000)
            if text:
                evidence += [
                    InteractionEvidence(label_of=owner, section=sec, excerpt=e)
                    for e in _excerpts(text, terms_for(other))
                ]
    return InteractionCheck(drug_a=a, drug_b=b, mentioned_in_labels=bool(evidence), evidence=evidence[:6])


# ---------------------------------------------------------------- resources & prompts


@mcp.resource("drug-safety://about", title="About this data", mime_type="text/plain")
def about() -> str:
    return DISCLAIMER


@mcp.prompt(title="Drug safety brief", description="Produce a cited one-page safety brief for a drug.")
def safety_brief(drug: str) -> str:
    return (
        f"Write a concise safety brief for {drug}. Use get_drug_label (boxed warning, contraindications, "
        f"interactions), top_adverse_events (serious_only=true) and recent_recalls. Structure: 1) key label "
        f"warnings, 2) most-reported serious adverse events with counts, 3) recalls in the last few years, "
        f"4) caveats. Cite the tool for every fact. End with: {DISCLAIMER}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Drug-safety MCP server")
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.http:
        mcp.settings.host, mcp.settings.port = args.host, args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
