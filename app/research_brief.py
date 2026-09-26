"""Weekly research brief: python -m app.research_brief

After a snapshot is published, Gemma 4 chooses a few local lookups and writes a short dated brief
(regional observations, seasonal context, coverage gaps), every claim citing evidence IDs. Claims with
medical implications go to MedGemma, which marks each supported / unsupported / unresolved; unsupported
ones are omitted. The original passages stay authoritative. Raw model responses are stored.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from typing import Literal

from pydantic import Field

from .config import SNAPSHOTS
from .evidence import current_snapshot_id, load_snapshot
from .models import GemmaClient, MedGemmaClient, ModelError, model_identity
from .schemas import LookupPlan, Strict
from .tools import ToolError, execute

BRIEF_TOOL_BUDGET = 3


class Observation(Strict):
    statement: str
    kind: Literal["regional_observation", "seasonal_context", "coverage_gap"]
    geography: str | None = None
    reporting_period: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    has_medical_implication: bool = False


class Brief(Strict):
    observations: list[Observation] = Field(default_factory=list)


class Verdict(Strict):
    index: int
    verdict: Literal["supported", "unsupported", "unresolved"]
    note: str


class BriefReview(Strict):
    items: list[Verdict] = Field(default_factory=list)


PLAN_PROMPT = """[brief_plan p1] You prepare a short weekly evidence brief for doctors in India from a LOCAL evidence
snapshot. Choose at most 3 search_evidence lookups (keywords) that find dated regional surveillance, seasonal
context, and coverage limits. Only the tool name search_evidence is allowed here. Return JSON only."""

WRITE_PROMPT = """[brief_write p1] Write a short dated evidence brief from ONLY the passages supplied.
Rules:
- Each observation cites the evidence_ids it comes from. No citation = do not write it.
- State reporting periods and geography exactly as given. An old report is dated context, not a current outbreak.
- Outbreak counts are context only; never turn them into a diagnosis or a probability for a patient.
- Add coverage_gap observations for what the snapshot does NOT cover (e.g. states that did not report).
- Mark has_medical_implication=true only if the statement says something clinical (e.g. what doctors should consider).
Return at most 6 observations. Return JSON only."""

REVIEW_PROMPT = """[brief_review p1] You are a medical reviewer. For each numbered statement, check it against the
supplied passages. verdict: supported (the passages support it as written), unsupported (they do not, or it
overreaches, e.g. treats surveillance as diagnosis), unresolved (cannot tell). Give a one-line note. Return JSON only."""


def passages_text(passages: list[dict]) -> str:
    return "\n\n".join(f"[{p['evidence_id']}] ({p['source_kind']}; {p['title']}; period "
                       f"{p.get('reporting_period_start') or '?'} to {p.get('reporting_period_end') or '?'}; "
                       f"{p['document_locator']})\n{p['exact_excerpt']}" for p in passages)


def build(snapshot_id: str | None = None) -> dict:
    snap = load_snapshot(snapshot_id or current_snapshot_id())
    gemma, med = GemmaClient(), MedGemmaClient()
    raw, rejected, passages = {}, [], {}

    plan, audit = gemma.structured(PLAN_PROMPT, "Snapshot scope: " + snap.meta["scope"]
                                   + "\nEvidence ids: " + ", ".join(snap.records), LookupPlan)
    raw["plan"] = [c["raw"] for c in audit["calls"]]
    for req in plan.model_dump()["lookups"][:BRIEF_TOOL_BUDGET]:
        try:
            if req["tool"] != "search_evidence":
                raise ToolError("only search_evidence is allowed for the brief")
            for r in execute(req, {}, snap, limit=4)["results"]:
                passages.setdefault(r["evidence_id"], r)
        except ToolError as exc:
            rejected.append({"request": req, "error": str(exc)})
    # surveillance coverage passages are always included so gaps are stated
    for r in snap.records.values():
        if r["source_kind"] == "surveillance":
            passages.setdefault(r["evidence_id"], r)
    plist = list(passages.values())[:8]

    brief, audit = gemma.structured(WRITE_PROMPT, passages_text(plist), Brief)
    raw["write"] = [c["raw"] for c in audit["calls"]]
    valid_ids = {p["evidence_id"] for p in plist}
    observations = []
    for o in brief.model_dump()["observations"][:6]:
        o["evidence_ids"] = [e for e in o["evidence_ids"] if e in valid_ids]
        if o["evidence_ids"]:
            observations.append(o)

    review_status, medgemma_error = "not_needed", None
    to_review = [i for i, o in enumerate(observations) if o["has_medical_implication"]]
    if to_review:
        numbered = "\n".join(f"{i}. {observations[i]['statement']} (cites {', '.join(observations[i]['evidence_ids'])})"
                             for i in to_review)
        try:
            rev, audit = med.structured(REVIEW_PROMPT, "STATEMENTS:\n" + numbered + "\n\nPASSAGES:\n"
                                        + passages_text(plist), BriefReview)
            raw["medgemma_review"] = [c["raw"] for c in audit["calls"]]
            verdicts = {v.index: v for v in rev.items}
            for i in to_review:
                v = verdicts.get(i)
                observations[i]["review"] = v.model_dump() if v else {"verdict": "unresolved", "note": "not reviewed"}
            review_status = "reviewed_by_medgemma"
        except ModelError as exc:
            medgemma_error = str(exc)[:200]
            for i in to_review:
                observations[i]["review"] = {"verdict": "unresolved", "note": "MedGemma unavailable"}
            review_status = "medgemma_unavailable"
    included = [o for o in observations if o.get("review", {}).get("verdict") not in ("unsupported",)
                and not (o["has_medical_implication"] and o.get("review", {}).get("verdict") != "supported")]
    omitted = [o for o in observations if o not in included]

    out = {"snapshot_id": snap.snapshot_id, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "models": {"gemma": model_identity(gemma.model), "medgemma": model_identity(med.model)},
           "review_status": review_status, "medgemma_error": medgemma_error,
           "observations": included, "omitted": omitted, "rejected_lookups": rejected,
           "passages_used": [p["evidence_id"] for p in plist], "raw_responses": raw,
           "note": "Model-written summary. The cited original passages are authoritative."}
    (SNAPSHOTS / snap.snapshot_id / "snapshot_brief.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def load_brief(snapshot_id: str) -> dict | None:
    p = SNAPSHOTS / snapshot_id / "snapshot_brief.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


if __name__ == "__main__":
    res = build(sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps({k: res[k] for k in ("snapshot_id", "review_status", "observations", "omitted")},
                     indent=1, ensure_ascii=False))
