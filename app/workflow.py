"""Application-controlled, bounded review workflow.

validate_input -> bind_snapshot -> gemma_lookup_plan -> execute_local_lookups -> gemma_assessment
-> medgemma_assessment -> reconcile_if_needed -> validate_result -> persist_result

State is saved after every completed stage so an interrupted run resumes from the last completed
stage. An in-flight model call that did not complete is simply re-run.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import traceback

from . import storage
from .config import CONFIG_SNAPSHOT, DATA, LIMITS, PROMPTS, RUN_BUDGET_S
from .evidence import load_snapshot
from .models import GemmaClient, MedGemmaClient, ModelError, model_identity
from .schemas import (NO_DISCREPANCY_MESSAGE, CaseRequest, FinalReview, LookupPlan, ModelAssessment,
                      VerifyResult)
from .tools import ToolError, execute, resolve_path

STAGES = ["validate_input", "bind_snapshot", "gemma_lookup_plan", "execute_local_lookups",
          "gemma_assessment", "medgemma_assessment", "reconcile_if_needed", "verify_result", "validate_result",
          "persist_result"]
STAGE_LABEL = {"gemma_lookup_plan": "Gathering evidence", "execute_local_lookups": "Gathering evidence",
               "gemma_assessment": "Gemma 4 assessment", "medgemma_assessment": "MedGemma review",
               "reconcile_if_needed": "MedGemma review", "gemma_fallback_review": "MedGemma review", "verify_result": "Result", "validate_result": "Result",
               "persist_result": "Result"}


def prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(case: dict, profile: dict, snapshot_id: str, identities: dict) -> str:
    prompt_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PROMPTS.glob("*.txt"))}
    rules_hash = hashlib.sha256((DATA / "medication_rules.json").read_bytes()).hexdigest()
    attachments = [a["file_id"] for a in case["presentation"].get("attachments", [])]
    material = {"case": case, "profile": profile, "snapshot": snapshot_id, "models": identities,
                "config": CONFIG_SNAPSHOT, "prompts": prompt_hashes, "rules": rules_hash,
                "attachments": attachments}
    return hashlib.sha256(canonical(material).encode()).hexdigest()


def case_sha(case: dict) -> str:
    return hashlib.sha256(canonical(case).encode()).hexdigest()


def passages_block(passages: list[dict]) -> str:
    return "\n\n".join(
        f"[{p['evidence_id']}] ({p['source_kind']}; {p['publisher']}; published {p['published_at'] or 'unknown'}; "
        f"{p['document_locator']})\n{p['exact_excerpt']}" for p in passages) or "(no passages retrieved)"


def brief_block(snapshot_id: str) -> str:
    from .research_brief import load_brief
    b = load_brief(snapshot_id)
    if not b or not b["observations"]:
        return ""
    lines = [f"- {o['statement']} [{', '.join(o['evidence_ids'])}]" for o in b["observations"]]
    return ("\n\nWEEKLY EVIDENCE SUMMARY (model-written context; the cited passages are authoritative; "
            "surveillance never establishes a patient's diagnosis):\n" + "\n".join(lines))


def case_block(case: dict, profile: dict) -> str:
    return ("DOCTOR PROFILE (tailors explanation only; does not decide correctness):\n" + canonical(profile)
            + "\n\nCASE (JSON; null = unknown):\n" + json.dumps(case, indent=1, ensure_ascii=False))


# ----------------------------------------------------------------------------- validation
def _valid_paths(paths: list[str], case: dict) -> list[str]:
    out = []
    for p in paths or []:
        if p.split(".")[0].split("[")[0] in ("patient", "presentation", "doctor_plan") and resolve_path(case, p)[0]:
            out.append(p)
    return out


def validate(final: dict, case: dict, allowed_ids: set[str], med_matches: list[dict]) -> dict:
    withheld, sections = [], {"secondary_diagnoses": [], "discrepancies": [], "suggested_checks": [],
                              "questions": []}
    proposed = (case["doctor_plan"].get("proposed_diagnosis") or "").lower()

    seen = set()
    for d in final.get("discrepancies", []):
        key = (d["issue_type"], d["summary"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        ev = [e for e in d["evidence_ids"] if e in allowed_ids]
        paths = _valid_paths(d["case_fact_paths"], case)
        if not paths:
            withheld.append({"item": d["summary"], "reason": "no referenced case fact exists in this revision"})
            continue
        if d["status"] == "supported" and not ev:
            d = {**d, "status": "unresolved"}
            withheld.append({"item": d["summary"], "reason": "no valid evidence reference; shown as unresolved"})
        sections["discrepancies"].append({**d, "evidence_ids": ev, "case_fact_paths": paths})

    for s in final.get("secondary_diagnoses", [])[:4]:
        ev = [e for e in s["evidence_ids"] if e in allowed_ids]
        paths = _valid_paths(s["case_fact_paths"], case)
        if s["condition"].lower() in proposed or (proposed and proposed in s["condition"].lower()):
            continue
        if not ev or not paths:
            withheld.append({"item": s["condition"], "reason": "not supported by both a case fact and a cited passage"})
            continue
        sections["secondary_diagnoses"].append({**s, "evidence_ids": ev, "case_fact_paths": paths})

    for c in final.get("suggested_checks", []):
        ev = [e for e in c["evidence_ids"] if e in allowed_ids]
        if not ev:
            withheld.append({"item": c["check"], "reason": "suggested check lacks a valid evidence reference"})
            continue
        sections["suggested_checks"].append({**c, "evidence_ids": ev})

    for q in final.get("questions", []):
        root = q["field_path"].split(".")[0].split("[")[0]
        if root in ("patient", "presentation", "doctor_plan"):
            sections["questions"].append(q)

    # Deterministic, source-backed medication rule matches are never silently dropped.
    covered = {p for d in sections["discrepancies"] for p in d["case_fact_paths"]}
    for m in med_matches:
        med_path = m["case_fact_paths"][0]
        if not any(c == med_path or c.startswith(med_path + ".") for c in covered):
            sections["discrepancies"].append({
                "issue_type": m["issue_type"], "summary": m["warning"],
                "why": "Matched a limited source-backed medication check; the final model review did not address it.",
                "case_fact_paths": m["case_fact_paths"], "evidence_ids": m["evidence_ids"],
                "status": "unresolved", "origin": f"rule:{m['rule_id']}"})

    sections["secondary_diagnoses"] = sections["secondary_diagnoses"][:LIMITS["max_secondary_dx"]]
    sections["discrepancies"] = sections["discrepancies"][:LIMITS["max_discrepancies"]]
    sections["questions"] = sections["questions"][:LIMITS["max_questions"]]

    supported = [d for d in sections["discrepancies"] if d["status"] == "supported"]
    unresolved = [d for d in sections["discrepancies"] if d["status"] == "unresolved"]
    disp = final["disposition"]
    if disp == "material_concern" and not supported:
        disp = "needs_clarification" if sections["questions"] else "insufficient_evidence"
    if disp == "no_material_discrepancy_identified" and (supported or unresolved):
        disp = "material_concern" if supported else "insufficient_evidence"
    if disp == "needs_clarification" and not sections["questions"]:
        disp = "insufficient_evidence"
    return {"disposition": disp, "sections": sections, "withheld": withheld,
            "disagreements": final.get("disagreements", []), "limitations": final.get("limitations", [])}


# ----------------------------------------------------------------------------- runner
class BudgetExceeded(RuntimeError):
    pass


def run(run_id: str) -> None:
    row = storage.get_run(run_id)
    case = json.loads(row["case_json"])
    profile = json.loads(row["profile_json"])
    state = json.loads(row["state_json"] or "{}")
    state.setdefault("done", [])
    state.setdefault("out", {})
    out = state["out"]
    t_start = time.perf_counter()
    gemma, med = GemmaClient(), MedGemmaClient()
    storage.update_run(run_id, status="running")

    times = out.setdefault("times", {})
    summ = out.setdefault("summaries", {})

    def persist_state(**extra):
        storage.update_run(run_id, state_json=json.dumps(state, ensure_ascii=False), **extra)

    def mark(name, actor):
        times[name] = {"start": time.time(), "end": None, "status": "running", "actor": actor}
        persist_state()

    def finish(name, status="done"):
        if name in times and times[name]["end"] is None:
            times[name]["end"] = time.time()
            times[name]["status"] = status

    def save(stage):
        state["done"].append(stage)
        finish(stage)
        summ[stage] = summarize(stage, out)
        storage.update_run(run_id, stage=stage, state_json=json.dumps(state, ensure_ascii=False))

    def budget():
        if time.perf_counter() - t_start + state.get("elapsed_before", 0) > RUN_BUDGET_S:
            raise BudgetExceeded(f"total run budget {RUN_BUDGET_S}s exceeded")

    def call(stage, client, system, user, schema):
        budget()
        storage.update_run(run_id, stage=stage + ":running")
        mark(stage, client.model)
        t0 = time.perf_counter()
        try:
            obj, audit = client.structured(system, user, schema)
        except ModelError as exc:
            finish(stage, "failed")
            persist_state()
            storage.event(run_id, stage, client.model, "failed", time.perf_counter() - t0,
                          {"error": str(exc), "calls": getattr(exc, "calls", None)})
            raise
        finish(stage)
        if stage not in STAGES:
            summ[stage] = summarize(stage, out, obj.model_dump())
        persist_state()
        storage.event(run_id, stage, client.model, "ok", time.perf_counter() - t0, audit)
        out.setdefault("model_calls", []).append({"stage": stage, "model": client.model,
                                                   "seconds": round(time.perf_counter() - t0, 2),
                                                   "attempts": len(audit["calls"])})
        out.setdefault("raw_responses", {})[stage] = [c["raw"] for c in audit["calls"]]
        return obj.model_dump()

    snap = load_snapshot(row["snapshot_id"])
    try:
        for stage in STAGES:
            if stage in state["done"]:
                continue
            mark(stage, "app")
            if stage == "validate_input":
                CaseRequest.model_validate(case)
            elif stage == "bind_snapshot":
                out["snapshot"] = {"snapshot_id": snap.snapshot_id, "created_at": snap.meta["created_at"],
                                   "coverage_notes": snap.meta["coverage_notes"],
                                   "failures": snap.meta["failures"]}
            elif stage == "gemma_lookup_plan":
                tools_desc = "Local evidence ids available: " + ", ".join(snap.records)
                out["plan"] = call(stage, gemma, prompt("gemma_plan"),
                                   case_block(case, profile) + "\n\n" + tools_desc, LookupPlan)
            elif stage == "execute_local_lookups":
                passages, results, rejected = {}, [], []
                for req in out["plan"]["lookups"][:LIMITS["initial_lookups"]]:
                    try:
                        res = execute(req, case, snap, limit=3)
                    except ToolError as exc:
                        rejected.append({"request": req, "error": str(exc)})
                        continue
                    results.append({"tool": res["tool"], "query": res["query"],
                                    "returned": [r["evidence_id"] for r in res["results"]],
                                    "medication_check": res.get("medication_check")})
                    for r in res["results"]:
                        passages.setdefault(r["evidence_id"], r)
                # Deterministic med check always runs so a model can't skip a source-backed rule.
                medres = execute({"tool": "check_medication_facts", "query": "app-enforced"}, case, snap, 3)
                out["medication_check"] = medres["medication_check"]
                for r in medres["results"]:
                    passages.setdefault(r["evidence_id"], r)
                out["lookups"] = {"executed": results, "rejected": rejected}
                out["passages"] = list(passages.values())[:LIMITS["max_passages"]]
                storage.event(run_id, stage, "app", "ok", None, out["lookups"])
            elif stage == "gemma_assessment":
                out["gemma_assessment"] = call(stage, gemma, prompt("gemma_assess"),
                                               case_block(case, profile) + "\n\nEVIDENCE PASSAGES:\n"
                                               + passages_block(out["passages"]) + brief_block(snap.snapshot_id),
                                               ModelAssessment)
            elif stage == "medgemma_assessment":
                try:
                    out["medgemma_assessment"] = call(stage, med, prompt("medgemma_assess"),
                                                      case_block(case, profile) + "\n\nEVIDENCE PASSAGES:\n"
                                                      + passages_block(out["passages"]) + brief_block(snap.snapshot_id),
                                                      ModelAssessment)
                except ModelError as exc:
                    # Degraded mode: Gemma finishes the review alone; clearly labelled, never "no discrepancy".
                    out["medgemma_assessment"] = None
                    out["medgemma_unavailable"] = str(exc)[:300]
            elif stage == "reconcile_if_needed":
                g, m = out["gemma_assessment"], out["medgemma_assessment"]
                rule_hits = out["medication_check"]["matched_rules"]
                material = any(a[k] for a in (g, m) if a for k in
                               ("candidate_conditions", "candidate_discrepancies", "missing_material_facts")) or rule_hits
                out["reconciled"] = bool(material)
                if not out["passages"]:
                    out["final"] = None
                elif not material and m is None:
                    out["final"] = {"disposition": "insufficient_evidence", "secondary_diagnoses": [],
                                    "discrepancies": [], "suggested_checks": [], "questions": [],
                                    "disagreements": [], "limitations": []}
                elif not material:
                    out["final"] = {"disposition": "no_material_discrepancy_identified", "secondary_diagnoses": [],
                                    "discrepancies": [], "suggested_checks": [], "questions": [],
                                    "disagreements": [], "limitations": []}
                else:
                    # one bounded extra lookup chosen by Gemma for the unresolved issue
                    issue = canonical({"gemma": g, "medgemma": m, "rule_hits": rule_hits})[:3000]
                    extra = call("reconcile_lookup", gemma, prompt("gemma_plan"),
                                 case_block(case, profile) + "\n\nUNRESOLVED ISSUES:\n" + issue
                                 + "\n\nRequest at most ONE targeted lookup about the specific disputed issue.",
                                 LookupPlan)
                    for req in extra["lookups"][:LIMITS["reconcile_lookups"]]:
                        try:
                            for r in execute(req, case, snap, limit=2)["results"]:
                                if r["evidence_id"] not in {p["evidence_id"] for p in out["passages"]}:
                                    out["passages"].append(r)
                        except ToolError as exc:
                            out["lookups"]["rejected"].append({"request": req, "error": str(exc)})
                    out["passages"] = out["passages"][:LIMITS["max_passages"]]
                    out["reconcile_lookup"] = extra
                    review_input = (case_block(case, profile)
                                    + "\n\nORCHESTRATOR ASSESSMENT (Gemma 4):\n" + canonical(g)
                                    + "\n\nINDEPENDENT ASSESSMENT (MedGemma):\n"
                                    + (canonical(m) if m else "UNAVAILABLE - MedGemma did not respond.")
                                    + "\n\nDETERMINISTIC MEDICATION CHECK:\n" + canonical(out["medication_check"])
                                    + "\n\nEVIDENCE PASSAGES:\n" + passages_block(out["passages"]))
                    final = None
                    if m is not None:
                        try:
                            final = call("medgemma_final_review", med, prompt("medgemma_review"), review_input,
                                         FinalReview)
                        except ModelError as exc:
                            out["medgemma_unavailable"] = str(exc)[:300]
                    if final is None:
                        final = call("gemma_fallback_review", gemma, prompt("medgemma_review")
                                     + "\n\nNOTE: The MedGemma specialist review is unavailable. You are producing a "
                                     "PROVISIONAL review without specialist confirmation. Never use "
                                     "no_material_discrepancy_identified.", review_input, FinalReview)
                    out["final"] = final
            elif stage == "verify_result":
                f = out.get("final")
                if f and any(f.get(k) for k in ("secondary_diagnoses", "discrepancies", "suggested_checks", "disagreements")):
                    shown = {k: list(enumerate(f.get(k, []))) for k in
                             ("secondary_diagnoses", "discrepancies", "suggested_checks")}
                    v = call("verify_result", gemma, prompt("verify"),
                             case_block(case, profile)
                             + "\n\nWHAT GEMMA 4 ACTUALLY SAID:\n" + canonical(out["gemma_assessment"])
                             + "\n\nWHAT MEDGEMMA ACTUALLY SAID:\n"
                             + (canonical(out["medgemma_assessment"]) if out.get("medgemma_assessment") else "UNAVAILABLE")
                             + "\n\nEVIDENCE PASSAGES:\n" + passages_block(out["passages"])
                             + brief_block(snap.snapshot_id)
                             + "\n\nFINAL REVIEW TO VERIFY (indexed):\n" + canonical(shown)
                             + "\nDISAGREEMENTS AS WRITTEN:\n" + canonical(f.get("disagreements", [])),
                             VerifyResult)
                    removed = []
                    for key, rk in (("secondary_diagnoses", "remove_secondary_diagnoses"),
                                    ("discrepancies", "remove_discrepancies"),
                                    ("suggested_checks", "remove_suggested_checks")):
                        drop = set(v[rk])
                        removed += [{"section": key, "item": it} for i, it in enumerate(f.get(key, [])) if i in drop]
                        f[key] = [it for i, it in enumerate(f.get(key, [])) if i not in drop]
                    f["disagreements"] = v["corrected_disagreements"]
                    out["verification"] = {"removed": removed, "reasons": v["reasons"], "verifier": gemma.model}
            elif stage == "validate_result":
                if out["final"] is None:
                    out["result"] = {"disposition": "insufficient_evidence", "sections": {}, "withheld": [],
                                     "disagreements": [], "limitations": ["No evidence passages were retrieved."]}
                else:
                    allowed = {p["evidence_id"] for p in out["passages"]}
                    out["result"] = validate(out["final"], case, allowed,
                                             out["medication_check"]["matched_rules"])
                if out.get("medgemma_unavailable"):
                    out["result"]["review_mode"] = "gemma_only"
                    out["result"]["limitations"] = [
                        "MedGemma specialist review was unavailable (" + out["medgemma_unavailable"][:120]
                        + "). This is a provisional Gemma 4 review without specialist confirmation."
                    ] + out["result"]["limitations"]
                    if out["result"]["disposition"] == "no_material_discrepancy_identified":
                        out["result"]["disposition"] = "insufficient_evidence"
                else:
                    out["result"]["review_mode"] = "two_model"
                if out["result"]["disposition"] == "no_material_discrepancy_identified":
                    out["result"]["message"] = NO_DISCREPANCY_MESSAGE
            elif stage == "persist_result":
                finish(stage)
                summ[stage] = summarize(stage, out)
                result = build_result(run_id, row, case, out)
                storage.update_run(run_id, status="completed", result_json=json.dumps(result, ensure_ascii=False),
                                   completed_at=storage.now())
            save(stage)
    except Exception as exc:  # noqa: BLE001 - failure is persisted and shown, never turned into "no discrepancy"
        state["elapsed_before"] = state.get("elapsed_before", 0) + time.perf_counter() - t_start
        for v in times.values():
            if v["end"] is None:
                v["end"], v["status"] = time.time(), "failed"
        storage.update_run(run_id, status="failed", state_json=json.dumps(state, ensure_ascii=False),
                           error=f"{type(exc).__name__}: {exc}")
        storage.event(run_id, "run", "app", "failed", None, traceback.format_exc()[-2000:])


PLURAL = {"alternative diagnosis": "alternative diagnoses", "discrepancy": "discrepancies"}


def _n(x, word):
    return f"{x} {word if x == 1 else PLURAL.get(word, word + 's')}"


def summarize(stage: str, out: dict, obj: dict | None = None) -> str:
    """One plain-language line describing what a step actually produced."""
    try:
        if stage == "validate_input":
            return "Case checked against the schema; unknowns kept as unknown"
        if stage == "bind_snapshot":
            return "Evidence pack " + out["snapshot"]["snapshot_id"] + " fixed for this run"
        if stage == "gemma_lookup_plan":
            p = out["plan"]
            return (_n(len(p["lookups"]), "lookup") + " planned: "
                    + "; ".join(f"{l['tool']} '{l['query']}'" for l in p["lookups"][:3]))
        if stage == "execute_local_lookups":
            rej = len(out["lookups"]["rejected"])
            mc = out["medication_check"]["matched_rules"]
            return (_n(len(out["passages"]), "passage") + " found, " + _n(len(mc), "medicine rule")
                    + " matched" + (f", {rej} request(s) refused" if rej else ""))
        if stage in ("gemma_assessment", "medgemma_assessment"):
            a = out.get(stage)
            if a is None:
                return "Unavailable: " + out.get("medgemma_unavailable", "model did not respond")[:120]
            return (_n(len(a["candidate_discrepancies"]), "possible problem") + ", "
                    + _n(len(a["candidate_conditions"]), "alternative diagnosis") + ", "
                    + _n(len(a["missing_material_facts"]), "missing fact"))
        if stage == "reconcile_lookup" and obj:
            return "; ".join(f"{l['tool']} '{l['query']}'" for l in obj["lookups"][:1]) or "No extra lookup needed"
        if stage in ("medgemma_final_review", "gemma_fallback_review") and obj:
            return (obj["disposition"].replace("_", " ") + ": " + _n(len(obj["discrepancies"]), "discrepancy")
                    + ", " + _n(len(obj["suggested_checks"]), "check") + ", " + _n(len(obj["questions"]), "question"))
        if stage == "reconcile_if_needed":
            return "Concern or disagreement found; final review ran" if out.get("reconciled") else \
                "No concern and no disagreement; final review not needed"
        if stage == "verify_result":
            v = out.get("verification")
            if not v:
                return "Nothing to verify"
            return (_n(len(v["removed"]), "item") + " removed as unsupported; disagreement text checked")
        if stage == "validate_result":
            r = out["result"]
            return ("Result: " + r["disposition"].replace("_", " ") + "; "
                    + _n(len(r["withheld"]), "item") + " withheld for missing sources")
        if stage == "persist_result":
            return "Advice saved and shown to the doctor"
    except (KeyError, TypeError):
        return ""
    return ""


def build_result(run_id: str, row: dict, case: dict, out: dict) -> dict:
    passages = {p["evidence_id"]: p for p in out["passages"]}
    return {
        **out["result"],
        "sources": list(passages.values()),
        "assessments": {"gemma": out.get("gemma_assessment"), "medgemma": out.get("medgemma_assessment"),
                        "reconciled": out.get("reconciled")},
        "medication_check": out.get("medication_check"),
        "verification": out.get("verification"),
        "lookups": out.get("lookups"),
        "execution": {
            "run_id": run_id, "case_id": case["case_id"], "case_revision": case["revision"],
            "snapshot": out.get("snapshot"), "models": json.loads(row["state_json"] or "{}").get("identities")
            or out.get("identities"), "config": CONFIG_SNAPSHOT, "model_calls": out.get("model_calls", []),
            "timeline": out.get("times", {}), "summaries": out.get("summaries", {}),
            "raw_responses": out.get("raw_responses", {}), "created_at": row["created_at"],
            "completed_at": storage.now(), "mode": "live_local", "technical_status": "completed"},
    }


def start_live(run_id: str) -> None:
    threading.Thread(target=run, args=(run_id,), daemon=True).start()


def identities() -> dict:
    return {"gemma": model_identity(GemmaClient().model), "medgemma": model_identity(MedGemmaClient().model)}
