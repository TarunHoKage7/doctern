"""Contract checks. These use MOCK model adapters and are tests, not a real-model demo."""
import copy
import json
import pathlib

import pytest
from pydantic import ValidationError

from app import storage, workflow
from app.evidence import current_snapshot_id, load_snapshot
from app.models import ModelError
from app.schemas import CaseRequest, FinalReview, LookupPlan, ModelAssessment
from app.tools import ToolError, check_medication_facts, execute

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "data" / "cases").glob("*.json")}
RAVI, MEERA, SANJAY = CASES["01-ravi-nsaid-dengue"], CASES["02-meera-no-discrepancy"], CASES["03-sanjay-incomplete"]
SNAP = load_snapshot(current_snapshot_id())


def test_unknown_vs_explicit_absence():
    c = CaseRequest.model_validate(SANJAY)
    assert c.patient.allergies is None            # unknown
    assert CaseRequest.model_validate(RAVI).patient.allergies == []   # explicitly none reported


def test_case_needs_symptom_or_question():
    bad = copy.deepcopy(MEERA)
    bad["presentation"]["symptoms"] = []
    with pytest.raises(ValidationError):
        CaseRequest.model_validate(bad)


def test_arbitrary_tool_rejected():
    with pytest.raises(ToolError):
        execute({"tool": "run_shell", "query": "rm -rf /"}, RAVI, SNAP, 3)
    with pytest.raises(ToolError):
        execute({"tool": "search_evidence", "query": "x" * 500}, RAVI, SNAP, 3)


def test_medication_rule_uses_clinical_facts_not_names():
    renamed = copy.deepcopy(RAVI)
    renamed["case_id"], renamed["patient"]["display_name"] = "zzz-other", "Someone"
    assert check_medication_facts(renamed)["matched_rules"]
    no_dengue = copy.deepcopy(RAVI)
    no_dengue["doctor_plan"]["proposed_diagnosis"] = "osteoarthritis flare"
    no_dengue["presentation"]["test_results"] = [{"name": "Dengue NS1 antigen (rapid)", "value": "negative",
                                                   "unit": None, "measured_at": None}]
    no_dengue["presentation"]["notes"] = None
    assert not check_medication_facts(no_dengue)["matched_rules"]


def test_missing_fact_is_unknown_not_clear():
    out = check_medication_facts(SANJAY)
    assert not out["matched_rules"] and out["unknowns"][0]["unknown_fields"] == ["presentation.test_results"]


def _final(**kw):
    base = {"disposition": "material_concern", "secondary_diagnoses": [], "discrepancies": [],
            "suggested_checks": [], "questions": [], "disagreements": [], "limitations": []}
    base.update(kw)
    return FinalReview.model_validate(base).model_dump()


def test_invalid_citations_and_fabricated_case_refs_withheld():
    f = _final(discrepancies=[
        {"issue_type": "guideline_conflict", "summary": "fake", "why": "x", "case_fact_paths": ["patient.nonexistent"],
         "evidence_ids": ["dengue-groupA-antipyretic"], "status": "supported"},
        {"issue_type": "guideline_conflict", "summary": "bad cite", "why": "x",
         "case_fact_paths": ["doctor_plan.proposed_medications[0]"], "evidence_ids": ["made-up-id"], "status": "supported"}])
    out = workflow.validate(f, RAVI, {"dengue-groupA-antipyretic"}, [])
    assert [d["summary"] for d in out["sections"]["discrepancies"]] == ["bad cite"]
    assert out["sections"]["discrepancies"][0]["status"] == "unresolved"
    assert out["disposition"] != "material_concern"


def test_rule_match_cannot_become_no_discrepancy():
    rule = check_medication_facts(RAVI)["matched_rules"]
    out = workflow.validate(_final(disposition="no_material_discrepancy_identified"), RAVI,
                            {"dengue-groupA-antipyretic"}, rule)
    assert out["disposition"] != "no_material_discrepancy_identified"
    assert out["sections"]["discrepancies"][0]["status"] == "unresolved"


def test_cache_key_changes_with_case_snapshot_model():
    ids = {"gemma": {"digest": "a"}, "medgemma": {"digest": "b"}}
    k = workflow.cache_key(RAVI, {"p": 1}, "s1", ids)
    changed = copy.deepcopy(RAVI); changed["patient"]["age_years"] = 35
    assert k != workflow.cache_key(changed, {"p": 1}, "s1", ids)
    assert k != workflow.cache_key(RAVI, {"p": 1}, "s2", ids)
    assert k != workflow.cache_key(RAVI, {"p": 1}, "s1", {**ids, "gemma": {"digest": "c"}})
    assert k == workflow.cache_key(copy.deepcopy(RAVI), {"p": 1}, "s1", ids)


# ------------------------------------------------------------------ mock end-to-end runs
class Mock:
    def __init__(self, model, fail=False):
        self.model, self.fail = model, fail

    def structured(self, system, user, out):
        if self.fail:
            raise ModelError("mock failure")
        if out is LookupPlan:
            obj = {"lookups": [{"tool": "search_evidence", "query": "dengue nsaid", "source_kinds": [], "reason": "r"}]}
        elif out is ModelAssessment:
            obj = {"candidate_discrepancies": [{"issue_type": "guideline_conflict", "summary": "NSAID in dengue",
                   "case_fact_paths": ["doctor_plan.proposed_medications[0]"],
                   "evidence_ids": ["dengue-groupA-antipyretic"], "rationale": "r"}]} if "ibuprofen" in user else {}
        else:
            obj = {"disposition": "material_concern", "discrepancies": [{
                "issue_type": "guideline_conflict", "summary": "Ibuprofen proposed with NS1-positive dengue",
                "why": "Guideline advises avoiding NSAIDs", "case_fact_paths": ["doctor_plan.proposed_medications[0]"],
                "evidence_ids": ["dengue-groupA-antipyretic"], "status": "supported"}],
                "disagreements": ["mock: none"]}
        return out.model_validate(obj), {"calls": [{"raw": json.dumps(obj)}]}


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "t.db")
    storage.init()
    return storage


def _mk(db, case, crid):
    rid = "run-" + crid
    db.create_run(run_id=rid, client_request_id=crid, case_id=case["case_id"], case_revision=1,
                  case_json=json.dumps(case), profile_json="{}", snapshot_id=SNAP.snapshot_id, cache_key="k",
                  mode="live_local", status="queued", stage="queued", state_json="{}")
    return rid


def test_mock_run_reaches_concern(db, monkeypatch):
    monkeypatch.setattr(workflow, "GemmaClient", lambda: Mock("mock-gemma"))
    monkeypatch.setattr(workflow, "MedGemmaClient", lambda: Mock("mock-med"))
    rid = _mk(db, RAVI, "t1")
    workflow.run(rid)
    row = db.get_run(rid)
    res = json.loads(row["result_json"])
    assert row["status"] == "completed" and res["disposition"] == "material_concern"
    assert res["sources"]


def test_schema_rejects_unlisted_tool_from_model():
    with pytest.raises(ValidationError):
        LookupPlan.model_validate({"lookups": [{"tool": "evil_tool", "query": "x", "reason": "r"}]})


def test_model_failure_never_no_discrepancy_and_resume(db, monkeypatch):
    monkeypatch.setattr(workflow, "GemmaClient", lambda: Mock("mock-gemma"))
    monkeypatch.setattr(workflow, "MedGemmaClient", lambda: Mock("mock-med", fail=True))
    rid = _mk(db, MEERA, "t2")
    workflow.run(rid)
    row = db.get_run(rid)
    assert row["status"] == "failed" and row["result_json"] is None
    done = json.loads(row["state_json"])["done"]
    assert "gemma_assessment" in done and "medgemma_assessment" not in done
    monkeypatch.setattr(workflow, "MedGemmaClient", lambda: Mock("mock-med"))
    workflow.run(rid)  # resume from last completed stage
    assert db.get_run(rid)["status"] == "completed"
