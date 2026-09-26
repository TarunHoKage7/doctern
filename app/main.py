"""FastAPI app. Serve on loopback: uvicorn app.main:app --host 127.0.0.1 --port 8000"""
from __future__ import annotations

import copy
import json
import uuid

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import storage, workflow
from .config import DATA, OLLAMA_URL, ROOT
from .evidence import current_snapshot_id, load_snapshot
from .schemas import (CaseRequest, ClarificationRequest, DoctorDecision, DoctorProfile, ReviewRequest)
from .tools import resolve_path

app = FastAPI(title="Doctern - Clinical Context Assistant")
storage.init()
for r in storage.interrupted_runs():  # a restart leaves in-flight runs resumable, never "completed"
    storage.update_run(r["run_id"], status="interrupted")

STATIC = ROOT / "app" / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def profiles() -> dict[str, dict]:
    raw = json.loads((DATA / "doctor_profiles.json").read_text(encoding="utf-8"))
    return {p["profile_id"]: DoctorProfile.model_validate(p).model_dump() for p in raw}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    try:
        tags = [m["name"] for m in httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5).json().get("models", [])]
        runtime = "ok"
    except Exception as exc:  # noqa: BLE001
        tags, runtime = [], f"unreachable: {type(exc).__name__}"
    ids = workflow.identities()
    return {"app": "ok", "runtime": runtime, "installed_models": tags, "models": ids,
            "snapshot_id": current_snapshot_id()}


@app.get("/api/profiles")
def get_profiles():
    return list(profiles().values())


@app.get("/api/cases")
def get_cases():
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((DATA / "cases").glob("*.json"))]


@app.get("/api/evidence/status")
def evidence_status():
    snap = load_snapshot(current_snapshot_id())
    return {"snapshot_id": snap.snapshot_id, "meta": snap.meta,
            "records": [{k: r[k] for k in ("evidence_id", "source_kind", "title", "published_at",
                                          "document_locator")} for r in snap.records.values()]}


@app.get("/api/evidence/brief")
def evidence_brief():
    from .research_brief import load_brief
    b = load_brief(current_snapshot_id())
    if not b:
        raise HTTPException(404, "no brief for the current snapshot; run python -m app.refresh --brief")
    return {k: b[k] for k in ("snapshot_id", "created_at", "models", "review_status", "observations", "omitted", "note")}


@app.get("/api/evidence/{evidence_id}")
def get_evidence(evidence_id: str, snapshot_id: str | None = None):
    snap = load_snapshot(snapshot_id or current_snapshot_id())
    rec = snap.records.get(evidence_id)
    if not rec:
        raise HTTPException(404, "evidence not in snapshot")
    return rec


def _create(case: CaseRequest, mode: str, crid: str) -> dict:
    existing = storage.by_client_request(crid)
    if existing:  # idempotent repeated clicks
        return {"run_id": existing["run_id"], "deduplicated": True}
    profs = profiles()
    if case.doctor_profile_id not in profs:
        raise HTTPException(422, "unknown doctor_profile_id")
    case_d, profile = case.model_dump(), profs[case.doctor_profile_id]
    snapshot_id = current_snapshot_id()  # bound once at creation
    ids = workflow.identities()
    key = workflow.cache_key(case_d, profile, snapshot_id, ids)
    run_id = "run-" + uuid.uuid4().hex[:12]
    base = dict(run_id=run_id, client_request_id=crid, case_id=case.case_id, case_revision=case.revision,
                case_json=json.dumps(case_d, ensure_ascii=False), profile_json=json.dumps(profile),
                snapshot_id=snapshot_id, cache_key=key)

    if mode == "recorded_replay":
        path = DATA / "replays" / f"{case.case_id}.json"
        if not path.exists():
            raise HTTPException(404, "no recorded replay exists for this case")
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("recorded_case_sha256") != workflow.case_sha(case_d):
            raise HTTPException(409, "This case was edited after the replay was recorded. "
                                     "Run a live review instead; the old result no longer applies.")
        rec["execution"] = {**rec["execution"], "mode": "recorded_replay",
                            "replay_note": "Recorded output of an earlier genuine local run; not a fresh model response."}
        storage.create_run(**base, mode="recorded_replay", status="completed", stage="persist_result",
                           result_json=json.dumps(rec, ensure_ascii=False), source_run_id=rec["execution"]["run_id"],
                           completed_at=storage.now())
        return {"run_id": run_id, "mode": "recorded_replay"}

    if mode == "auto":
        hit = storage.completed_by_cache_key(key)
        if hit:
            rec = json.loads(hit["result_json"])
            rec["execution"] = {**rec["execution"], "mode": "cache_hit", "source_run_id": hit["run_id"],
                                "cache_reason": "identical case, profile, snapshot, models, prompts, config and rules",
                                "original_generated_at": hit["completed_at"]}
            storage.create_run(**base, mode="cache_hit", status="completed", stage="persist_result",
                               result_json=json.dumps(rec, ensure_ascii=False), source_run_id=hit["run_id"],
                               completed_at=storage.now())
            return {"run_id": run_id, "mode": "cache_hit"}

    storage.create_run(**base, mode="live_local", status="queued", stage="queued",
                       state_json=json.dumps({"out": {"identities": ids}}))
    workflow.start_live(run_id)
    return {"run_id": run_id, "mode": "live_local"}


@app.post("/api/reviews")
def create_review(req: ReviewRequest):
    return _create(req.case, req.mode, req.client_request_id)


@app.get("/api/reviews")
def recent_reviews():
    return storage.list_runs()


@app.get("/api/reviews/{run_id}")
def get_review(run_id: str):
    row = storage.get_run(run_id)
    if not row:
        raise HTTPException(404, "run not found")
    stage = (row["stage"] or "").split(":")[0]
    return {"run_id": run_id, "status": row["status"], "stage": row["stage"],
            "stage_label": workflow.STAGE_LABEL.get(stage, stage), "mode": row["mode"],
            "error": row["error"], "case": json.loads(row["case_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "events": [{k: e[k] for k in ("at", "stage", "actor", "seconds", "outcome")}
                       for e in storage.events(run_id)],
            "decisions": storage.decisions(run_id), "created_at": row["created_at"],
            "completed_at": row["completed_at"]}


@app.post("/api/reviews/{run_id}/resume")
def resume(run_id: str):
    row = storage.get_run(run_id)
    if not row:
        raise HTTPException(404, "run not found")
    if row["status"] not in ("interrupted", "failed"):
        raise HTTPException(409, f"run is {row['status']}")
    workflow.start_live(run_id)
    return {"run_id": run_id, "resumed_from": row["stage"]}


@app.post("/api/reviews/{run_id}/clarifications")
def clarify(run_id: str, req: ClarificationRequest):
    row = storage.get_run(run_id)
    if not row:
        raise HTTPException(404, "run not found")
    case = copy.deepcopy(json.loads(row["case_json"]))
    for a in req.answers:
        parent_path, _, leaf = a.field_path.rpartition(".")
        ok, parent = resolve_path(case, parent_path) if parent_path else (True, case)
        if not ok or not isinstance(parent, dict) or leaf not in parent:
            raise HTTPException(422, f"unknown field {a.field_path}")
        parent[leaf] = a.value
    case["revision"] += 1  # new immutable revision; old review is not reused
    return _create(CaseRequest.model_validate(case), "live_local", req.client_request_id)


@app.post("/api/reviews/{run_id}/decision")
def decide(run_id: str, d: DoctorDecision):
    if not storage.get_run(run_id):
        raise HTTPException(404, "run not found")
    storage.add_decision(run_id, d.action, d.note)
    return {"recorded": True, "note": "Decision recorded. It does not change model weights or prove a diagnosis."}
