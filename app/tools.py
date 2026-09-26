"""Local tool allowlist. The application validates and executes every model-requested lookup."""
from __future__ import annotations

import json
import re
from typing import Any

from .config import DATA
from .evidence import Snapshot

ALLOWED_TOOLS = {"search_evidence", "get_evidence", "check_medication_facts"}
RULES = json.loads((DATA / "medication_rules.json").read_text(encoding="utf-8"))


class ToolError(ValueError):
    pass


def resolve_path(case: dict, path: str) -> tuple[bool, Any]:
    """Resolve 'patient.current_medications[0].name' inside a case. Returns (exists, value)."""
    cur: Any = case
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            i = int(part[1:-1])
            if not isinstance(cur, list) or i >= len(cur):
                return False, None
            cur = cur[i]
        else:
            if not isinstance(cur, dict) or part not in cur:
                return False, None
            cur = cur[part]
    return True, cur


def _text(v: Any) -> str:
    return json.dumps(v).lower() if v is not None else ""


def _eval(cond: dict, case: dict) -> tuple[bool, list[str], list[str]]:
    """Returns (matched, matched_paths, unknown_paths). Only declared operators are evaluated."""
    hits, unknown = [], []
    for c in cond["any"]:
        ok, val = resolve_path(case, c["field"])
        if not ok or val is None:
            unknown.append(c["field"])
            continue
        if c["op"] == "contains_any":
            if any(x in _text(val) for x in c["values"]):
                hits.append(c["field"])
        elif c["op"] == "item_matches":
            for i, item in enumerate(val or []):
                name, value = str(item.get("name", "")).lower(), str(item.get("value", "")).lower()
                if any(n in name for n in c["name_contains"]) and any(v in value for v in c["value_contains"]):
                    hits.append(f"{c['field']}[{i}]")
        else:
            raise ToolError(f"undeclared operator {c['op']}")
    return bool(hits), hits, unknown


def check_medication_facts(case: dict) -> dict:
    classes = RULES["medication_classes"]
    matched, unknowns = [], []
    for rule in RULES["rules"]:
        src = rule.get("source", "proposed_medications")
        base = "patient.current_medications" if src == "current_medications" else "doctor_plan.proposed_medications"
        ok, meds = resolve_path(case, base)
        if not ok or meds is None:
            unknowns.append({"rule_id": rule["rule_id"], "unknown_field": base})
            continue
        for i, m in enumerate(meds):
            if not any(member in m["name"].lower() for member in classes[rule["applies_to_class"]]):
                continue
            hit, paths, unk = _eval(rule["condition"], case)
            if hit:
                matched.append({"rule_id": rule["rule_id"], "issue_type": rule["issue_type"],
                                "warning": rule["warning"], "evidence_ids": rule["evidence_ids"],
                                "case_fact_paths": [f"{base}[{i}]"] + paths})
            elif unk:
                unknowns.append({"rule_id": rule["rule_id"], "medication_path": f"{base}[{i}]",
                                 "unknown_fields": unk,
                                 "meaning": "rule could not be evaluated; these facts are unknown"})
    return {"rule_version": RULES["rule_version"], "matched_rules": matched, "unknowns": unknowns,
            "scope_note": "Limited source-backed checks only; not a comprehensive safety check."}


def execute(request: dict, case: dict, snap: Snapshot, limit: int) -> dict:
    tool = request.get("tool")
    if tool not in ALLOWED_TOOLS:
        raise ToolError(f"tool not allowed: {tool!r}")
    query = request.get("query", "")
    if not isinstance(query, str) or len(query) > 200:
        raise ToolError("invalid query")
    if tool == "search_evidence":
        kinds = [k for k in request.get("source_kinds", []) if isinstance(k, str)]
        return {"tool": tool, "query": query, "results": snap.search(query, kinds or None, limit)}
    if tool == "get_evidence":
        ids = re.findall(r"[a-z0-9-]{3,80}", query)
        return {"tool": tool, "query": query, "results": snap.get(ids)}
    result = check_medication_facts(case)
    ids = sorted({e for m in result["matched_rules"] for e in m["evidence_ids"]})
    return {"tool": tool, "query": query, "medication_check": result, "results": snap.get(ids)}
