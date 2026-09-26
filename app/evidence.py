"""Immutable snapshot loading + local SQLite FTS5 retrieval with a small curated synonym map."""
from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache

from .config import SNAPSHOTS

SYNONYMS = {
    "nsaid": ["nsaid", "nsaids", "ibuprofen", "diclofenac", "naproxen", "aspirin", "anti-inflammatory"],
    "ibuprofen": ["nsaid", "nsaids"], "diclofenac": ["nsaid", "nsaids"], "naproxen": ["nsaid", "nsaids"],
    "aspirin": ["aspirin", "antiplatelet", "anti-platelet", "salicylates"],
    "fever": ["fever", "febrile", "temperature"], "paracetamol": ["paracetamol", "acetaminophen"],
    "platelet": ["platelet", "thrombocytopenia"], "diabetes": ["diabetes"],
    "bleeding": ["bleeding", "hemorrhagic", "haemorrhage"], "dengue": ["dengue"],
}


def current_snapshot_id() -> str:
    return (SNAPSHOTS / "CURRENT").read_text(encoding="utf-8").strip()


class Snapshot:
    def __init__(self, snapshot_id: str):
        folder = SNAPSHOTS / snapshot_id
        self.snapshot_id = snapshot_id
        self.meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        self.records = {r["evidence_id"]: r for r in
                        json.loads((folder / "records.json").read_text(encoding="utf-8"))}
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            self.db.execute("CREATE VIRTUAL TABLE idx USING fts5(evidence_id UNINDEXED, body)")
            self.fts = True
        except sqlite3.OperationalError:
            self.db.execute("CREATE TABLE idx (evidence_id TEXT, body TEXT)")
            self.fts = False
        for r in self.records.values():
            body = " ".join([r["title"], r["exact_excerpt"], " ".join(r["condition_tags"]),
                             " ".join(r["medication_tags"])])
            self.db.execute("INSERT INTO idx VALUES (?, ?)", (r["evidence_id"], body))

    def search(self, query: str, source_kinds: list[str] | None = None, limit: int = 3) -> list[dict]:
        terms = [t for t in re.findall(r"[a-zA-Z]{3,}", query.lower())]
        expanded = sorted({s for t in terms for s in SYNONYMS.get(t, [t])})
        if not expanded:
            return []
        if self.fts:
            q = " OR ".join('"' + t.replace('"', "") + '"' for t in expanded)
            rows = self.db.execute("SELECT evidence_id FROM idx WHERE idx MATCH ? ORDER BY bm25(idx)",
                                   (q,)).fetchall()
            ids = [r[0] for r in rows]
        else:  # deterministic fallback: term-overlap count
            scored = []
            for eid, body in self.db.execute("SELECT evidence_id, body FROM idx"):
                b = body.lower()
                scored.append((-sum(b.count(t) for t in expanded), eid))
            ids = [eid for s, eid in sorted(scored) if s < 0]
        out = [self.records[i] for i in ids
               if not source_kinds or self.records[i]["source_kind"] in source_kinds]
        return out[:limit]

    def get(self, ids: list[str]) -> list[dict]:
        return [self.records[i] for i in ids if i in self.records]


@lru_cache(maxsize=8)
def load_snapshot(snapshot_id: str) -> Snapshot:
    return Snapshot(snapshot_id)
