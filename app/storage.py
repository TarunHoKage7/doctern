"""SQLite persistence: runs, stage records, events, doctor decisions."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading

from .config import DB_PATH

_lock = threading.Lock()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init() -> None:
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
          run_id TEXT PRIMARY KEY, client_request_id TEXT UNIQUE, case_id TEXT, case_revision INT,
          case_json TEXT, profile_json TEXT, snapshot_id TEXT, cache_key TEXT, mode TEXT,
          status TEXT, stage TEXT, state_json TEXT, result_json TEXT, error TEXT,
          source_run_id TEXT, created_at TEXT, updated_at TEXT, completed_at TEXT);
        CREATE INDEX IF NOT EXISTS runs_cache ON runs(cache_key, status);
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, at TEXT, stage TEXT, actor TEXT,
          seconds REAL, outcome TEXT, detail TEXT);
        CREATE TABLE IF NOT EXISTS decisions (
          id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, at TEXT, action TEXT, note TEXT);
        """)


def create_run(**f) -> None:
    f.setdefault("created_at", now())
    f["updated_at"] = f["created_at"]
    cols = ",".join(f)
    with _lock, conn() as c:
        c.execute(f"INSERT INTO runs ({cols}) VALUES ({','.join('?' * len(f))})", list(f.values()))


def update_run(run_id: str, **f) -> None:
    f["updated_at"] = now()
    with _lock, conn() as c:
        c.execute(f"UPDATE runs SET {','.join(k + '=?' for k in f)} WHERE run_id=?", [*f.values(), run_id])


def get_run(run_id: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return dict(r) if r else None


def by_client_request(crid: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM runs WHERE client_request_id=?", (crid,)).fetchone()
    return dict(r) if r else None


def completed_by_cache_key(key: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM runs WHERE cache_key=? AND status='completed' AND mode='live_local' "
                      "ORDER BY completed_at DESC LIMIT 1", (key,)).fetchone()
    return dict(r) if r else None


def list_runs(limit: int = 30) -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT run_id, case_id, case_revision, mode, status, stage, created_at, "
                         "completed_at FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def interrupted_runs() -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT run_id FROM runs WHERE status IN ('running','queued')").fetchall()
    return [dict(r) for r in rows]


def event(run_id: str, stage: str, actor: str, outcome: str, seconds: float | None = None,
          detail: object = None) -> None:
    with _lock, conn() as c:
        c.execute("INSERT INTO events (run_id, at, stage, actor, seconds, outcome, detail) VALUES (?,?,?,?,?,?,?)",
                  (run_id, now(), stage, actor, seconds, outcome,
                   json.dumps(detail, ensure_ascii=False)[:20000] if detail is not None else None))


def events(run_id: str) -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT at, stage, actor, seconds, outcome, detail FROM events WHERE run_id=? ORDER BY id",
                         (run_id,)).fetchall()
    return [dict(r) for r in rows]


def add_decision(run_id: str, action: str, note: str | None) -> None:
    with _lock, conn() as c:
        c.execute("INSERT INTO decisions (run_id, at, action, note) VALUES (?,?,?,?)", (run_id, now(), action, note))


def decisions(run_id: str) -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT at, action, note FROM decisions WHERE run_id=? ORDER BY id",
                                           (run_id,)).fetchall()]
