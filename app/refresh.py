"""Research refresh: python -m app.refresh [--fetch]

Reads data/source_manifest.json, (optionally) downloads each official document, extracts the
configured passages verbatim from the PDF text, and atomically publishes a new immutable snapshot
under data/snapshots/<snapshot_id>/. A failed source is reported and never deletes the last good
snapshot. Publication dates come from the manifest, never from the download date.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys

import httpx
from pypdf import PdfReader

from .config import DATA, ROOT, SNAPSHOTS
from .schemas import EvidenceRecord


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_pages(pdf_path) -> list[str]:
    return [norm(p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages]


def cut(pages: list[str], start: str, end: str):
    s_n, e_n = norm(start), norm(end)
    for i, page in enumerate(pages):
        a = page.find(s_n)
        if a < 0:
            continue
        b = page.find(e_n, a)
        if b < 0:
            continue
        return i + 1, page[a:b + len(e_n)]
    return None, None


def fetch(url: str, dest) -> None:
    with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as c:
        r = c.get(url)
        r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)


def refresh(do_fetch: bool) -> dict:
    manifest = json.loads((DATA / "source_manifest.json").read_text(encoding="utf-8"))
    now = dt.datetime.now(dt.timezone.utc)
    snapshot_id = "snap-" + now.strftime("%Y%m%dT%H%M%SZ")
    records, failures, coverage = [], [], []

    for doc in manifest["documents"]:
        path = ROOT / doc["local_file"]
        try:
            if do_fetch or not path.exists():
                fetch(doc["source_url"], path)
            pages = extract_pages(path)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append({"document": doc["local_document_id"], "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        coverage.append({"document": doc["local_document_id"], "source_kind": doc["source_kind"],
                         "pages": len(pages), "passages": len(doc["passages"])})
        for p in doc["passages"]:
            page_no, excerpt = cut(pages, p["start"], p["end"])
            if excerpt is None:
                failures.append({"document": doc["local_document_id"], "passage": p["id"],
                                 "error": "anchor phrases not found in extracted text"})
                continue
            records.append(EvidenceRecord(
                evidence_id=p["id"], snapshot_id=snapshot_id, source_kind=doc["source_kind"],
                title=doc["title"], publisher=doc["publisher"], source_url=doc["source_url"],
                published_at=doc.get("published_at"), retrieved_at=now.date().isoformat(),
                reporting_period_start=doc.get("reporting_period_start"),
                reporting_period_end=doc.get("reporting_period_end"),
                geography=doc.get("geography"), condition_tags=p["condition_tags"],
                medication_tags=p["medication_tags"], jurisdiction=doc["jurisdiction"],
                exact_excerpt=excerpt, document_locator=f"PDF page {page_no}",
                content_sha256=sha, local_document_id=doc["local_document_id"],
            ).model_dump())

    if not records:
        return {"published": False, "reason": "no usable records", "failures": failures}

    meta = {"snapshot_id": snapshot_id, "created_at": now.isoformat(), "record_count": len(records),
            "coverage": coverage, "failures": failures,
            "scope": "Febrile illness / dengue clinical management (India, national guideline)",
            "coverage_notes": [
                "District-level surveillance coverage is missing unless a surveillance record is listed.",
                "No local report does not exclude a disease; old reports are dated context only."]}
    tmp = SNAPSHOTS / (snapshot_id + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "records.json").write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    final = SNAPSHOTS / snapshot_id
    if final.exists():
        shutil.rmtree(tmp)
    else:
        tmp.rename(final)
    (SNAPSHOTS / "CURRENT").write_text(snapshot_id, encoding="utf-8")
    return {"published": True, "snapshot_id": snapshot_id, "records": len(records), "failures": failures}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="re-download every manifest source")
    out = refresh(ap.parse_args().fetch)
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["published"] else 1)
