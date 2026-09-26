"""Submit a fixture case to the running app and print the genuine run trail."""
import json, sys, time, uuid
import httpx, os
BASE = os.getenv("DOCTERN_URL", "http://127.0.0.1:8000")
path, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "live_local")
case = json.load(open(path, encoding="utf-8"))
r = httpx.post(f"{BASE}/api/reviews", timeout=60,
               json={"case": case, "mode": mode, "client_request_id": "cli-" + uuid.uuid4().hex[:12]}).json()
rid = r["run_id"]; t = time.time()
while True:
    s = httpx.get(f"{BASE}/api/reviews/{rid}", timeout=30).json()
    if s["status"] in ("completed", "failed", "interrupted"): break
    time.sleep(3)
print(rid, s["mode"], s["status"], round(time.time() - t), "s", s["error"] or "")
for e in s["events"]: print("  ", e["stage"], e["actor"], e["outcome"], e["seconds"] and round(e["seconds"], 1))
if s["result"]:
    res = s["result"]; print("DISPOSITION:", res["disposition"])
    sec = res["sections"]
    for k in ("secondary_diagnoses", "discrepancies", "suggested_checks", "questions"):
        for it in sec.get(k, []): print(f"  [{k}]", json.dumps(it, ensure_ascii=False)[:300])
    print("  withheld:", res["withheld"]); print("  disagreements:", res["disagreements"]); print("  limitations:", res["limitations"])
