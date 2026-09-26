const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const STAGES = ["Gathering evidence", "Gemma 4 assessment", "MedGemma review", "Result"];
const MODE_LABEL = { live_local: "Live local", cache_hit: "Cached result", recorded_replay: "Recorded replay" };
const DISP_LABEL = {
  material_concern: "Material concern",
  no_material_discrepancy_identified: "No material discrepancy identified in this review.",
  needs_clarification: "Needs clarification",
  insufficient_evidence: "Insufficient evidence",
};
let cases = [], currentRun = null, pollTimer = null, snapshotId = null;

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body));
  return body;
}
const rid = () => "req-" + crypto.randomUUID().replace(/-/g, "").slice(0, 20);

function unk(v) { return v === null || v === undefined ? '<span class="unknown">unknown</span>' : esc(v); }
function listOrUnknown(v, f = (x) => x) {
  if (v === null || v === undefined) return '<span class="unknown">unknown</span>';
  if (!v.length) return "none reported";
  return esc(v.map(f).join(", "));
}
function preview(c) {
  const p = c.patient, pr = c.presentation, d = c.doctor_plan;
  const med = (m) => m.name + (m.dose && m.dose.value ? ` ${m.dose.value}${m.dose.unit || ""}` : "") + (m.indication ? ` (for ${m.indication})` : "");
  $("preview").innerHTML = `
    <b>${esc(p.display_name)}</b>, ${unk(p.age_years)} y, ${unk(p.sex)} · ${esc(p.residence ? [p.residence.district, p.residence.state].filter(Boolean).join(", ") : "residence unknown")} · rev ${c.revision}<br>
    <b>Symptoms:</b> ${esc(pr.symptoms.map((s) => s.name + (s.duration_days ? ` ${s.duration_days}d` : "")).join(", ") || "none")}<br>
    <b>Conditions:</b> ${listOrUnknown(p.known_conditions)} · <b>Allergies:</b> ${listOrUnknown(p.allergies, (a) => a.substance)}<br>
    <b>Current meds:</b> ${listOrUnknown(p.current_medications, med)}<br>
    <b>Tests:</b> ${listOrUnknown(pr.test_results, (t) => `${t.name}: ${t.value ?? "?"}${t.unit ? " " + t.unit : ""}`)}<br>
    <b>Proposed:</b> ${unk(d.proposed_diagnosis)} · ${listOrUnknown(d.proposed_medications, med)}`;
}

function readCase() {
  try { const c = JSON.parse($("caseJson").value); $("jsonErr").textContent = ""; preview(c); return c; }
  catch (e) { $("jsonErr").textContent = "Invalid JSON: " + e.message; return null; }
}

async function init() {
  const [profiles, cs, ev] = await Promise.all([api("/api/profiles"), api("/api/cases"), api("/api/evidence/status")]);
  cases = cs; snapshotId = ev.snapshot_id;
  $("profile").innerHTML = profiles.map((p) => `<option value="${esc(p.profile_id)}">${esc(p.display_name)}, ${esc(p.specialty)}</option>`).join("");
  $("caseSel").innerHTML = cs.map((c, i) => `<option value="${i}">${esc(c.patient.display_name)} · ${esc(c.case_id)}</option>`).join("");
  const failures = ev.meta.failures.map((f) => f.document).join(", ");
  $("snapInfo").innerHTML = `Evidence snapshot <b>${esc(ev.snapshot_id)}</b>, created ${esc(ev.meta.created_at.slice(0, 10))}, ${ev.meta.record_count} passages. Scope: ${esc(ev.meta.scope)}.` +
    (failures ? ` <span class="unknown">Unavailable sources: ${esc(failures)} (no district surveillance coverage).</span>` : "");
  loadCase(0);
  health();
}
function loadCase(i) {
  const c = cases[i];
  $("caseJson").value = JSON.stringify(c, null, 2);
  $("profile").value = c.doctor_profile_id;
  preview(c);
}
async function health() {
  try {
    const h = await api("/api/health");
    const ok = h.runtime === "ok" && h.models.gemma.digest && h.models.medgemma.digest;
    $("health").textContent = ok ? `Local: ${h.models.gemma.model} + ${h.models.medgemma.model}` : `Models not ready (${h.runtime})`;
    $("health").style.color = ok ? "var(--ok)" : "var(--warn)";
  } catch { $("health").textContent = "health check failed"; }
}

async function submit(mode) {
  const c = readCase(); if (!c) return;
  c.doctor_profile_id = $("profile").value;
  try {
    const r = await api("/api/reviews", { method: "POST", body: JSON.stringify({ case: c, mode, client_request_id: rid() }) });
    watch(r.run_id);
  } catch (e) { showPanel(); $("status").innerHTML = `<p class="err">${esc(e.message)}</p>`; }
}
function showPanel() { $("resultPanel").hidden = false; }

function watch(runId) {
  currentRun = runId; showPanel(); clearInterval(pollTimer);
  $("result").innerHTML = ""; $("clarify").innerHTML = ""; $("decision").hidden = true; $("decisionOut").textContent = "";
  const tick = async () => {
    const r = await api(`/api/reviews/${runId}`);
    render(r);
    if (["completed", "failed", "interrupted"].includes(r.status)) clearInterval(pollTimer);
  };
  tick(); pollTimer = setInterval(tick, 1500);
}

function stageBar(r) {
  const idx = r.status === "completed" ? 4 : Math.max(0, STAGES.indexOf(r.stage_label));
  $("stages").innerHTML = STAGES.map((s, i) => `<li class="${i < idx ? "done" : i === idx && r.status !== "completed" ? "active" : ""}">${s}</li>`).join("");
}

function cites(ids) { return (ids || []).map((e) => `<span class="cite" data-ev="${esc(e)}">${esc(e)}</span>`).join(""); }
function facts(paths) { return paths && paths.length ? `<div class="facts">case: ${esc(paths.join(", "))}</div>` : ""; }

function render(r) {
  stageBar(r);
  $("modeLabel").textContent = MODE_LABEL[r.mode] || r.mode;
  const elapsed = r.completed_at ? ((new Date(r.completed_at) - new Date(r.created_at)) / 1000).toFixed(0) + " s" : "";
  if (r.status === "failed" || r.status === "interrupted") {
    $("status").innerHTML = `<p class="err">Review ${esc(r.status)}: ${esc(r.error || "run stopped before completion")}. No clinical conclusion was produced.</p>
      <button id="resumeBtn">Resume from last completed stage</button>`;
    $("resumeBtn").onclick = async () => { await api(`/api/reviews/${r.run_id}/resume`, { method: "POST" }); watch(r.run_id); };
  } else if (r.status !== "completed") {
    $("status").innerHTML = `<p class="muted">${esc(r.stage_label || r.stage)}… (${esc(r.status)})</p>`;
  } else {
    const ex = r.result.execution;
    const when = ex.original_generated_at ? `originally generated ${ex.original_generated_at}` : `completed ${ex.completed_at}`;
    $("status").innerHTML = `<p class="muted">${esc(MODE_LABEL[r.mode])} · ${esc(when)}${r.mode === "live_local" ? " · " + elapsed : ""} · snapshot ${esc(ex.snapshot?.snapshot_id)}</p>`;
  }
  $("techBody").textContent = JSON.stringify({ events: r.events, execution: r.result?.execution, lookups: r.result?.lookups,
    medication_check: r.result?.medication_check, assessments: r.result?.assessments, withheld: r.result?.withheld }, null, 1);
  if (r.status !== "completed" || !r.result) return;
  const res = r.result, s = res.sections || {};
  let h = `<div class="disp ${res.disposition}">${esc(DISP_LABEL[res.disposition])}` +
    (res.disposition === "no_material_discrepancy_identified" ? `<span class="note">This is not approval of the diagnosis, prescription, or completeness of care.</span>` : "") + `</div>`;
  if (res.review_mode === "gemma_only") h += `<div class="disp insufficient_evidence">Gemma 4 only: MedGemma specialist review unavailable<span class="note">Provisional review without specialist confirmation.</span></div>`;
  if (s.secondary_diagnoses?.length) h += `<h3>Plausible secondary diagnosis</h3>` + s.secondary_diagnoses.map((d) =>
    `<div class="item"><b>${esc(d.condition)}</b><div>${esc(d.why)}</div>${facts(d.case_fact_paths)}${cites(d.evidence_ids)}</div>`).join("");
  if (s.discrepancies?.length) h += `<h3>Discrepancies</h3>` + s.discrepancies.map((d) =>
    `<div class="item ${d.status}"><span class="status">${esc(d.status)} · ${esc(d.issue_type)}</span><div><b>${esc(d.summary)}</b></div>${facts(d.case_fact_paths)}${cites(d.evidence_ids)}</div>`).join("");
  const whys = [...(s.discrepancies || []).map((d) => d.why), ...(s.secondary_diagnoses || []).map((d) => d.why)].filter(Boolean);
  if (s.discrepancies?.length && whys.length) h += `<h3>Why</h3><ul>${whys.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>`;
  if (s.suggested_checks?.length) h += `<h3>Suggested checks</h3>` + s.suggested_checks.map((c) =>
    `<div class="item"><b>${esc(c.check)}</b><div>${esc(c.why)}</div>${cites(c.evidence_ids)}</div>`).join("");
  if (res.disagreements?.length) h += `<h3>Unresolved disagreement</h3><ul>${res.disagreements.map((d) => `<li>${esc(d)}</li>`).join("")}</ul>`;
  if (res.limitations?.length) h += `<h3>Limitations</h3><ul>${res.limitations.map((d) => `<li>${esc(d)}</li>`).join("")}</ul>`;
  if (res.sources?.length) h += `<details id="sources"><summary>Sources (${res.sources.length})</summary>` + res.sources.map((p) =>
    `<div class="source" id="src-${esc(p.evidence_id)}"><b>${esc(p.evidence_id)}</b> · ${esc(p.source_kind)} · ${esc(p.title)} (${esc(p.published_at || "date unknown")}) · ${esc(p.document_locator)}
      <blockquote>${esc(p.exact_excerpt)}</blockquote><a href="${esc(p.source_url)}" target="_blank" rel="noopener">official source</a></div>`).join("") + `</details>`;
  $("result").innerHTML = h;
  document.querySelectorAll(".cite").forEach((el) => el.onclick = () => {
    $("sources").open = true; const t = $("src-" + el.dataset.ev);
    if (t) { t.scrollIntoView({ behavior: "smooth", block: "center" }); t.classList.add("flash"); setTimeout(() => t.classList.remove("flash"), 1500); }
  });
  renderQuestions(r, s.questions || []);
  $("decision").hidden = false;
}

function renderQuestions(r, qs) {
  if (!qs.length) { $("clarify").innerHTML = ""; return; }
  $("clarify").innerHTML = `<h3>Questions</h3>` + qs.map((q, i) => `
    <div class="item"><b>${esc(q.question)}</b><div class="muted">${esc(q.why_it_matters)}</div>
    <div class="facts">${esc(q.field_path)}</div>
    <input data-q="${i}" placeholder='Answer as JSON, e.g. [{"name":"platelet count","value":"85000","unit":"/microL","measured_at":null}]'></div>`).join("") +
    `<button id="clarifyBtn" class="primary">Submit clarification and recheck</button><div id="clarifyErr" class="err"></div>`;
  $("clarifyBtn").onclick = async () => {
    try {
      const answers = qs.map((q, i) => {
        const raw = document.querySelector(`[data-q="${i}"]`).value.trim();
        if (!raw) return null;
        let value; try { value = JSON.parse(raw); } catch { value = raw; }
        return { field_path: q.field_path, value };
      }).filter(Boolean);
      const out = await api(`/api/reviews/${r.run_id}/clarifications`, { method: "POST", body: JSON.stringify({ answers, client_request_id: rid() }) });
      const nr = await api(`/api/reviews/${out.run_id}`);
      $("caseJson").value = JSON.stringify(nr.case, null, 2); preview(nr.case);
      watch(out.run_id);
    } catch (e) { $("clarifyErr").textContent = e.message; }
  };
}

document.querySelectorAll("#decision button").forEach((b) => b.onclick = async () => {
  const out = await api(`/api/reviews/${currentRun}/decision`, { method: "POST", body: JSON.stringify({ action: b.dataset.act, note: $("decisionNote").value || null }) });
  $("decisionOut").textContent = `Recorded "${b.dataset.act}". ${out.note}`;
});
$("caseSel").onchange = (e) => loadCase(+e.target.value);
$("caseJson").oninput = readCase;
$("reviewBtn").onclick = () => submit("auto");
$("liveBtn").onclick = () => submit("live_local");
$("replayBtn").onclick = () => submit("recorded_replay");
init();
