// Live pipeline views driven by real run data: statuses and timings come from the server's timeline.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const TAG = { g: "Gemma 4", m: "MedGemma", a: "The app", d: "Doctor" };
  const LABEL = { idle: "Not started", run: "In progress", done: "Succeeded", failed: "Failed", skipped: "Not needed" };
  const fmt = (s) => (s < 60 ? s.toFixed(1) + "s" : Math.floor(s / 60) + "m " + Math.round(s % 60) + "s");

  const DOCTOR = [
    { keys: ["gemma_lookup_plan"], name: "Gemma plans the search", k: "g", inT: "Patient case", inS: "Plus doctor's specialty", outT: "Search list", outS: "What to look up, and why" },
    { keys: ["execute_local_lookups"], name: "App finds the passages", k: "a", inT: "Search list", inS: "Only allowed tools run", outT: "Evidence passages", outS: "Exact text and medicine rules" },
    { keys: ["gemma_assessment"], name: "Gemma's first opinion", k: "g", inT: "Case and found text", inS: "Exact guideline passages", outT: "Gemma's findings", outS: "Each tied to a source" },
    { keys: ["medgemma_assessment"], name: "MedGemma's own opinion", k: "m", inT: "Same case, same text", inS: "Can't see Gemma's view", outT: "MedGemma's findings", outS: "Made on its own" },
    { keys: ["reconcile_lookup"], name: "Gemma digs once more", k: "g", inT: "Open questions", inS: "Where the two differ", outT: "Extra evidence", outS: "On the disputed point", optional: true },
    { keys: ["medgemma_final_review", "gemma_fallback_review"], name: "MedGemma's final say", k: "m", inT: "Everything so far", inS: "Plus the medicine check", outT: "Final advice", outS: "Checks and questions", optional: true },
    { keys: ["verify_result"], name: "Fresh Gemma check", k: "g", inT: "Final advice", inS: "Case, both opinions, passages, weekly summary", outT: "Checked advice", outS: "Unsupported items removed" },
    { keys: ["validate_result"], name: "App checks every claim", k: "a", inT: "Checked advice", inS: "Every claim with its citation", outT: "Verified advice", outS: "Drops anything without a source" },
    { keys: ["persist_result"], name: "Doctor sees the advice", k: "a", inT: "Verified advice", inS: "Saved with the run", outT: "Second opinion", outS: "With the exact source text", voice: true },
  ];

  let run = null, steps = [], total = 0, t = 0, playing = false, sel = 0, speed = 5, last = 0, live = false, clockSkew = 0, note = "";

  function build(r) {
    run = r;
    live = !["completed", "failed", "interrupted"].includes(r.status);
    clockSkew = r.server_now ? r.server_now - Date.now() / 1000 : 0;
    const tl = r.timeline || {};
    const now = Date.now() / 1000 + clockSkew;
    note = "";
    let entries = Object.values(tl);
    let t0 = entries.length ? Math.min(...entries.map((e) => e.start)) : 0;
    let synthetic = null;
    if (!entries.length && (r.model_calls || []).length) {
      // Older recordings have no step clock; rebuild spacing from the recorded model-call durations.
      synthetic = {}; let acc = 0;
      for (const c of r.model_calls) { synthetic[c.stage] = { start: acc, end: acc + c.seconds, status: "done", actor: c.model }; acc += c.seconds; }
      t0 = 0; note = "Timings rebuilt from the recorded model-call durations.";
    }
    const src = synthetic || tl;
    steps = DOCTOR.map((d) => {
      const key = d.keys.find((k) => src[k]);
      const e = key ? src[key] : null;
      const fallback = key === "gemma_fallback_review";
      const start = e ? e.start - t0 : null;
      const end = e ? (e.end ?? now) - t0 : null;
      let status = e ? (e.status === "running" ? "run" : e.status === "failed" ? "failed" : "done") : (live ? "idle" : "skipped");
      if (!e && r.status === "failed") status = "idle";
      return { ...d, key, start, end, dur: e ? Math.max(0, end - start) : 0, status, actor: e ? e.actor : null,
        name: fallback ? "Gemma's final say (MedGemma down)" : d.name, k: fallback ? "g" : d.k,
        summary: (r.summaries || {})[key] || "" };
    });
    total = Math.max(0.1, ...steps.filter((s) => s.start !== null).map((s) => s.end));
    if (live) { t = total; playing = false; }
  }

  function statusAt(s) {
    if (live || s.start === null) return s.status;
    if (s.status === "skipped") return t >= total ? "skipped" : "idle";
    if (t >= s.end) return s.status;
    return t > s.start ? "run" : "idle";
  }

  function render() {
    if (!run) return;
    const finished = !live && t >= total;
    const state = live ? "Running live" : run.status === "failed" ? "Failed" : finished ? "Succeeded" : playing ? "Replaying" : "Paused";
    const dot = live ? "var(--run)" : run.status === "failed" ? "var(--err)" : finished ? "var(--ok)" : "var(--muted)";
    $("pipeStatus").innerHTML = `<span class="dot" style="background:${dot}"></span>${esc(state)} · <span class="mono">${fmt(Math.min(t, total))} / ${fmt(total)}</span>` +
      (note ? ` · <span class="muted">${esc(note)}</span>` : "");
    $("pipeToggle").textContent = finished ? "Replay" : playing ? "Pause" : "Play";
    $("pipeToggle").disabled = live; $("pipeRestart").disabled = live; $("pipeScrub").disabled = live;
    $("pipeScrub").max = total; $("pipeScrub").value = Math.min(t, total);

    $("pipeSteps").innerHTML = steps.map((s, i) => {
      const st = statusAt(s);
      const el = st === "idle" || st === "skipped" ? "" : fmt(Math.max(0, Math.min(s.dur, t - s.start)));
      return `<button class="step st-${st} ${sel === i ? "sel" : ""}" data-i="${i}">
        <span class="num">${i + 1}</span>
        <span class="sname">${esc(s.name)}</span>
        <span class="chip k-${s.k}">${TAG[s.k]}</span>
        <span class="sstat">${LABEL[st]}${el ? ` · <span class="mono">${el}</span>` : ""}</span>
        ${s.voice ? `<span class="voice">Doctor can answer questions by voice</span>` : ""}
      </button>`;
    }).join("");
    document.querySelectorAll("#pipeSteps .step").forEach((b) => b.onclick = () => { sel = +b.dataset.i; render(); });

    $("pipeGantt").innerHTML = steps.map((s, i) => {
      const st = statusAt(s);
      const left = s.start === null ? 0 : (s.start / total) * 100;
      const full = s.start === null ? 0 : (s.dur / total) * 100;
      const done = s.start === null ? 0 : (Math.max(0, Math.min(s.dur, t - s.start)) / total) * 100;
      return `<div class="grow ${sel === i ? "sel" : ""}" data-i="${i}"><span class="glabel">${i + 1}. ${esc(s.name)}</span>
        <span class="gtrack"><span class="gfull" style="left:${left}%;width:${full}%"></span><span class="gbar st-${st}" style="left:${left}%;width:${live ? full : done}%"></span></span></div>`;
    }).join("") + `<div class="gaxis"><span></span><span class="gticks">${[0, .25, .5, .75, 1].map((f) => `<span>${fmt(f * total)}</span>`).join("")}</span></div>`;
    document.querySelectorAll("#pipeGantt .grow").forEach((b) => b.onclick = () => { sel = +b.dataset.i; render(); });
    const head = $("pipeGantt").querySelector(".gtrack");
    if (head) $("pipeGantt").style.setProperty("--head", (Math.min(t, total) / total * 100) + "%");

    const s = steps[sel], st = statusAt(s);
    $("pipeDetail").innerHTML = `<div class="dhead"><span class="dot st-${st}"></span><b>${esc(s.name)}</b><span class="chip k-${s.k}">${TAG[s.k]}</span><span class="muted">${LABEL[st]}</span></div>
      <div class="dgrid">
        <div><div class="dlab">Started at</div><div class="mono">${s.start === null ? "-" : fmt(s.start)}</div></div>
        <div><div class="dlab">Took</div><div class="mono">${s.start === null ? "-" : fmt(s.dur)}</div></div>
        <div><div class="dlab">Ran on</div><div class="mono">${esc(s.actor || "-")}</div></div>
      </div>
      <div class="dio"><div><div class="dlab">Input</div><b>${esc(s.inT)}</b><div class="muted">${esc(s.inS)}</div></div>
        <div><div class="dlab">Output</div><b>${esc(s.outT)}</b><div class="muted">${esc(s.outS)}</div></div></div>
      <div><div class="dlab">What actually came out</div>${esc(s.summary || (st === "skipped" ? "This step was not needed for this case." : st === "idle" ? "Not run yet." : "Running..."))}</div>`;
  }

  function tick(now) {
    const dt = (now - last) / 1000; last = now;
    if (playing && !live) {
      t = Math.min(total, t + dt * speed);
      if (t >= total) playing = false;
      render();
    }
    requestAnimationFrame(tick);
  }

  window.Pipeline = {
    update(r) {
      const first = !run || run.run_id !== r.run_id;
      build(r);
      if (first && !live) { t = 0; playing = true; sel = 0; }
      if (live) { const cur = steps.findIndex((s) => s.status === "run"); if (cur >= 0) sel = cur; }
      $("doctorPipe").hidden = false;
      render();
    },
    research(ev, brief) {
      const m = ev.meta;
      const bOk = !!brief;
      const vStat = !bOk ? "idle" : brief.review_status === "medgemma_unavailable" ? "failed" : "done";
      const rs = [
        { name: "Gather relevant data", k: "a", st: "done", d: `${m.coverage.length} official source${m.coverage.length === 1 ? "" : "s"} read` + (m.failures.length ? `, ${m.failures.length} failed` : "") },
        { name: "Save and process the evidence pack", k: "a", st: "done", d: `${m.record_count} exact passages, pack ${ev.snapshot_id}` },
        { name: "Write a weekly summary", k: "g", st: bOk ? "done" : "idle", d: bOk ? `${brief.observations.length} cited observations` : "Not run for this pack" },
        { name: "Verification and confirmation", k: "m", st: vStat, d: !bOk ? "Waiting for the summary" : { reviewed_by_medgemma: "Clinical claims checked by MedGemma", not_needed: "No clinical claims needed checking", medgemma_unavailable: "MedGemma unavailable; clinical claims left out" }[brief.review_status] },
        { name: "Save with the research pack", k: "a", st: bOk ? "done" : "idle", d: bOk ? "Stored next to the passages" : "-" },
        { name: "Ready for doctor pipeline", k: "a", st: bOk ? "done" : "idle", d: bOk ? `Since ${new Date(brief.created_at).toLocaleString()}` : "-" },
      ];
      $("researchSteps").innerHTML = rs.map((s, i) => `<div class="step st-${s.st}"><span class="num">${i + 1}</span><span class="sname">${esc(s.name)}</span>
        <span class="chip k-${s.k}">${TAG[s.k]}</span><span class="sstat">${esc(s.d)}</span></div>`).join("");
      $("researchMeta").textContent = `Runs weekly with python -m app.refresh --brief. Pack created ${new Date(m.created_at).toLocaleString()}. Its output feeds every doctor review below.`;
    },
  };

  document.addEventListener("DOMContentLoaded", () => {
    $("pipeToggle").onclick = () => { if (t >= total) t = 0; playing = !playing; render(); };
    $("pipeRestart").onclick = () => { t = 0; playing = true; render(); };
    $("pipeScrub").oninput = (e) => { t = +e.target.value; playing = false; render(); };
    $("pipeSpeed").onchange = (e) => { speed = +e.target.value; };
    last = performance.now(); requestAnimationFrame(tick);
  });
})();
