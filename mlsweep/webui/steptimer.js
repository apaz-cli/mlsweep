// Step timer for running runs: the newest step, the logger's seconds per step,
// and a live count of seconds since the step last advanced.  The jobs API seeds
// it (last_step, step_age_seconds, step_seconds); the experiment WebSocket's
// metric events keep it current.  MLStepTimer.mount(el, () => runId) fills *el*
// once a second.
const MLStepTimer = (() => {
  const runs = new Map();   // run_id → { step, secs, at }  (at: Date.now() when step advanced)
  const mounts = [];

  function seed(job) {
    if (job.last_step == null) return;
    const prev = runs.get(job.run_id);
    if (prev && prev.step >= job.last_step) return;   // the WebSocket is already ahead
    runs.set(job.run_id, {
      step: job.last_step,
      secs: job.step_seconds ?? null,
      at: Date.now() - (job.step_age_seconds || 0) * 1000,
    });
  }

  function onMetric(evt) {
    const prev = runs.get(evt.run_id);
    const secs = evt.step_seconds ?? prev?.secs ?? null;
    if (!prev || evt.step > prev.step) runs.set(evt.run_id, { step: evt.step, secs, at: Date.now() });
    else prev.secs = secs;
    tick();
  }

  function clear() { runs.clear(); tick(); }

  function fmtSecs(s) {
    if (s < 1) return `${s.toFixed(3)} s`;
    if (s < 10) return `${s.toFixed(2)} s`;
    if (s < 120) return `${s.toFixed(1)} s`;
    return fmtAge(s);
  }
  function fmtAge(s) {
    s = Math.floor(s);
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
    return `${Math.floor(s / 3600)}h ${String(Math.floor(s / 60) % 60).padStart(2, "0")}m`;
  }

  function html(runId) {
    const r = runs.get(runId);
    if (!r) return `<span class="st-wait">waiting for the first step…</span>`;
    const age = (Date.now() - r.at) / 1000;
    // Late when the gap is well past the usual step time (or a minute when that is unknown).
    const late = age > Math.max(30, 3 * (r.secs || 20));
    return `<span class="st-item"><span class="st-k">step</span><b>${r.step.toLocaleString()}</b></span>`
      + `<span class="st-item"><span class="st-k">per step</span><b>${r.secs != null ? fmtSecs(r.secs) : "—"}</b></span>`
      + `<span class="st-item${late ? " st-late" : ""}"><span class="st-k">since last step</span><b>${fmtAge(age)}</b></span>`;
  }

  function tick() {
    for (const m of mounts) {
      const id = m.runId();
      m.el.style.display = id ? "" : "none";
      if (id) m.el.innerHTML = html(id);
    }
  }
  setInterval(tick, 1000);

  function mount(el, runId) { mounts.push({ el, runId }); tick(); }

  return { seed, onMetric, clear, mount, tick };
})();
