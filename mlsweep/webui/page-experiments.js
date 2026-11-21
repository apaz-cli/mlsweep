// ── State ─────────────────────────────────────────────────────────────────────
const expandedExps = new Set();
const expandedJobs = new Set();  // "expId/runId"
const jobsCache = {};  // experiment_id → [jobs]
let experiments = [];     // experiments in the selected campaign
let allExperiments = [];  // every experiment, for the campaign list
let expSearch = "";       // words to match in name, ID and note
let expFilter = "all";    // all | running | completed | failed

// What the View menu's entries lead to, counted the way each page lists them:
// results and export are finished runs, logs are every run, artifacts are the
// runs with artifact files.
function expCounts(expId) {
  const exp = allExperiments.find(e => e.experiment_id === expId) || {};
  const c = exp.job_counts || {};
  return {
    total: c.total || exp.expected_jobs || 0,
    finished: (c.done || 0) + (c.pruned || 0) + (c.failed || 0) + (c.xfailed || 0),
    withArtifacts: c.with_artifacts || 0,
  };
}

function toggleExpView(expId) {
  const n = expCounts(expId);
  toggleMenu(this, `view:${expId}`, [
    { label: "Results", icon: "results", count: n.finished, href: pageHref("results.html", { experiment: expId }) },
    { label: "Logs", icon: "logs", count: n.total, href: pageHref("logs.html", { experiment: expId }) },
    { label: "Artifacts", icon: "artifacts", count: n.withArtifacts, href: pageHref("files.html", { experiment: expId }) },
    { label: "Code", icon: "code", href: pageHref("code.html", { experiment: expId }),
      title: "The code this experiment ran, with its file tree and where it came from" },
    notesItem(expId),
    { label: "Export", icon: "export", count: n.finished, href: pageHref("export.html", { experiment: expId }) },
  ].filter(Boolean));
}

function toggleExpMenu(expId) {
  toggleMenu(this, `exp:${expId}`, [
    { label: "Rename…", act: act("renameExperiment", expId) },
    { label: "Move to campaign…", act: act("moveExperiment", expId) },
    { label: "Delete…", act: act("deleteExperiment", expId), danger: true, divider: true },
  ]);
}


function toggleRunMenu(expId, runId) {
  const j = (jobsCache[expId] || []).find(x => x.run_id === runId) || {};
  const items = [{ label: "Rename…", act: act("renameJob", expId, runId) }];
  if (ACTIVE_RUN.includes(j.status)) {
    items.push({ label: "Cancel", act: act("runAction", expId, runId, "cancel"), danger: true });
  }
  if (UNSUCCESSFUL_RUN.includes(j.status)) {
    items.push({ label: "Retry", act: act("runAction", expId, runId, "retry") });
  }
  if (FINISHED_RUN.has(j.status)) {
    items.push({ label: "Move to experiment…", act: act("moveRun", expId, runId), divider: true });
    items.push({ label: "Delete…", act: act("purgeRun", expId, runId), danger: true });
  }
  toggleMenu(this, `run:${expId}/${runId}`, items);
}

async function deleteExperiment(expId) {
  if (!await mlConfirm(`Delete experiment "${expId}" and all its jobs?`, { confirmLabel: "Delete" })) return;
  const r = await apiFetch(`/api/experiments/${encodeURIComponent(expId)}`, { method: "DELETE" });
  if (r.ok) {
    document.getElementById(`exp-row-${expId}`)?.remove();
    runRowsOf(expId).forEach(r => r.remove());
    experiments = experiments.filter(e => e.experiment_id !== expId);
    allExperiments = allExperiments.filter(e => e.experiment_id !== expId);
  } else {
    await mlAlert(`Delete failed: ${r.status}`);
  }
}

const FINISHED_RUN = new Set(["done", "pruned", "failed", "xfailed", "cancelled"]);
const ACTIVE_RUN = ["pending", "dispatched", "running"];
const RETRYABLE_RUN = ["failed", "cancelled"];        // what "Retry failed" re-queues
const UNSUCCESSFUL_RUN = ["failed", "cancelled", "xfailed"];

// POST to /api/experiments/{id}/jobs/{op}; reports a failure and returns false.
async function postRuns(expId, op, body, label) {
  const r = await apiFetch(`/api/experiments/${encodeURIComponent(expId)}/jobs/${op}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (r.ok) return true;
  const err = await r.json().catch(() => ({}));
  await mlAlert(`${label} failed: ${err.error || r.status}`);
  return false;
}

async function refreshRuns(expId) {
  delete jobsCache[expId];
  await fetchExperiments();
  if (expandedExps.has(expId)) { expandedExps.delete(expId); await toggleExpand(expId); }
}

async function setExpStatus(expId, status) {
  if (status === "aborted" && !(await mlConfirm(`Stop ${expId}?\n\nAll unfinished runs are cancelled.`, { confirmLabel: "Stop" }))) return;
  const r = await apiFetch(`/api/experiments/${encodeURIComponent(expId)}/status`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status }),
  });
  if (!r.ok) { const body = await r.json().catch(() => ({})); await mlAlert(`Failed: ${body.error || r.status}`); }
  await refreshRuns(expId);
}

async function runAction(expId, runId, action) {
  await postRunAction(expId, runId, action);
  await refreshRuns(expId);
}

async function bulkRuns(expId, action) {
  const jobs = jobsCache[expId] || [];
  const pick = { retry: RETRYABLE_RUN, cancel: ACTIVE_RUN, purge: UNSUCCESSFUL_RUN }[action];
  const ids = jobs.filter(j => pick.includes(j.status)).map(j => j.run_id);
  if (!ids.length) return;
  const verb = { retry: "Re-queue", cancel: "Cancel", purge: "Delete for good" }[action];
  if (!(await mlConfirm(`${verb} ${ids.length} run${ids.length === 1 ? "" : "s"} of ${expId}?`, { confirmLabel: verb, danger: action !== "retry" }))) return;
  if (action === "purge") {
    await postRuns(expId, "purge", { run_ids: ids }, "Delete");
  } else {
    const fails = [];
    for (const id of ids) {
      const r = await apiFetch(`/api/jobs/${encodeURIComponent(id)}/${action}?experiment_id=${encodeURIComponent(expId)}`, { method: "POST" });
      if (!r.ok) fails.push(id);
    }
    if (fails.length) await mlAlert(`${fails.length} run(s) could not be changed:\n${fails.slice(0, 10).join("\n")}`);
  }
  await refreshRuns(expId);
}

async function moveRun(expId, runId) {
  const here = MLCampaign.campaignOf(allExperiments.find(e => e.experiment_id === expId));
  const target = await mlPick(`Move run ${runId}, with its metrics, logs and files, to:`,
    mlExperimentItems(allExperiments, { exclude: [expId], campaign: here }), { placeholder: "Search experiments" });
  if (!target) return;
  if (!(await postRuns(expId, "move", { run_ids: [runId], to: target }, "Move"))) return;
  delete jobsCache[target];
  await refreshRuns(expId);
}

async function purgeRun(expId, runId) {
  if (!(await mlConfirm(`Delete run ${runId}?\n\nIts record, metrics, logs and files are removed for good.`, { confirmLabel: "Delete" }))) return;
  if (!(await postRuns(expId, "purge", { run_ids: [runId] }, "Delete"))) return;
  await refreshRuns(expId);
}

async function moveExperiment(expId) {
  const exp = allExperiments.find(e => e.experiment_id === expId);
  const current = MLCampaign.campaignOf(exp);
  const counts = new Map();
  for (const e of allExperiments) counts.set(MLCampaign.campaignOf(e), (counts.get(MLCampaign.campaignOf(e)) || 0) + 1);
  const items = MLCampaign.names(allExperiments).map(c => ({
    value: c, label: c, sub: `${counts.get(c) || 0} experiment${counts.get(c) === 1 ? "" : "s"}` + (c === current ? " · here now" : ""),
    disabled: c === current,
  }));
  const target = await mlPick(`Move "${exp?.name || expId}" to campaign:`, items,
                              { placeholder: "Search campaigns", create: { label: "New campaign…", prompt: "Name of the new campaign:" } });
  if (!target || target.trim() === current) return;
  const r = await apiFetch(`/api/experiments/${encodeURIComponent(expId)}/campaign`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ campaign: target.trim() }),
  });
  if (r.ok) {
    await fetchExperiments();
  } else {
    const body = await r.json().catch(() => ({}));
    await mlAlert(`Move failed: ${body.error || r.status}`);
  }
}

async function renameExperiment(expId) {
  const exp = experiments.find(e => e.experiment_id === expId);
  const current = exp?.name || expId;
  const newName = await mlPrompt("Rename experiment:", current, { okLabel: "Rename" });
  if (newName === null || newName === current) return;
  if (!newName) { await mlAlert("Name cannot be empty."); return; }
  const r = await apiFetch(`/api/experiments/${encodeURIComponent(expId)}/name`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: newName }),
  });
  if (r.ok) {
    if (exp) exp.name = newName;
    renderCurrent();
  } else {
    await mlAlert(`Rename failed: ${r.status}`);
  }
}

async function renameJob(expId, runId) {
  const job = (jobsCache[expId] || []).find(j => j.run_id === runId);
  const label = await mlRenameRun(expId, runId, job?.label);
  if (label === undefined) return;
  if (job) job.label = label;
  renderRuns(expId);
  // Then redraw from the manager, which also brings in anything else that changed.
  await refreshAll();
}

// ── Job detail panel ──────────────────────────────────────────────────────────
function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
       + " " + d.toLocaleDateString([], { month: "short", day: "numeric" });
}

function renderJobDetail(j) {
  const cmd = (() => { try { return JSON.parse(j.command || "[]").join(" "); } catch { return j.command || ""; } })();
  const gpuIds = (() => { try { return JSON.parse(j.dispatched_gpu_ids || "null"); } catch { return null; } })();
  const retryStr = j.retry_count > 0 ? `${j.retry_count} / ${j.max_retries}` : `0 / ${j.max_retries}`;
  const exitStr  = j.exit_code != null ? String(j.exit_code) : "—";
  const resources = [
    j.gpus_per_run > 1  ? `${j.gpus_per_run} GPUs/run` : "1 GPU/run",
    j.nodes_per_run > 1 ? `${j.nodes_per_run} nodes` : null,
    gpuIds              ? `GPU IDs: ${gpuIds.join(",")}` : null,
  ].filter(Boolean).join(" · ");

  return `<div class="job-detail-panel">
    <div class="job-detail-section">
      <div class="job-detail-item"><span class="job-detail-label">Submitted</span><span class="job-detail-value">${fmtTime(j.submit_time)}</span></div>
      <div class="job-detail-item"><span class="job-detail-label">Dispatched</span><span class="job-detail-value">${fmtTime(j.dispatch_time)}</span></div>
      <div class="job-detail-item"><span class="job-detail-label">Started</span><span class="job-detail-value">${fmtTime(j.start_time)}</span></div>
      <div class="job-detail-item"><span class="job-detail-label">Finished</span><span class="job-detail-value">${fmtTime(j.finish_time)}</span></div>
      <div class="job-detail-item"><span class="job-detail-label">Exit code</span><span class="job-detail-value ${j.exit_code ? 'muted' : ''}">${exitStr}</span></div>
      <div class="job-detail-item"><span class="job-detail-label">Retries</span><span class="job-detail-value ${j.retry_count === 0 ? 'muted' : ''}">${retryStr}</span></div>
      <div class="job-detail-item"><span class="job-detail-label">Priority</span><span class="job-detail-value">${j.priority}</span></div>
      ${resources ? `<div class="job-detail-item"><span class="job-detail-label">Resources</span><span class="job-detail-value">${escHtml(resources)}</span></div>` : ""}
      ${j.artifact_id ? `<div class="job-detail-item"><span class="job-detail-label">Code</span><span class="job-detail-value"><a class="code-btn" href="${pageHref("code.html", { experiment: j.experiment_id, run: j.run_id })}" title="The code this run used"><svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="m5.5 4.5-3.5 3.5 3.5 3.5M10.5 4.5 14 8l-3.5 3.5"/></svg>Code <span class="snap">${escHtml(j.artifact_id.slice(0, 17))}</span></a></span></div>` : ""}
    </div>
    ${cmd ? `<div><div class="job-detail-label" style="margin-bottom:3px">Command</div><div class="job-cmd-block">${cmd.replace(/&/g,"&amp;").replace(/</g,"&lt;")}</div></div>` : ""}
  </div>`;
}

function toggleJobDetail(expId, runId) {
  const key = `${expId}/${runId}`;
  const detailRow = document.getElementById(`job-detail-${expId}-${runId}`);
  const toggleEl  = document.getElementById(`job-toggle-${expId}-${runId}`);
  if (!detailRow) return;
  const isOpen = expandedJobs.has(key);
  if (isOpen) {
    expandedJobs.delete(key);
    detailRow.style.display = "none";
    if (toggleEl) toggleEl.textContent = "▶";
  } else {
    expandedJobs.add(key);
    detailRow.style.display = "";
    if (toggleEl) toggleEl.textContent = "▼";
    // Populate if empty (in case job was added after initial render)
    const jobs = jobsCache[expId] || [];
    const j = jobs.find(x => x.run_id === runId);
    if (j) detailRow.querySelector("td").innerHTML = renderJobDetail(j);
  }
}

// A pencil and count beside a name that notes mention, linking to those notes.
function noteRefHtml(expId, runId) {
  const n = notesEnabled ? noteCount(expId, runId) : 0;
  if (!n) return "";
  const what = runId || expId;
  const title = runId
    ? `${n} note${n === 1 ? "" : "s"} mention this run`
    : `${n} note${n === 1 ? "" : "s"} mention this experiment or its runs`;
  return `<a class="note-ref" href="${escHtml(pageHref("notes.html", { mentions: what }))}" title="${title}" data-act="">`
    + `${VIEW_ICONS.notes}${n}</a>`;
}

// ── Render runs ───────────────────────────────────────────────────────────────
// An experiment's runs are rows of the experiments table, tagged with
// data-runs-of so they can be replaced together.
function runsHtml(expId) {
  const jobs = jobsCache[expId] || [];
  const rows = jobs.map(j => {
    const comboObj = j.combo ? (typeof j.combo === "string" ? (() => { try { return JSON.parse(j.combo); } catch { return {}; } })() : j.combo) : {};
    const comboStr = escHtml(Object.entries(comboObj).map(([k,v]) => `${k}=${v}`).join(" "));
    const key = `${expId}/${j.run_id}`;
    const isOpen = expandedJobs.has(key);
    return `<tr class="run-row" style="cursor:pointer" ${act("toggleJobDetail", expId, j.run_id)}>
      <td class="mono"><span class="job-toggle" id="${escHtml(`job-toggle-${expId}-${j.run_id}`)}">${isOpen ? "▼" : "▶"}</span>${runNameHtml(j)}</td>
      <td${j.reason ? ` title="${escHtml(j.reason)}"` : ""}>${badge(j.status)}</td>
      <td colspan="2">${j.worker_id ? escHtml(j.worker_id.split(":")[0]) : "—"}</td>
      <td>${fmtElapsed(j.elapsed)}</td>
      <td colspan="3" class="combo-cell mono" title="${comboStr}"><div class="combo-clip">${comboStr}</div></td>
      <td class="notes-cell" data-act="">${noteRefHtml(expId, j.run_id)}</td>
      <td class="actions-cell" data-act=""><div class="actions">
        <button class="btn-link btn-menu" aria-haspopup="true" title="Results, Artifacts, Logs, Code, Notes" ${act("toggleRunView", expId, j.run_id)}>View ▾</button>
        <button class="btn-link btn-manage" aria-haspopup="true" title="Rename, cancel, move or delete" ${act("toggleRunMenu", expId, j.run_id)}>Manage</button>
      </div></td>
    </tr>
    <tr id="${escHtml(`job-detail-${expId}-${j.run_id}`)}" class="job-detail-row" style="display:${isOpen ? "" : "none"}">
      <td colspan="10">${isOpen ? renderJobDetail(j) : ""}</td>
    </tr>`;
  }).join("");
  const head = `<tr class="runs-head"><td>Run ID</td><td>Status</td><td colspan="2">Worker</td>
      <td>Elapsed</td><td colspan="3">Combo</td><td>Notes</td><td></td></tr>`;
  return runTools(expId, jobs) + (rows ? head + rows : `<tr class="runs-empty"><td colspan="10">No jobs</td></tr>`);
}

// Experiment controls (the same as mlsweep pause/unpause/stop/retry/cancel/purge).
function runTools(expId, jobs) {
  const exp = allExperiments.find(e => e.experiment_id === expId) || {};
  const n = st => jobs.filter(j => st.includes(j.status)).length;
  const retryable = n(RETRYABLE_RUN), active = n(ACTIVE_RUN);
  const purgeable = n(UNSUCCESSFUL_RUN);
  const b = (label, action, title, cls = "") => `<button class="btn-link ${cls}" title="${title}" ${act(...action)}>${label}</button>`;
  const btns = [];
  if (exp.status === "running") btns.push(b("Pause", ["setExpStatus", expId, "paused"], "Stop dispatching new runs (mlsweep pause)"));
  if (exp.status === "paused") btns.push(b("Unpause", ["setExpStatus", expId, "running"], "Resume dispatching (mlsweep unpause)"));
  if (exp.status === "running" || exp.status === "paused") btns.push(b("Stop", ["setExpStatus", expId, "aborted"], "Abort: cancel in-flight runs and stop dispatching (mlsweep stop)", "btn-delete"));
  if (retryable) btns.push(b(`Retry failed (${retryable})`, ["bulkRuns", expId, "retry"], "Re-queue failed and cancelled runs (mlsweep resume)"));
  if (active) btns.push(b(`Cancel active (${active})`, ["bulkRuns", expId, "cancel"], "Cancel pending and running runs (mlsweep cancel --all)", "btn-delete"));
  if (purgeable) btns.push(b(`Delete failed (${purgeable})`, ["bulkRuns", expId, "purge"], "Delete failed, cancelled and xfailed runs with their files (mlsweep purge)", "btn-delete"));
  if (!btns.length) return "";
  return `<tr class="runs-tools"><td colspan="10"><div class="actions">${btns.join("")}</div></td></tr>`;
}

function runRowsOf(expId) {
  return [...document.querySelectorAll(`#exp-tbody tr[data-runs-of="${CSS.escape(expId)}"]`)];
}

// Replace an experiment's run rows: drawn under its row when it is expanded,
// removed when it is not.
function renderRuns(expId) {
  runRowsOf(expId).forEach(r => r.remove());
  const expRow = document.getElementById(`exp-row-${expId}`);
  if (!expRow || !expandedExps.has(expId) || !jobsCache[expId]) return;
  const tmp = document.createElement("tbody");
  tmp.innerHTML = runsHtml(expId);
  const rows = [...tmp.rows];
  rows.forEach(r => { r.dataset.runsOf = expId; });
  expRow.after(...rows);
}

// ── Toggle expand ──────────────────────────────────────────────────────────────
async function toggleExpand(expId) {
  const expRow  = document.getElementById(`exp-row-${expId}`);
  const toggle  = expRow ? expRow.querySelector(".exp-toggle") : null;

  if (expandedExps.has(expId)) {
    expandedExps.delete(expId);
    renderRuns(expId);
    if (toggle) toggle.textContent = "▶";
    expRow && expRow.classList.remove("expanded");
    return;
  }

  expandedExps.add(expId);
  if (toggle) toggle.textContent = "▼";
  expRow && expRow.classList.add("expanded");

  if (!jobsCache[expId]) {
    try {
      const r = await apiFetch(`/api/experiments/${encodeURIComponent(expId)}/jobs`);
      jobsCache[expId] = await r.json();
    } catch {
      jobsCache[expId] = [];
    }
  }

  renderRuns(expId);
}

// ── Refresh jobs for expanded experiments ─────────────────────────────────────
async function refreshExpandedJobs() {
  for (const expId of expandedExps) {
    try {
      const r = await apiFetch(`/api/experiments/${encodeURIComponent(expId)}/jobs`);
      jobsCache[expId] = await r.json();
      renderRuns(expId);
    } catch {}
  }
}

// ── Empty state: run instructions ────────────────────────────────────────────
let publicUrl = null;  // set when the manager is reachable from its public IP

function cmdBlock(url) {
  const lines = [
    `mlsweep_run`,
    `     --manager ${url}`,
  ];
  if (TOKEN) lines.push(`     --token ${TOKEN}`);
  const c = MLCampaign.get();
  if (c !== MLCampaign.ALL && c !== MLCampaign.DEFAULT) lines.push(`     --campaign ${c}`);
  lines.push(`    sweep.py`);
  const maxLen = Math.max(...lines.map(l => l.length));
  return `<code class="run-cmd">${lines.map((l, i) =>
    escHtml(i < lines.length - 1 ? l + " ".repeat(maxLen - l.length + 1) + "\\" : l)
  ).join("\n")}</code>`;
}

function emptyHtml() {
  const c = MLCampaign.get();
  const lead = allExperiments.length && c !== MLCampaign.ALL
    ? `No experiments in campaign <b>${escHtml(c)}</b>. Submit one with:`
    : `No experiments yet. Submit one with:`;
  let html = `${lead}<br>${cmdBlock(window.location.origin)}`;
  if (publicUrl) html += `<div class="run-label">From a remote machine:</div>${cmdBlock(publicUrl)}`;
  return html;
}

(async function findPublicUrl() {
  try {
    const r = await apiFetch("/api/health");
    if (r.ok) publicUrl = (await r.json()).public_url || null;
  } catch (_) {}
  if (!filtering()) document.getElementById("empty-msg").innerHTML = emptyHtml();
})();

// ── Search and status filter ─────────────────────────────────────────────────
// Running covers anything still scheduling work; failed is an aborted
// experiment or a finished one with failed runs.
function expCategory(exp) {
  if (["running", "pending", "paused"].includes(exp.status)) return "running";
  if (exp.status === "aborted" || (exp.job_counts?.failed || 0) > 0) return "failed";
  if (exp.status === "completed") return "completed";
  return "other";
}

function expMatches(exp) {
  if (expFilter !== "all" && expCategory(exp) !== expFilter) return false;
  const words = expSearch.toLowerCase().split(/\s+/).filter(Boolean);
  const hay = `${exp.name || ""} ${exp.experiment_id} ${exp.note || ""}`.toLowerCase();
  return words.every(w => hay.includes(w));
}

function clearExpFilters() {
  expSearch = "";
  expFilter = "all";
  document.getElementById("exp-search").value = "";
  renderCurrent();
}

const filtering = () => !!expSearch.trim() || expFilter !== "all";

function renderCurrent() {
  const counts = { all: experiments.length, running: 0, completed: 0, failed: 0 };
  for (const exp of experiments) {
    const cat = expCategory(exp);
    if (cat in counts) counts[cat]++;
  }
  document.querySelectorAll("#exp-filters .filter-chip").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.filter === expFilter);
    btn.querySelector(".n").textContent = counts[btn.dataset.filter];
  });
  const shown = experiments.filter(expMatches);
  document.getElementById("exp-shown").textContent =
    filtering() && experiments.length ? `${shown.length} of ${experiments.length} shown` : "";
  renderExperiments(shown);
}

document.getElementById("exp-search").addEventListener("input", e => {
  expSearch = e.target.value;
  renderCurrent();
});
document.getElementById("exp-filters").addEventListener("click", e => {
  const btn = e.target.closest(".filter-chip");
  if (!btn) return;
  expFilter = btn.dataset.filter;
  renderCurrent();
});

// ── Render experiments table ──────────────────────────────────────────────────
function renderExperiments(exps) {
  const tbody = document.getElementById("exp-tbody");
  const emptyMsg = document.getElementById("empty-msg");
  const table = document.getElementById("exp-table");

  if (exps.length === 0) {
    tbody.innerHTML = "";
    table.style.display = "none";
    emptyMsg.innerHTML = filtering() && experiments.length
      ? `No experiments match. <button class="btn-link" ${act("clearExpFilters")}>Clear search and filter</button>`
      : emptyHtml();
    emptyMsg.style.display = "";
    return;
  }
  table.style.display = "";
  emptyMsg.style.display = "none";

  // Keep track of which rows already exist to avoid full re-render.
  const existingIds = new Set(
    [...tbody.querySelectorAll(".exp-row")].map(r => r.dataset.expId)
  );
  const newIds = new Set(exps.map(e => e.experiment_id));

  // Remove stale rows.
  for (const id of existingIds) {
    if (!newIds.has(id)) {
      document.getElementById(`exp-row-${id}`)?.remove();
      runRowsOf(id).forEach(r => r.remove());
    }
  }

  // Insert/update rows.
  exps.forEach((exp, idx) => {
    const expId = exp.experiment_id;
    const existingRow = document.getElementById(`exp-row-${expId}`);
    const isExpanded = expandedExps.has(expId);

    const c = exp.job_counts || {};
    const running = (c.running || 0) + (c.dispatched || 0);
    const total = c.total || exp.expected_jobs || 0;
    const done = exp.status && exp.status !== "running" && exp.status !== "pending";
    const html = `<td>
        <div class="exp-name-cell">
          <span class="exp-toggle" ${act("toggleExpand", expId)}>${isExpanded ? "▼" : "▶"}</span>
          <span class="exp-name" ${act("toggleExpand", expId)} style="cursor:pointer">${escHtml(exp.name || expId)}</span>
          ${exp.note ? `<span class="exp-note">${escHtml(exp.note)}</span>` : ""}
          ${MLCampaign.get() === MLCampaign.ALL ? `<span class="campaign-tag">${escHtml(MLCampaign.campaignOf(exp))}</span>` : ""}
        </div>
      </td>
      <td>${badge(exp.status || "unknown")}</td>
      <td class="count-cell">${c.done || 0}${c.pruned ? ` <span style="color:var(--text-muted)" title="stopped early by PRUNE">+${c.pruned} pruned</span>` : ""}</td>
      <td class="count-cell">${c.failed ? `<span style="color:var(--bad)">${c.failed}</span>` : (done ? "0" : "—")}</td>
      <td class="count-cell">${c.xfailed || (done ? "0" : "—")}</td>
      <td class="count-cell">${running ? `<span style="color:var(--ok)">${running}</span>` : (done ? "0" : "—")}</td>
      <td class="count-cell">${total ? `${(c.done||0)+(c.pruned||0)+(c.failed||0)+(c.xfailed||0)}/${total}` : "—"}</td>
      <td>${relTime(exp.submit_time)}</td>
      <td class="notes-cell">${noteRefHtml(expId)}</td>
      <td class="actions-cell"><div class="actions">
        <button class="btn-link btn-menu" aria-haspopup="true" title="Results, Logs, Artifacts, Code, Notes, Export" ${act("toggleExpView", expId)}>View ▾</button>
        <button class="btn-link btn-manage" aria-haspopup="true" title="Rename, move or delete" ${act("toggleExpMenu", expId)}>Manage</button>
      </div></td>`;

    if (existingRow) {
      existingRow.innerHTML = html;
      existingRow.classList.toggle("expanded", isExpanded);
    } else {
      // Insert new row at correct position.
      const tr = document.createElement("tr");
      tr.className = "exp-row";
      tr.id = `exp-row-${expId}`;
      tr.dataset.expId = expId;
      tr.innerHTML = html;

      // Before the idx-th experiment row; the previous experiment's runs stay above it.
      const allRows = tbody.querySelectorAll(".exp-row");
      tbody.insertBefore(tr, idx < allRows.length ? allRows[idx] : null);
    }

    if (isExpanded) renderRuns(expId);
  });
}

// ── Fetch and refresh ─────────────────────────────────────────────────────────
async function fetchExperiments() {
  try {
    const r = await apiFetch("/api/experiments");
    if (!r.ok) {
      setConnStatus("disconnected");
      document.getElementById("error-msg").textContent = `API error: ${r.status}`;
      document.getElementById("error-msg").style.display = r.status === 401 ? "none" : "";
      document.getElementById("empty-msg").style.display = "none";
      document.getElementById("exp-table").style.display = "none";
      return;
    }
    allExperiments = await r.json();
    showCampaign();
    setConnStatus("connected");
    document.getElementById("error-msg").style.display = "none";
  } catch (e) {
    setConnStatus("disconnected");
    document.getElementById("error-msg").textContent = `Failed to fetch: ${e}`;
    document.getElementById("error-msg").style.display = "";
  }
}

// ── Campaign ──────────────────────────────────────────────────────────────────
const campaignSelect = document.getElementById("campaign-select");

function showCampaign() {
  MLCampaign.fill(campaignSelect, allExperiments);
  experiments = MLCampaign.filter(allExperiments);
  renderCurrent();
}

MLCampaign.bind(campaignSelect, showCampaign);

watchNoteMentions(() => { if (allExperiments.length) showCampaign(); });

async function refreshAll() {
  await fetchExperiments();
  await refreshExpandedJobs();
}

fetchExperiments();
setInterval(fetchExperiments, 5000);
setInterval(refreshExpandedJobs, 10000);
// A rename made in another tab or from the CLI shows as soon as you come back.
document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshAll(); });

bindActions({
  toggleJobDetail, renameJob, runAction, moveRun, purgeRun, setExpStatus, bulkRuns,
  toggleExpand, renameExperiment, moveExperiment, deleteExperiment,
  toggleExpView, toggleExpMenu, toggleRunView, toggleRunMenu, clearExpFilters,
});
