import { AnsiUp } from './ansi_up.js';

const params = new URLSearchParams(location.search);

function parseJSON(s, fallback) {
  try { return JSON.parse(s); } catch { return fallback; }
}




// ── DOM refs ──────────────────────────────────────────────────────────────────
const expSelect    = document.getElementById("exp-select");
const jobListEl    = document.getElementById("job-list");
const logHeader    = document.getElementById("log-header");
const logWrap      = document.getElementById("log-wrap");
const logOutput    = document.getElementById("log-output");
const noSel        = document.getElementById("no-sel");
const jumpBtn      = document.getElementById("jump-btn");
const autoScrollCb = document.getElementById("auto-scroll");
const logRunIdEl   = document.getElementById("log-run-id");
const logBadgeEl   = document.getElementById("log-badge");
const logDlBtn     = document.getElementById("log-dl-btn");
const expDlAllBtn  = document.getElementById("exp-dl-all");
const refreshBtn   = document.getElementById("refresh-btn");
const renameBtn    = document.getElementById("rename-btn");

// ── Auto-scroll ───────────────────────────────────────────────────────────────
logOutput.addEventListener("scroll", () => {
  const atBottom = logOutput.scrollHeight - logOutput.scrollTop - logOutput.clientHeight < 60;
  jumpBtn.style.display = atBottom ? "none" : "";
  if (atBottom) autoScrollCb.checked = true;
});
jumpBtn.addEventListener("click", () => {
  logOutput.scrollTop = logOutput.scrollHeight;
  jumpBtn.style.display = "none";
  autoScrollCb.checked = true;
});

// ── State ─────────────────────────────────────────────────────────────────────
let jobsData      = [];
let jobFilter     = "all";
let currentExpId  = null;
let currentRunId  = null;
let ansiUp        = new AnsiUp();
let ws            = null;
let wsExpId       = null;
let wsReconnTimer = null;
let loadAbort     = null;   // AbortController for in-flight history fetches

// ── Log rendering ─────────────────────────────────────────────────────────────
function appendLog(text) {
  if (!text) return;
  logOutput.insertAdjacentHTML("beforeend", ansiUp.ansi_to_html(text));
  if (autoScrollCb.checked) logOutput.scrollTop = logOutput.scrollHeight;
}

// ── Job selection ─────────────────────────────────────────────────────────────
async function selectJob(job) {
  // Abort any previous in-flight history fetch.
  if (loadAbort) { loadAbort.abort(); loadAbort = null; }

  currentRunId = job.run_id;

  logRunIdEl.innerHTML = runNameHtml(job);
  setBadge(job.status);
  logHeader.style.display = "";
  logWrap.style.display   = "flex";
  noSel.style.display     = "none";

  // Wire download button.
  logDlBtn.href            = authURL(`/api/experiments/${encodeURIComponent(job.experiment_id)}/jobs/${encodeURIComponent(job.run_id)}/logs`);
  logDlBtn.download        = `${job.run_id}.log`;
  logDlBtn.style.display   = "";

  // Highlight in sidebar.
  jobListEl.querySelectorAll(".job-item").forEach(el => {
    el.classList.toggle("selected", el.dataset.runId === job.run_id);
  });

  // Reset output.
  logOutput.innerHTML  = "";
  ansiUp               = new AnsiUp();
  autoScrollCb.checked = true;
  jumpBtn.style.display = "none";

  // Fetch history.
  const ctrl = new AbortController();
  loadAbort  = ctrl;
  try {
    const r = await apiFetch(
      `/api/experiments/${encodeURIComponent(job.experiment_id)}/jobs/${encodeURIComponent(job.run_id)}/logs`,
      { signal: ctrl.signal },
    );
    if (!ctrl.signal.aborted && r.ok) {
      const text = await r.text();
      if (!ctrl.signal.aborted) appendLog(text);
    }
  } catch (e) {
    if (e.name !== "AbortError") console.warn("Log fetch failed:", e);
  }
}

function setBadge(status) {
  logBadgeEl.innerHTML = `<span class="badge badge-${escHtml(status)}">${escHtml(status)}</span>`;
}

// ── Job list ──────────────────────────────────────────────────────────────────
function renderJobList() {
  if (!jobsData.length) {
    jobListEl.innerHTML = `<div class="job-list-msg">No jobs</div>`;
    return;
  }

  const FAILED_STATUSES = new Set(["failed", "xfailed", "cancelled"]);
  const filtered = jobsData.filter(j =>
    jobFilter === "all"    ? true :
    jobFilter === "done"   ? j.status === "done" :
    jobFilter === "failed" ? FAILED_STATUSES.has(j.status) : true
  );

  if (!filtered.length) {
    jobListEl.innerHTML = `<div class="job-list-msg">No ${jobFilter} jobs</div>`;
    return;
  }

  const sorted = [...filtered].sort((a, b) => new Date(b.submit_time) - new Date(a.submit_time));

  jobListEl.innerHTML = sorted.map(j => {
    const combo    = parseJSON(typeof j.combo === "string" ? j.combo : JSON.stringify(j.combo || {}), {});
    const comboStr = Object.entries(combo).map(([k, v]) => `${k}=${v}`).join(" ");
    const sel      = j.run_id === currentRunId;
    return `<div class="job-item${sel ? " selected" : ""}" data-run-id="${escHtml(j.run_id)}" data-exp-id="${escHtml(j.experiment_id)}">
      <div class="job-item-top">
        <span class="job-run-id" title="${escHtml(j.run_id)}">${runNameHtml(j)}</span>
        <span class="badge badge-${escHtml(j.status)}">${escHtml(j.status)}</span>
      </div>
      ${comboStr ? `<div class="job-combo" title="${escHtml(comboStr)}">${escHtml(comboStr)}</div>` : ""}
    </div>`;
  }).join("");

  jobListEl.querySelectorAll(".job-item").forEach(el => {
    el.addEventListener("click", () => {
      const job = jobsData.find(j => j.run_id === el.dataset.runId);
      if (job) selectJob(job);
    });
  });
}

// Show a run's new display name in the list and, if selected, the header.
function setJobLabel(runId, label) {
  const job = jobsData.find(j => j.run_id === runId);
  if (!job) return;
  job.label = label;
  const el = Array.from(jobListEl.querySelectorAll(".job-item"))
    .find(el => el.dataset.runId === runId);
  if (el) el.querySelector(".job-run-id").innerHTML = runNameHtml(job);
  if (runId === currentRunId) logRunIdEl.innerHTML = runNameHtml(job);
}

async function renameCurrentJob() {
  const job = jobsData.find(j => j.run_id === currentRunId);
  if (!job) return;
  const label = await mlRenameRun(job.experiment_id, job.run_id, job.label);
  if (label !== undefined) setJobLabel(job.run_id, label);
}
renameBtn.addEventListener("click", renameCurrentJob);

function setJobStatus(runId, status) {
  const job = jobsData.find(j => j.run_id === runId);
  if (job) job.status = status;

  // Update badge in list without full re-render.
  const el = Array.from(jobListEl.querySelectorAll(".job-item"))
    .find(el => el.dataset.runId === runId);
  if (el) {
    const b = el.querySelector(".badge");
    if (b) { b.className = `badge badge-${status}`; b.textContent = status; }
  }

  if (runId === currentRunId) setBadge(status);
}

// ── Experiment selection ──────────────────────────────────────────────────────
async function loadJobs(expId) {
  try {
    const r = await apiFetch(`/api/experiments/${encodeURIComponent(expId)}/jobs`);
    if (!r.ok) return;
    jobsData = await r.json();
    renderJobList();
  } catch {}
}

async function selectExperiment(expId) {
  if (loadAbort) { loadAbort.abort(); loadAbort = null; }

  currentExpId = expId;
  currentRunId = null;
  jobsData     = [];

  logOutput.innerHTML      = "";
  logHeader.style.display  = "none";
  logWrap.style.display    = "none";
  noSel.style.display      = "";
  logDlBtn.style.display   = "none";
  jobListEl.innerHTML      = `<div class="job-list-msg">Loading…</div>`;

  expDlAllBtn.href         = authURL(`/api/experiments/${encodeURIComponent(expId)}/logs.zip`);
  expDlAllBtn.style.display = "";

  await loadJobs(expId);
  connectWS(expId);

  // Auto-select run from URL on first load only.
  const urlRun = params.get("run");
  if (urlRun) {
    const job = jobsData.find(j => j.run_id === urlRun);
    if (job) await selectJob(job);
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS(expId) {
  if (wsReconnTimer) { clearTimeout(wsReconnTimer); wsReconnTimer = null; }
  if (ws) { ws.close(); ws = null; }

  wsExpId = expId;
  const sock = new WebSocket(wsURL(`/ws/experiments/${encodeURIComponent(expId)}`));
  ws = sock;

  ws.addEventListener("open", () => setConnStatus("connected"));

  ws.addEventListener("message", e => {
    try { handleWsEvent(JSON.parse(e.data)); } catch {}
  });

  ws.addEventListener("close", () => {
    if (ws !== sock) return;  // replaced or closed on purpose, not dropped
    setConnStatus("reconnecting");
    if (wsExpId === expId)
      wsReconnTimer = setTimeout(() => connectWS(expId), 3000);
  });

  ws.addEventListener("error", () => setConnStatus("disconnected"));
}

function handleWsEvent(evt) {
  if (evt.type === "log" && evt.run_id === currentRunId) {
    appendLog(evt.data);
  } else if (evt.type === "job_started") {
    setJobStatus(evt.run_id, "running");
  } else if (evt.type === "job_dispatched") {
    setJobStatus(evt.run_id, "dispatched");
  } else if (evt.type === "job_done") {
    setJobStatus(evt.run_id, evt.status || (evt.success ? "done" : "failed"));
  } else if (evt.type === "job_updated" && "label" in evt) {
    setJobLabel(evt.run_id, evt.label);
  }
}

// ── Experiment list ───────────────────────────────────────────────────────────
const campaignSelect = document.getElementById("campaign-select");
let allExps = [];
let firstLoad = true;

function clearExperiment() {
  if (wsReconnTimer) { clearTimeout(wsReconnTimer); wsReconnTimer = null; }
  wsExpId = null;
  if (ws) { ws.close(); ws = null; }
  currentExpId = null;
  currentRunId = null;
  jobsData = [];
  logOutput.innerHTML       = "";
  logHeader.style.display   = "none";
  logWrap.style.display     = "none";
  noSel.style.display       = "";
  expDlAllBtn.style.display = "none";
  jobListEl.innerHTML       = `<div class="job-list-msg">No experiments in this campaign</div>`;
}

// Fill the experiment select from the selected campaign and load *prefer*,
// else the experiment already shown, else the newest one.  Returns true when
// it switched experiments (which loads that experiment's jobs).
async function showCampaign(prefer) {
  MLCampaign.fill(campaignSelect, allExps);
  const targetId = MLCampaign.fillExperiments(expSelect, MLCampaign.filter(allExps), prefer, currentExpId);
  if (!targetId) { clearExperiment(); return false; }
  if (targetId === currentExpId) return false;
  await selectExperiment(targetId);
  return true;
}

// Returns true when the shown experiment changed (see showCampaign).
async function loadExperiments() {
  try {
    const r = await apiFetch("/api/experiments");
    if (!r.ok) { setConnStatus("disconnected"); return false; }
    allExps = await r.json();
    setConnStatus("connected");

    const urlExp = firstLoad ? params.get("experiment") : null;
    firstLoad = false;
    MLCampaign.follow(allExps, urlExp);
    return await showCampaign(urlExp);
  } catch {
    setConnStatus("disconnected");
    return false;
  }
}

MLCampaign.bind(campaignSelect, () => showCampaign(null));
expSelect.addEventListener("change", () => selectExperiment(expSelect.value));
refreshBtn.addEventListener("click", async () => {
  if (!(await loadExperiments()) && currentExpId) loadJobs(currentExpId);
});

["all", "done", "failed"].forEach(f => {
  document.getElementById(`filter-${f}`).addEventListener("click", () => {
    jobFilter = f;
    document.querySelectorAll(".quick-btn").forEach(b => b.classList.toggle("active", b.id === `filter-${f}`));
    renderJobList();
  });
});

// Refresh job list periodically to pick up newly submitted jobs.
setInterval(() => { if (currentExpId) loadJobs(currentExpId); }, 15000);

// ── Init ──────────────────────────────────────────────────────────────────────
loadExperiments();
