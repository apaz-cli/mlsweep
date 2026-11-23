import { AnsiUp } from './ansi_up.js';

const params = new URLSearchParams(location.search);

function parseJSON(s, fallback) {
  try { return JSON.parse(s); } catch { return fallback; }
}




MLChrome.collapsibleSidebar(document.getElementById("sidebar"));
MLChrome.resizable(document.getElementById("sidebar"), { name: "sidebar_width_logs", min: 180, max: 520 });

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
const codeBtn      = document.getElementById("code-btn");

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
let expSeq        = 0;      // bumped when another experiment is selected
let expsSeq       = 0;      // bumped when another experiment-list fetch starts
let jobsSeq       = 0;      // bumped when another job-list fetch starts
let historyLoading = false; // a selected run's history snapshot is in flight
let pendingLogEvents = [];  // live log chunks that arrive while history loads
let urlRunPending = params.get("run");  // ?run=, applied once

// "Wrap lines" off scrolls long lines sideways instead.
const wrapCb = document.getElementById("wrap-lines");
wrapCb.checked = !!MLPrefs.get("log_wrap", true);
logOutput.classList.toggle("nowrap", !wrapCb.checked);
wrapCb.addEventListener("change", () => {
  MLPrefs.set("log_wrap", wrapCb.checked);
  logOutput.classList.toggle("nowrap", !wrapCb.checked);
});

// ── Log rendering ─────────────────────────────────────────────────────────────
// With "Collapse JSON" on, a line that is a whole JSON object shows as one
// summary row that opens to the pretty-printed object.  Lines are only known to
// be whole once their newline arrives, so the unfinished last line is drawn in
// a tail element and redrawn as more of it comes.
const collapseCb = document.getElementById("collapse-json");
collapseCb.checked = !!MLPrefs.get("log_collapse_json", true);
let logRaw  = "";     // everything shown so far, to redraw when the toggle changes
let pending = "";     // the last line, until its newline arrives
let tailEl  = null;

function jsonOf(line) {
  const t = line.trim();
  if (t.length < 2 || t[0] !== "{" || t[t.length - 1] !== "}") return null;
  try {
    const obj = JSON.parse(t);
    return obj && typeof obj === "object" && !Array.isArray(obj) ? obj : null;
  } catch { return null; }
}

function shortValue(v) {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4)));
  const s = typeof v === "string" ? v : JSON.stringify(v);
  return s.length > 40 ? s.slice(0, 37) + "…" : s;
}

function jsonLineHtml(obj) {
  const keys = Object.keys(obj);
  const first = keys.includes("step") ? ["step", ...keys.filter(k => k !== "step")] : keys;
  const shown = first.slice(0, 6);
  const parts = shown.map(k => `<span class="jk">${escHtml(k)}</span> <span class="jv">${escHtml(shortValue(obj[k]))}</span>`);
  const more = keys.length - shown.length;
  return `<details class="json-line"><summary>${parts.join('<span class="jsep">·</span>')}`
    + `${more > 0 ? `<span class="jmore">+${more} keys</span>` : ""}</summary>`
    + `<pre>${escHtml(JSON.stringify(obj, null, 2))}</pre></details>`;
}

// Append *text* to the output, without recording it (see appendLog).
function renderChunk(text) {
  if (!collapseCb.checked) {
    logOutput.insertAdjacentHTML("beforeend", ansiUp.ansi_to_html(text));
    return;
  }
  const lines = (pending + text).split("\n");
  pending = lines.pop();
  let plain = "";   // consecutive non-JSON lines go through ansi_up together
  const flush = () => { if (plain) { tailEl.insertAdjacentHTML("beforebegin", ansiUp.ansi_to_html(plain)); plain = ""; } };
  if (!tailEl) { tailEl = document.createElement("span"); logOutput.append(tailEl); }
  for (const line of lines) {
    const obj = jsonOf(line);
    if (obj) { flush(); tailEl.insertAdjacentHTML("beforebegin", jsonLineHtml(obj)); }
    else plain += line + "\n";
  }
  flush();
  // Draw the unfinished line with a copy of the ANSI state, so the real state
  // only advances once the line is complete.
  const peek = Object.assign(Object.create(Object.getPrototypeOf(ansiUp)), ansiUp);
  tailEl.innerHTML = pending ? peek.ansi_to_html(pending) : "";
}

function appendLog(text) {
  if (!text) return;
  logRaw += text;
  renderChunk(text);
  if (autoScrollCb.checked) logOutput.scrollTop = logOutput.scrollHeight;
}

function resetLog() {
  logOutput.innerHTML = "";
  ansiUp  = new AnsiUp();
  logRaw  = "";
  pending = "";
  tailEl  = null;
}

collapseCb.addEventListener("change", () => {
  MLPrefs.set("log_collapse_json", collapseCb.checked);
  const text = logRaw, atBottom = autoScrollCb.checked;
  resetLog();
  appendLog(text);
  if (atBottom) logOutput.scrollTop = logOutput.scrollHeight;
});

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
  codeBtn.href = pageHref("code.html", { experiment: job.experiment_id, run: job.run_id });
  codeBtn.style.display = job.artifact_id ? "" : "none";

  // Highlight in sidebar.
  jobListEl.querySelectorAll(".job-item").forEach(el => {
    el.classList.toggle("selected", el.dataset.runId === job.run_id);
  });

  // Reset output.
  resetLog();
  autoScrollCb.checked = true;
  MLStepTimer.tick();
  jumpBtn.style.display = "none";
  await loadLogHistory(job);
}

// Draw the stored log of *job* (the selected run), replacing what is shown.
// Live log events for this run keep arriving on the experiment WebSocket;
// they are held until the snapshot is drawn.  The snapshot's X-Log-Seq
// header says which of them it already contains, so overlapping chunks are
// dropped rather than shown twice, out of order.  Called on selection and
// again whenever the socket (re)opens, since chunks broadcast while it was
// not subscribed are not replayed.
async function loadLogHistory(job) {
  if (loadAbort) { loadAbort.abort(); loadAbort = null; }
  const runId = job.run_id;
  historyLoading = true;
  pendingLogEvents = [];
  const ctrl = new AbortController();
  loadAbort  = ctrl;
  let text = null;
  let lastSeq = -1;
  try {
    const r = await apiFetch(
      `/api/experiments/${encodeURIComponent(job.experiment_id)}/jobs/${encodeURIComponent(runId)}/logs`,
      { signal: ctrl.signal },
    );
    if (r.ok) {
      const header = r.headers.get("X-Log-Seq");
      if (header != null && header !== "") {
        const n = Number(header);
        if (Number.isFinite(n)) lastSeq = n;
      }
      text = await r.text();
    }
  } catch (e) {
    if (e.name !== "AbortError") console.warn("Log fetch failed:", e);
  }

  // A newer job or experiment was selected, or a newer reload started.
  if (ctrl.signal.aborted || runId !== currentRunId) return;
  loadAbort = null;
  historyLoading = false;
  const buffered = pendingLogEvents;
  pendingLogEvents = [];
  // A failed reload keeps what is already shown.
  if (text == null && logRaw) return;
  const scroll = logOutput.scrollTop;
  resetLog();
  appendLog(text || "");
  for (const evt of buffered) {
    if (evt.run_id !== currentRunId) continue;
    if (lastSeq >= 0 && typeof evt.seq === "number" && evt.seq <= lastSeq) continue;
    if (evt.data) appendLog(evt.data);
  }
  if (!autoScrollCb.checked) logOutput.scrollTop = scroll;
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
        <span class="badge badge-${escHtml(j.status)}"${j.reason ? ` title="${escHtml(j.reason)}"` : ""}>${escHtml(j.status)}</span>
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

// Right-click a run: show it, rename it, or go to its other pages.
jobListEl.addEventListener("contextmenu", e => {
  const item = e.target.closest(".job-item");
  const job = item && jobsData.find(j => j.run_id === item.dataset.runId);
  if (!job || mlNativeMenuWanted(e, item)) return;
  mlContextMenu(e, [
    { title: job.label || job.run_id },
    job.run_id === currentRunId ? null : { label: "Show logs", run: () => selectJob(job) },
    { label: "Download log", run: () => {
      const a = Object.assign(document.createElement("a"), {
        href: authURL(`/api/experiments/${encodeURIComponent(job.experiment_id)}/jobs/${encodeURIComponent(job.run_id)}/logs`),
        download: `${job.run_id}.log`,
      });
      a.click();
    } },
    { label: "Rename…", run: async () => {
      const label = await mlRenameRun(job.experiment_id, job.run_id, job.label);
      if (label !== undefined) setJobLabel(job.run_id, label);
    } },
    "-",
    ...mlRunPageItems(job.experiment_id, job.run_id, { skip: "logs", code: Boolean(job.artifact_id) }),
    "-",
    { label: "Copy run ID", run: () => copyText(job.run_id, "Copied the run ID") },
  ]);
});

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
  const seq = ++jobsSeq;
  try {
    const r = await apiFetch(`/api/experiments/${encodeURIComponent(expId)}/jobs`);
    if (!r.ok) return;
    const data = await r.json();
    // Another experiment was selected, or a newer job-list fetch (the 15s
    // poll, a refresh) already answered.
    if (seq !== jobsSeq || expId !== currentExpId) return;
    jobsData = data;
    jobsData.forEach(MLStepTimer.seed);
    renderJobList();
    MLStepTimer.tick();
  } catch {}
}

async function selectExperiment(expId) {
  const token = ++expSeq;
  if (loadAbort) { loadAbort.abort(); loadAbort = null; }
  historyLoading = false;
  pendingLogEvents = [];

  currentExpId = expId;
  currentRunId = null;
  jobsData     = [];
  MLStepTimer.clear();

  resetLog();
  logHeader.style.display  = "none";
  logWrap.style.display    = "none";
  noSel.style.display      = "";
  logDlBtn.style.display   = "none";
  jobListEl.innerHTML      = `<div class="job-list-msg">Loading…</div>`;

  expDlAllBtn.href         = authURL(`/api/experiments/${encodeURIComponent(expId)}/logs.zip`);
  expDlAllBtn.style.display = "";

  await loadJobs(expId);
  if (token !== expSeq || currentExpId !== expId) return;
  connectWS(expId);

  // Auto-select the run from the URL, on the first experiment shown only.
  const urlRun = urlRunPending;
  urlRunPending = null;
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

  ws.addEventListener("open", () => {
    if (ws !== sock) return;
    setConnStatus("connected");
    // Log chunks sent before this socket subscribed are not replayed; reload
    // the selected run's snapshot so they are not missing.
    const job = jobsData.find(j => j.run_id === currentRunId);
    if (job) loadLogHistory(job);
  });

  ws.addEventListener("message", e => {
    // A message queued on the previous socket must not be taken for the new
    // experiment's stream: run IDs repeat across experiments.
    if (ws !== sock) return;
    try { handleWsEvent(JSON.parse(e.data)); } catch {}
  });

  ws.addEventListener("close", () => {
    if (ws !== sock) return;  // replaced or closed on purpose, not dropped
    setConnStatus("reconnecting");
    if (wsExpId === expId)
      wsReconnTimer = setTimeout(() => connectWS(expId), 3000);
  });

  ws.addEventListener("error", () => { if (ws === sock) setConnStatus("disconnected"); });
}

function handleWsEvent(evt) {
  if (evt.type === "log" && evt.run_id === currentRunId) {
    if (historyLoading) pendingLogEvents.push(evt);
    else appendLog(evt.data);
  } else if (evt.type === "metric") {
    MLStepTimer.onMetric(evt);
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
  expSeq += 1;
  if (wsReconnTimer) { clearTimeout(wsReconnTimer); wsReconnTimer = null; }
  wsExpId = null;
  if (ws) { ws.close(); ws = null; }
  currentExpId = null;
  currentRunId = null;
  historyLoading = false;
  pendingLogEvents = [];
  jobsData = [];
  MLStepTimer.clear();
  resetLog();
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
  const seq = ++expsSeq;
  try {
    const r = await apiFetch("/api/experiments");
    if (!r.ok) { if (seq === expsSeq) setConnStatus("disconnected"); return false; }
    const data = await r.json();
    if (seq !== expsSeq) return false;   // a newer list was fetched meanwhile
    allExps = data;
    setConnStatus("connected");

    const urlExp = firstLoad ? params.get("experiment") : null;
    firstLoad = false;
    MLCampaign.follow(allExps, urlExp);
    return await showCampaign(urlExp);
  } catch {
    if (seq === expsSeq) setConnStatus("disconnected");
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

// The step timer shows while the selected run is running.
MLStepTimer.mount(document.getElementById("step-timer"), () => {
  const job = jobsData.find(j => j.run_id === currentRunId);
  return job && job.status === "running" ? job.run_id : null;
});

// ── Init ──────────────────────────────────────────────────────────────────────
loadExperiments();
