"""Embedded HTTP web UI for the mlsweep sweep controller.

Serves a live dashboard showing worker status, in-flight/pending/completed jobs,
and exposes a cancel endpoint so users can stop individual runs from the browser.

The ControllerUI object is thread-safe: the controller's event loop calls
update_* methods, while concurrent HTTP handler threads read state and serve SSE.
"""

import copy
import json
import queue
import socket
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


# ── Shared state ───────────────────────────────────────────────────────────────


class ControllerUI:
    """Thread-safe shared state between the sweep controller and its HTTP UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "experiment": "",
            "exp_dir": "",
            "output_dir": "",
            "total_expected": 0,
            "method": "grid",
            "started_at": time.time(),
            "workers": {},   # str(worker_id) → worker dict
            "jobs": {},      # run_id → job dict
        }
        self._cancel_q: "queue.Queue[str]" = queue.Queue()
        self._sse_lock = threading.Lock()
        self._sse_subs: "list[queue.Queue[bytes | None]]" = []

    # ── Controller-facing API ──────────────────────────────────────────────────

    def update(self, **kwargs: Any) -> None:
        """Update top-level state fields and broadcast to SSE subscribers."""
        with self._lock:
            self._data.update(kwargs)
        if kwargs:
            self._broadcast("update", kwargs)

    def update_worker(self, worker_id: int, **kwargs: Any) -> None:
        """Upsert a worker entry and broadcast the delta."""
        key = str(worker_id)
        with self._lock:
            w = self._data["workers"].setdefault(key, {"id": worker_id})
            w.update(kwargs)
        self._broadcast("worker", {"id": worker_id, **kwargs})

    def update_job(self, run_id: str, **kwargs: Any) -> None:
        """Upsert a job entry and broadcast the delta."""
        with self._lock:
            j = self._data["jobs"].setdefault(run_id, {"run_id": run_id})
            j.update(kwargs)
        self._broadcast("job", {"run_id": run_id, **kwargs})

    def snapshot(self) -> "dict[str, Any]":
        """Return a deep copy of the current state (safe to serialize to JSON)."""
        with self._lock:
            return copy.deepcopy(self._data)

    def request_cancel(self, run_id: str) -> None:
        """Queue a cancel request (HTTP layer → controller loop)."""
        self._cancel_q.put(run_id)

    def pop_cancel_requests(self) -> "list[str]":
        """Drain and return all pending cancel requests (called by controller loop)."""
        reqs: list[str] = []
        while True:
            try:
                reqs.append(self._cancel_q.get_nowait())
            except queue.Empty:
                break
        return reqs

    # ── HTTP/SSE-facing API ────────────────────────────────────────────────────

    def subscribe_sse(self) -> "queue.Queue[bytes | None]":
        q: "queue.Queue[bytes | None]" = queue.Queue(maxsize=200)
        with self._sse_lock:
            self._sse_subs.append(q)
        return q

    def unsubscribe_sse(self, q: "queue.Queue[bytes | None]") -> None:
        with self._sse_lock:
            if q in self._sse_subs:
                self._sse_subs.remove(q)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _broadcast(self, event_type: str, data: "dict[str, Any]") -> None:
        msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()
        with self._sse_lock:
            subs = list(self._sse_subs)
        dead = []
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        if dead:
            with self._sse_lock:
                self._sse_subs = [q for q in self._sse_subs if q not in dead]

    def start(self, port: int) -> "tuple[ThreadingHTTPServer, int] | None":
        """Start the HTTP server daemon thread. Returns (server, bound_port) or None."""
        try:
            handler_cls = _make_handler(self)
            server = ThreadingHTTPServer(("", port), handler_cls)
            actual_port: int = server.server_address[1]
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            return server, actual_port
        except OSError:
            return None


# ── HTML / JS ──────────────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<title>Sweep Controller</title>
<link rel="icon" type="image/png" href="/favicon.png">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #f5f5f5; --surface: #fff; --border: #ddd; --border-input: #ccc;
  --text: #222; --text-muted: #999; --text-dim: #888; --text-label: #555; --text-strong: #333;
  --btn-bg: white; --btn-hover: #f0f0f0; --btn-text: #666;
  --radio-bg: #fafafa; --radio-checked-bg: #e8f0fe;
  --radio-checked-border: #4c9be8; --radio-checked-color: #1a56c4;
  --plot-bg: white; --plot-grid: #eee; --plot-text: #444;
  --swatch-border: rgba(0,0,0,0.15);
}
[data-theme=dark] {
  --bg: #1c1c1c; --surface: #252525; --border: #383838; --border-input: #444;
  --text: #ddd; --text-muted: #666; --text-dim: #777; --text-label: #aaa; --text-strong: #ccc;
  --btn-bg: #2e2e2e; --btn-hover: #383838; --btn-text: #aaa;
  --radio-bg: #2e2e2e; --radio-checked-bg: #1e2f50;
  --radio-checked-border: #4c9be8; --radio-checked-color: #82b4f0;
  --plot-bg: #1e1e1e; --plot-grid: #2e2e2e; --plot-text: #aaa;
  --swatch-border: rgba(255,255,255,0.15);
}

body { font-family: system-ui, -apple-system, sans-serif; display: flex;
       height: 100vh; background: var(--bg); color: var(--text); overflow: hidden; }

#sidebar { width: 270px; min-width: 270px; background: var(--surface);
           border-right: 1px solid var(--border); overflow-y: auto;
           padding: 14px 12px; display: flex; flex-direction: column; gap: 16px; }

#main { flex: 1; display: flex; flex-direction: column; min-width: 0; padding: 12px; gap: 8px; }

h1 { font-size: 13px; font-weight: 700; color: var(--text-strong); }
.exp-name { font-size: 10px; color: var(--text-muted); word-break: break-all; margin-top: 2px; }

.section { display: flex; flex-direction: column; gap: 6px; }
.section-title { font-size: 10px; font-weight: 700; color: var(--text-dim);
                 text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Tab bar ── */
#topbar { display: flex; align-items: center; justify-content: space-between; gap: 8px;
          flex-shrink: 0; }
#tabs { display: flex; gap: 3px; flex-wrap: wrap; }
.tab { font-size: 11px; padding: 3px 12px; border: 1px solid var(--border-input);
       border-radius: 4px; cursor: pointer; background: var(--btn-bg); color: var(--btn-text);
       font-family: inherit; white-space: nowrap; }
.tab:hover { background: var(--btn-hover); }
.tab.active { background: var(--radio-checked-bg); border-color: var(--radio-checked-border);
              color: var(--radio-checked-color); font-weight: 600; }

#statusbar { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
#status { font-size: 11px; color: var(--text-muted); }
#theme-toggle { background: none; border: none; font-size: 16px; cursor: pointer;
                color: var(--text-dim); padding: 0; line-height: 1; }
#theme-toggle:hover { color: var(--text); }

/* ── Tab panes ── */
.tab-pane { flex: 1; min-height: 0; overflow-y: auto; border: 1px solid var(--border);
            border-radius: 6px; background: var(--surface); }

/* ── Job table ── */
.job-table { width: 100%; border-collapse: collapse; }
.job-table th {
  position: sticky; top: 0; z-index: 1;
  background: var(--surface);
  text-align: left; padding: 6px 10px;
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--text-dim);
  border-bottom: 1px solid var(--border);
}
.job-table td {
  padding: 5px 10px; border-bottom: 1px solid var(--border);
  font-size: 11px; color: var(--text);
  font-variant-numeric: tabular-nums;
}
.job-table td.mono { font-family: ui-monospace, "Cascadia Code", monospace; font-size: 10.5px; }
.job-table tr:last-child td { border-bottom: none; }
.job-table tbody tr:hover td { background: var(--btn-hover); }

/* ── Status badges ── */
.badge { display: inline-block; padding: 1px 6px; border-radius: 3px;
         font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }
:root .badge-run  { background: #ddeeff; color: #1a56c4; }
:root .badge-ok   { background: #e6f4ea; color: #2e7d32; }
:root .badge-fail { background: #fdecea; color: #c62828; }
:root .badge-pend { background: #f0f0f0;  color: #888; }
:root .badge-warn { background: #fff8e1; color: #f57f17; }
[data-theme=dark] .badge-run  { background: #1e2f50; color: #82b4f0; }
[data-theme=dark] .badge-ok   { background: #1a3020; color: #6dbf7e; }
[data-theme=dark] .badge-fail { background: #2f1a1a; color: #e57373; }
[data-theme=dark] .badge-pend { background: #2e2e2e; color: #777; }
[data-theme=dark] .badge-warn { background: #2f2300; color: #ffc107; }

/* ── Cancel button ── */
.cancel-btn { font-size: 9px; padding: 1px 7px; border: 1px solid var(--border-input);
              background: var(--btn-bg); border-radius: 3px; cursor: pointer;
              color: var(--btn-text); font-family: inherit; }
.cancel-btn:hover { border-color: #ef5350; color: #ef5350; background: var(--btn-hover); }

/* ── Progress bar ── */
.progress-outer { width: 100%; height: 5px; background: var(--border);
                  border-radius: 3px; overflow: hidden; }
.progress-inner { height: 100%; background: #4c9be8; border-radius: 3px; transition: width 0.4s; }

/* ── Sidebar stat rows ── */
.stat-row { display: flex; justify-content: space-between; align-items: baseline;
            font-size: 11px; }
.stat-label { color: var(--text-muted); }
.stat-value { color: var(--text-strong); font-weight: 600; font-variant-numeric: tabular-nums; }

/* ── Sidebar worker summary ── */
.worker-row { display: flex; align-items: center; gap: 6px; font-size: 11px; padding: 2px 0; }
.worker-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-ok   { background: #4caf50; }
.dot-warn { background: #ffc107; }
.dot-dead { background: #ef5350; }
.dot-conn { background: var(--text-muted); }
.worker-host-label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); }
.worker-slots-label { font-size: 10px; color: var(--text-muted); font-variant-numeric: tabular-nums; }

/* ── Worker cards (workers tab) ── */
#worker-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                gap: 10px; padding: 12px; align-content: start; }
.worker-card { border: 1px solid var(--border); border-radius: 6px;
               padding: 10px 12px; display: flex; flex-direction: column; gap: 7px; }
.card-header { display: flex; align-items: center; justify-content: space-between; gap: 6px; }
.card-host { font-size: 12px; font-weight: 600; color: var(--text-strong);
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.card-meta { font-size: 10px; color: var(--text-muted); }
.slot-bar { display: flex; gap: 3px; flex-wrap: wrap; }
.slot-chip { width: 20px; height: 20px; border-radius: 3px; border: 1px solid var(--border);
             display: flex; align-items: center; justify-content: center;
             font-size: 8px; color: var(--text-dim); }
:root .slot-chip.busy { border-color: #4c9be8; background: #ddeeff; color: #1a56c4; }
[data-theme=dark] .slot-chip.busy { border-color: #4c9be8; background: #1e2f50; color: #82b4f0; }
.card-jobs { font-size: 10px; color: var(--text-muted); word-break: break-all;
             font-family: ui-monospace, monospace; }

/* ── Pending list ── */
.pending-row { display: flex; align-items: baseline; gap: 8px; padding: 5px 10px;
               border-bottom: 1px solid var(--border); }
.pending-row:last-child { border-bottom: none; }
.pending-num { color: var(--text-dim); min-width: 26px; text-align: right;
               font-size: 10px; font-variant-numeric: tabular-nums; }
.pending-id { font-family: ui-monospace, monospace; font-size: 10.5px; color: var(--text-strong); }
.pending-combo { font-size: 10px; color: var(--text-muted); flex: 1; }

/* ── Empty state ── */
.empty-msg { padding: 40px; text-align: center; color: var(--text-muted); font-size: 12px; }
</style>
</head>
<body>

<div id="sidebar">
  <div>
    <h1>Sweep Controller</h1>
    <div id="exp-name" class="exp-name">—</div>
  </div>

  <div class="section">
    <div class="section-title">Progress</div>
    <div style="display:flex;flex-direction:column;gap:5px;">
      <div class="progress-outer">
        <div class="progress-inner" id="progress-bar" style="width:0%"></div>
      </div>
      <div class="stat-row">
        <span class="stat-label">Done</span>
        <span class="stat-value" id="stat-done">0 / 0</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Running</span>
        <span class="stat-value" id="stat-running">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Pending</span>
        <span class="stat-value" id="stat-pending">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Elapsed</span>
        <span class="stat-value" id="stat-elapsed">—</span>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Workers</div>
    <div id="worker-summary"></div>
  </div>
</div>

<div id="main">
  <div id="topbar">
    <div id="tabs">
      <button class="tab active" data-tab="running">Running</button>
      <button class="tab" data-tab="pending">Pending</button>
      <button class="tab" data-tab="done">Done</button>
      <button class="tab" data-tab="workers">Workers</button>
    </div>
    <div id="statusbar">
      <div id="status">Connecting…</div>
      <button id="theme-toggle" title="Switch to light mode">☀</button>
    </div>
  </div>

  <!-- Running tab -->
  <div id="tab-running" class="tab-pane">
    <table class="job-table">
      <thead><tr>
        <th>Run</th><th>Combo</th><th>Worker</th><th>GPUs</th><th>Elapsed</th><th></th>
      </tr></thead>
      <tbody id="running-tbody"></tbody>
    </table>
    <div id="running-empty" class="empty-msg">No running jobs</div>
  </div>

  <!-- Pending tab -->
  <div id="tab-pending" class="tab-pane" style="display:none">
    <div id="pending-list"></div>
    <div id="pending-empty" class="empty-msg">No pending jobs</div>
  </div>

  <!-- Done tab -->
  <div id="tab-done" class="tab-pane" style="display:none">
    <table class="job-table">
      <thead><tr>
        <th>Run</th><th>Combo</th><th>Status</th><th>Elapsed</th>
      </tr></thead>
      <tbody id="done-tbody"></tbody>
    </table>
    <div id="done-empty" class="empty-msg">No completed jobs yet</div>
  </div>

  <!-- Workers tab -->
  <div id="tab-workers" class="tab-pane" style="display:none">
    <div id="worker-cards"></div>
    <div id="workers-empty" class="empty-msg">No workers</div>
  </div>
</div>

<script>
// ── State ─────────────────────────────────────────────────────────────────────

let STATE = {experiment:"", exp_dir:"", output_dir:"", total_expected:0,
             method:"grid", started_at:0, workers:{}, jobs:{}};
let activeTab = "running";

// ── Formatters ────────────────────────────────────────────────────────────────

function fmtElapsed(s) {
  s = Math.max(0, s);
  if (s < 60)   return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s/60)}m ${Math.round(s%60)}s`;
  return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;
}

function fmtCombo(combo) {
  if (!combo || !Object.keys(combo).length) return "—";
  return Object.entries(combo).map(([k,v]) => `${k}=${v}`).join("  ");
}

function fmtGpus(gpu_ids) {
  if (!gpu_ids || !gpu_ids.length) return "—";
  return gpu_ids.map(String).join(", ");
}

function badge(status) {
  const labels = {
    running:"running", ok:"ok", failed:"failed", pending:"pending",
    CONNECTED:"connected", CONNECTING:"connecting", RECONNECTING:"reconnect", DEAD:"dead",
  };
  const cls = {
    running:"badge-run", ok:"badge-ok", failed:"badge-fail", pending:"badge-pend",
    CONNECTED:"badge-ok", CONNECTING:"badge-pend", RECONNECTING:"badge-warn", DEAD:"badge-fail",
  };
  const text = labels[status] || status;
  const c = cls[status] || "badge-pend";
  return `<span class="badge ${c}">${text}</span>`;
}

// ── Data helpers ──────────────────────────────────────────────────────────────

function getJobs(status) {
  return Object.values(STATE.jobs).filter(j => j.status === status);
}

function getWorkers() {
  return Object.values(STATE.workers).sort((a,b) => (a.id||0) - (b.id||0));
}

function countDone() {
  return Object.values(STATE.jobs).filter(j => j.status === "ok" || j.status === "failed").length;
}

// ── Sidebar rendering ─────────────────────────────────────────────────────────

function renderSidebar() {
  document.getElementById("exp-name").textContent = STATE.experiment || "—";

  const running = getJobs("running").length;
  const pending = getJobs("pending").length;
  const done    = countDone();
  const total   = STATE.total_expected || 0;
  const pct     = total > 0 ? Math.min(100, (done / total) * 100) : 0;

  document.getElementById("progress-bar").style.width = `${pct}%`;
  document.getElementById("stat-done").textContent    = `${done} / ${total}`;
  document.getElementById("stat-running").textContent = running;
  document.getElementById("stat-pending").textContent = pending;

  const ws = getWorkers();
  const container = document.getElementById("worker-summary");
  if (!ws.length) {
    container.innerHTML = `<div style="font-size:11px;color:var(--text-muted)">No workers yet</div>`;
    return;
  }
  container.innerHTML = "";
  for (const w of ws) {
    const dotCls = {
      CONNECTED:"dot-ok", CONNECTING:"dot-conn", RECONNECTING:"dot-warn", DEAD:"dot-dead",
    }[w.status] || "dot-conn";
    const busy  = w.slots_busy  || 0;
    const total = w.slots_total || 0;
    const row = document.createElement("div");
    row.className = "worker-row";
    row.innerHTML = `
      <div class="worker-dot ${dotCls}"></div>
      <span class="worker-host-label">${w.host || `worker ${w.id}`}</span>
      <span class="worker-slots-label">${busy}/${total}</span>
    `;
    container.appendChild(row);
  }
}

// ── Tab: Running ──────────────────────────────────────────────────────────────

function renderRunning() {
  const jobs   = getJobs("running").sort((a,b) => (a.started_at||0) - (b.started_at||0));
  const tbody  = document.getElementById("running-tbody");
  const empty  = document.getElementById("running-empty");
  tbody.innerHTML = "";
  empty.style.display = jobs.length ? "none" : "";
  if (!jobs.length) return;
  const now = Date.now() / 1000;
  for (const j of jobs) {
    const elapsed = j.started_at ? now - j.started_at : 0;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${esc(j.run_id)}</td>
      <td>${esc(fmtCombo(j.combo))}</td>
      <td>${esc(j.worker_host || "—")}</td>
      <td>${esc(fmtGpus(j.gpu_ids))}</td>
      <td data-started-at="${j.started_at||0}">${fmtElapsed(elapsed)}</td>
      <td><button class="cancel-btn" data-run-id="${esc(j.run_id)}">Cancel</button></td>
    `;
    tbody.appendChild(tr);
  }
  tbody.querySelectorAll(".cancel-btn").forEach(btn => {
    btn.addEventListener("click", () => cancelJob(btn.dataset.runId));
  });
}

// ── Tab: Pending ──────────────────────────────────────────────────────────────

function renderPending() {
  const jobs  = getJobs("pending").sort((a,b) => (a.run_id||"").localeCompare(b.run_id||""));
  const list  = document.getElementById("pending-list");
  const empty = document.getElementById("pending-empty");
  list.innerHTML = "";
  empty.style.display = jobs.length ? "none" : "";
  if (!jobs.length) return;
  jobs.forEach((j, i) => {
    const row = document.createElement("div");
    row.className = "pending-row";
    row.innerHTML = `
      <span class="pending-num">${i+1}</span>
      <span class="pending-id">${esc(j.run_id)}</span>
      <span class="pending-combo">${esc(fmtCombo(j.combo))}</span>
    `;
    list.appendChild(row);
  });
}

// ── Tab: Done ─────────────────────────────────────────────────────────────────

function renderDone() {
  const jobs  = Object.values(STATE.jobs)
    .filter(j => j.status === "ok" || j.status === "failed")
    .sort((a,b) => (a.run_id||"").localeCompare(b.run_id||""));
  const tbody = document.getElementById("done-tbody");
  const empty = document.getElementById("done-empty");
  tbody.innerHTML = "";
  empty.style.display = jobs.length ? "none" : "";
  if (!jobs.length) return;
  for (const j of jobs) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${esc(j.run_id)}</td>
      <td>${esc(fmtCombo(j.combo))}</td>
      <td>${badge(j.status)}</td>
      <td>${j.elapsed != null ? fmtElapsed(j.elapsed) : "—"}</td>
    `;
    tbody.appendChild(tr);
  }
}

// ── Tab: Workers ──────────────────────────────────────────────────────────────

function renderWorkers() {
  const workers = getWorkers();
  const cards   = document.getElementById("worker-cards");
  const empty   = document.getElementById("workers-empty");
  cards.innerHTML = "";
  empty.style.display = workers.length ? "none" : "";
  if (!workers.length) return;
  for (const w of workers) {
    const busy  = w.slots_busy  || 0;
    const total = w.slots_total || 0;
    const gpus  = w.gpus || [];

    let chipsHtml = "";
    for (let i = 0; i < total; i++) {
      const isBusy = i < busy;
      const gid = gpus[i] != null ? gpus[i] : "";
      chipsHtml += `<div class="slot-chip${isBusy?" busy":""}" title="GPU ${gid}">${gid}</div>`;
    }

    const wJobs = Object.values(STATE.jobs).filter(
      j => j.status === "running" && j.worker_host === w.host
    );
    const jobsText = wJobs.length
      ? wJobs.map(j => esc(j.run_id)).join("<br>")
      : `<span style="color:var(--text-dim)">idle</span>`;

    const card = document.createElement("div");
    card.className = "worker-card";
    card.innerHTML = `
      <div class="card-header">
        <div class="card-host">${esc(w.host || `worker ${w.id}`)}</div>
        ${badge(w.status || "CONNECTING")}
      </div>
      <div class="card-meta">
        ${gpus.length} GPU${gpus.length===1?"":"s"}
        &nbsp;·&nbsp;
        ${busy} / ${total} slot${total===1?"":"s"} busy
      </div>
      ${total > 0 ? `<div class="slot-bar">${chipsHtml}</div>` : ""}
      <div class="card-jobs">${jobsText}</div>
    `;
    cards.appendChild(card);
  }
}

// ── Cancel ────────────────────────────────────────────────────────────────────

function cancelJob(runId) {
  fetch("/api/cancel", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({run_id: runId}),
  }).catch(() => {});
}

// ── Rendering orchestration ───────────────────────────────────────────────────

function renderCurrentTab() {
  if      (activeTab === "running") renderRunning();
  else if (activeTab === "pending") renderPending();
  else if (activeTab === "done")    renderDone();
  else if (activeTab === "workers") renderWorkers();
}

function updateTabCounts() {
  const counts = {
    running: getJobs("running").length,
    pending: getJobs("pending").length,
    done:    countDone(),
    workers: getWorkers().length,
  };
  const labels = {running:"Running", pending:"Pending", done:"Done", workers:"Workers"};
  document.querySelectorAll(".tab").forEach(btn => {
    const t = btn.dataset.tab;
    btn.textContent = `${labels[t]} (${counts[t]??0})`;
    btn.classList.toggle("active", t === activeTab);
  });
}

function renderAll() {
  renderSidebar();
  renderCurrentTab();
  updateTabCounts();
}

function setTab(name) {
  activeTab = name;
  document.querySelectorAll(".tab-pane").forEach(p => p.style.display = "none");
  const pane = document.getElementById(`tab-${name}`);
  if (pane) pane.style.display = "";
  updateTabCounts();
  renderCurrentTab();
}

// ── XSS-safe escaping ─────────────────────────────────────────────────────────

function esc(s) {
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

// ── Theme ─────────────────────────────────────────────────────────────────────

(function() {
  const btn  = document.getElementById("theme-toggle");
  const root = document.documentElement;
  function applyTheme(dark) {
    dark ? root.setAttribute("data-theme","dark") : root.removeAttribute("data-theme");
    btn.textContent = dark ? "☀" : "☾";
    btn.title = dark ? "Switch to light mode" : "Switch to dark mode";
  }
  applyTheme(localStorage.getItem("theme") !== "light");
  btn.addEventListener("click", () => {
    const dark = root.getAttribute("data-theme") !== "dark";
    localStorage.setItem("theme", dark ? "dark" : "light");
    applyTheme(dark);
  });
})();

// ── SSE connection ────────────────────────────────────────────────────────────

function connectSSE() {
  const es = new EventSource("/events");

  es.addEventListener("init", e => {
    STATE = JSON.parse(e.data);
    document.getElementById("status").textContent = "Live";
    renderAll();
  });

  es.addEventListener("update", e => {
    Object.assign(STATE, JSON.parse(e.data));
    renderSidebar();
    updateTabCounts();
  });

  es.addEventListener("worker", e => {
    const d = JSON.parse(e.data);
    const key = String(d.id);
    STATE.workers[key] = Object.assign(STATE.workers[key] || {id: d.id}, d);
    renderSidebar();
    if (activeTab === "workers") renderWorkers();
    updateTabCounts();
  });

  es.addEventListener("job", e => {
    const d = JSON.parse(e.data);
    STATE.jobs[d.run_id] = Object.assign(STATE.jobs[d.run_id] || {run_id: d.run_id}, d);
    renderAll();
  });

  es.onerror = () => {
    document.getElementById("status").textContent = "Disconnected — retrying…";
  };
}

// ── Elapsed ticker (updates running job timers client-side) ───────────────────

setInterval(() => {
  const now = Date.now() / 1000;
  document.querySelectorAll("[data-started-at]").forEach(el => {
    const t = parseFloat(el.dataset.startedAt);
    if (t > 0) el.textContent = fmtElapsed(now - t);
  });
  const elEl = document.getElementById("stat-elapsed");
  if (elEl && STATE.started_at) elEl.textContent = fmtElapsed(now - STATE.started_at);
}, 1000);

// ── Tab wire-up & bootstrap ───────────────────────────────────────────────────

document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => setTab(btn.dataset.tab));
});

connectSSE();
</script>
</body>
</html>
"""


# ── HTTP handler ───────────────────────────────────────────────────────────────


def _make_handler(ui: ControllerUI) -> type:
    class _Handler(BaseHTTPRequestHandler):
        _ui = ui

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)

            if parsed.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", _HTML.encode())

            elif parsed.path == "/api/state":
                body = json.dumps(self._ui.snapshot()).encode()
                self._send(200, "application/json", body)

            elif parsed.path == "/events":
                self._serve_sse()

            elif parsed.path == "/favicon.png":
                fav = Path(__file__).parent / "static" / "favicon.png"
                if fav.exists():
                    self._send(200, "image/png", fav.read_bytes())
                else:
                    self.send_response(404)
                    self.end_headers()

            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:
            if self.path == "/api/cancel":
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length))
                    run_id = str(body["run_id"])
                    self._ui.request_cancel(run_id)
                    self._send(200, "application/json", b'{"ok":true}')
                except (KeyError, ValueError, json.JSONDecodeError):
                    self.send_response(400)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def _serve_sse(self) -> None:
            q = self._ui.subscribe_sse()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                snapshot = self._ui.snapshot()
                init_msg = f"event: init\ndata: {json.dumps(snapshot)}\n\n".encode()
                self.wfile.write(init_msg)
                self.wfile.flush()
                while True:
                    try:
                        chunk = q.get(timeout=15)
                        if chunk is None:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except OSError:
                pass
            finally:
                self._ui.unsubscribe_sse(q)

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            pass  # suppress request logs

    return _Handler
