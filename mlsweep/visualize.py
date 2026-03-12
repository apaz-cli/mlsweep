#!/usr/bin/env python3

"""
Sweep experiment visualizer.

Reads training metrics from local JSONL files and serves an interactive browser
UI for exploring loss curves across experimental dims.

Usage:
    mlsweep_viz                                                     # newest experiment
    mlsweep_viz debug_smoke_...                                     # specific experiment
    mlsweep_viz --open-browser
    mlsweep_viz --port 43801
    mlsweep_viz --dir ./outputs/sweeps                              # local files
"""

import argparse
import base64
import dataclasses
import importlib.metadata
import queue
from typing import Any
import json
import math
import os
import socket
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mlsweep._shared import _detect_sub_dims, _parse_tag_value, _val_sort_key

# ── Data loading (file-based) ──────────────────────────────────────────────────


def list_experiments(output_dir: str) -> list[str]:
    """Return all experiment names, sorted newest-first."""
    root = Path(output_dir)
    latest: dict[str, float] = {}

    # New format: directories with sweep_manifest.json
    for manifest_path in root.glob("*/sweep_manifest.json"):
        exp = manifest_path.parent.name
        t = manifest_path.stat().st_mtime
        latest[exp] = max(latest.get(exp, 0.0), t)

    # Old format: run_meta.json files (backward compat)
    for meta_path in root.glob("*/*/run_meta.json"):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            exp = meta.get("experiment")
            if exp:
                t = meta.get("start_time", 0.0)
                latest[exp] = max(latest.get(exp, 0.0), t)
        except (OSError, json.JSONDecodeError):
            pass

    return sorted(latest, key=lambda e: latest[e], reverse=True)


def load_experiment_meta(experiment_name: str, output_dir: str) -> dict[str, Any]:
    """Return dims, run combos, and metric names — no metric data loaded.

    Returns {"experiment", "dims", "runs", "metricNames", "subDims"}.
    Each run contains {"name", "hash", "combo"} where "hash" == run_name.
    """
    root = Path(output_dir) / experiment_name

    # New format: sweep_manifest.json has everything
    manifest_path = root / "sweep_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest: dict[str, Any] = json.load(f)
            # Populate metricNames if missing
            if not manifest.get("metricNames"):
                metric_names: set[str] = set()
                for metrics_path in sorted(root.glob("*/metrics.jsonl")):
                    try:
                        with open(metrics_path) as f:
                            first_line = f.readline()
                        if first_line.strip():
                            rec = json.loads(first_line)
                            for k in rec:
                                if k not in ("step",):
                                    metric_names.add(k)
                        if metric_names:
                            break
                    except (OSError, json.JSONDecodeError):
                        continue
                manifest["metricNames"] = sorted(metric_names)
            return manifest
        except (OSError, json.JSONDecodeError):
            pass

    # Old format: scan run_meta.json files
    dim_values: dict[str, set[Any]] = {}
    runs: list[dict[str, Any]] = []
    old_metric_names: set[str] = set()

    for meta_path in sorted(root.glob("*/run_meta.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        run_name = meta.get("run_name") or meta_path.parent.name
        tags = meta.get("tags", {})
        combo = {k: _parse_tag_value(str(v)) for k, v in tags.items()}
        for k, v in combo.items():
            dim_values.setdefault(k, set()).add(v)

        runs.append({"name": run_name, "hash": run_name, "combo": combo})

        metrics_path = meta_path.parent / "metrics.jsonl"
        try:
            with open(metrics_path) as f:
                first_line = f.readline()
            if first_line.strip():
                rec = json.loads(first_line)
                for k in rec:
                    if k not in ("step", "t"):
                        old_metric_names.add(k)
        except (OSError, json.JSONDecodeError):
            pass

    dims = {k: sorted(vs, key=_val_sort_key) for k, vs in dim_values.items()}
    sub_dims = _detect_sub_dims(runs, dims)

    return {"experiment": experiment_name, "dims": dims, "runs": runs,
            "metricNames": sorted(old_metric_names), "subDims": sub_dims}


def load_manifest(experiment_name: str, output_dir: str) -> dict[str, Any] | None:
    """Return the sweep manifest dict, or None if not found."""
    p = Path(output_dir) / experiment_name / "sweep_manifest.json"
    try:
        with open(p) as f:
            data: dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    metric_names_val = data.get("metricNames")
    if not metric_names_val:
        metric_names: set[str] = set()
        root = Path(output_dir) / experiment_name
        for metrics_path in sorted(root.glob("*/metrics.jsonl")):
            try:
                with open(metrics_path) as f:
                    first_line = f.readline()
                if first_line.strip():
                    rec = json.loads(first_line)
                    for k in rec:
                        if k not in ("step", "t"):
                            metric_names.add(k)
                    if metric_names:
                        break
            except (OSError, json.JSONDecodeError):
                continue
        data["metricNames"] = sorted(metric_names)

    return data



# ── Server-side file watcher ───────────────────────────────────────────────────


@dataclasses.dataclass
class _RunState:
    offset: int = 0    # byte offset in metrics.jsonl (last fully read position)
    mtime: float = 0.0
    # metric_name → (steps, values)
    metrics: dict[str, tuple[list[Any], list[Any]]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _ExpState:
    runs: dict[str, _RunState] = dataclasses.field(default_factory=dict)
    status: dict[str, str] = dataclasses.field(default_factory=dict)
    status_mtime: float = 0.0


_state_lock = threading.Lock()
_experiments: dict[str, _ExpState] = {}
_sub_lock = threading.Lock()
_subscribers: dict[str, list["queue.Queue[bytes | None]"]] = {}
_poll_interval: float = 1.0  # seconds; updated via /poll-interval endpoint


def _broadcast(exp_name: str, event_type: str, data: dict[str, Any]) -> None:
    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()
    with _sub_lock:
        subs = list(_subscribers.get(exp_name, []))
    dead = []
    for q in subs:
        try:
            q.put_nowait(msg)
        except queue.Full:
            dead.append(q)
    if dead:
        with _sub_lock:
            _subscribers[exp_name] = [q for q in _subscribers.get(exp_name, []) if q not in dead]


def _broadcast_experiments(output_dir: str) -> None:
    """Tell all connected clients about the current experiment list."""
    exps = list_experiments(output_dir)
    msg = f"event: experiments\ndata: {json.dumps({'experiments': exps})}\n\n".encode()
    with _sub_lock:
        all_queues = [q for subs in _subscribers.values() for q in subs]
    for q in all_queues:
        try:
            q.put_nowait(msg)
        except queue.Full:
            pass


def _check_exp_status(exp_name: str, exp_dir: Path) -> None:
    status_path = exp_dir / "sweep_status.json"
    try:
        mtime = status_path.stat().st_mtime
    except OSError:
        return
    with _state_lock:
        state = _experiments.get(exp_name)
        if state is None or mtime <= state.status_mtime:
            return
        state.status_mtime = mtime
    try:
        with open(status_path) as f:
            data: dict[str, Any] = json.load(f)
        new_status = {k: v.get("status", "unknown") for k, v in data.items()}
    except (OSError, json.JSONDecodeError):
        return
    with _state_lock:
        state = _experiments.get(exp_name)
        if state is None:
            return
        changed = {k: v for k, v in new_status.items() if state.status.get(k) != v}
        state.status.update(new_status)
    for run, st in changed.items():
        _broadcast(exp_name, "status", {"run": run, "status": st})


def _check_run_metrics(
    exp_name: str, run_name: str, metrics_path: Path
) -> dict[str, dict[str, Any]] | None:
    """Read new metric lines for one run. Returns {metric: {steps, values}} or None if no new data."""
    try:
        mtime = metrics_path.stat().st_mtime
    except OSError:
        return None
    with _state_lock:
        state = _experiments.get(exp_name)
        if state is None:
            return None
        if run_name not in state.runs:
            state.runs[run_name] = _RunState()
        run_state = state.runs[run_name]
        if mtime <= run_state.mtime:
            return None
        run_state.mtime = mtime
        offset = run_state.offset
    try:
        with open(metrics_path, "rb") as f:
            f.seek(offset)
            new_bytes = f.read()
        new_offset = offset + len(new_bytes)
    except OSError:
        return None
    if not new_bytes:
        return None
    updates: dict[str, tuple[list[Any], list[Any]]] = {}
    for raw_line in new_bytes.decode(errors="replace").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            rec: dict[str, Any] = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        step = rec.get("step")
        if step is None:
            continue
        for k, v in rec.items():
            if k == "step":
                continue
            if isinstance(v, float) and math.isnan(v):
                v = None
            if k not in updates:
                updates[k] = ([], [])
            updates[k][0].append(step)
            updates[k][1].append(v)
    if not updates:
        return None
    with _state_lock:
        state = _experiments.get(exp_name)
        if state is None or run_name not in state.runs:
            return None
        run_state = state.runs[run_name]
        run_state.offset = new_offset
        for metric_name, (steps, values) in updates.items():
            if metric_name not in run_state.metrics:
                run_state.metrics[metric_name] = ([], [])
            run_state.metrics[metric_name][0].extend(steps)
            run_state.metrics[metric_name][1].extend(values)
    return {k: {"steps": s, "values": v} for k, (s, v) in updates.items()}


def _scan(output_dir: str) -> None:
    root = Path(output_dir)
    found: set[str] = set()
    for p in root.glob("*/sweep_manifest.json"):
        found.add(p.parent.name)
    for p in root.glob("*/*/run_meta.json"):
        try:
            with open(p) as f:
                meta = json.load(f)
            if exp := meta.get("experiment"):
                found.add(exp)
        except (OSError, json.JSONDecodeError):
            pass
    with _state_lock:
        new_exps = found - set(_experiments)
        for exp in new_exps:
            _experiments[exp] = _ExpState()
    if new_exps:
        _broadcast_experiments(output_dir)
    for exp_name in list(_experiments):
        exp_dir = root / exp_name
        _check_exp_status(exp_name, exp_dir)
        # Collect all run updates for this tick, then broadcast once.
        batch: dict[str, Any] = {}
        for metrics_path in sorted(exp_dir.glob("*/metrics.jsonl")):
            run_name = metrics_path.parent.name
            run_updates = _check_run_metrics(exp_name, run_name, metrics_path)
            if run_updates:
                batch[run_name] = run_updates
        if batch:
            _broadcast(exp_name, "metrics", {"runs": batch})


def _watch_loop(output_dir: str) -> None:
    while True:
        try:
            _scan(output_dir)
        except Exception:
            pass
        time.sleep(_poll_interval)


def _sse_init_snapshot(exp_name: str) -> dict[str, Any]:
    """Build the initial snapshot for a new SSE subscriber (status only; metric data loaded on demand)."""
    with _state_lock:
        state = _experiments.get(exp_name)
        if state is None:
            return {"status": {}}
        return {"status": dict(state.status)}


def _metric_data_snapshot(exp_name: str, metric_name: str) -> dict[str, Any]:
    """Return all accumulated data for one metric across all runs.

    Returns {run_name: {steps: [...], values: [...]}}.
    """
    with _state_lock:
        state = _experiments.get(exp_name)
        if state is None:
            return {}
        result: dict[str, Any] = {}
        for run_name, run_state in state.runs.items():
            if metric_name in run_state.metrics:
                steps, values = run_state.metrics[metric_name]
                result[run_name] = {"steps": list(steps), "values": list(values)}
        return result



# ── HTML / JS ─────────────────────────────────────────────────────────────────

HTML = """\
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<title>Sweep Visualizer</title>
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
       height: 100vh; background: var(--bg); color: var(--text); }

#sidebar { width: 270px; min-width: 270px; background: var(--surface);
           border-right: 1px solid var(--border); overflow-y: auto;
           padding: 14px 12px; display: flex; flex-direction: column; gap: 16px; }

#main { flex: 1; display: flex; flex-direction: column; min-width: 0; padding: 12px; gap: 8px; }

.chart-pane { flex: 1; min-height: 0; border: 1px solid var(--border);
              border-radius: 6px; background: var(--plot-bg); }

h1 { font-size: 13px; font-weight: 700; color: var(--text-strong); }
.exp-name { font-size: 10px; color: var(--text-muted); word-break: break-all; margin-top: 2px; }

@keyframes rainbow-sweep {
  0%   { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}
.exp-name.loading {
  background: linear-gradient(90deg, #f00,#ff0,#0f0,#0ff,#00f,#f0f,#f00,#ff0,#0f0);
  background-size: 200% auto;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700;
  font-size: 12px;
  animation: rainbow-sweep 1s linear infinite;
}

.section { display: flex; flex-direction: column; gap: 6px; }
.section-title { font-size: 10px; font-weight: 700; color: var(--text-dim);
                 text-transform: uppercase; letter-spacing: 0.06em; }

select { width: 100%; padding: 5px 7px; border: 1px solid var(--border-input);
         border-radius: 4px; font-size: 12px; background: var(--btn-bg); color: var(--text); }
select:focus { outline: none; border-color: #4c9be8; }

#refresh-experiments {
  flex-shrink: 0; width: 28px; height: 28px;
  background: var(--btn-bg); border: 1px solid var(--border-input);
  border-radius: 4px; font-size: 14px; cursor: pointer;
  color: var(--text); display: flex; align-items: center; justify-content: center;
}
#refresh-experiments:hover { background: var(--btn-hover); }

input[type=number] { padding: 4px 6px; border: 1px solid var(--border-input);
                      border-radius: 4px; background: var(--btn-bg); color: var(--text); }
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button { opacity: 1; }
[data-theme=dark] input[type=number] { color-scheme: dark; }

.radio-row { display: flex; flex-wrap: wrap; gap: 3px; }
.radio-row input[type=radio] { display: none; }
.radio-row label { font-size: 11px; padding: 2px 7px; border: 1px solid var(--border-input);
                   border-radius: 3px; cursor: pointer; background: var(--radio-bg);
                   white-space: nowrap; color: var(--text); }
.radio-row label:hover { background: var(--btn-hover); }
.radio-row label.checked { background: var(--radio-checked-bg); border-color: var(--radio-checked-border);
                           color: var(--radio-checked-color); font-weight: 600; }

.filter-group { display: flex; flex-direction: column; gap: 4px; }
.filter-header { display: flex; justify-content: space-between; align-items: center; }
.filter-header .axis-label { font-size: 11px; font-weight: 600; color: var(--text-label); }
.filter-btns { display: flex; gap: 3px; }
.filter-btns button { font-size: 9px; padding: 1px 5px; border: 1px solid var(--border-input);
                      background: var(--btn-bg); border-radius: 3px; cursor: pointer; color: var(--btn-text); }
.filter-btns button:hover { background: var(--btn-hover); }

.checkbox-list { display: flex; flex-direction: column; gap: 3px; padding-left: 2px; }
.check-row { display: flex; align-items: center; gap: 6px; font-size: 12px; cursor: pointer;
             padding: 1px 0; }
.check-row input[type=checkbox] { cursor: pointer; width: 13px; height: 13px; }
.color-swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
                border: 1px solid var(--swatch-border); }
.check-label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.sub-filter-group { margin-top: 4px; margin-left: 10px; padding-left: 4px;
                    border-left: 2px solid var(--border); }
.sub-filter-group .axis-label { font-size: 10px; color: var(--text-dim); }
.sub-filter-group.hidden { display: none; }

.row { display: flex; gap: 8px; align-items: center; }
.row label { font-size: 12px; white-space: nowrap; }
.row input[type=range] { flex: 1; }
#smooth-val { font-size: 11px; color: var(--text-dim); min-width: 28px; text-align: right; }

/* ── Tab bar ── */
#topbar { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
#tabs { display: flex; gap: 3px; }
.tab { font-size: 11px; padding: 3px 12px; border: 1px solid var(--border-input);
       border-radius: 4px; cursor: pointer; background: var(--btn-bg); color: var(--btn-text);
       font-family: inherit; }
.tab:hover { background: var(--btn-hover); }
.tab.active { background: var(--radio-checked-bg); border-color: var(--radio-checked-border);
              color: var(--radio-checked-color); font-weight: 600; }

#statusbar { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
#status { font-size: 11px; color: var(--text-muted); }

#theme-toggle { background: none; border: none; font-size: 16px; cursor: pointer;
                color: var(--text-dim); padding: 0; line-height: 1; flex-shrink: 0; }
#theme-toggle:hover { color: var(--text); }

#sweep-note { font-size: 11px; color: var(--text-muted); font-style: italic;
              margin-top: 4px; word-break: break-word; }
</style>
</head>
<body>
<div id="sidebar">
  <div>
    <h1>Sweep Visualizer</h1>
    <div class="exp-name loading" id="exp-name">Fetching data…</div>
    <div id="sweep-note" style="display:none"></div>
  </div>

  <div class="section">
    <div class="section-title">Experiment</div>
    <div class="row">
      <select id="experiment-sel"></select>
      <button id="refresh-experiments" title="Refresh experiment list">&#8635;</button>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Metric</div>
    <select id="metric-sel"></select>
  </div>

  <div class="section">
    <div class="section-title">Poll interval (sec)</div>
    <div class="row">
      <input type="number" id="poll-interval" min="0.5" max="300" step="0.5" value="1" style="width:60px">
    </div>
  </div>

  <!-- Curves-only controls -->
  <div class="section curves-only">
    <div class="section-title">Color by</div>
    <div id="color-by" class="radio-row"></div>
  </div>

  <div class="section curves-only">
    <div class="section-title">Smoothing (EMA)</div>
    <div class="row">
      <input type="range" id="smoothing" min="0" max="0.99" step="0.01" value="0">
      <span id="smooth-val">0.00</span>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Filters</div>
    <div id="filters"></div>
  </div>
</div>

<div id="main">
  <div id="topbar">
    <div id="tabs">
      <button class="tab active" data-tab="curves">Curves</button>
      <button class="tab" data-tab="sensitivity">Sensitivity</button>
    </div>
    <div id="statusbar">
      <div id="status"></div>
      <button id="theme-toggle" title="Switch to dark mode">☾</button>
    </div>
  </div>
  <div id="chart-curves" class="chart-pane"></div>
  <div id="chart-sensitivity" class="chart-pane" style="display:none"></div>
</div>

<script>
// ── Constants & state ─────────────────────────────────────────────────────────

const PALETTE = [
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
  "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
];

let DATA         = null;  // experiment metadata (dims, runs, metricNames, subDims)
let METRIC_CACHE = {};    // metric_name → {run_hash: {steps, values}}, loaded on demand
let colorBy      = null;  // dim name to color by (curves tab)
let metric       = "loss";
let smoothAlpha  = 0;     // EMA alpha; 0 = disabled
let filters      = {};    // dim → Set of visible values
let METRIC_NAMES = [];    // populated by buildMetricSelect

let activeTab    = "curves";
const dirtyTabs  = new Set(["curves", "sensitivity"]);

// ── SSE state ─────────────────────────────────────────────────────────────────

let LIVE_RUNS      = new Set();  // run hashes currently status=running
let _evtSource     = null;       // EventSource for SSE updates
// run_hash → line-trace index in chart-curves (populated by buildCurvesChart)
let _traceIndexMap = {};
// last EMA accumulator: {metric: {run_hash: s_prev}} for incremental smoothing
let LAST_EMA_S     = {};

// ── Tab management ────────────────────────────────────────────────────────────

function renderTab(name) {
  if (!DATA || !METRIC_CACHE[metric]) return;
  dirtyTabs.delete(name);
  if (name === "curves") buildCurvesChart();
  else if (name === "sensitivity") buildSensitivityChart();
}

function setTab(name) {
  activeTab = name;
  document.querySelectorAll(".tab").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".curves-only").forEach(el =>
    el.style.display = (name === "curves") ? "" : "none");
  // Show the active pane, hide the other.
  document.querySelectorAll(".chart-pane").forEach(el => el.style.display = "none");
  const pane = document.getElementById(`chart-${name}`);
  pane.style.display = "";
  // Only re-render if data changed since last render; otherwise just reveal.
  if (dirtyTabs.has(name)) renderTab(name);
  else Plotly.relayout(pane, {});
}

// Mark all tabs dirty and re-render the active one. Called when data changes.
function buildActiveChart() {
  if (typeof Plotly === "undefined") return;  // Plotly not yet loaded
  dirtyTabs.add("curves");
  dirtyTabs.add("sensitivity");
  renderTab(activeTab);
}

// ── Theme ─────────────────────────────────────────────────────────────────────

(function() {
  const btn  = document.getElementById("theme-toggle");
  const root = document.documentElement;

  function applyTheme(dark) {
    dark ? root.setAttribute("data-theme", "dark") : root.removeAttribute("data-theme");
    btn.textContent = dark ? "☀" : "☾";
    btn.title = dark ? "Switch to light mode" : "Switch to dark mode";
  }

  applyTheme(localStorage.getItem("theme") !== "light");

  btn.addEventListener("click", () => {
    const dark = root.getAttribute("data-theme") !== "dark";
    localStorage.setItem("theme", dark ? "dark" : "light");
    applyTheme(dark);
    if (DATA) buildActiveChart();
  });
})();

// ── Shared helpers ─────────────────────────────────────────────────────────────

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Exponential moving average; treats null/non-finite values as gaps.
function ema(vals, alpha) {
  if (alpha === 0) return vals;
  const out = [];
  let s = null;
  for (const v of vals) {
    if (v === null || !isFinite(v)) { out.push(v); continue; }
    s = s === null ? v : alpha * s + (1 - alpha) * v;
    out.push(s);
  }
  return out;
}

// Map an array of dim values to palette colors.
function colorMap(dimValues) {
  const m = {};
  dimValues.forEach((v, i) => { m[v] = PALETTE[i % PALETTE.length]; });
  return m;
}

// Interpolate blue (#1565C0) → red (#C62828).
function lerpColor(t) {
  const r = Math.round(0x15 + (0xC6 - 0x15) * t);
  const g = Math.round(0x65 + (0x28 - 0x65) * t);
  const b = Math.round(0xC0 + (0x28 - 0xC0) * t);
  return `rgb(${r},${g},${b})`;
}

// Return a run-name → color map based on each run's final value of metricName.
function metricColorsForRuns(runs, metricName) {
  const cache = METRIC_CACHE[metricName] || {};
  const finals = runs.map(r => {
    const vals = (cache[r.hash] || {}).values || [];
    for (let i = vals.length - 1; i >= 0; i--)
      if (vals[i] !== null && isFinite(vals[i])) return vals[i];
    return null;
  });
  const finite = finals.filter(v => v !== null);
  if (!finite.length) return new Map(runs.map(r => [r.name, "#888"]));
  const lo = Math.min(...finite), hi = Math.max(...finite);
  return new Map(runs.map((r, i) => {
    const v = finals[i];
    const t = v === null ? 0.5 : (hi === lo ? 0.5 : (v - lo) / (hi - lo));
    return [r.name, lerpColor(t)];
  }));
}

function orderedComboEntries(combo, skipDim) {
  return Object.keys(DATA.dims)
    .filter(k => k !== skipDim && k in combo)
    .map(k => [k, combo[k]]);
}

function comboLabel(combo, skipDim) {
  return orderedComboEntries(combo, skipDim)
    .map(([k, v]) => `${k}=${v}`)
    .join("  ");
}

function hoverText(combo) {
  return orderedComboEntries(combo, null)
    .map(([k, v]) => `<b>${k}</b>: ${v}`)
    .join("<br>");
}

function visibleRuns() {
  return DATA.runs.filter(r =>
    Object.entries(filters).every(([dim, vals]) => !(dim in r.combo) || vals.has(r.combo[dim]))
  );
}

function yRangeOf(runs) {
  const cache = METRIC_CACHE[metric] || {};
  let lo = Infinity, hi = -Infinity;
  for (const r of runs)
    for (const v of ((cache[r.hash] || {}).values || []))
      if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!isFinite(lo)) return undefined;
  const pad = Math.max((hi - lo) * 0.05, 1e-6);
  return [lo - pad, hi + pad];
}

// ── Shared helpers for analysis tabs ──────────────────────────────────────────

// Return the final logged value for a run from the currently-loaded metric.
function runScalar(runHash) {
  const cache = METRIC_CACHE[metric] || {};
  const vals = (cache[runHash]?.values || []).filter(v => v !== null && isFinite(v));
  return vals.length ? vals[vals.length - 1] : null;
}

// For each dim, compute per-value means and the overall effect size
// (max_mean - min_mean). Returns array sorted descending by effect size.
// Each element: { dim, effectSize, valueMeans: [{val, mean, scalars, count}] }
function computeDimEffects(runs) {
  return Object.entries(DATA.dims).map(([dim, values]) => {
    const valueMeans = values.map(val => {
      const matching = runs.filter(r => r.combo[dim] === val);
      const scalars = matching.map(r => runScalar(r.hash)).filter(v => v !== null);
      const mean = scalars.length ? scalars.reduce((a, b) => a + b, 0) / scalars.length : null;
      return { val, mean, scalars, count: scalars.length };
    }).filter(vm => vm.mean !== null);
    const means = valueMeans.map(vm => vm.mean);
    const effectSize = means.length >= 2 ? Math.max(...means) - Math.min(...means) : 0;
    return { dim, effectSize, valueMeans };
  }).sort((a, b) => b.effectSize - a.effectSize);
}

// ── Tab: Curves ───────────────────────────────────────────────────────────────
// The original loss-curve view. Color by any axis, smoothing, filters.

function buildCurvesChart() {
  let runs = visibleRuns();

  // When coloring by a subdim, hide runs that don't have that dim.
  const subDimInfo = DATA.subDims || {};
  if (!colorBy.startsWith("_m:") && subDimInfo[colorBy])
    runs = runs.filter(r => colorBy in r.combo);

  const isMetricColor = colorBy.startsWith("_m:");
  let getColor, getGroup, showLegendFor, legendGroupTitle;

  if (isMetricColor) {
    const colorMap_ = metricColorsForRuns(DATA.runs, colorBy.slice(3));
    getColor        = r     => colorMap_.get(r.name) || "#888";
    getGroup        = r     => r.name;
    showLegendFor   = ()    => false;
    legendGroupTitle = ()   => undefined;
  } else {
    const cmap      = colorMap(DATA.dims[colorBy] || []);
    const firstSeen = new Set();
    getColor  = r => cmap[r.combo[colorBy]] || "#888";
    getGroup  = r => String(r.combo[colorBy]);
    showLegendFor   = group => { const f = !firstSeen.has(group); if (f) firstSeen.add(group); return f; };
    legendGroupTitle = (group, isFirst) => isFirst ? { text: colorBy, font: { size: 11 } } : undefined;
  }

  const plotBg   = cssVar("--plot-bg");
  const plotGrid = cssVar("--plot-grid");
  const plotText = cssVar("--plot-text");

  const mcache = METRIC_CACHE[metric] || {};
  const lineTraces = runs.map(r => {
    const color   = getColor(r);
    const group   = getGroup(r);
    const isFirst = showLegendFor(group);
    const mdata   = mcache[r.hash] || {};
    return {
      x: mdata.steps  || [],
      y: ema(mdata.values || [], smoothAlpha),
      mode: "lines",
      line: { color, width: 1.5 },
      name: group,
      legendgroup: group,
      legendgrouptitle: legendGroupTitle(group, isFirst),
      showlegend: isFirst,
      hovertemplate: hoverText(r.combo) + "<br><b>step</b>: %{x}<br><b>" + metric + "</b>: %{y:.4f}<extra></extra>",
      customdata: [comboLabel(r.combo, colorBy)],
    };
  });

  // One dot per run at the final data point.
  const dotTraces = runs.map(r => {
    const color = getColor(r);
    const group = getGroup(r);
    const mdata = mcache[r.hash] || {};
    const steps  = mdata.steps  || [];
    const vals   = ema(mdata.values || [], smoothAlpha);
    const lastX  = steps.length  ? [steps[steps.length - 1]]  : [];
    const lastY  = vals.length   ? [vals[vals.length - 1]]    : [];
    return {
      x: lastX, y: lastY,
      mode: "markers",
      marker: { color, size: 6, line: { color: plotBg, width: 1.5 } },
      legendgroup: group,
      showlegend: false,
      hoverinfo: "skip",
    };
  });

  const traces = [...lineTraces, ...dotTraces];

  Plotly.react("chart-curves", traces, {
    margin: { t: 20, r: 20, b: 50, l: 60 },
    xaxis: { title: "step", gridcolor: plotGrid, color: plotText },
    yaxis: { title: metric, gridcolor: plotGrid, color: plotText,
             range: yRangeOf(DATA.runs), autorange: yRangeOf(DATA.runs) === undefined },
    paper_bgcolor: plotBg, plot_bgcolor: plotBg,
    legend: { groupclick: "toggleitem", font: { size: 11, color: plotText } },
    hovermode: "closest",
  }, { responsive: true, scrollZoom: true });

  // Capture trace-index map for incremental poll updates.
  _traceIndexMap = {};
  runs.forEach((r, i) => { _traceIndexMap[r.hash] = i; });

  // Resync EMA state from full cache (after a full rebuild, accumulator is reset).
  if (!LAST_EMA_S[metric]) LAST_EMA_S[metric] = {};
  for (const r of runs) {
    if (smoothAlpha === 0) { LAST_EMA_S[metric][r.hash] = null; continue; }
    const vals = (mcache[r.hash] || {}).values || [];
    let s = null;
    for (const v of vals) {
      if (v === null || !isFinite(v)) continue;
      s = s === null ? v : smoothAlpha * s + (1 - smoothAlpha) * v;
    }
    LAST_EMA_S[metric][r.hash] = s;
  }

  const liveCount = runs.filter(r => LIVE_RUNS.has(r.hash)).length;
  document.getElementById("status").textContent =
    `${runs.length} / ${DATA.runs.length} runs visible` +
    (liveCount ? `  •  ${liveCount} live` : "");
}

// ── Tab: Sensitivity ──────────────────────────────────────────────────────────
// Dumbbell dot chart: one row per dimension, dots at per-value means,
// sorted by effect size (max_mean - min_mean). Tells you which knob matters most.

function buildSensitivityChart() {
  const runs    = visibleRuns();
  const plotBg  = cssVar("--plot-bg");
  const plotText = cssVar("--plot-text");
  const plotGrid = cssVar("--plot-grid");

  const effects = computeDimEffects(runs);
  if (!effects.length) {
    Plotly.react("chart-sensitivity", [], { paper_bgcolor: plotBg, plot_bgcolor: plotBg }, { responsive: true });
    document.getElementById("status").textContent = "No data";
    return;
  }

  const traces = [];

  // Dumbbell connector lines: from min_mean to max_mean for each dim.
  effects.forEach(e => {
    if (e.valueMeans.length < 2) return;
    const xVals = e.valueMeans.map(vm => vm.mean);
    const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
    traces.push({
      type: "scatter",
      mode: "lines",
      x: [xMin, xMax],
      y: [e.dim, e.dim],
      line: { color: cssVar("--text-dim"), width: 2 },
      showlegend: false,
      hoverinfo: "skip",
    });
  });

  // One scatter trace per unique value label for consistent cross-dim coloring.
  const allLabels = [...new Set(
    effects.flatMap(e => e.valueMeans.map(vm => String(vm.val)))
  )];
  const labelColorMap = Object.fromEntries(allLabels.map((l, i) => [l, PALETTE[i % PALETTE.length]]));

  allLabels.forEach(label => {
    const xs = [], ys = [], customdata = [];
    effects.forEach(e => {
      const vm = e.valueMeans.find(v => String(v.val) === label);
      if (!vm) return;
      xs.push(vm.mean);
      ys.push(e.dim);
      customdata.push({
        dim: e.dim, val: label, count: vm.count,
        effectSize: e.effectSize.toFixed(4),
      });
    });
    if (!xs.length) return;
    traces.push({
      type: "scatter",
      mode: "markers",
      x: xs,
      y: ys,
      name: label,
      marker: {
        size: 12,
        color: labelColorMap[label],
        line: { color: plotBg, width: 2 },
      },
      customdata,
      hovertemplate:
        "<b>%{customdata.dim} = %{customdata.val}</b><br>" +
        `Mean ${metric}: %{x:.4f}<br>` +
        "n=%{customdata.count}<br>" +
        "Effect size: %{customdata.effectSize}<extra></extra>",
    });
  });

  // Effect size text annotations on the right.
  const allMeans = effects.flatMap(e => e.valueMeans.map(vm => vm.mean));
  const xMax = allMeans.length ? Math.max(...allMeans) : 1;
  const xMin = allMeans.length ? Math.min(...allMeans) : 0;
  const xPad = Math.max((xMax - xMin) * 0.22, 1e-6);

  const annotations = effects.map(e => ({
    x: xMax + xPad * 0.05,
    y: e.axis,
    xanchor: "left",
    yanchor: "middle",
    text: `Δ${e.effectSize.toFixed(4)}`,
    showarrow: false,
    font: { size: 10, color: plotText },
  }));

  Plotly.react("chart-sensitivity", traces, {
    margin: { t: 50, r: 20, b: 60, l: 110 },
    xaxis: {
      title: `Mean ${metric} (final value)`,
      gridcolor: plotGrid, color: plotText,
      range: [xMin - xPad * 0.1, xMax + xPad],
    },
    yaxis: {
      autorange: "reversed",   // top = highest effect size
      color: plotText,
      tickfont: { size: 11 },
    },
    paper_bgcolor: plotBg, plot_bgcolor: plotBg,
    legend: { font: { size: 10, color: plotText } },
    annotations,
    title: {
      text: "Dimension sensitivity — ranked by effect size (top = most impact)",
      font: { color: plotText, size: 13 }, x: 0.5,
    },
    hovermode: "closest",
  }, { responsive: true });

  document.getElementById("status").textContent =
    `${runs.length} runs · ${effects.length} dimensions`;
}

// ── Sidebar builders ──────────────────────────────────────────────────────────

function buildFilters() {
  const subDimInfo = DATA.subDims || {};
  const subDimSet  = new Set(Object.keys(subDimInfo));

  // childrenOf["parentDim:parentValue"] = [childDim, ...]
  const childrenOf = {};
  for (const [dim, info] of Object.entries(subDimInfo)) {
    const key = `${info.parentDim}:${info.parentValue}`;
    (childrenOf[key] = childrenOf[key] || []).push(dim);
  }

  const container = document.getElementById("filters");
  container.innerHTML = "";

  function makeFilterGroup(dim, values, extraClass) {
    const isColorDim = !colorBy.startsWith("_m:") && dim === colorBy;
    const cmap_      = isColorDim ? colorMap(values) : {};

    const group = document.createElement("div");
    group.className = "filter-group" + (extraClass ? " " + extraClass : "");

    const header = document.createElement("div");
    header.className = "filter-header";
    header.innerHTML = `<span class="axis-label">${dim}</span>`;
    const btns = document.createElement("div");
    btns.className = "filter-btns";
    ["all", "none"].forEach(action => {
      const b = document.createElement("button");
      b.textContent = action;
      b.onclick = () => {
        filters[dim] = action === "all" ? new Set(values) : new Set();
        group.querySelectorAll(`input[data-dim="${CSS.escape(dim)}"]`)
             .forEach(cb => { cb.checked = action === "all"; });
        buildActiveChart();
      };
      btns.appendChild(b);
    });
    header.appendChild(btns);
    group.appendChild(header);

    const list = document.createElement("div");
    list.className = "checkbox-list";
    for (const val of values) {
      const row = document.createElement("label");
      row.className = "check-row";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.dataset.dim = dim;
      cb.checked = filters[dim]?.has(val) ?? true;
      cb.addEventListener("change", () => {
        if (cb.checked) filters[dim].add(val); else filters[dim].delete(val);
        buildActiveChart();
      });
      row.appendChild(cb);

      if (isColorDim) {
        const swatch = document.createElement("div");
        swatch.className = "color-swatch";
        swatch.style.background = cmap_[val] || "#ccc";
        row.appendChild(swatch);
      }

      const lbl = document.createElement("span");
      lbl.className = "check-label";
      lbl.textContent = val;
      row.appendChild(lbl);
      list.appendChild(row);

      // Nest any child dims under this value, hidden when unchecked.
      for (const childDim of (childrenOf[`${dim}:${val}`] || [])) {
        const childGroup = makeFilterGroup(childDim, DATA.dims[childDim] || [], "sub-filter-group");
        if (!cb.checked) childGroup.classList.add("hidden");
        cb.addEventListener("change", () => childGroup.classList.toggle("hidden", !cb.checked));
        list.appendChild(childGroup);
      }
    }

    group.appendChild(list);
    return group;
  }

  for (const [dim, values] of Object.entries(DATA.dims)) {
    if (subDimSet.has(dim)) continue;  // rendered inline under parent
    container.appendChild(makeFilterGroup(dim, values, null));
  }
}

function buildColorBySelect() {
  const container  = document.getElementById("color-by");
  const subDimInfo = DATA.subDims || {};
  container.innerHTML = "";

  for (const dim of Object.keys(DATA.dims)) {
    const id    = "cbr-" + dim;
    const input = document.createElement("input");
    input.type = "radio"; input.name = "color-by"; input.id = id; input.value = dim;

    const lbl = document.createElement("label");
    lbl.htmlFor = id;
    const info = subDimInfo[dim];
    lbl.textContent = info ? `${dim} (${info.parentValue})` : dim;
    if (dim === colorBy) lbl.classList.add("checked");

    input.addEventListener("change", () => {
      container.querySelectorAll("label").forEach(l => l.classList.remove("checked"));
      lbl.classList.add("checked");
      colorBy = dim;
      buildFilters();
      buildActiveChart();
    });

    container.appendChild(input);
    container.appendChild(lbl);
  }
}

function buildMetricSelect() {
  METRIC_NAMES = DATA.metricNames;

  const sel = document.getElementById("metric-sel");
  sel.innerHTML = "";
  for (const m of METRIC_NAMES) {
    const opt = document.createElement("option");
    opt.value = m; opt.textContent = m;
    if (m === metric) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.onchange = () => loadMetric(sel.value);
}

// ── Init & loading ────────────────────────────────────────────────────────────

// Tracks metrics whose initial HTTP fetch is in-flight to avoid duplicate
// population from concurrent SSE metric events.
const _metricFetching = new Set();

function loadMetric(name) {
  metric = name;
  if (METRIC_CACHE[name] !== undefined) {
    buildActiveChart();
    return;
  }
  if (_metricFetching.has(name)) return;
  const expName = document.getElementById("experiment-sel").value;
  if (!expName) return;
  _metricFetching.add(name);
  fetch(`/metric_data.json?experiment=${encodeURIComponent(expName)}&metric=${encodeURIComponent(name)}`)
    .then(r => r.json())
    .then(data => {
      _metricFetching.delete(name);
      // Merge: SSE may have added live updates to METRIC_CACHE[name] while we
      // were fetching; keep whichever is more complete per run.
      if (!METRIC_CACHE[name]) {
        METRIC_CACHE[name] = data;
      } else {
        // SSE already created entries; historical fetch wins for any run not
        // already in cache (SSE events are incremental from live offset).
        for (const [run, runData] of Object.entries(data)) {
          if (!METRIC_CACHE[name][run]) {
            METRIC_CACHE[name][run] = runData;
          }
        }
      }
      if (metric === name) buildActiveChart();
    })
    .catch(() => { _metricFetching.delete(name); });
}

function init(data) {
  DATA         = data;
  METRIC_CACHE = {};
  colorBy      = Object.keys(DATA.dims)[0];

  const noteEl = document.getElementById("sweep-note");
  if (DATA.note) {
    noteEl.textContent = DATA.note;
    noteEl.style.display = "";
  } else {
    noteEl.style.display = "none";
  }

  filters = Object.fromEntries(
    Object.entries(DATA.dims).map(([dim, vals]) => [dim, new Set(vals)])
  );

  // Wire up one-time event listeners.
  if (!init._wired) {
    init._wired = true;

    // Smoothing slider (curves-only).
    document.getElementById("smoothing").addEventListener("input", e => {
      smoothAlpha = parseFloat(e.target.value);
      document.getElementById("smooth-val").textContent = smoothAlpha.toFixed(2);
      if (activeTab === "curves") buildCurvesChart();
    });

    // Tab buttons.
    document.querySelectorAll(".tab").forEach(btn => {
      btn.addEventListener("click", () => setTab(btn.dataset.tab));
    });

  }

  metric = DATA.metricNames.includes(metric) ? metric
    : (DATA.metricNames.find(m => m.includes("loss")) || DATA.metricNames[0]);

  buildMetricSelect();
  buildColorBySelect();
  buildFilters();

  // Re-apply tab visibility in case experiment was switched.
  setTab(activeTab);

  loadMetric(metric);
}

// ── SSE connection ────────────────────────────────────────────────────────────

function _applyEmaIncremental(newValues, lastS) {
  // Apply EMA to newValues using lastS as the seed accumulator.
  // Returns [smoothedValues, newS].
  if (smoothAlpha === 0) return [newValues, null];
  const out = [];
  let s = lastS;
  for (const v of newValues) {
    if (v === null || !isFinite(v)) { out.push(v); continue; }
    s = s === null ? v : smoothAlpha * s + (1 - smoothAlpha) * v;
    out.push(s);
  }
  return [out, s];
}

function connectSSE(expName) {
  if (_evtSource) { _evtSource.close(); _evtSource = null; }
  LIVE_RUNS  = new Set();
  LAST_EMA_S = {};
  _metricFetching.clear();

  _evtSource = new EventSource(`/events?experiment=${encodeURIComponent(expName)}`);

  _evtSource.addEventListener("init", e => {
    const msg = JSON.parse(e.data);
    LIVE_RUNS = new Set(Object.entries(msg.status)
      .filter(([, s]) => s === "running").map(([n]) => n));
    // Metric data is loaded on demand via /metric_data.json — clear cache so
    // loadMetric fetches fresh data for the current experiment.
    METRIC_CACHE = {};
    const expNameEl = document.getElementById("exp-name");
    expNameEl.classList.remove("loading");
    expNameEl.textContent = DATA ? DATA.experiment : expName;
    // Load initial metric data.
    loadMetric(metric);
  });

  _evtSource.addEventListener("metrics", e => {
    const msg = JSON.parse(e.data);
    // msg.runs: {run_name: {metric_name: {steps, values}}}
    const incrementalUpdates = [];
    let gotNewData = false;

    for (const [run, perRunUpdates] of Object.entries(msg.runs)) {
      for (const [mname, {steps, values}] of Object.entries(perRunUpdates)) {
        if (!METRIC_CACHE[mname]) METRIC_CACHE[mname] = {};
        if (!METRIC_CACHE[mname][run]) METRIC_CACHE[mname][run] = {steps: [], values: []};
        METRIC_CACHE[mname][run].steps.push(...steps);
        METRIC_CACHE[mname][run].values.push(...values);
        gotNewData = true;

        if (mname === metric) {
          if (!LAST_EMA_S[metric]) LAST_EMA_S[metric] = {};
          const [newSmoothed, newS] = _applyEmaIncremental(values, LAST_EMA_S[metric][run] ?? null);
          LAST_EMA_S[metric][run] = newS;
          const lineIdx = _traceIndexMap[run];
          if (lineIdx !== undefined) {
            const n = Object.keys(_traceIndexMap).length;
            const dotIdx = n + lineIdx;
            const lastStep = steps[steps.length - 1];
            const lastY    = newSmoothed[newSmoothed.length - 1];
            incrementalUpdates.push({lineIdx, dotIdx, newSteps: steps, newSmoothed, lastStep, lastY});
          }
        }
      }
    }

    if (!gotNewData) return;
    if (activeTab === "curves" && !dirtyTabs.has("curves") && smoothAlpha === 0) {
      // Batch all extendTraces/restyle calls: collect indices and data together
      // to minimize Plotly re-renders.
      const lineIdxs = [], lineXs = [], lineYs = [];
      const dotIdxs  = [], dotXs  = [], dotYs  = [];
      for (const u of incrementalUpdates) {
        lineIdxs.push(u.lineIdx); lineXs.push(u.newSteps); lineYs.push(u.newSmoothed);
        dotIdxs.push(u.dotIdx);   dotXs.push([u.lastStep]); dotYs.push([u.lastY]);
      }
      if (lineIdxs.length) {
        Plotly.extendTraces("chart-curves", {x: lineXs, y: lineYs}, lineIdxs);
        Plotly.restyle("chart-curves",      {x: dotXs,  y: dotYs},  dotIdxs);
      }
      const vis = Object.keys(_traceIndexMap).length;
      document.getElementById("status").textContent =
        `${vis} / ${DATA.runs.length} runs visible` +
        (LIVE_RUNS.size ? `  •  ${LIVE_RUNS.size} live` : "");
    } else {
      buildActiveChart();
    }
  });

  _evtSource.addEventListener("status", e => {
    const msg = JSON.parse(e.data);
    if (msg.status === "running") LIVE_RUNS.add(msg.run);
    else LIVE_RUNS.delete(msg.run);
  });

  _evtSource.addEventListener("experiments", e => {
    const msg = JSON.parse(e.data);
    _updateExperimentList(msg.experiments);
  });
}

function _updateExperimentList(experiments) {
  const sel = document.getElementById("experiment-sel");
  const current = sel.value;
  sel.innerHTML = "";
  for (const name of experiments) {
    const opt = document.createElement("option");
    opt.value = name; opt.textContent = name;
    if (name === current) opt.selected = true;
    sel.appendChild(opt);
  }
}

function loadExperiment(name) {
  const expNameEl = document.getElementById("exp-name");
  expNameEl.classList.add("loading");
  expNameEl.textContent = "Fetching data…";
  if (_evtSource) { _evtSource.close(); _evtSource = null; }

  fetch(`/manifest.json?name=${encodeURIComponent(name)}`)
    .then(r => {
      if (r.status === 404) return null;
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then(manifest => {
      if (manifest !== null) {
        metric = "loss";
        init(manifest);
      } else {
        return fetch(`/data.json?name=${encodeURIComponent(name)}`)
          .then(r => r.json())
          .then(data => { metric = "loss"; init(data); });
      }
    })
    .then(() => connectSSE(name))
    .catch(e => {
      expNameEl.classList.remove("loading");
      expNameEl.textContent = "Error";
      document.getElementById("status").textContent = "Error loading data: " + e;
    });
}

function refreshExperiments() {
  fetch("/experiments")
    .then(r => r.json())
    .then(({experiments}) => _updateExperimentList(experiments))
    .catch(e => {
      document.getElementById("status").textContent = "Error refreshing: " + e;
    });
}

// Bootstrap: fetch config, then experiment list, then load the default.
fetch("/config.json").then(r => r.json()).then(cfg => {
  if (cfg.poll_interval) document.getElementById("poll-interval").value = cfg.poll_interval;
}).catch(() => {});

document.getElementById("poll-interval").addEventListener("change", e => {
  const val = parseFloat(e.target.value);
  if (val >= 0.5 && val <= 300) {
    fetch("/poll-interval", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({seconds: val}),
    }).catch(() => {});
  }
});

// Bootstrap: fetch experiment list then load the default.
fetch("/experiments")
  .then(r => r.json())
  .then(({ experiments, default: def }) => {
    const sel = document.getElementById("experiment-sel");
    for (const name of experiments) {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      if (name === def) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.onchange = () => loadExperiment(sel.value);
    document.getElementById("refresh-experiments").onclick = refreshExperiments;
    loadExperiment(def);
  })
  .catch(e => {
    const el = document.getElementById("exp-name");
    el.classList.remove("loading");
    el.textContent = "Error";
    document.getElementById("status").textContent = "Error fetching experiments: " + e;
  });
</script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" onload="if(DATA)buildActiveChart()"></script>
</body>
</html>
"""


# ── HTTP server ────────────────────────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    _default: str | None = None
    exp_source: str = ""

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs     = urllib.parse.parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", HTML.encode())

        elif parsed.path == "/experiments":
            names = list_experiments(self.exp_source)
            body  = json.dumps({"experiments": names, "default": self._default}).encode()
            self._send(200, "application/json", body)

        elif parsed.path == "/data.json":
            name = qs.get("name", [None])[0]
            if not name:
                self.send_response(400); self.end_headers(); return
            data = load_experiment_meta(name, self.exp_source)
            self._send(200, "application/json", json.dumps(data).encode())

        elif parsed.path == "/manifest.json":
            name = qs.get("name", [None])[0]
            if not name:
                self.send_response(400); self.end_headers(); return
            manifest_data = load_manifest(name, self.exp_source)
            if manifest_data is None:
                self.send_response(404); self.end_headers(); return
            body = json.dumps(manifest_data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/metric_data.json":
            name   = qs.get("experiment", [None])[0]
            metric = qs.get("metric",     [None])[0]
            if not name or not metric:
                self.send_response(400); self.end_headers(); return
            data = _metric_data_snapshot(name, metric)
            self._send(200, "application/json", json.dumps(data).encode())

        elif parsed.path == "/config.json":
            self._send(200, "application/json",
                       json.dumps({"poll_interval": _poll_interval}).encode())

        elif parsed.path == "/events":
            name = qs.get("experiment", [None])[0]
            if not name:
                self.send_response(400); self.end_headers(); return
            self._serve_sse(name)

        elif parsed.path == "/favicon.png":
            favicon_path = Path(__file__).parent / "static" / "favicon.png"
            if favicon_path.exists():
                self._send(200, "image/png", favicon_path.read_bytes())
            else:
                self.send_response(404); self.end_headers()

        else:
            self.send_response(404); self.end_headers()

    def _serve_sse(self, exp_name: str) -> None:
        q: "queue.Queue[bytes | None]" = queue.Queue(maxsize=200)
        with _sub_lock:
            _subscribers.setdefault(exp_name, []).append(q)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            snapshot = _sse_init_snapshot(exp_name)
            self.wfile.write(
                f"event: init\ndata: {json.dumps(snapshot)}\n\n".encode()
            )
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
            with _sub_lock:
                subs = _subscribers.get(exp_name, [])
                if q in subs:
                    subs.remove(q)

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        global _poll_interval
        if self.path == "/poll-interval":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length))
                val = float(body["seconds"])
                if 0.5 <= val <= 300:
                    _poll_interval = val
                self._send(200, "application/json", b"{}")
            except (KeyError, ValueError, json.JSONDecodeError):
                self.send_response(400); self.end_headers()
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # suppress request logs


def main() -> None:
    try:
        parser = argparse.ArgumentParser(description=__doc__,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("experiment", nargs="?",
                            help="Experiment name to open (default: newest)")
        parser.add_argument("--dir", default=None, metavar="PATH",
                            help="Path to sweeps directory (default: ./outputs/sweeps)")
        parser.add_argument("--port", type=int, default=43801)
        parser.add_argument("--poll-interval", type=float, default=1.0, metavar="SECONDS",
                            help="File-watcher poll interval in seconds (default: 1)")
        parser.add_argument("--open-browser", action="store_true",
                            help="Open the browser automatically")
        parser.add_argument(
            "--version", action="version",
            version=f"%(prog)s {importlib.metadata.version('mlsweep')}")
        args = parser.parse_args()

        global _poll_interval
        output_dir = args.dir or os.path.join(os.getcwd(), "outputs", "sweeps")
        Handler.exp_source = output_dir
        _poll_interval = args.poll_interval

        experiments = list_experiments(output_dir)
        if not experiments:
            sys.exit(
                f"No experiments found in: {output_dir}\n"
                "  Run a sweep with mlsweep_run first."
            )

        default_exp = args.experiment or experiments[0]
        if default_exp not in experiments:
            sys.exit(f"Experiment not found: {default_exp}")

        Handler._default = default_exp

        # Initial scan so SSE init events have data on first connect.
        _scan(output_dir)

        # Background watcher: polls files every 1s and pushes SSE events.
        watcher = threading.Thread(target=_watch_loop, args=(output_dir,), daemon=True)
        watcher.start()

        url = f"http://localhost:{args.port}"
        print(f"Sweep visualizer")
        print(f"  Source:  {output_dir}")
        print(f"  Browser: {url}  (Ctrl+C to stop)")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                public_ip = s.getsockname()[0]
            if public_ip != "127.0.0.1":
                print(f"           http://{public_ip}:{args.port}")
        except OSError:
            pass

        if args.open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()

        ThreadingHTTPServer(("", args.port), Handler).serve_forever()


    except KeyboardInterrupt:
        sys.exit(130)

if __name__ == "__main__":
    main()
