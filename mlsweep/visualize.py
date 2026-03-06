#!/usr/bin/env python3

"""
Sweep experiment visualizer.

Loads training metrics from JSONL files (or mlsweep_server) and serves an
interactive browser UI for exploring loss curves across experimental axes.

Usage:
    mlsweep_viz                                                     # newest experiment
    mlsweep_viz debug_smoke_...                                     # specific experiment
    mlsweep_viz --open-browser
    mlsweep_viz --port 43801
    mlsweep_viz --server http://host:53800                          # remote server
    mlsweep_viz --dir ./outputs/sweeps                              # local files

Environment variables:
    EXP_SERVER      Fallback for --server (http://host:port for mlsweep_server)
    MLSWEEP_TOKEN   Fallback for --token (bearer auth token for mlsweep_server)
"""

import argparse
import base64
import importlib.metadata
from typing import Any
import json
import math
import os
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mlsweep._utils import _detect_sub_axes, _parse_tag_value, _val_sort_key

# Default exp source: check EXP_SERVER env var, fall back to local sweeps dir.

# ── Data loading (dual-mode: HTTP server OR direct file reads) ─────────────────


def _http_get_json(base_url: str, path: str) -> dict:
    """GET JSON from an HTTP server. Returns parsed dict."""
    url = base_url.rstrip("/") + path
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


def list_experiments(source: str) -> list[str]:
    """Return all experiment names, sorted newest-first.

    source: http://host:port  OR  local/path/to/sweeps
    """
    if source.startswith("http"):
        return _http_get_json(source, "/experiments")["experiments"]  # type: ignore[no-any-return]

    # File mode: walk subdirs, find experiment names from run_meta.json files
    root = Path(source)
    latest: dict[str, float] = {}
    for meta_path in root.glob("*/*/run_meta.json"):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            exp = meta.get("experiment")
            if exp:
                t = meta.get("start_time", 0.0)
                if exp not in latest or t > latest[exp]:
                    latest[exp] = t
        except (OSError, json.JSONDecodeError):
            pass
    return sorted(latest, key=lambda e: latest[e], reverse=True)


def load_experiment_meta(experiment_name: str, source: str) -> dict:
    """Return axes, run combos, and metric names — no metric data loaded.

    Returns {"experiment", "axes", "runs", "metricNames", "subAxes"}.
    Each run contains {"name", "hash", "combo"} where "hash" == run_name.
    """
    if source.startswith("http"):
        path = f"/data.json?name={urllib.parse.quote(experiment_name)}"
        return _http_get_json(source, path)

    # File mode: walk source/experiment_name/*/run_meta.json
    root = Path(source) / experiment_name
    axis_values: dict[str, set] = {}
    runs: list[dict[str, Any]] = []
    metric_names: set[str] = set()

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
            axis_values.setdefault(k, set()).add(v)

        # Use run_name as "hash" so JS METRIC_CACHE can key into it
        runs.append({"name": run_name, "hash": run_name, "combo": combo})

        # Discover metric names from first line of metrics.jsonl (fast)
        metrics_path = meta_path.parent / "metrics.jsonl"
        try:
            with open(metrics_path) as f:
                first_line = f.readline()
            if first_line.strip():
                rec = json.loads(first_line)
                for k in rec:
                    if k not in ("step", "t"):
                        metric_names.add(k)
        except (OSError, json.JSONDecodeError):
            pass

    axes = {k: sorted(vs, key=_val_sort_key) for k, vs in axis_values.items()}
    sub_axes = _detect_sub_axes(runs, axes)

    return {"experiment": experiment_name, "axes": axes, "runs": runs,
            "metricNames": sorted(metric_names), "subAxes": sub_axes}


def load_manifest(experiment_name: str, source: str) -> dict | None:
    """Return the sweep manifest dict, or None if not found."""
    if source.startswith("http"):
        try:
            path = f"/manifest.json?name={urllib.parse.quote(experiment_name)}"
            return _http_get_json(source, path)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        except (urllib.error.URLError, OSError):
            return None
    p = Path(source) / experiment_name / "sweep_manifest.json"
    try:
        with open(p) as f:
            data: dict = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    metric_names_val = data.get("metricNames")
    if not metric_names_val:
        metric_names = set()
        root = Path(source) / experiment_name
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


def load_metric_since(experiment_name: str, metric_name: str, since_step: int, source: str) -> dict[str, dict[str, list[Any]]]:
    """Return {run_name: {steps, values}} for steps strictly greater than since_step."""
    if source.startswith("http"):
        path = (f"/metric_since.json?name={urllib.parse.quote(experiment_name)}"
                f"&metric={urllib.parse.quote(metric_name)}"
                f"&since_step={int(since_step)}")
        return _http_get_json(source, path)
    root = Path(source) / experiment_name
    result = {}
    for metrics_path in sorted(root.glob("*/metrics.jsonl")):
        run_name = metrics_path.parent.name
        steps, values = [], []
        try:
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    step = rec.get("step")
                    if step is None or step <= since_step:
                        continue
                    if metric_name in rec:
                        steps.append(step)
                        v = rec[metric_name]
                        values.append(None if (isinstance(v, float) and math.isnan(v)) else v)
        except OSError:
            continue
        if steps:
            result[run_name] = {"steps": steps, "values": values}
    return result


def load_run_status(experiment_name: str, source: str) -> dict[str, str]:
    """Return {run_name: status_string} for all known runs."""
    if source.startswith("http"):
        path = f"/run_status.json?name={urllib.parse.quote(experiment_name)}"
        return _http_get_json(source, path)
    root = Path(source) / experiment_name
    result = {}
    for meta_path in sorted(root.glob("*/run_meta.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        run_name = meta.get("run_name") or meta_path.parent.name
        result[run_name] = meta.get("status", "unknown")
    return result


def load_metric_data(experiment_name: str, metric_name: str, source: str) -> dict[str, dict[str, list[Any]]]:
    """Load one metric's values for all runs in an experiment.

    Returns {run_name: {"steps": [...], "values": [...]}} — keyed by run_name
    (same as "hash" in load_experiment_meta, so JS METRIC_CACHE works correctly).
    """
    if source.startswith("http"):
        path = (f"/metric.json?name={urllib.parse.quote(experiment_name)}"
                f"&metric={urllib.parse.quote(metric_name)}")
        return _http_get_json(source, path)

    # File mode: walk source/experiment_name/*/metrics.jsonl
    root = Path(source) / experiment_name
    result = {}

    for metrics_path in sorted(root.glob("*/metrics.jsonl")):
        run_name = metrics_path.parent.name
        steps = []
        values = []
        try:
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # skip truncated last line on live runs
                    if metric_name in rec:
                        steps.append(rec["step"])
                        v = rec[metric_name]
                        values.append(None if (isinstance(v, float) and math.isnan(v)) else v)
        except OSError:
            continue
        if steps:
            result[run_name] = {"steps": steps, "values": values}

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
      <input type="number" id="poll-interval" min="0.5" max="300" step="0.5" value="2" style="width:60px">
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

let DATA         = null;  // experiment metadata (axes, runs, metricNames, subAxes)
let METRIC_CACHE = {};    // metric_name → {run_hash: {steps, values}}, loaded on demand
let colorBy      = null;  // axis name to color by (curves tab)
let metric       = "loss";
let smoothAlpha  = 0;     // EMA alpha; 0 = disabled
let filters      = {};    // axis → Set of visible values
let METRIC_NAMES = [];    // populated by buildMetricSelect

let activeTab    = "curves";
const dirtyTabs  = new Set(["curves", "sensitivity"]);

// ── Live-poll state ────────────────────────────────────────────────────────────

const CONFIG      = { poll_interval: 2 };   // seconds between poll ticks
let LIVE_RUNS     = new Set();               // run hashes currently status=running
let LAST_STEP     = {};                      // run_hash → last known step (int)
let _pollActive   = false;                   // re-entrancy guard
let _pollTimer    = null;                    // setInterval handle
// run_hash → line-trace index in chart-curves (populated by buildCurvesChart)
let _traceIndexMap = {};
// last EMA accumulator: {metric: {run_hash: s_prev}} for incremental smoothing
const LAST_EMA_S  = {};

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

// Map an array of axis values to palette colors.
function colorMap(axisValues) {
  const m = {};
  axisValues.forEach((v, i) => { m[v] = PALETTE[i % PALETTE.length]; });
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

function orderedComboEntries(combo, skipAxis) {
  return Object.keys(DATA.axes)
    .filter(k => k !== skipAxis && k in combo)
    .map(k => [k, combo[k]]);
}

function comboLabel(combo, skipAxis) {
  return orderedComboEntries(combo, skipAxis)
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
    Object.entries(filters).every(([axis, vals]) => !(axis in r.combo) || vals.has(r.combo[axis]))
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

// For each axis, compute per-value means and the overall effect size
// (max_mean - min_mean). Returns array sorted descending by effect size.
// Each element: { axis, effectSize, valueMeans: [{val, mean, scalars, count}] }
function computeAxisEffects(runs) {
  return Object.entries(DATA.axes).map(([axis, values]) => {
    const valueMeans = values.map(val => {
      const matching = runs.filter(r => r.combo[axis] === val);
      const scalars = matching.map(r => runScalar(r.hash)).filter(v => v !== null);
      const mean = scalars.length ? scalars.reduce((a, b) => a + b, 0) / scalars.length : null;
      return { val, mean, scalars, count: scalars.length };
    }).filter(vm => vm.mean !== null);
    const means = valueMeans.map(vm => vm.mean);
    const effectSize = means.length >= 2 ? Math.max(...means) - Math.min(...means) : 0;
    return { axis, effectSize, valueMeans };
  }).sort((a, b) => b.effectSize - a.effectSize);
}

// ── Tab: Curves ───────────────────────────────────────────────────────────────
// The original loss-curve view. Color by any axis, smoothing, filters.

function buildCurvesChart() {
  let runs = visibleRuns();

  // When coloring by a sub-axis, hide runs that don't have that axis.
  const subAxisInfo = DATA.subAxes || {};
  if (!colorBy.startsWith("_m:") && subAxisInfo[colorBy])
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
    const cmap      = colorMap(DATA.axes[colorBy] || []);
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

  const effects = computeAxisEffects(runs);
  if (!effects.length) {
    Plotly.react("chart-sensitivity", [], { paper_bgcolor: plotBg, plot_bgcolor: plotBg }, { responsive: true });
    document.getElementById("status").textContent = "No data";
    return;
  }

  const traces = [];

  // Dumbbell connector lines: from min_mean to max_mean for each axis.
  effects.forEach(e => {
    if (e.valueMeans.length < 2) return;
    const xVals = e.valueMeans.map(vm => vm.mean);
    const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
    traces.push({
      type: "scatter",
      mode: "lines",
      x: [xMin, xMax],
      y: [e.axis, e.axis],
      line: { color: cssVar("--text-dim"), width: 2 },
      showlegend: false,
      hoverinfo: "skip",
    });
  });

  // One scatter trace per unique value label for consistent cross-axis coloring.
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
      ys.push(e.axis);
      customdata.push({
        axis: e.axis, val: label, count: vm.count,
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
        "<b>%{customdata.axis} = %{customdata.val}</b><br>" +
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
  const subAxisInfo = DATA.subAxes || {};
  const subAxisSet  = new Set(Object.keys(subAxisInfo));

  // childrenOf["parentAxis:parentValue"] = [childAxis, ...]
  const childrenOf = {};
  for (const [axis, info] of Object.entries(subAxisInfo)) {
    const key = `${info.parentAxis}:${info.parentValue}`;
    (childrenOf[key] = childrenOf[key] || []).push(axis);
  }

  const container = document.getElementById("filters");
  container.innerHTML = "";

  function makeFilterGroup(axis, values, extraClass) {
    const isColorAxis = !colorBy.startsWith("_m:") && axis === colorBy;
    const cmap_       = isColorAxis ? colorMap(values) : {};

    const group = document.createElement("div");
    group.className = "filter-group" + (extraClass ? " " + extraClass : "");

    const header = document.createElement("div");
    header.className = "filter-header";
    header.innerHTML = `<span class="axis-label">${axis}</span>`;
    const btns = document.createElement("div");
    btns.className = "filter-btns";
    ["all", "none"].forEach(action => {
      const b = document.createElement("button");
      b.textContent = action;
      b.onclick = () => {
        filters[axis] = action === "all" ? new Set(values) : new Set();
        group.querySelectorAll(`input[data-axis="${CSS.escape(axis)}"]`)
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
      cb.dataset.axis = axis;
      cb.checked = filters[axis]?.has(val) ?? true;
      cb.addEventListener("change", () => {
        if (cb.checked) filters[axis].add(val); else filters[axis].delete(val);
        buildActiveChart();
      });
      row.appendChild(cb);

      if (isColorAxis) {
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

      // Nest any child axes under this value, hidden when unchecked.
      for (const childAxis of (childrenOf[`${axis}:${val}`] || [])) {
        const childGroup = makeFilterGroup(childAxis, DATA.axes[childAxis] || [], "sub-filter-group");
        if (!cb.checked) childGroup.classList.add("hidden");
        cb.addEventListener("change", () => childGroup.classList.toggle("hidden", !cb.checked));
        list.appendChild(childGroup);
      }
    }

    group.appendChild(list);
    return group;
  }

  for (const [axis, values] of Object.entries(DATA.axes)) {
    if (subAxisSet.has(axis)) continue;  // rendered inline under parent
    container.appendChild(makeFilterGroup(axis, values, null));
  }
}

function buildColorBySelect() {
  const container   = document.getElementById("color-by");
  const subAxisInfo = DATA.subAxes || {};
  container.innerHTML = "";

  for (const axis of Object.keys(DATA.axes)) {
    const id    = "cbr-" + axis;
    const input = document.createElement("input");
    input.type = "radio"; input.name = "color-by"; input.id = id; input.value = axis;

    const lbl = document.createElement("label");
    lbl.htmlFor = id;
    const info = subAxisInfo[axis];
    lbl.textContent = info ? `${axis} (${info.parentValue})` : axis;
    if (axis === colorBy) lbl.classList.add("checked");

    input.addEventListener("change", () => {
      container.querySelectorAll("label").forEach(l => l.classList.remove("checked"));
      lbl.classList.add("checked");
      colorBy = axis;
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

function loadMetric(name) {
  const expNameEl = document.getElementById("exp-name");

  if (METRIC_CACHE[name]) {
    metric = name;
    buildActiveChart();
    return;
  }

  const expName = document.getElementById("experiment-sel").value;
  expNameEl.classList.add("loading");
  expNameEl.textContent = `Loading ${name}…`;

  fetch(`/metric.json?name=${encodeURIComponent(expName)}&metric=${encodeURIComponent(name)}`)
    .then(r => r.json())
    .then(data => {
      METRIC_CACHE[name] = data;
      metric = name;
      // Initialize LAST_STEP so the poll loop fetches only new data.
      for (const [runName, mdata] of Object.entries(data)) {
        const steps = (mdata || {}).steps || [];
        if (steps.length) LAST_STEP[runName] = steps[steps.length - 1];
      }
      expNameEl.classList.remove("loading");
      expNameEl.textContent = DATA.experiment;
      buildActiveChart();
    })
    .catch(e => {
      expNameEl.classList.remove("loading");
      expNameEl.textContent = DATA.experiment;
      document.getElementById("status").textContent = "Error loading metric: " + e;
    });
}

function init(data) {
  DATA         = data;
  METRIC_CACHE = {};
  colorBy      = Object.keys(DATA.axes)[0];

  const noteEl = document.getElementById("sweep-note");
  if (DATA.note) {
    noteEl.textContent = DATA.note;
    noteEl.style.display = "";
  } else {
    noteEl.style.display = "none";
  }

  filters = Object.fromEntries(
    Object.entries(DATA.axes).map(([axis, vals]) => [axis, new Set(vals)])
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

// ── Live poll ─────────────────────────────────────────────────────────────────

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

async function pollTick() {
  if (_pollActive || document.hidden || !DATA) return;
  _pollActive = true;
  const expName = document.getElementById("experiment-sel").value;
  try {
    // Step 1: fetch run statuses
    let statusMap;
    try {
      const r = await fetch(`/run_status.json?name=${encodeURIComponent(expName)}`);
      if (!r.ok) { _pollActive = false; return; }
      statusMap = await r.json();
    } catch (e) { _pollActive = false; return; }

    // Update LIVE_RUNS based on runs that exist in DATA.runs
    const knownRuns = new Set(DATA.runs.map(r => r.name));
    const statusMapFiltered = Object.fromEntries(
      Object.entries(statusMap).filter(([name]) => knownRuns.has(name))
    );
    LIVE_RUNS = new Set(
      Object.entries(statusMapFiltered)
        .filter(([, s]) => s === "running")
        .map(([n]) => n)
    );

    // Step 2: fetch incremental metric data for ALL runs (to catch newly completed)
    if (knownRuns.size === 0) { _pollActive = false; return; }

    let gotNewData = false;
    const incrementalUpdates = [];

    const fetchResults = await Promise.all(
      [...knownRuns].map(async runName => {
          const since = LAST_STEP[runName] ?? -1;
          try {
            const r = await fetch(
              `/metric_since.json?name=${encodeURIComponent(expName)}` +
              `&metric=${encodeURIComponent(metric)}` +
              `&since_step=${since}`
            );
            return [runName, await r.json()];
          } catch (e) { return [runName, null]; }
        })
      );

      // Step 3: merge new data into METRIC_CACHE
      if (!METRIC_CACHE[metric]) METRIC_CACHE[metric] = {};
      const mcache = METRIC_CACHE[metric];
      if (!LAST_EMA_S[metric]) LAST_EMA_S[metric] = {};

      for (const [runName, data] of fetchResults) {
        if (!data) continue;
        const newPiece = data[runName];
        if (!newPiece || !newPiece.steps.length) continue;
        gotNewData = true;

        if (!mcache[runName]) mcache[runName] = { steps: [], values: [] };
        mcache[runName].steps.push(...newPiece.steps);
        mcache[runName].values.push(...newPiece.values);
        LAST_STEP[runName] = newPiece.steps[newPiece.steps.length - 1];

        const [newSmoothed, newS] = _applyEmaIncremental(
          newPiece.values, LAST_EMA_S[metric][runName] ?? null);
        LAST_EMA_S[metric][runName] = newS;

        const lineIdx = _traceIndexMap[runName];
        if (lineIdx === undefined) continue;
        const n = Object.keys(_traceIndexMap).length;
        const dotIdx = n + lineIdx;
        const lastStep = newPiece.steps[newPiece.steps.length - 1];
        const lastY    = newSmoothed[newSmoothed.length - 1];
        incrementalUpdates.push({ lineIdx, dotIdx, newSteps: newPiece.steps, newSmoothed, lastStep, lastY });
      }

    // Step 4: update Plotly
    if (!gotNewData) { _pollActive = false; return; }

    if (activeTab === "curves" && !dirtyTabs.has("curves") && smoothAlpha === 0) {
      // Incremental path: safe when no EMA and chart is not stale
      for (const u of incrementalUpdates) {
        Plotly.extendTraces("chart-curves",
          { x: [u.newSteps], y: [u.newSmoothed] }, [u.lineIdx]);
        Plotly.restyle("chart-curves",
          { x: [[u.lastStep]], y: [[u.lastY]] }, [u.dotIdx]);
      }
      // Update status line
      const liveCount = incrementalUpdates.length;
      const vis = Object.keys(_traceIndexMap).length;
      document.getElementById("status").textContent =
        `${vis} / ${DATA.runs.length} runs visible` + (liveCount ? `  •  ${liveCount} live` : "");
    } else if (gotNewData) {
      buildActiveChart();
    }
  } finally {
    _pollActive = false;
  }
}

function startPollLoop() {
  if (_pollTimer !== null) { clearInterval(_pollTimer); _pollTimer = null; }
  LIVE_RUNS  = new Set();
  LAST_STEP  = {};
  _pollActive = false;
  // First tick after a short delay (let init() and loadMetric() settle first)
  setTimeout(pollTick, 2000);
  _pollTimer = setInterval(pollTick, CONFIG.poll_interval * 1000);
}

function loadExperiment(name) {
  const expNameEl = document.getElementById("exp-name");
  expNameEl.classList.add("loading");
  expNameEl.textContent = "Fetching data…";

  // Stop existing poll before switching experiments
  if (_pollTimer !== null) { clearInterval(_pollTimer); _pollTimer = null; }

  fetch(`/manifest.json?name=${encodeURIComponent(name)}`)
    .then(r => {
      console.log("manifest.json response status:", r.status);
      if (r.status === 404) return null;
      if (!r.ok) throw new Error(`manifest HTTP ${r.status}`);
      return r.json();
    })
    .then(manifest => {
      console.log("manifest loaded:", manifest);
      if (manifest !== null) {
        expNameEl.textContent = "Parsing JSON…";
        metric = "loss";
        init(manifest);
        startPollLoop();
      } else {
        return fetch(`/data.json?name=${encodeURIComponent(name)}`)
          .then(r => { expNameEl.textContent = "Parsing JSON…"; return r.json(); })
          .then(data => {
            metric = "loss";
            init(data);
            startPollLoop();
          });
      }
    })
    .catch(e => {
      expNameEl.classList.remove("loading");
      expNameEl.textContent = "Error";
      document.getElementById("status").textContent = "Error loading data: " + e;
    });
}

// Refresh experiment list (for when new experiments are added)
function refreshExperiments() {
  fetch("/experiments")
    .then(r => r.json())
    .then(({ experiments, default: def }) => {
      const sel = document.getElementById("experiment-sel");
      const current = sel.value;
      sel.innerHTML = "";
      for (const name of experiments) {
        const opt = document.createElement("option");
        opt.value = name; opt.textContent = name;
        if (name === current || ( !current && name === def)) opt.selected = true;
        sel.appendChild(opt);
      }
      if (sel.value !== current && sel.value) {
        loadExperiment(sel.value);
      }
    })
    .catch(e => {
      document.getElementById("status").textContent = "Error refreshing experiments: " + e;
    });
}

// Bootstrap: fetch config, then experiment list, then load the default.
fetch("/config.json").then(r => r.json()).then(cfg => {
  if (cfg.poll_interval) {
    CONFIG.poll_interval = cfg.poll_interval;
    document.getElementById("poll-interval").value = cfg.poll_interval;
  }
}).catch(() => {});  // ignore errors — default CONFIG is fine

document.getElementById("poll-interval").addEventListener("change", e => {
  const val = parseFloat(e.target.value);
  if (val >= 0.5 && val <= 300) {
    CONFIG.poll_interval = val;
    if (_pollTimer) {
      clearInterval(_pollTimer);
      _pollTimer = setInterval(pollTick, CONFIG.poll_interval * 1000);
    }
  }
});

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

_META_CACHE_TTL = 30.0  # seconds for live experiment discovery


class Handler(BaseHTTPRequestHandler):
    # experiment name → (timestamp, json_bytes); 30s TTL for live runs
    _meta_cache: dict[str, tuple[float, bytes]] = {}
    _default: str | None = None
    exp_source: str = ""
    poll_interval: int = 2
    token: str | None = None

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs     = urllib.parse.parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", HTML.encode())

        elif parsed.path == "/config.json":
            cfg = {"poll_interval": self.poll_interval}
            self._send(200, "application/json", json.dumps(cfg).encode())

        elif parsed.path == "/experiments":
            names = list_experiments(self.exp_source)
            body  = json.dumps({"experiments": names, "default": self._default}).encode()
            self._send(200, "application/json", body)

        elif parsed.path == "/data.json":
            name = qs.get("name", [None])[0]
            if not name:
                self.send_response(400); self.end_headers(); return
            now = time.time()
            if name in Handler._meta_cache:
                ts, cached_body = Handler._meta_cache[name]
                if now - ts < _META_CACHE_TTL:
                    self._send(200, "application/json", cached_body)
                    return
            print(f"Loading metadata: {name}", flush=True)
            data = load_experiment_meta(name, self.exp_source)
            print(f"  {len(data['runs'])} runs, {len(data['axes'])} axes, "
                  f"{len(data['metricNames'])} metrics", flush=True)
            body = json.dumps(data).encode()
            Handler._meta_cache[name] = (now, body)
            self._send(200, "application/json", body)

        elif parsed.path == "/metric.json":
            name   = qs.get("name",   [None])[0]
            metric = qs.get("metric", [None])[0]
            if not name or not metric:
                self.send_response(400); self.end_headers(); return
            # Not cached server-side: JS already caches, and data changes on live runs
            print(f"  Loading metric '{metric}' for {name}…", flush=True)
            data = load_metric_data(name, metric, self.exp_source)
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
            self.send_header("Cache-Control", "public, max-age=31536000")
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/metric_since.json":
            name   = qs.get("name",       [None])[0]
            met    = qs.get("metric",     [None])[0]
            since  = qs.get("since_step", [None])[0]
            if not name or not met:
                self.send_response(400); self.end_headers(); return
            since_step = int(since) if since is not None else -1
            data = load_metric_since(name, met, since_step, self.exp_source)
            self._send(200, "application/json", json.dumps(data).encode())

        elif parsed.path == "/run_status.json":
            name = qs.get("name", [None])[0]
            if not name:
                self.send_response(400); self.end_headers(); return
            data = load_run_status(name, self.exp_source)
            self._send(200, "application/json", json.dumps(data).encode())

        elif parsed.path == "/favicon.png":
            favicon_path = Path(__file__).parent / "static" / "favicon.png"
            if favicon_path.exists():
                self._send(200, "image/png", favicon_path.read_bytes())
            else:
                self.send_response(404)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # suppress request logs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?",
                        help="Experiment name to pre-load (default: newest)")
    parser.add_argument("--server", default=None, metavar="URL",
                        help="http://host:port for mlsweep_server "
                             "(also read from EXP_SERVER env var)")
    parser.add_argument("--dir", default=None, metavar="PATH",
                        help="Local path to sweeps directory (default: ./outputs/sweeps)")
    parser.add_argument("--port", type=int, default=43801)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--poll-interval", type=int, default=2, metavar="SECONDS",
        help="Live-update poll interval in seconds (default: 10)")
    parser.add_argument(
        "--token", default=None, metavar="SECRET",
        help="Bearer token to include in requests to mlsweep_server "
             "(also read from MLSWEEP_TOKEN env var)")
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {importlib.metadata.version('mlsweep')}")
    args = parser.parse_args()

    exp_source = (
        args.server
        or args.dir
        or os.environ.get("EXP_SERVER")
        or os.path.join(os.getcwd(), "outputs", "sweeps")
    )
    Handler.exp_source     = exp_source
    Handler.poll_interval  = args.poll_interval
    Handler.token          = args.token or os.environ.get("MLSWEEP_TOKEN")

    # Find and pre-load the default experiment so the first page load is fast.
    experiments = list_experiments(exp_source)
    if not experiments:
        sys.exit(
            f"No experiments found in: {exp_source}\n"
            "  Run a training job with the mlsweep logger, or start the mlsweep server and pass --server http://host:port."
        )

    default_exp = args.experiment or experiments[0]
    if default_exp not in experiments:
        sys.exit(f"Experiment not found: {default_exp}")

    print(f"Loading metadata: {default_exp}")
    data = load_experiment_meta(default_exp, exp_source)
    print(f"  {len(data['runs'])} runs, {len(data['axes'])} axes, "
          f"{len(data['metricNames'])} metrics")

    Handler._meta_cache[default_exp] = (time.time(), json.dumps(data).encode())
    Handler._default = default_exp

    url = f"http://localhost:{args.port}"
    print(f"  Source: {exp_source}")
    print(f"  Serving at {url}  (Ctrl+C to stop)")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            public_ip = s.getsockname()[0]
        if public_ip != "127.0.0.1":
            print(f"             http://{public_ip}:{args.port}")
    except OSError:
        pass

    if args.open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    ThreadingHTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
