#!/usr/bin/env python3
"""
mlsweep Web UI — job management, metrics visualization, and system observability.

Usage:
    mlsweep_ui                           # serve on port 43802, CWD outputs
    mlsweep_ui --port 8080
    mlsweep_ui --dir ./outputs/sweeps
    mlsweep_ui --open-browser
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, IO

from mlsweep._shared import _detect_sub_dims, _parse_tag_value, _val_sort_key


# ── Data loading (same as visualize.py) ───────────────────────────────────────


def list_experiments(output_dir: str) -> list[str]:
    root = Path(output_dir)
    latest: dict[str, float] = {}
    for manifest_path in root.glob("*/sweep_manifest.json"):
        exp = manifest_path.parent.name
        t = manifest_path.stat().st_mtime
        latest[exp] = max(latest.get(exp, 0.0), t)
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
    root = Path(output_dir) / experiment_name
    manifest_path = root / "sweep_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest: dict[str, Any] = json.load(f)
            if not manifest.get("metricNames"):
                metric_names: set[str] = set()
                for mp in sorted(root.glob("*/metrics.jsonl")):
                    try:
                        with open(mp) as f:
                            line = f.readline()
                        if line.strip():
                            for k in json.loads(line):
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
        mp = meta_path.parent / "metrics.jsonl"
        try:
            with open(mp) as f:
                fl = f.readline()
            if fl.strip():
                for k in json.loads(fl):
                    if k not in ("step", "t"):
                        old_metric_names.add(k)
        except (OSError, json.JSONDecodeError):
            pass
    dims = {k: sorted(vs, key=_val_sort_key) for k, vs in dim_values.items()}
    sub_dims = _detect_sub_dims(runs, dims)
    return {"experiment": experiment_name, "dims": dims, "runs": runs,
            "metricNames": sorted(old_metric_names), "subDims": sub_dims}


def load_manifest(experiment_name: str, output_dir: str) -> dict[str, Any] | None:
    p = Path(output_dir) / experiment_name / "sweep_manifest.json"
    try:
        with open(p) as f:
            data: dict[str, Any] = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not data.get("metricNames"):
        mnames: set[str] = set()
        root = Path(output_dir) / experiment_name
        for mp in sorted(root.glob("*/metrics.jsonl")):
            try:
                with open(mp) as f:
                    line = f.readline()
                if line.strip():
                    for k in json.loads(line):
                        if k not in ("step", "t"):
                            mnames.add(k)
                    if mnames:
                        break
            except (OSError, json.JSONDecodeError):
                continue
        data["metricNames"] = sorted(mnames)
    return data


# ── File watcher + SSE state ───────────────────────────────────────────────────


@dataclasses.dataclass
class _RunState:
    offset: int = 0
    mtime: float = 0.0
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
_poll_interval: float = 1.0


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
    with _state_lock:
        state = _experiments.get(exp_name)
        if state is None:
            return {"status": {}}
        return {"status": dict(state.status)}


def _metric_data_snapshot(exp_name: str, metric_name: str) -> dict[str, Any]:
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


# ── Job management ─────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ManagedJob:
    job_id: str
    sweep_file: str
    workers_file: str | None
    output_dir: str
    extra_args: list[str]
    note: str | None
    status: str  # "starting" | "running" | "done" | "failed" | "killed"
    started_at: float
    ended_at: float | None
    returncode: int | None
    experiment: str | None
    proc: "subprocess.Popen[str] | None"

    def to_dict(self) -> dict[str, Any]:
        elapsed: float | None = None
        if self.ended_at is not None:
            elapsed = self.ended_at - self.started_at
        elif self.status == "running":
            elapsed = time.time() - self.started_at
        return {
            "job_id": self.job_id,
            "sweep_file": self.sweep_file,
            "workers_file": self.workers_file,
            "output_dir": self.output_dir,
            "extra_args": self.extra_args,
            "note": self.note,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed": elapsed,
            "returncode": self.returncode,
            "experiment": self.experiment,
            "progress": _job_progress(self),
        }


_JOBS: dict[str, ManagedJob] = {}
_JOBS_LOCK = threading.Lock()


def _job_progress(job: ManagedJob) -> dict[str, Any]:
    if not job.experiment:
        return {}
    status_path = Path(job.output_dir) / job.experiment / "sweep_status.json"
    try:
        with open(status_path) as f:
            data: dict[str, Any] = json.load(f)
        total = len(data)
        ok = sum(1 for s in data.values() if s.get("status") == "ok")
        failed = sum(1 for s in data.values() if s.get("status") == "failed")
        running = sum(1 for s in data.values() if s.get("status") == "in-progress")
        done = ok + failed
        return {
            "total": total, "done": done, "ok": ok,
            "failed": failed, "running": running,
            "pct": round(done / max(total, 1) * 100),
        }
    except (OSError, json.JSONDecodeError):
        return {}


def _find_newest_experiment(output_dir: str, after: float) -> str | None:
    """Find the most recently created experiment in output_dir created after `after`."""
    try:
        root = Path(output_dir)
        candidates: list[tuple[float, str]] = []
        for mp in root.glob("*/sweep_manifest.json"):
            t = mp.stat().st_mtime
            if t >= after - 10:
                candidates.append((t, mp.parent.name))
        if candidates:
            return max(candidates)[1]
    except OSError:
        pass
    return None


def _monitor_jobs_loop() -> None:
    while True:
        time.sleep(2.0)
        with _JOBS_LOCK:
            for job in list(_JOBS.values()):
                if job.status in ("done", "failed", "killed"):
                    continue
                if job.proc is not None:
                    rc = job.proc.poll()
                    if rc is not None:
                        job.status = "done" if rc == 0 else "failed"
                        job.returncode = rc
                        job.ended_at = time.time()
                    elif job.status == "starting":
                        job.status = "running"
                if job.experiment is None:
                    job.experiment = _find_newest_experiment(job.output_dir, job.started_at)


def _start_job(
    sweep_file: str,
    workers_file: str | None,
    output_dir: str,
    extra_args: list[str],
    note: str | None,
) -> ManagedJob:
    job_id = uuid.uuid4().hex[:12]
    # Prefer mlsweep_run on PATH; fall back to running as module
    mlsweep_run = shutil.which("mlsweep_run") or sys.executable
    cmd: list[str] = []
    if mlsweep_run == sys.executable:
        cmd = [sys.executable, "-m", "mlsweep.run_sweep"]
    else:
        cmd = [mlsweep_run]
    cmd.append(sweep_file)
    if workers_file:
        cmd += ["--workers", workers_file]
    cmd += ["--output_dir", output_dir]
    if note:
        cmd += ["--note", note]
    if extra_args:
        cmd += extra_args

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    job = ManagedJob(
        job_id=job_id,
        sweep_file=sweep_file,
        workers_file=workers_file,
        output_dir=output_dir,
        extra_args=extra_args,
        note=note,
        status="starting",
        started_at=time.time(),
        ended_at=None,
        returncode=None,
        experiment=None,
        proc=proc,
    )
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    return job


# ── System monitoring ──────────────────────────────────────────────────────────


_gpu_stats: list[dict[str, Any]] = []
_sys_stats: dict[str, Any] = {}
_stats_lock = threading.Lock()
_cpu_prev: tuple[int, int] | None = None


def _read_gpu_stats() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                mem_used = int(parts[3])
                mem_total = int(parts[4])
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util": int(parts[2]),
                    "mem_used": mem_used,
                    "mem_total": mem_total,
                    "temp": int(parts[5]),
                    "mem_pct": round(mem_used / max(mem_total, 1) * 100),
                })
            except (ValueError, ZeroDivisionError):
                continue
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def _read_mem_stats() -> dict[str, Any]:
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem: dict[str, int] = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mem[parts[0].rstrip(":")] = int(parts[1])
                except ValueError:
                    pass
        total = mem.get("MemTotal", 0)
        available = mem.get("MemAvailable", 0)
        used = total - available
        return {
            "total_gb": round(total / 1024 / 1024, 1),
            "used_gb": round(used / 1024 / 1024, 1),
            "pct": round(used / max(total, 1) * 100),
        }
    except OSError:
        return {}


def _read_cpu_pct() -> float:
    global _cpu_prev
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        fields = list(map(int, line.split()[1:]))
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        total = sum(fields)
        if _cpu_prev is None:
            _cpu_prev = (idle, total)
            return 0.0
        prev_idle, prev_total = _cpu_prev
        d_idle = idle - prev_idle
        d_total = total - prev_total
        _cpu_prev = (idle, total)
        if d_total == 0:
            return 0.0
        return round((1 - d_idle / d_total) * 100, 1)
    except (OSError, ValueError, IndexError):
        return 0.0


def _poll_system_loop() -> None:
    global _gpu_stats, _sys_stats
    while True:
        gpus = _read_gpu_stats()
        mem = _read_mem_stats()
        cpu = _read_cpu_pct()
        try:
            hostname = os.uname().nodename
        except AttributeError:
            hostname = "unknown"
        with _stats_lock:
            _gpu_stats = gpus
            _sys_stats = {"gpus": gpus, "cpu_pct": cpu, "mem": mem, "hostname": hostname}
        time.sleep(2.0)


# ── HTML ───────────────────────────────────────────────────────────────────────

HTML = """\
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<title>mlsweep</title>
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

/* ── Sidebar ── */
#sidebar { width: 270px; min-width: 270px; background: var(--surface);
           border-right: 1px solid var(--border); overflow-y: auto;
           padding: 14px 12px; display: flex; flex-direction: column; gap: 16px; }

/* ── Main ── */
#main { flex: 1; display: flex; flex-direction: column; min-width: 0; overflow: hidden; }

/* ── Topbar ── */
#topbar { display: flex; align-items: center; justify-content: space-between;
          gap: 8px; padding: 8px 12px; border-bottom: 1px solid var(--border);
          background: var(--surface); flex-shrink: 0; }
#page-tabs { display: flex; gap: 3px; }
.page-tab { font-size: 11px; padding: 3px 12px; border: 1px solid var(--border-input);
            border-radius: 4px; cursor: pointer; background: var(--btn-bg);
            color: var(--btn-text); font-family: inherit; }
.page-tab:hover { background: var(--btn-hover); }
.page-tab.active { background: var(--radio-checked-bg); border-color: var(--radio-checked-border);
                   color: var(--radio-checked-color); font-weight: 600; }
#statusbar { display: flex; align-items: center; gap: 8px; }
#status-text { font-size: 11px; color: var(--text-muted); }
#theme-toggle { background: none; border: none; font-size: 16px; cursor: pointer;
                color: var(--text-dim); padding: 0; line-height: 1; }
#theme-toggle:hover { color: var(--text); }

/* ── Page content ── */
.page-view { flex: 1; overflow: auto; display: flex; flex-direction: column; min-height: 0; }

/* ── Shared sidebar components ── */
h1 { font-size: 13px; font-weight: 700; color: var(--text-strong); }
.exp-name { font-size: 10px; color: var(--text-muted); word-break: break-all; margin-top: 2px; }

@keyframes rainbow-sweep {
  0%   { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}
.exp-name.loading {
  background: linear-gradient(90deg, #f00,#ff0,#0f0,#0ff,#00f,#f0f,#f00,#ff0,#0f0);
  background-size: 200% auto;
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700; font-size: 12px;
  animation: rainbow-sweep 1s linear infinite;
}

.section { display: flex; flex-direction: column; gap: 6px; }
.section-title { font-size: 10px; font-weight: 700; color: var(--text-dim);
                 text-transform: uppercase; letter-spacing: 0.06em; }

select { width: 100%; padding: 5px 7px; border: 1px solid var(--border-input);
         border-radius: 4px; font-size: 12px; background: var(--btn-bg); color: var(--text); }
select:focus { outline: none; border-color: #4c9be8; }

input[type=text], input[type=number] {
  padding: 5px 7px; border: 1px solid var(--border-input);
  border-radius: 4px; font-size: 12px; background: var(--btn-bg); color: var(--text);
  font-family: inherit; width: 100%; }
input[type=text]:focus, input[type=number]:focus { outline: none; border-color: #4c9be8; }
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button { opacity: 1; }
[data-theme=dark] input[type=number] { color-scheme: dark; }

.row { display: flex; gap: 8px; align-items: center; }
.row label { font-size: 12px; white-space: nowrap; }
.row input[type=range] { flex: 1; }
#smooth-val { font-size: 11px; color: var(--text-dim); min-width: 28px; text-align: right; }

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
.check-row { display: flex; align-items: center; gap: 6px; font-size: 12px; cursor: pointer; padding: 1px 0; }
.check-row input[type=checkbox] { cursor: pointer; width: 13px; height: 13px; }
.color-swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
                border: 1px solid var(--swatch-border); }
.check-label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sub-filter-group { margin-top: 4px; margin-left: 10px; padding-left: 4px;
                    border-left: 2px solid var(--border); }
.sub-filter-group .axis-label { font-size: 10px; color: var(--text-dim); }
.sub-filter-group.hidden { display: none; }

#refresh-experiments {
  flex-shrink: 0; width: 28px; height: 28px;
  background: var(--btn-bg); border: 1px solid var(--border-input);
  border-radius: 4px; font-size: 14px; cursor: pointer;
  color: var(--text); display: flex; align-items: center; justify-content: center; }
#refresh-experiments:hover { background: var(--btn-hover); }
#sweep-note { font-size: 11px; color: var(--text-muted); font-style: italic;
              margin-top: 4px; word-break: break-word; }

/* ── Metrics chart tab bar ── */
#metric-tab-bar { display: flex; gap: 3px; padding: 8px 12px 0; flex-shrink: 0; }
.tab { font-size: 11px; padding: 3px 12px; border: 1px solid var(--border-input);
       border-radius: 4px; cursor: pointer; background: var(--btn-bg); color: var(--btn-text);
       font-family: inherit; }
.tab:hover { background: var(--btn-hover); }
.tab.active { background: var(--radio-checked-bg); border-color: var(--radio-checked-border);
              color: var(--radio-checked-color); font-weight: 600; }
.chart-pane { flex: 1; min-height: 0; border: 1px solid var(--border);
              border-radius: 6px; background: var(--plot-bg); margin: 8px 12px 12px; }

/* ── Jobs page ── */
.jobs-page-content { padding: 12px; display: flex; flex-direction: column; gap: 12px; flex: 1; }

.empty-state { text-align: center; padding: 60px 20px; color: var(--text-muted); }
.empty-state .empty-icon { font-size: 28px; margin-bottom: 10px; color: var(--border-input); }
.empty-state .empty-title { font-size: 13px; font-weight: 600; color: var(--text-dim); margin-bottom: 4px; }
.empty-state .empty-sub { font-size: 12px; }

.jobs-table-wrap { border: 1px solid var(--border); border-radius: 6px;
                   background: var(--surface); overflow: hidden; }
table.jobs-table { width: 100%; border-collapse: collapse; font-size: 12px; }
table.jobs-table th { font-size: 10px; font-weight: 700; color: var(--text-dim);
                      text-transform: uppercase; letter-spacing: 0.06em;
                      padding: 7px 10px; border-bottom: 1px solid var(--border); text-align: left;
                      background: var(--bg); white-space: nowrap; }
table.jobs-table td { padding: 8px 10px; border-bottom: 1px solid var(--border);
                      color: var(--text); vertical-align: middle; }
table.jobs-table tr:last-child td { border-bottom: none; }
table.jobs-table tr:hover td { background: var(--bg); }

.status-badge { display: inline-flex; align-items: center; gap: 4px;
                padding: 2px 7px; border-radius: 3px; font-size: 10px; font-weight: 600;
                white-space: nowrap; }
.badge-running  { background: rgba(76,155,232,0.15); color: #4c9be8; }
.badge-done     { background: rgba(44,160,44,0.15); color: #2ca02c; }
.badge-failed   { background: rgba(214,39,40,0.15); color: #d62728; }
.badge-killed   { background: rgba(127,127,127,0.15); color: #7f7f7f; }
.badge-starting { background: rgba(255,127,14,0.15); color: #ff7f0e; }

.dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-running  { background: #4c9be8; animation: pulse 1.5s ease-in-out infinite; }
.dot-done     { background: #2ca02c; }
.dot-failed   { background: #d62728; }
.dot-killed   { background: #7f7f7f; }
.dot-starting { background: #ff7f0e; animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

.prog-wrap { display: flex; align-items: center; gap: 6px; }
.prog-bar { width: 70px; height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; }
.prog-fill { height: 100%; border-radius: 3px; background: #4c9be8; transition: width .4s; }
.prog-fill.all-ok { background: #2ca02c; }
.prog-fill.has-fail { background: #d62728; }
.prog-txt { font-size: 11px; color: var(--text-dim); white-space: nowrap; }

.btn-sm { font-size: 10px; padding: 2px 8px; border: 1px solid var(--border-input);
          border-radius: 3px; cursor: pointer; background: var(--btn-bg);
          color: var(--btn-text); font-family: inherit; white-space: nowrap; }
.btn-sm:hover { background: var(--btn-hover); }
.btn-sm.danger { color: #d62728; border-color: rgba(214,39,40,0.35); }
.btn-sm.danger:hover { background: rgba(214,39,40,0.08); }
.btn-sm.primary { color: var(--radio-checked-color); border-color: var(--radio-checked-border); }
.btn-sm.primary:hover { background: var(--radio-checked-bg); }
.btn-actions { display: flex; gap: 4px; }

.exp-link { color: var(--radio-checked-color); cursor: pointer; text-decoration: none;
            font-weight: 600; }
.exp-link:hover { text-decoration: underline; }
.path-text { color: var(--text-dim); font-size: 11px; font-family: monospace; }

/* ── Launch form (sidebar) ── */
.launch-btn { width: 100%; padding: 7px; background: var(--radio-checked-bg);
              border: 1px solid var(--radio-checked-border); border-radius: 4px;
              color: var(--radio-checked-color); font-size: 12px; font-weight: 600;
              cursor: pointer; font-family: inherit; margin-top: 2px; }
.launch-btn:hover { filter: brightness(1.1); }
.launch-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.launch-error { font-size: 11px; color: #d62728; padding: 4px 6px;
                background: rgba(214,39,40,0.08); border-radius: 3px;
                border: 1px solid rgba(214,39,40,0.2); display: none; word-break: break-word; }

/* ── System page ── */
.system-page-content { padding: 12px; display: flex; flex-direction: column; gap: 12px; flex: 1; overflow-y: auto; }

.sys-section-title { font-size: 10px; font-weight: 700; color: var(--text-dim);
                     text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }

.gpu-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; }

.gpu-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
            padding: 12px; display: flex; flex-direction: column; gap: 9px; }
.gpu-header { display: flex; justify-content: space-between; align-items: flex-start; }
.gpu-name-text { font-size: 12px; font-weight: 600; color: var(--text-strong); }
.gpu-idx { font-size: 10px; color: var(--text-muted); }
.gpu-temp { font-size: 11px; color: var(--text-dim); }

.stat-row { display: flex; align-items: center; gap: 7px; }
.stat-label { font-size: 10px; font-weight: 600; color: var(--text-dim);
              text-transform: uppercase; letter-spacing: 0.04em; min-width: 32px; }
.bar-bg { flex: 1; height: 7px; background: var(--border); border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; transition: width .5s ease; }
.bar-green  { background: #2ca02c; }
.bar-yellow { background: #bcbd22; }
.bar-red    { background: #d62728; }
.bar-blue   { background: #4c9be8; }
.stat-val { font-size: 11px; color: var(--text-dim); min-width: 72px; text-align: right; }

.sys-cards { display: flex; gap: 10px; flex-wrap: wrap; }
.sys-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
            padding: 12px 16px; display: flex; flex-direction: column; gap: 8px;
            min-width: 200px; flex: 1; }

.no-gpu { color: var(--text-muted); font-size: 12px; padding: 20px 0; }

.procs-table-wrap { border: 1px solid var(--border); border-radius: 6px;
                    background: var(--surface); overflow: hidden; }
table.procs-table { width: 100%; border-collapse: collapse; font-size: 11px; }
table.procs-table th { font-size: 10px; font-weight: 700; color: var(--text-dim);
                       text-transform: uppercase; letter-spacing: 0.06em;
                       padding: 5px 10px; border-bottom: 1px solid var(--border);
                       background: var(--bg); text-align: left; }
table.procs-table td { padding: 5px 10px; border-bottom: 1px solid var(--border);
                       color: var(--text); vertical-align: middle; }
table.procs-table tr:last-child td { border-bottom: none; }
.cmd-cell { color: var(--text-dim); font-family: monospace; font-size: 10px;
            max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.no-procs { font-size: 12px; color: var(--text-muted); padding: 14px 10px; }
</style>
</head>
<body>

<!-- ── Sidebar ─────────────────────────────────────────────────────────────── -->
<div id="sidebar">

  <!-- Jobs sidebar -->
  <div id="sb-jobs">
    <div>
      <h1>mlsweep</h1>
      <div class="exp-name">Control Panel</div>
    </div>

    <div class="section">
      <div class="section-title">Launch Sweep</div>
      <div id="launch-form" style="display:flex;flex-direction:column;gap:7px">
        <div style="display:flex;flex-direction:column;gap:3px">
          <div class="section-title" style="font-size:9px">Sweep file</div>
          <input type="text" id="sweep-file-input" placeholder="path/to/sweep.toml">
        </div>
        <div style="display:flex;flex-direction:column;gap:3px">
          <div class="section-title" style="font-size:9px">Workers file (optional)</div>
          <input type="text" id="workers-file-input" placeholder="path/to/workers.toml">
        </div>
        <div style="display:flex;flex-direction:column;gap:3px">
          <div class="section-title" style="font-size:9px">Note (optional)</div>
          <input type="text" id="note-input" placeholder="experiment note…">
        </div>
        <div style="display:flex;flex-direction:column;gap:3px">
          <div class="section-title" style="font-size:9px">Extra args (optional)</div>
          <input type="text" id="extra-args-input" placeholder="--resume --dry-run …">
        </div>
        <div id="launch-error" class="launch-error"></div>
        <button class="launch-btn" id="launch-btn">&#9654; Launch</button>
      </div>
    </div>
  </div>

  <!-- Metrics sidebar -->
  <div id="sb-metrics" style="display:none">
    <div>
      <h1>mlsweep</h1>
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

  <!-- System sidebar -->
  <div id="sb-system" style="display:none">
    <div>
      <h1>mlsweep</h1>
      <div class="exp-name" id="sys-hostname">System Monitor</div>
    </div>
    <div class="section">
      <div class="section-title">Refresh</div>
      <div style="font-size:11px;color:var(--text-muted)">Updates every 2 seconds</div>
    </div>
  </div>

</div>

<!-- ── Main ──────────────────────────────────────────────────────────────────── -->
<div id="main">

  <!-- Top navigation bar -->
  <div id="topbar">
    <div id="page-tabs">
      <button class="page-tab active" data-page="jobs">Jobs</button>
      <button class="page-tab" data-page="metrics">Metrics</button>
      <button class="page-tab" data-page="system">System</button>
    </div>
    <div id="statusbar">
      <div id="status-text"></div>
      <button id="theme-toggle" title="Toggle theme">&#9788;</button>
    </div>
  </div>

  <!-- Jobs page -->
  <div id="view-jobs" class="page-view">
    <div class="jobs-page-content">
      <div id="jobs-empty" class="empty-state" style="display:none">
        <div class="empty-icon">&#9634;</div>
        <div class="empty-title">No jobs yet</div>
        <div class="empty-sub">Use the launch form on the left to start a sweep.</div>
      </div>
      <div id="jobs-table-wrap" class="jobs-table-wrap" style="display:none">
        <table class="jobs-table">
          <thead>
            <tr>
              <th>Experiment</th>
              <th>Sweep File</th>
              <th>Status</th>
              <th>Progress</th>
              <th>Elapsed</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="jobs-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Metrics page -->
  <div id="view-metrics" class="page-view" style="display:none">
    <div id="metric-tab-bar">
      <button class="tab active" data-tab="curves">Curves</button>
      <button class="tab" data-tab="sensitivity">Sensitivity</button>
    </div>
    <div id="chart-curves" class="chart-pane"></div>
    <div id="chart-sensitivity" class="chart-pane" style="display:none"></div>
  </div>

  <!-- System page -->
  <div id="view-system" class="page-view" style="display:none">
    <div class="system-page-content">
      <div id="sys-overview" class="sys-cards"></div>
      <div>
        <div class="sys-section-title">GPUs</div>
        <div id="gpu-grid" class="gpu-grid"></div>
        <div id="no-gpu" class="no-gpu" style="display:none">
          No NVIDIA GPUs detected (nvidia-smi not found or no GPUs present).
        </div>
      </div>
      <div>
        <div class="sys-section-title">mlsweep Processes</div>
        <div id="procs-wrap" class="procs-table-wrap">
          <div id="no-procs" class="no-procs">No mlsweep processes found.</div>
          <table class="procs-table" id="procs-table" style="display:none">
            <thead>
              <tr><th>PID</th><th>CPU%</th><th>MEM%</th><th>Command</th></tr>
            </thead>
            <tbody id="procs-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
// ── Constants ─────────────────────────────────────────────────────────────────

const PALETTE = [
  "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
  "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
  "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5",
];

// ── Theme ─────────────────────────────────────────────────────────────────────

(function() {
  const btn  = document.getElementById("theme-toggle");
  const root = document.documentElement;
  function applyTheme(dark) {
    dark ? root.setAttribute("data-theme", "dark") : root.removeAttribute("data-theme");
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

// ── Page navigation ───────────────────────────────────────────────────────────

let activePage = "jobs";

function setPage(name) {
  activePage = name;
  document.querySelectorAll(".page-tab").forEach(b =>
    b.classList.toggle("active", b.dataset.page === name));
  document.querySelectorAll(".page-view").forEach(el => el.style.display = "none");
  document.getElementById("view-" + name).style.display = "flex";
  // Sidebar
  document.getElementById("sb-jobs").style.display    = name === "jobs"    ? "" : "none";
  document.getElementById("sb-metrics").style.display = name === "metrics" ? "" : "none";
  document.getElementById("sb-system").style.display  = name === "system"  ? "" : "none";
  // Trigger re-render if switching to metrics
  if (name === "metrics" && DATA && typeof Plotly !== "undefined") {
    if (dirtyTabs.has(activeTab)) renderTab(activeTab);
    else Plotly.relayout("chart-" + activeTab, {});
  }
  if (name === "system") fetchSystem();
}

document.querySelectorAll(".page-tab").forEach(btn => {
  btn.addEventListener("click", () => setPage(btn.dataset.page));
});

// ── Jobs page ─────────────────────────────────────────────────────────────────

function fmtTime(s) {
  if (!s) return "--";
  if (s < 60) return s.toFixed(0) + "s";
  if (s < 3600) return (s / 60).toFixed(0) + "m";
  return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m";
}

function fmtTS(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleDateString() + " " + d.toTimeString().slice(0,5);
}

function statusBadge(status) {
  const labels = {running:"Running",done:"Done",failed:"Failed",killed:"Killed",starting:"Starting"};
  return `<span class="status-badge badge-${status}"><span class="dot dot-${status}"></span>${labels[status]||status}</span>`;
}

function progressCell(prog) {
  if (!prog || prog.total === 0) return '<span style="color:var(--text-muted);font-size:11px">—</span>';
  const pct = prog.pct || 0;
  let cls = "prog-fill";
  if (prog.done === prog.total && prog.failed === 0) cls += " all-ok";
  else if (prog.failed > 0) cls += " has-fail";
  return `<div class="prog-wrap">
    <div class="prog-bar"><div class="${cls}" style="width:${pct}%"></div></div>
    <span class="prog-txt">${prog.done}/${prog.total}</span>
  </div>`;
}

function renderJobs(jobs) {
  const empty = document.getElementById("jobs-empty");
  const wrap  = document.getElementById("jobs-table-wrap");
  const tbody = document.getElementById("jobs-tbody");

  if (!jobs || jobs.length === 0) {
    empty.style.display = "";
    wrap.style.display  = "none";
    document.getElementById("status-text").textContent = "";
    return;
  }
  empty.style.display = "none";
  wrap.style.display  = "";

  const running = jobs.filter(j => j.status === "running" || j.status === "starting").length;
  document.getElementById("status-text").textContent =
    running ? `${running} job${running > 1 ? "s" : ""} running` : "";

  tbody.innerHTML = jobs.map(job => {
    const expCell = job.experiment
      ? `<a class="exp-link" onclick="viewMetrics('${job.experiment}')">${job.experiment}</a>`
      : `<span style="color:var(--text-muted);font-size:11px">${
          job.status === "starting" ? "detecting…" : "—"}</span>`;
    const sweepName = job.sweep_file.split("/").pop() || job.sweep_file;
    const alive = job.status === "running" || job.status === "starting";
    const canView = !!job.experiment;
    return `<tr>
      <td>${expCell}</td>
      <td><span class="path-text" title="${job.sweep_file}">${sweepName}</span></td>
      <td>${statusBadge(job.status)}</td>
      <td>${progressCell(job.progress)}</td>
      <td style="font-size:11px;color:var(--text-dim)">${fmtTime(job.elapsed)}</td>
      <td><div class="btn-actions">
        ${alive ? `<button class="btn-sm danger" onclick="killJob('${job.job_id}')">Stop</button>` : ""}
        ${canView ? `<button class="btn-sm primary" onclick="viewMetrics('${job.experiment}')">Metrics</button>` : ""}
      </div></td>
    </tr>`;
  }).join("");
}

async function fetchJobs() {
  try {
    const resp = await fetch("/jobs");
    const jobs = await resp.json();
    renderJobs(jobs);
  } catch (e) {}
}

async function killJob(jobId) {
  try {
    await fetch("/jobs/kill", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({job_id: jobId}),
    });
    fetchJobs();
  } catch (e) {}
}

function viewMetrics(expName) {
  setPage("metrics");
  const sel = document.getElementById("experiment-sel");
  // If the experiment is already in the selector, switch to it
  for (const opt of sel.options) {
    if (opt.value === expName) {
      sel.value = expName;
      loadExperiment(expName);
      return;
    }
  }
  // Otherwise, refresh the list first
  refreshExperiments(() => {
    sel.value = expName;
    loadExperiment(expName);
  });
}

// Launch form
document.getElementById("launch-btn").addEventListener("click", async () => {
  const sweepFile  = document.getElementById("sweep-file-input").value.trim();
  const workersFile = document.getElementById("workers-file-input").value.trim();
  const note       = document.getElementById("note-input").value.trim();
  const extraArgs  = document.getElementById("extra-args-input").value.trim();
  const errEl      = document.getElementById("launch-error");
  const btn        = document.getElementById("launch-btn");

  errEl.style.display = "none";
  if (!sweepFile) { errEl.textContent = "Sweep file path is required."; errEl.style.display = ""; return; }

  btn.disabled = true;
  btn.textContent = "Launching…";
  try {
    const resp = await fetch("/jobs/start", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        sweep_file: sweepFile,
        workers_file: workersFile || null,
        note: note || null,
        extra_args: extraArgs || null,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({error: "Unknown error"}));
      errEl.textContent = err.error || "Launch failed.";
      errEl.style.display = "";
    } else {
      document.getElementById("sweep-file-input").value = "";
      document.getElementById("note-input").value = "";
      document.getElementById("extra-args-input").value = "";
      fetchJobs();
    }
  } catch (e) {
    errEl.textContent = String(e);
    errEl.style.display = "";
  } finally {
    btn.disabled = false;
    btn.textContent = "▶ Launch";
  }
});

// Auto-refresh jobs
setInterval(fetchJobs, 2000);
fetchJobs();

// ── System page ───────────────────────────────────────────────────────────────

function utilColor(pct) {
  if (pct < 50) return "bar-green";
  if (pct < 80) return "bar-yellow";
  return "bar-red";
}

function renderSystem(data) {
  // hostname
  if (data.hostname) {
    document.getElementById("sys-hostname").textContent = data.hostname;
  }

  // Overview cards
  const overviewEl = document.getElementById("sys-overview");
  const cpuPct = data.cpu_pct || 0;
  const mem = data.mem || {};
  overviewEl.innerHTML = `
    <div class="sys-card">
      <div class="sys-section-title" style="margin-bottom:0">CPU</div>
      <div class="stat-row">
        <div class="bar-bg"><div class="bar-fill ${utilColor(cpuPct)}" style="width:${cpuPct}%"></div></div>
        <span class="stat-val" style="min-width:44px">${cpuPct.toFixed(1)}%</span>
      </div>
    </div>
    <div class="sys-card">
      <div class="sys-section-title" style="margin-bottom:0">RAM</div>
      <div class="stat-row">
        <div class="bar-bg"><div class="bar-fill ${utilColor(mem.pct||0)}" style="width:${mem.pct||0}%"></div></div>
        <span class="stat-val" style="min-width:100px">${(mem.used_gb||0).toFixed(1)} / ${(mem.total_gb||0).toFixed(1)} GB</span>
      </div>
    </div>
  `;

  // GPUs
  const gpuGrid = document.getElementById("gpu-grid");
  const noGpu   = document.getElementById("no-gpu");
  const gpus    = data.gpus || [];
  if (gpus.length === 0) {
    gpuGrid.style.display = "none";
    noGpu.style.display   = "";
  } else {
    noGpu.style.display   = "none";
    gpuGrid.style.display = "";
    gpuGrid.innerHTML = gpus.map(g => `
      <div class="gpu-card">
        <div class="gpu-header">
          <div>
            <div class="gpu-idx">GPU ${g.index}</div>
            <div class="gpu-name-text">${g.name}</div>
          </div>
          <div class="gpu-temp">${g.temp}&deg;C</div>
        </div>
        <div class="stat-row">
          <span class="stat-label">Util</span>
          <div class="bar-bg"><div class="bar-fill ${utilColor(g.util)}" style="width:${g.util}%"></div></div>
          <span class="stat-val">${g.util}%</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">VRAM</span>
          <div class="bar-bg"><div class="bar-fill bar-blue" style="width:${g.mem_pct}%"></div></div>
          <span class="stat-val">${(g.mem_used/1024).toFixed(1)} / ${(g.mem_total/1024).toFixed(1)} GB</span>
        </div>
      </div>
    `).join("");
  }
}

async function fetchSystem() {
  try {
    const resp = await fetch("/system");
    const data = await resp.json();
    renderSystem(data);
    // Fetch processes separately
    fetchProcs();
  } catch (e) {}
}

async function fetchProcs() {
  try {
    const resp = await fetch("/procs");
    const procs = await resp.json();
    const noProcs   = document.getElementById("no-procs");
    const procsTable = document.getElementById("procs-table");
    const tbody     = document.getElementById("procs-tbody");
    if (!procs || procs.length === 0) {
      noProcs.style.display   = "";
      procsTable.style.display = "none";
    } else {
      noProcs.style.display   = "none";
      procsTable.style.display = "";
      tbody.innerHTML = procs.map(p => `<tr>
        <td>${p.pid}</td>
        <td>${p.cpu.toFixed(1)}</td>
        <td>${p.mem.toFixed(1)}</td>
        <td class="cmd-cell" title="${p.cmd}">${p.cmd}</td>
      </tr>`).join("");
    }
  } catch (e) {}
}

// Auto-refresh system when on system page
setInterval(() => { if (activePage === "system") fetchSystem(); }, 2000);

// ── Metrics tab (loss curves + sensitivity) ───────────────────────────────────
// Identical logic to mlsweep_viz

let DATA         = null;
let METRIC_CACHE = {};
let colorBy      = null;
let metric       = "loss";
let smoothAlpha  = 0;
let filters      = {};
let METRIC_NAMES = [];
let activeTab    = "curves";
const dirtyTabs  = new Set(["curves", "sensitivity"]);
let LIVE_RUNS    = new Set();
let _evtSource   = null;
let _traceIndexMap = {};
let LAST_EMA_S   = {};

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
  document.getElementById("chart-curves").style.display      = name === "curves"      ? "" : "none";
  document.getElementById("chart-sensitivity").style.display = name === "sensitivity" ? "" : "none";
  const pane = document.getElementById("chart-" + name);
  if (dirtyTabs.has(name)) renderTab(name);
  else if (typeof Plotly !== "undefined") Plotly.relayout(pane, {});
}

function buildActiveChart() {
  if (typeof Plotly === "undefined") return;
  dirtyTabs.add("curves");
  dirtyTabs.add("sensitivity");
  renderTab(activeTab);
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function ema(vals, alpha) {
  if (alpha === 0) return vals;
  const out = []; let s = null;
  for (const v of vals) {
    if (v === null || !isFinite(v)) { out.push(v); continue; }
    s = s === null ? v : alpha * s + (1 - alpha) * v;
    out.push(s);
  }
  return out;
}

function colorMap(dimValues) {
  const m = {};
  dimValues.forEach((v, i) => { m[v] = PALETTE[i % PALETTE.length]; });
  return m;
}

function lerpColor(t) {
  const r = Math.round(0x15 + (0xC6 - 0x15) * t);
  const g = Math.round(0x65 + (0x28 - 0x65) * t);
  const b = Math.round(0xC0 + (0x28 - 0xC0) * t);
  return `rgb(${r},${g},${b})`;
}

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
  return orderedComboEntries(combo, skipDim).map(([k,v]) => `${k}=${v}`).join("  ");
}

function hoverText(combo) {
  return orderedComboEntries(combo, null).map(([k,v]) => `<b>${k}</b>: ${v}`).join("<br>");
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

function runScalar(runHash) {
  const cache = METRIC_CACHE[metric] || {};
  const vals = (cache[runHash]?.values || []).filter(v => v !== null && isFinite(v));
  return vals.length ? vals[vals.length - 1] : null;
}

function computeDimEffects(runs) {
  return Object.entries(DATA.dims).map(([dim, values]) => {
    const valueMeans = values.map(val => {
      const matching = runs.filter(r => r.combo[dim] === val);
      const scalars = matching.map(r => runScalar(r.hash)).filter(v => v !== null);
      const mean = scalars.length ? scalars.reduce((a,b) => a+b, 0) / scalars.length : null;
      return { val, mean, scalars, count: scalars.length };
    }).filter(vm => vm.mean !== null);
    const means = valueMeans.map(vm => vm.mean);
    const effectSize = means.length >= 2 ? Math.max(...means) - Math.min(...means) : 0;
    return { dim, effectSize, valueMeans };
  }).sort((a, b) => b.effectSize - a.effectSize);
}

function buildCurvesChart() {
  let runs = visibleRuns();
  const subDimInfo = DATA.subDims || {};
  if (!colorBy.startsWith("_m:") && subDimInfo[colorBy])
    runs = runs.filter(r => colorBy in r.combo);

  const isMetricColor = colorBy.startsWith("_m:");
  let getColor, getGroup, showLegendFor, legendGroupTitle;
  if (isMetricColor) {
    const cmap_ = metricColorsForRuns(DATA.runs, colorBy.slice(3));
    getColor         = r     => cmap_.get(r.name) || "#888";
    getGroup         = r     => r.name;
    showLegendFor    = ()    => false;
    legendGroupTitle = ()    => undefined;
  } else {
    const cmap     = colorMap(DATA.dims[colorBy] || []);
    const firstSeen = new Set();
    getColor         = r => cmap[r.combo[colorBy]] || "#888";
    getGroup         = r => String(r.combo[colorBy]);
    showLegendFor    = group => { const f = !firstSeen.has(group); if (f) firstSeen.add(group); return f; };
    legendGroupTitle = (group, isFirst) => isFirst ? { text: colorBy, font: { size: 11 } } : undefined;
  }

  const plotBg   = cssVar("--plot-bg");
  const plotGrid = cssVar("--plot-grid");
  const plotText = cssVar("--plot-text");
  const mcache   = METRIC_CACHE[metric] || {};

  const lineTraces = runs.map(r => {
    const color = getColor(r), group = getGroup(r), isFirst = showLegendFor(group);
    const mdata = mcache[r.hash] || {};
    return {
      x: mdata.steps || [], y: ema(mdata.values || [], smoothAlpha),
      mode: "lines", line: { color, width: 1.5 },
      name: group, legendgroup: group,
      legendgrouptitle: legendGroupTitle(group, isFirst),
      showlegend: isFirst,
      hovertemplate: hoverText(r.combo) + "<br><b>step</b>: %{x}<br><b>" + metric + "</b>: %{y:.4f}<extra></extra>",
    };
  });

  const dotTraces = runs.map(r => {
    const color = getColor(r), group = getGroup(r);
    const mdata = mcache[r.hash] || {};
    const steps = mdata.steps || [], vals = ema(mdata.values || [], smoothAlpha);
    return {
      x: steps.length ? [steps[steps.length-1]] : [],
      y: vals.length  ? [vals[vals.length-1]]   : [],
      mode: "markers",
      marker: { color, size: 6, line: { color: plotBg, width: 1.5 } },
      legendgroup: group, showlegend: false, hoverinfo: "skip",
    };
  });

  Plotly.react("chart-curves", [...lineTraces, ...dotTraces], {
    margin: { t:20, r:20, b:50, l:60 },
    xaxis: { title: "step", gridcolor: plotGrid, color: plotText },
    yaxis: { title: metric, gridcolor: plotGrid, color: plotText,
             range: yRangeOf(DATA.runs), autorange: yRangeOf(DATA.runs) === undefined },
    paper_bgcolor: plotBg, plot_bgcolor: plotBg,
    legend: { groupclick: "toggleitem", font: { size: 11, color: plotText } },
    hovermode: "closest",
  }, { responsive: true, scrollZoom: true });

  _traceIndexMap = {};
  runs.forEach((r, i) => { _traceIndexMap[r.hash] = i; });
  if (!LAST_EMA_S[metric]) LAST_EMA_S[metric] = {};
  for (const r of runs) {
    if (smoothAlpha === 0) { LAST_EMA_S[metric][r.hash] = null; continue; }
    const vals = (mcache[r.hash] || {}).values || [];
    let s = null;
    for (const v of vals) {
      if (v === null || !isFinite(v)) continue;
      s = s === null ? v : smoothAlpha * s + (1-smoothAlpha) * v;
    }
    LAST_EMA_S[metric][r.hash] = s;
  }

  const liveCount = runs.filter(r => LIVE_RUNS.has(r.hash)).length;
  document.getElementById("status-text").textContent =
    `${runs.length} / ${DATA.runs.length} runs visible` +
    (liveCount ? `  •  ${liveCount} live` : "");
}

function buildSensitivityChart() {
  const runs = visibleRuns();
  const plotBg   = cssVar("--plot-bg");
  const plotText = cssVar("--plot-text");
  const plotGrid = cssVar("--plot-grid");
  const effects  = computeDimEffects(runs);
  if (!effects.length) {
    Plotly.react("chart-sensitivity", [], { paper_bgcolor: plotBg, plot_bgcolor: plotBg }, { responsive: true });
    return;
  }
  const traces = [];
  effects.forEach(e => {
    if (e.valueMeans.length < 2) return;
    const xVals = e.valueMeans.map(vm => vm.mean);
    const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
    traces.push({
      type: "scatter", mode: "lines",
      x: [xMin, xMax], y: [e.dim, e.dim],
      line: { color: cssVar("--text-dim"), width: 2 },
      showlegend: false, hoverinfo: "skip",
    });
  });
  const allLabels = [...new Set(effects.flatMap(e => e.valueMeans.map(vm => String(vm.val))))];
  const labelColorMap = Object.fromEntries(allLabels.map((l,i) => [l, PALETTE[i % PALETTE.length]]));
  allLabels.forEach(label => {
    const xs = [], ys = [], customdata = [];
    effects.forEach(e => {
      const vm = e.valueMeans.find(v => String(v.val) === label);
      if (!vm) return;
      xs.push(vm.mean); ys.push(e.dim);
      customdata.push({ dim: e.dim, val: label, count: vm.count, effectSize: e.effectSize.toFixed(4) });
    });
    if (!xs.length) return;
    traces.push({
      type: "scatter", mode: "markers",
      x: xs, y: ys, name: label,
      marker: { size: 12, color: labelColorMap[label], line: { color: plotBg, width: 2 } },
      customdata,
      hovertemplate: "<b>%{customdata.dim} = %{customdata.val}</b><br>" +
        `Mean ${metric}: %{x:.4f}<br>n=%{customdata.count}<br>Effect: %{customdata.effectSize}<extra></extra>`,
    });
  });
  const allMeans = effects.flatMap(e => e.valueMeans.map(vm => vm.mean));
  const xMax = allMeans.length ? Math.max(...allMeans) : 1;
  const xMin = allMeans.length ? Math.min(...allMeans) : 0;
  const xPad = Math.max((xMax - xMin) * 0.22, 1e-6);
  const annotations = effects.map(e => ({
    x: xMax + xPad * 0.05, y: e.dim,
    xanchor: "left", yanchor: "middle",
    text: `Δ${e.effectSize.toFixed(4)}`, showarrow: false,
    font: { size: 10, color: plotText },
  }));
  Plotly.react("chart-sensitivity", traces, {
    margin: { t:50, r:20, b:60, l:110 },
    xaxis: { title: `Mean ${metric} (final)`, gridcolor: plotGrid, color: plotText,
             range: [xMin - xPad*0.1, xMax + xPad] },
    yaxis: { autorange: "reversed", color: plotText, tickfont: { size: 11 } },
    paper_bgcolor: plotBg, plot_bgcolor: plotBg,
    legend: { font: { size: 10, color: plotText } },
    annotations,
    title: { text: "Dimension sensitivity — ranked by effect size",
             font: { color: plotText, size: 13 }, x: 0.5 },
    hovermode: "closest",
  }, { responsive: true });
  document.getElementById("status-text").textContent =
    `${runs.length} runs · ${effects.length} dimensions`;
}

// ── Metrics sidebar builders ──────────────────────────────────────────────────

function buildFilters() {
  const subDimInfo = DATA.subDims || {};
  const subDimSet  = new Set(Object.keys(subDimInfo));
  const childrenOf = {};
  for (const [dim, info] of Object.entries(subDimInfo)) {
    const key = `${info.parentDim}:${info.parentValue}`;
    (childrenOf[key] = childrenOf[key] || []).push(dim);
  }
  const container = document.getElementById("filters");
  container.innerHTML = "";

  function makeFilterGroup(dim, values, extraClass) {
    const isColorDim = !colorBy.startsWith("_m:") && dim === colorBy;
    const cmap_ = isColorDim ? colorMap(values) : {};
    const group = document.createElement("div");
    group.className = "filter-group" + (extraClass ? " " + extraClass : "");
    const header = document.createElement("div");
    header.className = "filter-header";
    header.innerHTML = `<span class="axis-label">${dim}</span>`;
    const btns = document.createElement("div");
    btns.className = "filter-btns";
    ["all","none"].forEach(action => {
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
      cb.type = "checkbox"; cb.dataset.dim = dim;
      cb.checked = filters[dim]?.has(val) ?? true;
      cb.addEventListener("change", () => {
        if (cb.checked) filters[dim].add(val); else filters[dim].delete(val);
        buildActiveChart();
      });
      row.appendChild(cb);
      if (isColorDim) {
        const sw = document.createElement("div");
        sw.className = "color-swatch"; sw.style.background = cmap_[val] || "#ccc";
        row.appendChild(sw);
      }
      const lbl = document.createElement("span");
      lbl.className = "check-label"; lbl.textContent = val;
      row.appendChild(lbl);
      list.appendChild(row);
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
    if (subDimSet.has(dim)) continue;
    container.appendChild(makeFilterGroup(dim, values, null));
  }
}

function buildColorBySelect() {
  const container  = document.getElementById("color-by");
  const subDimInfo = DATA.subDims || {};
  container.innerHTML = "";
  for (const dim of Object.keys(DATA.dims)) {
    const id = "cbr-" + dim;
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

const _metricFetching = new Set();

function loadMetric(name) {
  metric = name;
  if (METRIC_CACHE[name] !== undefined) { buildActiveChart(); return; }
  if (_metricFetching.has(name)) return;
  const expName = document.getElementById("experiment-sel").value;
  if (!expName) return;
  _metricFetching.add(name);
  fetch(`/metric_data.json?experiment=${encodeURIComponent(expName)}&metric=${encodeURIComponent(name)}`)
    .then(r => r.json())
    .then(data => {
      _metricFetching.delete(name);
      if (!METRIC_CACHE[name]) {
        METRIC_CACHE[name] = data;
      } else {
        for (const [run, runData] of Object.entries(data))
          if (!METRIC_CACHE[name][run]) METRIC_CACHE[name][run] = runData;
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
  if (DATA.note) { noteEl.textContent = DATA.note; noteEl.style.display = ""; }
  else            { noteEl.style.display = "none"; }

  filters = Object.fromEntries(
    Object.entries(DATA.dims).map(([dim, vals]) => [dim, new Set(vals)])
  );

  if (!init._wired) {
    init._wired = true;
    document.getElementById("smoothing").addEventListener("input", e => {
      smoothAlpha = parseFloat(e.target.value);
      document.getElementById("smooth-val").textContent = smoothAlpha.toFixed(2);
      if (activeTab === "curves") buildCurvesChart();
    });
    document.querySelectorAll(".tab").forEach(btn => {
      btn.addEventListener("click", () => setTab(btn.dataset.tab));
    });
  }

  metric = DATA.metricNames.includes(metric) ? metric
    : (DATA.metricNames.find(m => m.includes("loss")) || DATA.metricNames[0]);

  buildMetricSelect();
  buildColorBySelect();
  buildFilters();
  setTab(activeTab);
  loadMetric(metric);
}

// ── SSE ───────────────────────────────────────────────────────────────────────

function _applyEmaIncremental(newValues, lastS) {
  if (smoothAlpha === 0) return [newValues, null];
  const out = []; let s = lastS;
  for (const v of newValues) {
    if (v === null || !isFinite(v)) { out.push(v); continue; }
    s = s === null ? v : smoothAlpha * s + (1 - smoothAlpha) * v;
    out.push(s);
  }
  return [out, s];
}

function connectSSE(expName) {
  if (_evtSource) { _evtSource.close(); _evtSource = null; }
  LIVE_RUNS = new Set(); LAST_EMA_S = {}; _metricFetching.clear();
  _evtSource = new EventSource(`/events?experiment=${encodeURIComponent(expName)}`);

  _evtSource.addEventListener("init", e => {
    const msg = JSON.parse(e.data);
    LIVE_RUNS = new Set(Object.entries(msg.status)
      .filter(([,s]) => s === "running").map(([n]) => n));
    METRIC_CACHE = {};
    const el = document.getElementById("exp-name");
    el.classList.remove("loading");
    el.textContent = DATA ? DATA.experiment : expName;
    loadMetric(metric);
  });

  _evtSource.addEventListener("metrics", e => {
    const msg = JSON.parse(e.data);
    const incrementalUpdates = [];
    let gotNewData = false;
    for (const [run, perRunUpdates] of Object.entries(msg.runs)) {
      for (const [mname, {steps, values}] of Object.entries(perRunUpdates)) {
        if (!METRIC_CACHE[mname]) METRIC_CACHE[mname] = {};
        if (!METRIC_CACHE[mname][run]) METRIC_CACHE[mname][run] = {steps:[], values:[]};
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
            const lastStep = steps[steps.length-1];
            const lastY    = newSmoothed[newSmoothed.length-1];
            incrementalUpdates.push({lineIdx, dotIdx, newSteps:steps, newSmoothed, lastStep, lastY});
          }
        }
      }
    }
    if (!gotNewData) return;
    if (activePage === "metrics" && activeTab === "curves" && !dirtyTabs.has("curves") && smoothAlpha === 0) {
      const lineIdxs=[], lineXs=[], lineYs=[], dotIdxs=[], dotXs=[], dotYs=[];
      for (const u of incrementalUpdates) {
        lineIdxs.push(u.lineIdx); lineXs.push(u.newSteps);  lineYs.push(u.newSmoothed);
        dotIdxs.push(u.dotIdx);   dotXs.push([u.lastStep]); dotYs.push([u.lastY]);
      }
      if (lineIdxs.length) {
        Plotly.extendTraces("chart-curves", {x:lineXs, y:lineYs}, lineIdxs);
        Plotly.restyle("chart-curves",      {x:dotXs,  y:dotYs},  dotIdxs);
      }
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
  const el = document.getElementById("exp-name");
  el.classList.add("loading");
  el.textContent = "Fetching data…";
  if (_evtSource) { _evtSource.close(); _evtSource = null; }

  fetch(`/manifest.json?name=${encodeURIComponent(name)}`)
    .then(r => { if (r.status === 404) return null; if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(manifest => {
      if (manifest !== null) {
        metric = "loss"; init(manifest);
      } else {
        return fetch(`/data.json?name=${encodeURIComponent(name)}`).then(r => r.json())
          .then(data => { metric = "loss"; init(data); });
      }
    })
    .then(() => connectSSE(name))
    .catch(e => {
      el.classList.remove("loading"); el.textContent = "Error";
      document.getElementById("status-text").textContent = "Error: " + e;
    });
}

function refreshExperiments(cb) {
  fetch("/experiments").then(r => r.json())
    .then(({experiments}) => { _updateExperimentList(experiments); if (cb) cb(); })
    .catch(() => {});
}

// Bootstrap metrics tab
fetch("/config.json").then(r => r.json()).then(cfg => {
  if (cfg.poll_interval) document.getElementById("poll-interval").value = cfg.poll_interval;
}).catch(() => {});

document.getElementById("poll-interval").addEventListener("change", e => {
  const val = parseFloat(e.target.value);
  if (val >= 0.5 && val <= 300)
    fetch("/poll-interval", { method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({seconds:val}) }).catch(()=>{});
});

fetch("/experiments")
  .then(r => r.json())
  .then(({experiments, default: def}) => {
    const sel = document.getElementById("experiment-sel");
    for (const name of experiments) {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      if (name === def) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.onchange = () => loadExperiment(sel.value);
    document.getElementById("refresh-experiments").onclick = () => refreshExperiments(null);
    if (def) loadExperiment(def);
    else {
      document.getElementById("exp-name").classList.remove("loading");
      document.getElementById("exp-name").textContent = "No experiments yet";
    }
  })
  .catch(() => {
    document.getElementById("exp-name").classList.remove("loading");
    document.getElementById("exp-name").textContent = "No experiments";
  });
</script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" onload="if(DATA&&activePage==='metrics')buildActiveChart()"></script>
</body>
</html>
"""


# ── HTTP handler ───────────────────────────────────────────────────────────────


def _read_procs() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
        procs: list[dict[str, Any]] = []
        for line in result.stdout.splitlines()[1:]:
            if "mlsweep" not in line:
                continue
            parts = line.split(None, 10)
            if len(parts) < 11:
                continue
            try:
                procs.append({
                    "pid": int(parts[1]),
                    "cpu": float(parts[2]),
                    "mem": float(parts[3]),
                    "cmd": parts[10][:120],
                })
            except (ValueError, IndexError):
                continue
        return procs[:20]
    except (OSError, subprocess.TimeoutExpired):
        return []


class Handler(BaseHTTPRequestHandler):
    _default: str | None = None
    _output_dir: str = ""

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
            mdata = load_manifest(name, self.exp_source)
            if mdata is None:
                self.send_response(404); self.end_headers(); return
            body = json.dumps(mdata).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/metric_data.json":
            name   = qs.get("experiment", [None])[0]
            mname  = qs.get("metric",     [None])[0]
            if not name or not mname:
                self.send_response(400); self.end_headers(); return
            data = _metric_data_snapshot(name, mname)
            self._send(200, "application/json", json.dumps(data).encode())

        elif parsed.path == "/config.json":
            self._send(200, "application/json",
                       json.dumps({"poll_interval": _poll_interval}).encode())

        elif parsed.path == "/events":
            name = qs.get("experiment", [None])[0]
            if not name:
                self.send_response(400); self.end_headers(); return
            self._serve_sse(name)

        elif parsed.path == "/system":
            with _stats_lock:
                data = dict(_sys_stats)
            self._send(200, "application/json", json.dumps(data).encode())

        elif parsed.path == "/procs":
            procs = _read_procs()
            self._send(200, "application/json", json.dumps(procs).encode())

        elif parsed.path == "/jobs":
            with _JOBS_LOCK:
                jobs = [j.to_dict() for j in reversed(list(_JOBS.values()))]
            self._send(200, "application/json", json.dumps(jobs).encode())

        elif parsed.path == "/favicon.png":
            favicon_path = Path(__file__).parent / "static" / "favicon.png"
            if favicon_path.exists():
                self._send(200, "image/png", favicon_path.read_bytes())
            else:
                self.send_response(404); self.end_headers()

        else:
            self.send_response(404); self.end_headers()

    def do_POST(self) -> None:
        global _poll_interval

        length = int(self.headers.get("Content-Length", 0))
        raw    = self.rfile.read(length)

        if self.path == "/poll-interval":
            try:
                body = json.loads(raw)
                val  = float(body["seconds"])
                if 0.5 <= val <= 300:
                    _poll_interval = val
                self._send(200, "application/json", b"{}")
            except (KeyError, ValueError, json.JSONDecodeError):
                self.send_response(400); self.end_headers()

        elif self.path == "/jobs/start":
            try:
                body = json.loads(raw)
                sweep_file   = str(body.get("sweep_file", "")).strip()
                workers_file = body.get("workers_file") or None
                note         = body.get("note") or None
                raw_extra    = body.get("extra_args") or ""
                extra_args: list[str] = raw_extra.split() if raw_extra else []

                if not sweep_file:
                    self._send(400, "application/json",
                               json.dumps({"error": "sweep_file is required"}).encode())
                    return

                sweep_path = Path(sweep_file)
                if not sweep_path.is_absolute():
                    sweep_path = Path(os.getcwd()) / sweep_path
                if not sweep_path.exists():
                    self._send(400, "application/json",
                               json.dumps({"error": f"File not found: {sweep_file}"}).encode())
                    return
                if workers_file:
                    wp = Path(workers_file)
                    if not wp.is_absolute():
                        wp = Path(os.getcwd()) / wp
                    if not wp.exists():
                        self._send(400, "application/json",
                                   json.dumps({"error": f"Workers file not found: {workers_file}"}).encode())
                        return
                    workers_file = str(wp)

                job = _start_job(
                    sweep_file   = str(sweep_path),
                    workers_file = workers_file,
                    output_dir   = self.exp_source,
                    extra_args   = extra_args,
                    note         = note,
                )
                self._send(200, "application/json", json.dumps({"job_id": job.job_id}).encode())
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self._send(400, "application/json", json.dumps({"error": str(e)}).encode())
            except OSError as e:
                self._send(500, "application/json", json.dumps({"error": str(e)}).encode())

        elif self.path == "/jobs/kill":
            try:
                body   = json.loads(raw)
                job_id = str(body["job_id"])
                with _JOBS_LOCK:
                    kill_job: ManagedJob | None = _JOBS.get(job_id)
                    if kill_job is None:
                        self._send(404, "application/json",
                                   json.dumps({"error": "Job not found"}).encode())
                        return
                    if kill_job.proc is not None and kill_job.proc.poll() is None:
                        kill_job.proc.terminate()
                    kill_job.status   = "killed"
                    kill_job.ended_at = time.time()
                self._send(200, "application/json", b"{}")
            except (KeyError, json.JSONDecodeError) as e:
                self._send(400, "application/json", json.dumps({"error": str(e)}).encode())

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

    def log_message(self, fmt: str, *args: object) -> None:
        pass

    @property
    def exp_source(self) -> str:
        return Handler._output_dir


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    try:
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("--dir", default=None, metavar="PATH",
                            help="Sweeps output directory (default: ./outputs/sweeps)")
        parser.add_argument("--port", type=int, default=43802)
        parser.add_argument("--poll-interval", type=float, default=1.0, metavar="SECONDS")
        parser.add_argument("--open-browser", action="store_true")
        parser.add_argument(
            "--version", action="version",
            version=f"%(prog)s {importlib.metadata.version('mlsweep')}")
        args = parser.parse_args()

        global _poll_interval
        output_dir = os.path.abspath(args.dir or os.path.join(os.getcwd(), "outputs", "sweeps"))
        os.makedirs(output_dir, exist_ok=True)

        Handler._output_dir = output_dir
        _poll_interval = args.poll_interval

        experiments = list_experiments(output_dir)
        Handler._default = experiments[0] if experiments else None

        # Initial scan
        _scan(output_dir)

        # Background threads
        threading.Thread(target=_watch_loop,      args=(output_dir,), daemon=True).start()
        threading.Thread(target=_poll_system_loop,                    daemon=True).start()
        threading.Thread(target=_monitor_jobs_loop,                   daemon=True).start()

        url = f"http://localhost:{args.port}"
        print("mlsweep Web UI")
        print(f"  Output:  {output_dir}")
        print(f"  Browser: {url}  (Ctrl+C to stop)")
        try:
            import socket as _sock
            with _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                public_ip = s.getsockname()[0]
            if public_ip != "127.0.0.1":
                print(f"           http://{public_ip}:{args.port}")
        except OSError:
            pass

        if args.open_browser:
            import threading as _t
            _t.Timer(0.5, lambda: webbrowser.open(url)).start()

        ThreadingHTTPServer(("", args.port), Handler).serve_forever()

    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
