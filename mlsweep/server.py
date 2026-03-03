#!/usr/bin/env python3
"""Experiment tracking server for ECO/torchtitan experiments.

Lightweight HTTP server that stores and serves training metrics.
Data is stored as JSONL files co-located with training output — no database needed.

Storage layout:
    <exp-dir>/<experiment>/<run_name>/
        run_meta.json   — written once at run start, updated at end
        metrics.jsonl   — one JSON line per log() call, append-only

Usage:
    python exp_server.py --exp-dir outputs/sweeps --port 53800 --host 0.0.0.0

API (write):
    POST /run/start    — register a new run, write run_meta.json
    POST /run/end      — finalize a run with end_time + status
    POST /metrics      — append one JSONL line to a run's metrics.jsonl
    POST /run/replay   — bulk-append records with step-based deduplication

API (read):
    GET  /experiments                            — list experiments sorted newest-first
    GET  /data.json?name=...                     — experiment metadata (30s TTL cache)
    GET  /metric.json?name=...&metric=...        — full metric time-series for all runs
    GET  /manifest.json?name=...                 — sweep manifest (axes, runs, metricNames)
    GET  /metric_since.json?name=...&metric=...&since_step=N  — incremental metric data
    GET  /run_status.json?name=...               — per-run status dict (O(1), no disk)
"""

import argparse
import json
import os
import socketserver
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path


_verbose: bool = False


def _vlog(msg: str) -> None:
    if _verbose:
        print(msg, flush=True)


# ── In-memory index ────────────────────────────────────────────────────────────

# run_key = f"{experiment}/{run_name}"
_experiments: dict[str, list[str]] = {}   # experiment → [run_name, ...]
_run_dirs: dict[str, str] = {}             # run_key → absolute path
_open_files: dict[str, object] = {}        # run_key → open append file handle

# Per-run status: "pending" | "running" | "completed" | "failed" | "interrupted"
_run_status: dict[str, str] = {}           # run_key → status string

_lock = threading.Lock()        # guards _experiments, _run_dirs, _run_status
_files_lock = threading.Lock()  # guards _open_files

_root_dir: str = ""

# GET /data.json cache: experiment_name → (timestamp, json_bytes)
_meta_cache: dict[str, tuple[float, bytes]] = {}
_META_CACHE_TTL = 30.0  # seconds

# Heartbeat-based interrupt detection: run is interrupted if last_heartbeat is
# present and more than _HEARTBEAT_STALE seconds old.
_HEARTBEAT_STALE = 300.0   # 5 minutes
_STARTUP_GRACE   = 3600.0  # runs without heartbeat, started < 1h ago, are not interrupted


def _run_key(experiment: str, run_name: str) -> str:
    return f"{experiment}/{run_name}"


def _is_interrupted(meta: dict) -> bool:
    """Heuristic: is a 'running' run actually interrupted?"""
    hb = meta.get("last_heartbeat")
    now = time.time()
    if hb is not None:
        return (now - hb) > _HEARTBEAT_STALE
    # No heartbeat field: use start_time grace period
    return meta.get("start_time", now) < (now - _STARTUP_GRACE)


def _scan_disk(root_dir: str) -> None:
    """Rebuild in-memory index from disk. Called once at startup."""
    global _root_dir
    _root_dir = os.path.abspath(root_dir)

    with _lock:
        for meta_path in sorted(Path(_root_dir).glob("*/*/run_meta.json")):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            run_name = meta.get("run_name")
            experiment = meta.get("experiment")
            if not run_name or not experiment:
                continue

            run_dir = str(meta_path.parent)
            key = _run_key(experiment, run_name)
            _run_dirs[key] = run_dir

            if experiment not in _experiments:
                _experiments[experiment] = []
            if run_name not in _experiments[experiment]:
                _experiments[experiment].append(run_name)

            status = meta.get("status", "unknown")
            if status == "running" and _is_interrupted(meta):
                status = "interrupted"
                meta["status"] = "interrupted"
                tmp = str(meta_path) + ".tmp"
                try:
                    with open(tmp, "w") as f:
                        json.dump(meta, f, indent=2)
                    os.replace(tmp, str(meta_path))
                except OSError:
                    pass
            _run_status[key] = status

    n_exp = len(_experiments)
    n_runs = len(_run_dirs)
    print(f"Scanned {_root_dir}: {n_runs} runs across {n_exp} experiments", flush=True)


def _get_or_open_file(experiment: str, run_name: str):
    """Return the open (line-buffered) append file handle for a run, opening if needed."""
    key = _run_key(experiment, run_name)
    with _files_lock:
        if key in _open_files:
            return _open_files[key]
        with _lock:
            run_dir = _run_dirs.get(key)
        if run_dir is None:
            return None
        path = os.path.join(run_dir, "metrics.jsonl")
        fh = open(path, "a", buffering=1)  # line-buffered: flush on each newline
        _open_files[key] = fh
        return fh


# ── HTTP handler ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    _token: str | None = None  # set by main() if --token is provided

    def _check_token(self) -> bool:
        """Return True if the request is authorized (or no token configured)."""
        if self._token is None:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {self._token}"

    def do_POST(self):
        if not self._check_token():
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Bearer realm="exp-server"')
            self.end_headers()
            return
        try:
            self._handle_post()
        except Exception:
            traceback.print_exc()
            try:
                self.send_response(500)
                self.end_headers()
            except Exception:
                pass

    def do_GET(self):
        try:
            self._handle_get()
        except Exception:
            traceback.print_exc()
            try:
                self.send_response(500)
                self.end_headers()
            except Exception:
                pass

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw)

    def _send(self, code: int, content_type: str, body: bytes,
              cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: object, code: int = 200,
                   cache: str = "no-store") -> None:
        self._send(code, "application/json", json.dumps(data).encode(), cache=cache)

    def _handle_post(self) -> None:
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/run/start":
            meta = self._read_body()
            run_name = meta.get("run_name")
            experiment = meta.get("experiment")
            _vlog(f"POST /run/start   {experiment}/{run_name}")
            if not run_name or not experiment:
                self.send_response(400)
                self.end_headers()
                return

            run_dir = os.path.join(_root_dir, experiment, run_name)
            os.makedirs(run_dir, exist_ok=True)

            meta_path = os.path.join(run_dir, "run_meta.json")
            tmp = meta_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp, meta_path)

            key = _run_key(experiment, run_name)
            with _lock:
                _run_dirs[key] = run_dir
                if experiment not in _experiments:
                    _experiments[experiment] = []
                if run_name not in _experiments[experiment]:
                    _experiments[experiment].append(run_name)
                _run_status[key] = "running"
                # Invalidate TTL cache so next /data.json picks up the new run
                _meta_cache.pop(experiment, None)

            print(f"[start] {experiment}/{run_name}", flush=True)
            self._send_json({"ok": True})

        elif parsed.path == "/run/end":
            data = self._read_body()
            run_name = data.get("run_name")
            experiment = data.get("experiment")
            _vlog(f"POST /run/end     {experiment}/{run_name}")
            if not run_name or not experiment:
                self.send_response(400)
                self.end_headers()
                return

            key = _run_key(experiment, run_name)
            with _lock:
                run_dir = _run_dirs.get(key)
            if run_dir is None:
                self.send_response(404)
                self.end_headers()
                return

            meta_path = os.path.join(run_dir, "run_meta.json")
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                meta = {}
            meta["end_time"] = data.get("end_time", time.time())
            status = data.get("status", "completed")
            meta["status"] = status
            tmp = meta_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp, meta_path)

            with _lock:
                _run_status[key] = status

            # Close and remove the cached file handle for this run
            with _files_lock:
                fh = _open_files.pop(key, None)
                if fh:
                    try:
                        fh.close()
                    except OSError:
                        pass

            print(f"[end]   {experiment}/{run_name}  status={status}", flush=True)
            self._send_json({"ok": True})

        elif parsed.path == "/metrics":
            data = self._read_body()
            run_name = data.get("run_name")
            experiment = data.get("experiment")
            record = data.get("record")
            _vlog(f"POST /metrics     {experiment}/{run_name}  step={record.get('step') if record else '?'}")
            if not run_name or not experiment or record is None:
                self.send_response(400)
                self.end_headers()
                return

            fh = _get_or_open_file(experiment, run_name)
            if fh is None:
                self.send_response(404)
                self.end_headers()
                return

            fh.write(json.dumps(record) + "\n")
            self._send_json({"ok": True})

        elif parsed.path == "/run/replay":
            # Bulk append with step-based deduplication.
            data = self._read_body()
            run_name = data.get("run_name")
            experiment = data.get("experiment")
            records = data.get("records")
            _vlog(f"POST /run/replay  {experiment}/{run_name}  records={len(records) if isinstance(records, list) else '?'}")
            if not run_name or not experiment or not isinstance(records, list):
                self.send_response(400)
                self.end_headers()
                return

            key = _run_key(experiment, run_name)
            with _lock:
                run_dir = _run_dirs.get(key)
            if run_dir is None:
                # Run not registered yet — auto-register from first replay
                # (handles the case where /run/start was lost during outage)
                self.send_response(404)
                self.end_headers()
                return

            metrics_path = os.path.join(run_dir, "metrics.jsonl")

            # Read existing steps to deduplicate
            existing_steps: set[int] = set()
            try:
                with open(metrics_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            s = rec.get("step")
                            if s is not None:
                                existing_steps.add(s)
                        except json.JSONDecodeError:
                            continue
            except OSError:
                pass  # file doesn't exist yet

            # Append only new records; use open file handle for efficiency
            fh = _get_or_open_file(experiment, run_name)
            if fh is None:
                self.send_response(404)
                self.end_headers()
                return

            written = 0
            for rec in records:
                step = rec.get("step")
                if step is not None and step in existing_steps:
                    continue  # deduplicate
                fh.write(json.dumps(rec) + "\n")
                if step is not None:
                    existing_steps.add(step)
                written += 1

            self._send_json({"ok": True, "written": written})

        else:
            self.send_response(404)
            self.end_headers()

    def _handle_get(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/experiments":
            _vlog("GET  /experiments")
            with _lock:
                exps = list(_experiments.keys())

            def newest_start(exp: str) -> float:
                best = 0.0
                with _lock:
                    runs = list(_experiments.get(exp, []))
                for rn in runs:
                    key = _run_key(exp, rn)
                    with _lock:
                        rd = _run_dirs.get(key)
                    if rd:
                        try:
                            with open(os.path.join(rd, "run_meta.json")) as f:
                                m = json.load(f)
                            best = max(best, m.get("start_time", 0.0))
                        except (OSError, json.JSONDecodeError):
                            pass
                return best

            experiments = sorted(exps, key=newest_start, reverse=True)
            self._send_json({
                "experiments": experiments,
                "default": experiments[0] if experiments else None,
            })

        elif parsed.path == "/data.json":
            name = qs.get("name", [None])[0]
            _vlog(f"GET  /data.json   name={name}")
            if not name:
                self.send_response(400)
                self.end_headers()
                return

            now = time.time()
            with _lock:
                cached = _meta_cache.get(name)
            if cached:
                ts, cached_body = cached
                if now - ts < _META_CACHE_TTL:
                    self._send(200, "application/json", cached_body)
                    return

            data = _build_experiment_meta(name)
            body = json.dumps(data).encode()
            with _lock:
                _meta_cache[name] = (now, body)
            self._send(200, "application/json", body)

        elif parsed.path == "/metric.json":
            name = qs.get("name", [None])[0]
            metric = qs.get("metric", [None])[0]
            _vlog(f"GET  /metric.json  name={name}  metric={metric}")
            if not name or not metric:
                self.send_response(400)
                self.end_headers()
                return
            data = _load_metric_data(name, metric)
            self._send_json(data)

        elif parsed.path == "/manifest.json":
            name = qs.get("name", [None])[0]
            _vlog(f"GET  /manifest.json  name={name}")
            if not name:
                self.send_response(400)
                self.end_headers()
                return
            data = _load_manifest(name)
            if data is None:
                self._send_json({"error": "manifest not found"}, code=404)
                return
            # Populate metricNames dynamically if empty
            if not data.get("metricNames"):
                data["metricNames"] = _discover_metric_names(name)
            # Manifest is immutable once written; cache indefinitely
            self._send_json(data, cache="public, max-age=31536000")

        elif parsed.path == "/metric_since.json":
            name = qs.get("name", [None])[0]
            metric = qs.get("metric", [None])[0]
            since_raw = qs.get("since_step", [None])[0]
            _vlog(f"GET  /metric_since.json  name={name}  metric={metric}  since_step={since_raw}")
            if not name or not metric:
                self.send_response(400)
                self.end_headers()
                return
            since_step = int(since_raw) if since_raw is not None else -1
            data = _load_metric_since(name, metric, since_step)
            self._send_json(data)

        elif parsed.path == "/run_status.json":
            name = qs.get("name", [None])[0]
            _vlog(f"GET  /run_status.json  name={name}")
            if not name:
                self.send_response(400)
                self.end_headers()
                return
            with _lock:
                run_names = list(_experiments.get(name, []))
                result = {
                    rn: _run_status.get(_run_key(name, rn), "unknown")
                    for rn in run_names
                }
            self._send_json(result)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        if _verbose:
            print(f"  -> {fmt % args}", flush=True)


# ── Data builders ──────────────────────────────────────────────────────────────

def _parse_tag_value(s: str):
    """Convert a tag value string to a typed Python value."""
    if s == "True":
        return True
    if s == "False":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _detect_sub_axes(runs: list[dict], axes: dict) -> dict:
    """Detect axes that only appear when a parent axis has a specific value.

    Shared utility used by both _build_experiment_meta (backward compat) and
    sweep_manifest population.
    """
    all_names = {r["hash"] for r in runs}
    names_with = {ax: {r["hash"] for r in runs if ax in r["combo"]} for ax in axes}
    sub_axes: dict = {}
    for axis in axes:
        if names_with[axis] == all_names:
            continue  # universal axis — not a sub-axis
        for parent_axis in axes:
            if parent_axis == axis:
                continue
            for parent_val in axes[parent_axis]:
                names_with_parent = {
                    r["hash"] for r in runs if r["combo"].get(parent_axis) == parent_val
                }
                if names_with_parent == names_with[axis]:
                    sub_axes[axis] = {"parentAxis": parent_axis, "parentValue": parent_val}
                    break
            if axis in sub_axes:
                break
    return sub_axes


def _build_experiment_meta(experiment_name: str) -> dict:
    """Build metadata dict for an experiment by reading run_meta.json files."""
    with _lock:
        run_names = list(_experiments.get(experiment_name, []))

    axis_values: dict[str, set] = {}
    runs = []
    metric_names: set[str] = set()

    for run_name in run_names:
        key = _run_key(experiment_name, run_name)
        with _lock:
            run_dir = _run_dirs.get(key)
        if not run_dir:
            continue

        try:
            with open(os.path.join(run_dir, "run_meta.json")) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        tags = meta.get("tags", {})
        combo = {k: _parse_tag_value(str(v)) for k, v in tags.items()}
        for k, v in combo.items():
            axis_values.setdefault(k, set()).add(v)

        runs.append({"name": run_name, "hash": run_name, "combo": combo})

        metrics_path = os.path.join(run_dir, "metrics.jsonl")
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

    def val_sort_key(v):
        if isinstance(v, bool):
            return (0, str(v))
        if isinstance(v, (int, float)):
            return (1, v)
        return (2, str(v))

    axes = {k: sorted(vs, key=val_sort_key) for k, vs in axis_values.items()}
    sub_axes = _detect_sub_axes(runs, axes)

    return {
        "experiment": experiment_name,
        "axes": axes,
        "runs": runs,
        "metricNames": sorted(metric_names),
        "subAxes": sub_axes,
    }


def _load_metric_data(experiment_name: str, metric_name: str) -> dict:
    """Load one metric's values for all runs in an experiment (full history)."""
    with _lock:
        run_names = list(_experiments.get(experiment_name, []))

    result = {}
    for run_name in run_names:
        key = _run_key(experiment_name, run_name)
        with _lock:
            run_dir = _run_dirs.get(key)
        if not run_dir:
            continue

        steps = []
        values = []
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
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
                    if metric_name in rec:
                        steps.append(rec["step"])
                        v = rec[metric_name]
                        values.append(None if (isinstance(v, float) and v != v) else v)
        except OSError:
            continue

        if steps:
            result[run_name] = {"steps": steps, "values": values}

    return result


def _load_manifest(experiment_name: str) -> dict | None:
    """Load sweep_manifest.json for an experiment, or None if not present."""
    manifest_path = os.path.join(_root_dir, experiment_name, "sweep_manifest.json")
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _discover_metric_names(experiment_name: str) -> list[str]:
    """Discover metric names from first available metrics.jsonl in an experiment."""
    with _lock:
        run_names = list(_experiments.get(experiment_name, []))
    with _lock:
        run_names = list(_experiments.get(experiment_name, []))

    metric_names: set[str] = set()
    for run_name in run_names:
        key = _run_key(experiment_name, run_name)
        with _lock:
            run_dir = _run_dirs.get(key)
        if not run_dir:
            continue
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        try:
            with open(metrics_path) as f:
                first_line = f.readline()
            if first_line.strip():
                rec = json.loads(first_line)
                for k in rec:
                    if k not in ("step", "t"):
                        metric_names.add(k)
                if metric_names:
                    break  # found metrics, stop scanning
        except (OSError, json.JSONDecodeError):
            continue

    return sorted(metric_names)


def _load_metric_since(experiment_name: str, metric_name: str,
                       since_step: int) -> dict:
    """Load metric values with step strictly greater than since_step.

    Returns {run_name: {"steps": [...], "values": [...]}} with only new records.
    No server-side caching — this endpoint is always live.
    """
    with _lock:
        run_names = list(_experiments.get(experiment_name, []))

    result = {}
    for run_name in run_names:
        key = _run_key(experiment_name, run_name)
        with _lock:
            run_dir = _run_dirs.get(key)
        if not run_dir:
            continue

        steps = []
        values = []
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
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
                        values.append(None if (isinstance(v, float) and v != v) else v)
        except OSError:
            continue

        if steps:
            result[run_name] = {"steps": steps, "values": values}

    return result


# ── Watch mode ─────────────────────────────────────────────────────────────────

def _watch_loop(interval: float = 60.0) -> None:
    """Background thread: periodically scan for run_meta.json files not yet in index.

    This picks up runs that were written directly to disk (e.g. during a server
    outage) without sending /run/start. Runs forever; starts as a daemon thread.
    """
    while True:
        time.sleep(interval)
        try:
            new_runs = []
            for meta_path in sorted(Path(_root_dir).glob("*/*/run_meta.json")):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                run_name   = meta.get("run_name")
                experiment = meta.get("experiment")
                if not run_name or not experiment:
                    continue
                key = _run_key(experiment, run_name)
                with _lock:
                    if key in _run_dirs:
                        continue  # already known
                    # New run found — register it
                    run_dir = str(meta_path.parent)
                    _run_dirs[key] = run_dir
                    if experiment not in _experiments:
                        _experiments[experiment] = []
                    if run_name not in _experiments[experiment]:
                        _experiments[experiment].append(run_name)
                    status = meta.get("status", "unknown")
                    if status == "running" and _is_interrupted(meta):
                        status = "interrupted"
                    _run_status[key] = status
                    _meta_cache.pop(experiment, None)  # invalidate cache
                    new_runs.append(f"{experiment}/{run_name}")
            if new_runs:
                print(f"[watch] Discovered {len(new_runs)} new run(s): {new_runs}", flush=True)
        except Exception as e:
            print(f"[watch] Error during scan: {e}", flush=True)


# ── Server ─────────────────────────────────────────────────────────────────────

class ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-dir",
        default="outputs/sweeps",
        help="Root directory containing experiment runs (default: outputs/sweeps)",
    )
    parser.add_argument("--port", type=int, default=53800)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--watch-interval", type=float, default=60.0, metavar="SECONDS",
        help="Seconds between scans for newly-appeared run_meta.json files (default: 60)")
    parser.add_argument(
        "--token", default=None, metavar="SECRET",
        help="Optional shared token; if set, require 'Authorization: Bearer SECRET' "
             "on all POST requests (also read from MLSWEEP_TOKEN env var)")
    parser.add_argument(
        "--request-timeout", type=float, default=30.0, metavar="SECONDS",
        help="Timeout for long-running GET requests like /metric.json (default: 30)")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Log every request received and response sent")
    args = parser.parse_args()

    global _verbose
    _verbose = args.verbose

    Handler._request_timeout = args.request_timeout

    _scan_disk(args.exp_dir)

    # Start background watch thread
    wt = threading.Thread(target=_watch_loop, args=(args.watch_interval,),
                          daemon=True, name="exp-server-watch")
    wt.start()

    # Store optional token for POST auth (--token flag or MLSWEEP_TOKEN env var)
    Handler._token = args.token or os.environ.get("MLSWEEP_TOKEN")

    try:
        server = ThreadingServer((args.host, args.port), Handler)
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"Error: port {args.port} is already in use.", flush=True)
            raise SystemExit(1)
        raise
    print(
        f"Experiment server at http://{args.host}:{args.port}  (Ctrl+C to stop)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        server.shutdown()
        with _files_lock:
            for fh in list(_open_files.values()):
                try:
                    fh.close()
                except OSError:
                    pass


if __name__ == "__main__":
    main()
