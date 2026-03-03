#!/usr/bin/env python3

"""Run experiment sweeps. See docs/sweep_configuration.md for format and usage."""

import argparse
import importlib.util
import itertools
import json
import os
import queue
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ANSI colors
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_RESET = "\033[0m"
_DIM_COLORS = [_CYAN, _YELLOW, _MAGENTA, _BLUE]

# Metadata keys in a dimension spec (no dot prefix). Dot-prefixed keys are subdimensions.
_METADATA_KEYS = {"values", "flags", "name", "singular", "monotonic"}

_log_file = None


def _parse_flag_list(f, ctx):
    """Normalize a flags field to a list of CLI strings (branch/fixed dims only)."""
    if f is None:
        return []
    if isinstance(f, str):
        return [f]
    if isinstance(f, list):
        return f
    raise ValueError(f"{ctx}: flags must be str or list, got {type(f).__name__}")


def sweep_print(msg, end="\n"):
    """Print to stdout (colored) and log file (plain)."""
    print(msg, end=end, flush=True)
    if _log_file is not None:
        _log_file.write(re.sub(r"\033\[[0-9;]*m", "", msg) + end)
        _log_file.flush()


def fmt_time(s):
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.0f}m"
    return f"{int(s // 3600)}h {int((s % 3600) // 60)}m"


# ── Sweep loading ──────────────────────────────────────────────────────────


def _load_module(path):
    path = Path(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_sweep_file(path):
    """Load a single sweep .py file, returning a sweep-info dict."""
    mod = _load_module(path)
    command = getattr(mod, "COMMAND", None)
    if command is None:
        raise ValueError(f"{path}: COMMAND is required (e.g. COMMAND = ['python', 'train.py'])")
    if isinstance(command, str):
        command = shlex.split(command)
    elif not isinstance(command, list):
        raise ValueError(f"{path}: COMMAND must be a str or list, got {type(command).__name__}")
    gpus_per_run = getattr(mod, "GPUS_PER_RUN", 1)
    if not isinstance(gpus_per_run, int) or gpus_per_run < 1:
        raise ValueError(f"{path}: GPUS_PER_RUN must be a positive integer, got {gpus_per_run!r}")
    return {
        "name": Path(path).stem,
        "options": mod.OPTIONS,
        "command": command,
        "exclude": getattr(mod, "EXCLUDE", None),
        "extra_flags": getattr(mod, "EXTRA_FLAGS", []),
        "abbrev": getattr(mod, "ABBREV", None),
        "gpus_per_run": gpus_per_run,
    }


def load_sweeps():
    """Import all sweep files from sweeps/ directory."""
    return {
        f.stem: load_sweep_file(f)
        for f in sorted((Path(os.getcwd()) / "sweeps").glob("[!_]*.py"))
    }


def validate_options(options, _ancestor_keys=None):
    """Validate and normalize OPTIONS dict.

    Each key in options must start with '.' to identify it as a dimension.
    Within a dim spec, dot-prefixed keys are subdimensions; non-dot keys are metadata.

    Three dimension types (determined by content):
      Value dim   — has 'values'; sweeps over that list with per-value flags.
      Branch dim  — no 'values', has dot-prefixed subdim keys; each subdim is a
                    mutually-exclusive branch (the dim's implicit "values").
      Fixed dim   — no 'values', no subdims; flags are always appended (one combo).

    Synthesizes _values, _flags, _sub_opts_map on each dim spec for use by
    _expand_tree and related functions.
    """
    if _ancestor_keys is None:
        _ancestor_keys = frozenset()
    current_keys = frozenset(options.keys())

    for key, opt in options.items():
        if not key.startswith("."):
            raise ValueError(
                f"Dimension key {key!r} must start with '.' to mark it as a dimension "
                f"(metadata keys inside a dim spec have no dot)"
            )
        subdim_keys = [k for k in opt if k.startswith(".")]
        for mk in opt:
            if not mk.startswith(".") and not mk.startswith("_") and mk not in _METADATA_KEYS:
                raise ValueError(f"Unknown metadata key {mk!r} in dimension {key!r}")

        m = opt.get("monotonic")
        if m is not None and m not in ("increasing", "decreasing"):
            raise ValueError(f"Dimension {key!r} monotonic must be 'increasing'|'decreasing'|None")

        has_values = "values" in opt
        has_subdims = bool(subdim_keys)

        if has_values and has_subdims:
            raise ValueError(
                f"Dimension {key!r} has both 'values' and subdimensions {subdim_keys!r}. "
                f"Use 'values' for a value dim or dot-prefixed subdim keys for a branch dim, not both."
            )

        if has_values:
            # VALUE DIM — explicit list of values
            values = opt["values"]
            flags = opt.get("flags")
            if flags is None:
                flags_dict = {v: [] for v in values}
            elif isinstance(flags, str):
                flags_dict = {v: [flags, str(v)] for v in values}
            elif isinstance(flags, dict):
                flags_dict = dict(flags)
            else:
                raise ValueError(f"Dimension {key!r} flags must be str or dict, got {type(flags).__name__}")
            for v in values:
                if v not in flags_dict:
                    raise ValueError(f"Dimension {key!r} missing flags for value {v!r}")
            opt["_values"] = list(values)
            opt["_flags"] = flags_dict
            opt["_sub_opts_map"] = {}

        elif has_subdims:
            # BRANCH DIM — subdim keys are the mutually-exclusive branches
            branch_values, flags_dict, sub_opts_map = [], {}, {}
            for sdkey in subdim_keys:
                sdspec = opt[sdkey]
                val = sdkey[1:]  # branch value name = subdim key without leading dot
                branch_values.append(val)
                flags_dict[val] = _parse_flag_list(sdspec.get("flags"), f"Branch {sdkey!r} in {key!r}")
                branch_subdims = {k: v for k, v in sdspec.items() if k.startswith(".")}
                if branch_subdims:
                    sub_opts_map[val] = branch_subdims
                    for bk in branch_subdims:
                        if bk in _ancestor_keys or bk in current_keys:
                            raise ValueError(
                                f"Subdim key {bk!r} (under branch {sdkey!r} of {key!r}) "
                                f"collides with ancestor or sibling dim"
                            )
                    validate_options(branch_subdims, _ancestor_keys | current_keys)
            opt["_values"] = branch_values
            opt["_flags"] = flags_dict
            opt["_sub_opts_map"] = sub_opts_map

        else:
            # FIXED DIM — no values, no subdims; flags always appended
            f_list = _parse_flag_list(opt.get("flags"), f"Fixed dim {key!r}")
            opt["_values"] = [None]
            opt["_flags"] = {None: f_list}
            opt["_sub_opts_map"] = {}


# ── Variation generation ───────────────────────────────────────────────────


def _make_part(nm, val):
    """Build a name part string from dim name and value, or None if no name/value."""
    if nm is None:
        return None
    if isinstance(val, bool):
        return f"{nm}{'T' if val else 'F'}"
    return f"{nm}{val}"


def _flatten_tokens(tokens):
    """Flatten a name_tokens list to a plain list of strings."""
    parts = []
    for tok in tokens:
        if isinstance(tok, list):
            parts.extend(tok)
        else:
            parts.append(tok)
    return parts


def _build_level_tokens(all_keys, vals, options, contributing_keys=(), child_tokens=()):
    """Build name tokens for one level of the expansion tree.

    contributing_keys: keys whose selected branch has child sub-dims (child_tokens
                       are attributed to them, dotted onto this level's name part).
    child_tokens:      name tokens from the recursive child expansion.
    """
    tokens = []
    for key, val in zip(all_keys, vals):
        nm = options[key].get("name", key[1:])
        part = _make_part(nm, val)
        if key in contributing_keys:
            if part is not None:
                tokens.append([part] + _flatten_tokens(child_tokens))
            else:
                # No name for this dim: pass child tokens through flat
                tokens.extend(child_tokens)
        elif part is not None:
            tokens.append(part)
    return tokens


def _expand_tree(options, combo_so_far, effective_so_far):
    """Recursively expand an options tree, yielding (combo, effective_options, name_tokens).

    options: dict with dot-prefixed dimension keys (e.g. {".treatment": {...}}).
    combo keys and effective_options keys use dim names WITHOUT the leading dot.

    name_tokens is a list of:
      - str       — a simple name part (from a non-branching dim)
      - list[str] — a dot group: [parent_part, *child_parts], joined with '.'

    Non-singular (lex) dims vary fastest in sorted order.
    Singular (diag) dims vary slowest in diagonal order.
    Branch dims expand their selected branch's sub-dims as additional dims.
    """
    lex_keys = sorted(k for k in options if not options[k].get("singular"))
    diag_keys = sorted(k for k in options if options[k].get("singular"))
    all_keys = lex_keys + diag_keys

    if not all_keys:
        yield combo_so_far, effective_so_far, []
        return

    lex_combos = (list(itertools.product(*(options[k]["_values"] for k in lex_keys)))
                  if lex_keys else [()])

    if diag_keys:
        raw = list(itertools.product(*(range(len(options[k]["_values"])) for k in diag_keys)))
        raw.sort(key=lambda idx: (sum(idx), idx))
        diag_combos = [
            tuple(options[diag_keys[i]]["_values"][j] for i, j in enumerate(idx))
            for idx in raw
        ]
    else:
        diag_combos = [()]

    for dv in diag_combos:
        for lv in lex_combos:
            vals = lv + dv
            # Combo and effective keys strip the leading dot from dim keys
            combo = {**combo_so_far, **{k[1:]: v for k, v in zip(all_keys, vals)}}
            effective = {**effective_so_far, **{k[1:]: v for k, v in options.items()}}

            # Collect sub-options from dims whose selected value has branch sub-dims
            sub_opts = {}
            contributing_keys = set()
            for key, val in zip(all_keys, vals):
                opt = options[key]
                children = opt["_sub_opts_map"].get(val, {})
                if children:
                    sub_opts.update(children)
                    contributing_keys.add(key)

            if sub_opts:
                for child_combo, child_effective, child_tokens in _expand_tree(
                    sub_opts, combo, effective
                ):
                    yield child_combo, child_effective, _build_level_tokens(
                        all_keys, vals, options, contributing_keys, child_tokens)
            else:
                yield combo, effective, _build_level_tokens(all_keys, vals, options)


def generate_variations(sweep_name, options, exclude_fn=None, extra_flags=()):
    """Generate all config variations using tree expansion.

    Singular dims vary slowest (diagonal order — advances all singular dims
    at roughly the same rate).  Non-singular dims vary fastest (lex order —
    interleaves treatments for better parallel probing).

    Branch dims expand their selected branch's sub-dims as additional dims.
    Run names use '.' to signal branch ancestry and '_' to separate peer dims.
    """
    variations = []
    for combo, effective, name_tokens in _expand_tree(options, {}, {}):
        if exclude_fn and exclude_fn(combo):
            continue
        # Flatten name tokens: dot groups → "parent.child", peers → "_"-joined
        segments = []
        for tok in name_tokens:
            if isinstance(tok, list):
                segments.append(".".join(tok))
            else:
                segments.append(tok)
        name = f"{sweep_name}_{'_'.join(segments) if segments else 'default'}"
        # Build CLI overrides from all dims in this combo
        overrides = list(extra_flags)
        for key, val in combo.items():
            if key in effective:
                overrides.extend(effective[key]["_flags"].get(val, []))
        variations.append({
            "name": name,
            "overrides": overrides,
            "combo": combo,
            "effective_options": effective,
        })
    return variations


def _treatment_key(combo, options):
    """Non-singular dims identify a treatment. Both combo and options use stripped keys (no dot)."""
    return tuple(combo[k] for k in sorted(options) if not options[k].get("singular"))


def count_expected(options):
    """Expected runs, computed recursively over the options tree.

    Singular dims contribute 1 (we only need one working value).
    Non-singular dims with no sub-options multiply by their value count.
    Non-singular dims with sub-options: non-branching values count as 1,
    branching values multiply by their subtree count.
    """
    n = 1
    for key, opt in options.items():
        if opt.get("singular"):
            continue
        n_vals = len(opt["_values"])
        sub_opts_map = opt["_sub_opts_map"]
        if not sub_opts_map:
            n *= n_vals
        else:
            branch_sum = sum(count_expected(sub) for sub in sub_opts_map.values())
            n *= (n_vals - len(sub_opts_map)) + branch_sum
    return n


# ── Skip logic ─────────────────────────────────────────────────────────────


def should_skip(combo, failed, succeeded, options):
    """Check monotonic (skip worse on failure) and singular (skip others on success).

    Both combo and options use stripped keys (no dot prefix).
    """
    # Monotonic: skip values worse than a known failure (all other dims must match)
    for fc in failed:
        for key, opt in options.items():
            m = opt.get("monotonic")
            if not m:
                continue
            if not all(fc.get(k) == combo.get(k) for k in options if k != key):
                continue
            vals = opt["_values"]
            try:
                fi, ci = vals.index(fc[key]), vals.index(combo[key])
            except (ValueError, TypeError, KeyError):
                continue
            if (m == "increasing" and fi <= ci) or (m == "decreasing" and fi >= ci):
                return True

    # Singular: skip different values once one succeeds
    # Only require non-singular dims to match (multiple singular dims resolve independently)
    for sc in succeeded:
        for key, opt in options.items():
            if not opt.get("singular"):
                continue
            if not all(sc.get(k) == combo.get(k)
                       for k in options if k != key and not options[k].get("singular")):
                continue
            if sc.get(key) != combo.get(key):
                return True

    return False


# ── Execution ──────────────────────────────────────────────────────────────




def _visible_devices():
    """Get list of visible GPU device indices."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        devs = []
        for p in cvd.split(","):
            p = p.strip()
            if "-" in p:
                a, b = p.split("-", 1)
                devs.extend(range(int(a), int(b) + 1))
            else:
                devs.append(int(p))
        return devs
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True)
        return [int(x) for x in r.stdout.strip().splitlines() if x.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return [0]


def _discover_remote_gpus(workers):
    """SSH to each worker, count GPUs. Returns [(worker, gpu_id), ...]."""
    slots = []
    for w in workers:
        try:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", w,
                 "nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                sweep_print(f"  {_RED}WARN{_RESET}  Cannot reach {w}: {result.stderr.strip()}")
                continue
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line:
                    slots.append((w, int(line)))
        except (subprocess.TimeoutExpired, ValueError) as e:
            sweep_print(f"  {_RED}WARN{_RESET}  Cannot reach {w}: {e}")
            continue
    return slots


def _topo_score(conn_type):
    """Convert an nvidia-smi topology connection type to a numeric score (higher = better)."""
    if conn_type.startswith("NV"):
        try:
            return 100 + int(conn_type[2:])  # NV12/NV18 -> 12, 18, etc.
        except ValueError:
            return 100
    return {"PIX": 50, "PXB": 40, "PHB": 30, "NODE": 20, "SYS": 10}.get(conn_type, 0)


def _parse_topo_output(text):
    """Parse `nvidia-smi topo -m` stdout. Returns {(gpu_a, gpu_b): score}."""
    lines = [l for l in text.splitlines() if l.strip()]
    # Header line starts with whitespace (tab) before GPU0
    col_gpus = None
    for line in lines:
        if "GPU0" in line and not line[0].isalpha():
            col_gpus = [int(m.group(1)) for m in re.finditer(r"GPU(\d+)", line)]
            break
    if not col_gpus:
        return {}
    scores = {}
    for line in lines:
        if not line.startswith("GPU"):
            continue
        parts = line.split()
        try:
            row_gpu = int(parts[0][3:])
        except (ValueError, IndexError):
            continue
        for ci, col_gpu in enumerate(col_gpus):
            if col_gpu == row_gpu or ci + 1 >= len(parts):
                continue
            val = parts[ci + 1]
            if val != "X":
                scores[(row_gpu, col_gpu)] = _topo_score(val)
    return scores


def _gpu_topology(worker=None):
    """Query GPU interconnect topology via `nvidia-smi topo -m`.

    Returns {(gpu_a, gpu_b): score} where higher score means better connectivity
    (NVLink >> PCIe switch >> PCIe host bridge >> NUMA >> cross-NUMA).
    Falls back to {} if the query fails or nvidia-smi is unavailable.
    worker: None for local; SSH target string for remote.
    """
    cmd = ["nvidia-smi", "topo", "-m"]
    if worker:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", worker] + cmd
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return {}
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {}
    return _parse_topo_output(r.stdout)


def _best_gpu_groups(devices, group_size, n_groups, worker=None):
    """Select n_groups non-overlapping groups of group_size GPUs from devices,
    preferring groups with the best NVLink/PCIe interconnect.

    Uses a greedy algorithm: seeds each group with the highest-scoring pair,
    then expands by adding the GPU that maximises total score to the existing group.
    Falls back to sequential grouping when topology is unavailable (all scores zero).
    worker=None means local; SSH target string for remote topology query.
    """
    if group_size == 1:
        return [[d] for d in devices[:n_groups]]

    topo = _gpu_topology(worker)

    def pair_score(a, b):
        return topo.get((a, b), 0) + topo.get((b, a), 0)

    available = list(devices)
    groups = []

    for _ in range(n_groups):
        if len(available) < group_size:
            break
        # Seed with the highest-scoring pair (O(n^2), fine for ≤64 GPUs)
        best_pair = (available[0], available[1] if len(available) > 1 else available[0])
        best_pair_score = -1
        for a, b in itertools.combinations(available, 2):
            s = pair_score(a, b)
            if s > best_pair_score:
                best_pair_score, best_pair = s, (a, b)
        # Greedily expand to group_size
        group = list(best_pair)
        remaining = [d for d in available if d not in set(group)]
        while len(group) < group_size and remaining:
            best_g = max(remaining, key=lambda g: sum(pair_score(g, e) for e in group))
            group.append(best_g)
            remaining.remove(best_g)

        if len(group) < group_size:
            break
        groups.append(group)
        used = set(group)
        available = [g for g in available if g not in used]

    return groups


def _parse_workers(workers_arg):
    """Parse --workers argument: comma-separated list or @file."""
    if workers_arg.startswith("@"):
        path = workers_arg[1:]
        with open(path) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [w.strip() for w in workers_arg.split(",") if w.strip()]


def _get_default_exp_server():
    """Get default exp server address (this machine's hostname + default port)."""
    try:
        # Get the first non-loopback IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return f"{ip}:53800"
    except Exception:
        return f"{socket.gethostname()}:53800"


def _run_env_dict(device, run_dir, name, experiment, tag_str, exp_server):
    """Common env vars for a training run, as a plain dict."""
    env = {
        "CUDA_VISIBLE_DEVICES": device,
        "HIP_VISIBLE_DEVICES": device,
        "MLSWEEP_RUN_DIR": run_dir,
        "MLSWEEP_RUN_NAME": name,
        "EXP_EXPERIMENT": experiment,
    }
    if tag_str:
        env["EXP_TAGS"] = tag_str
    if exp_server:
        env["EXP_SERVER"] = f"http://{exp_server}"
    return env


def _run_one(command, var, output_dir, experiment, extra, gpu, log,
             worker=None, remote_dir=None, exp_server=None, sync_artifacts=False):
    """Execute one training run. Returns (var, success, elapsed, log_file).

    When worker is None, runs locally.
    When worker is an SSH target (e.g. 'user@host'), dispatches via SSH.
    command: list[str] from the sweep file's COMMAND.
    """
    name = var["name"]
    run_dir = os.path.join(output_dir, experiment, name)
    os.makedirs(run_dir, exist_ok=True)

    cmd = list(command) + var["overrides"] + list(extra)
    tag_str = ",".join(f"{k}={v}" for k, v in var["combo"].items())

    device = ",".join(str(g) for g in gpu)
    t0 = time.time()
    try:
        if worker is None:
            # === LOCAL execution ===
            run_vars = _run_env_dict(device, run_dir, name, experiment, tag_str, exp_server)
            env = {**os.environ, **run_vars}

            with open(log, "w", buffering=1) as f:
                subprocess.run(cmd, env=env, stdout=f,
                               stderr=subprocess.STDOUT, check=True)
        else:
            # === REMOTE execution via SSH ===
            remote_run_dir = os.path.join(remote_dir, "outputs", "sweeps", experiment, name)
            remote_log = os.path.join(remote_run_dir, "training.log")

            run_vars = _run_env_dict(device, remote_run_dir, name, experiment, tag_str, exp_server)
            env_parts = [f"{k}={shlex.quote(str(v))}" for k, v in run_vars.items()]
            if os.environ.get("MLSWEEP_TOKEN"):
                env_parts.append(f"MLSWEEP_TOKEN={shlex.quote(os.environ['MLSWEEP_TOKEN'])}")
            env_str = " ".join(env_parts)

            remote_cmd_str = " ".join(shlex.quote(c) for c in cmd)

            remote_cmd = (
                f"mkdir -p {shlex.quote(remote_run_dir)} && "
                f"cd {shlex.quote(remote_dir)} && "
                f"{env_str} {remote_cmd_str}"
                f" 2>&1 | tee {shlex.quote(remote_log)}"
            )

            with open(log, "w", buffering=1) as f:
                subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                     worker, remote_cmd],
                    stdout=f, stderr=subprocess.STDOUT, check=True)

            # Optional: sync artifacts (checkpoints, profiling) back to controller
            if sync_artifacts:
                subprocess.run(
                    ["rsync", "-a", "--exclude=training.log",
                     f"{worker}:{remote_run_dir}/", f"{run_dir}/"],
                    capture_output=True)

        return var, True, time.time() - t0, log
    except subprocess.CalledProcessError:
        return var, False, time.time() - t0, log


def _singular_desc(combo, options, abbrev=None):
    """Short description of singular dim values, e.g. 'bs=64, ac=full'.

    Both combo and options use stripped keys (no dot prefix).
    abbrev: optional dict mapping dim names to short labels (configurable per sweep).
    """
    _abbrev = abbrev or {}
    return ", ".join(
        f"{_abbrev.get(k, k[:4])}={combo[k]}"
        for k in sorted(options) if options[k].get("singular")
    )


# ── Manifest & status helpers ──────────────────────────────────────────────────


def _manifest_axes_from_variations(variations):
    """Extract axes and sub_axes from a list of variations.

    Returns (axes, sub_axes_fmt) where:
      axes:          {dim_name: [sorted values]}
      sub_axes_fmt:  {dim_name: {"parentAxis": ..., "parentValue": ...}}
    """
    axis_values: dict[str, set] = {}
    all_names: set = set()
    names_with: dict[str, set] = {}

    for var in variations:
        combo = var["combo"]
        all_names.add(var["name"])
        for k, v in combo.items():
            axis_values.setdefault(k, set()).add(v)
            names_with.setdefault(k, set()).add(var["name"])

    def val_sort_key(v):
        if isinstance(v, bool):
            return (0, str(v))
        if isinstance(v, (int, float)):
            return (1, v)
        return (2, str(v))

    axes = {k: sorted(vs, key=val_sort_key) for k, vs in axis_values.items()}

    # Detect sub-axes: axes that only appear when a parent has a specific value
    sub_axes_fmt: dict = {}
    for axis in axes:
        if names_with.get(axis) == all_names:
            continue  # universal axis
        for parent_axis in axes:
            if parent_axis == axis:
                continue
            for parent_val in axes[parent_axis]:
                names_with_parent = {
                    var["name"] for var in variations
                    if var["combo"].get(parent_axis) == parent_val
                }
                if names_with_parent == names_with.get(axis, set()):
                    sub_axes_fmt[axis] = {
                        "parentAxis": parent_axis,
                        "parentValue": parent_val,
                    }
                    break
            if axis in sub_axes_fmt:
                break

    return axes, sub_axes_fmt


def _write_manifest(exp_dir: str, experiment: str, variations: list) -> None:
    """Write the initial sweep_manifest.json before any jobs are dispatched."""
    axes, sub_axes = _manifest_axes_from_variations(variations)
    manifest = {
        "experiment": experiment,
        "axes": axes,
        "subAxes": sub_axes,
        "runs": [],         # populated as jobs are dispatched
        "metricNames": [],  # populated dynamically by server
    }
    path = os.path.join(exp_dir, "sweep_manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


_manifest_lock = threading.Lock()
_status_lock = threading.Lock()


def _append_manifest_run(exp_dir: str, var: dict) -> None:
    """Append a dispatched run entry to sweep_manifest.json (thread-safe)."""
    path = os.path.join(exp_dir, "sweep_manifest.json")
    with _manifest_lock:
        try:
            with open(path) as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        manifest["runs"].append({"name": var["name"], "hash": var["name"], "combo": var["combo"]})
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, path)


def _load_sweep_status(exp_dir: str) -> dict:
    """Load sweep_status.json if present. Returns {} if missing or corrupt."""
    path = os.path.join(exp_dir, "sweep_status.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _update_sweep_status(exp_dir: str, run_name: str, status: str,
                          elapsed: float, combo: dict) -> None:
    """Append/update a run's entry in sweep_status.json (thread-safe)."""
    path = os.path.join(exp_dir, "sweep_status.json")
    with _status_lock:
        try:
            with open(path) as f:
                status_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            status_data = {}
        status_data[run_name] = {
            "status": status,
            "elapsed": round(elapsed, 2),
            "combo": combo,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(status_data, f, indent=2)
        os.replace(tmp, path)


# ── Sweep execution ────────────────────────────────────────────────────────


def run_sweep(variations, command, output_dir, experiment, expected, extra,
              gpu_slots, jobs_per_gpu, remote_dir=None, exp_server=None,
              sync_artifacts=False, exp_dir=None, abbrev=None):
    """Execute all variations. Single code path for sequential and parallel.

    gpu_slots: list of (worker, gpu_group) tuples. worker=None means local; gpu_group is a list of GPU ids.
    expected:  expected number of resolved treatments (from count_expected).
    command:   list[str] from the sweep file's COMMAND.
    exp_dir:   experiment output directory (for manifest/status updates).
    abbrev:    optional abbreviation dict for singular dim display.
    """
    num_slots = len(gpu_slots) * jobs_per_gpu
    has_singular = any(
        o.get("singular")
        for var in variations
        for o in var["effective_options"].values()
    )

    # Shared state (protected by lock)
    results = []          # (var, success, elapsed, log_file)
    failed, succeeded = [], []
    resolved = set()      # treatment keys with at least one success
    lock = threading.Lock()
    t0 = time.time()
    pad = max((len(v["name"]) for v in variations), default=0)

    # GPU pool: each slot appears jobs_per_gpu times
    # Slots are (worker, gpu_id) tuples — worker=None for local
    gpu_q = queue.Queue()
    for worker, gpu_id in gpu_slots:
        for _ in range(jobs_per_gpu):
            gpu_q.put((worker, gpu_id))

    skipped_count = 0

    def _job_worker(var):
        nonlocal skipped_count
        worker, gpu = gpu_q.get()
        opts = var["effective_options"]

        # Pre-flight: re-check skip in case treatment was resolved while queued
        with lock:
            if should_skip(var["combo"], failed, succeeded, opts):
                gpu_q.put((worker, gpu))
                skipped_count += 1
                return

        nm = var["name"].ljust(pad)
        sdesc = _singular_desc(var["combo"], opts, abbrev) if has_singular else ""
        # Compute log path here so we can print it at START
        run_dir = os.path.join(output_dir, experiment, var["name"])
        log = os.path.join(run_dir, "training.log")
        n = 0
        while os.path.exists(log):
            n += 1
            log = os.path.join(run_dir, f"training.{n}.log")
        gpu_str = ",".join(str(g) for g in gpu)
        gpu_label = f"gpu{'s' if len(gpu) > 1 else ''} {gpu_str}"
        with lock:
            if worker:
                loc = f"{_MAGENTA}{worker}{_RESET} {gpu_label}"
            else:
                loc = gpu_label
            if sdesc:
                sweep_print(f"  {_CYAN}START{_RESET}  {_GREEN}{nm}{_RESET} ({sdesc}) {loc} {_BLUE}{log}{_RESET}")
            else:
                sweep_print(f"  {_CYAN}START{_RESET}  {_GREEN}{nm}{_RESET} {loc} {_BLUE}{log}{_RESET}")

        # Record run in manifest as dispatched
        if exp_dir:
            _append_manifest_run(exp_dir, var)

        job_t0 = time.time()
        try:
            _, ok, elapsed, log = _run_one(
                command, var, output_dir, experiment, extra, gpu, log,
                worker=worker, remote_dir=remote_dir, exp_server=exp_server,
                sync_artifacts=sync_artifacts)
        except Exception:
            ok, elapsed, log = False, time.time() - job_t0, "?"
        finally:
            gpu_q.put((worker, gpu))

        # Update sweep_status.json
        if exp_dir:
            _update_sweep_status(exp_dir, var["name"],
                                  "ok" if ok else "failed", elapsed, var["combo"])

        with lock:
            results.append((var, ok, elapsed, log))
            tk = _treatment_key(var["combo"], opts)
            (succeeded if ok else failed).append(var["combo"])
            if ok:
                resolved.add(tk)
            nr = len(resolved)

            if ok:
                tag = f"{_GREEN}   OK{_RESET}"
            elif has_singular and tk not in resolved:
                tag = f"{_YELLOW}PROBE{_RESET}"
            else:
                tag = f"{_RED} FAIL{_RESET}"
            if worker:
                loc = f"{_MAGENTA}{worker}{_RESET} {gpu_label}"
            else:
                loc = gpu_label
            sweep_print(f"  {tag}  {_GREEN}{nm}{_RESET} {loc} {_BLUE}{log}{_RESET} {elapsed:.1f}s [{nr}/{expected} resolved]")

        return

    def _drain_completed():
        """Process any already-completed futures to update state before skip checks."""
        for f in [f for f in futures if f.done()]:
            f.result()
            del futures[f]

    inflight = {}  # treatment_key -> most recent future for that treatment

    with ThreadPoolExecutor(max_workers=num_slots) as pool:
        futures = {}
        remaining = list(variations)
        while remaining:
            deferred = []
            for var in remaining:
                _drain_completed()

                # Defer if previous run of same treatment is still in-flight
                tk = _treatment_key(var["combo"], var["effective_options"])
                prev = inflight.get(tk)
                if prev is not None and not prev.done():
                    deferred.append(var)
                    continue

                # Wait for a slot if pool is full
                while len(futures) >= num_slots:
                    done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()
                        del futures[f]

                # Check skip with most up-to-date state
                with lock:
                    skip = should_skip(var["combo"], failed, succeeded,
                                       var["effective_options"])
                if skip:
                    skipped_count += 1
                    continue

                f = pool.submit(_job_worker, var)
                futures[f] = var
                inflight[tk] = f

            remaining = deferred
            if remaining and futures:
                done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()
                    del futures[f]

        # Drain remaining futures
        for f in list(futures):
            f.result()

    return results, skipped_count, time.time() - t0


# ── Summary ────────────────────────────────────────────────────────────────


def print_summary(results, skipped, elapsed, abbrev=None):
    """Print per-treatment summary."""
    has_singular = any(
        o.get("singular")
        for var, _, _, _ in results
        for o in var["effective_options"].values()
    )

    # Group by treatment
    treatments = {}
    for var, ok, el, log in results:
        tk = _treatment_key(var["combo"], var["effective_options"])
        treatments.setdefault(tk, []).append((var, ok, el, log))

    ok_count = sum(1 for runs in treatments.values() if any(s for _, s, _, _ in runs))
    total_treatments = len(treatments)
    total_runs = len(results)
    probe_fails = sum(
        sum(1 for _, s, _, _ in runs if not s)
        for runs in treatments.values() if any(s for _, s, _, _ in runs)
    ) if has_singular else 0

    parts = [f"{total_runs} runs"]
    if probe_fails:
        parts.append(f"{probe_fails} probe failures")
    if skipped:
        parts.append(f"{skipped} skipped")

    sweep_print(f"\n{'=' * 80}")
    sweep_print(f"SUMMARY — {ok_count}/{total_treatments} OK in {fmt_time(elapsed)}"
                f" ({', '.join(parts)})")
    sweep_print(f"{'=' * 80}")

    for tk in sorted(treatments, key=lambda t: tuple('' if v is None else str(v) for v in t)):
        runs = treatments[tk]
        successes = [(v, e, lf) for v, s, e, lf in runs if s]
        if successes:
            best_var, best_el, best_log = min(successes, key=lambda x: x[1])
            sdesc = _singular_desc(best_var["combo"], best_var["effective_options"], abbrev) if has_singular else ""
            n_probes = sum(1 for _, s, _, _ in runs if not s)
            pnote = f", {n_probes} probes" if n_probes else ""
            sweep_print(f"  {_GREEN}   OK{_RESET}  {best_var['name']:<40}"
                        f" {sdesc:<20} ({best_el:.1f}s{pnote})")
            sweep_print(f"         {_BLUE}{best_log}{_RESET}")
        else:
            last_var, _, _, last_log = runs[-1]
            sweep_print(f"  {_RED} FAIL{_RESET}  {last_var['name']:<40}"
                        f" (all {len(runs)} combos failed)")
            sweep_print(f"         {_BLUE}{last_log}{_RESET}")

    return ok_count < total_treatments


# ── Main ───────────────────────────────────────────────────────────────────


def _setup_gpu_slots(args, gpus_per_run, exp_server=None):
    """Resolve and return (gpu_slots, exp_server).

    Remote mode: discovers GPUs via SSH, groups them topology-aware per worker.
    Local mode: uses visible GPUs, groups them topology-aware locally.
    """
    if args.workers:
        # Remote mode: discover GPUs on each worker via SSH
        worker_list = _parse_workers(args.workers)
        sweep_print(f"Discovering GPUs on {len(worker_list)} workers...")
        gpu_slots = _discover_remote_gpus(worker_list)
        if not gpu_slots:
            sweep_print(f"{_RED}Error: no reachable workers with GPUs{_RESET}")
            sys.exit(1)
        if exp_server is None:
            exp_server = _get_default_exp_server()
        # Group raw [(worker, gpu_id)] by worker
        worker_gpu_map = {}
        for w, g in gpu_slots:
            worker_gpu_map.setdefault(w, []).append(g)
        # Optional per-worker GPU cap (--gpus)
        if args.gpus is not None and args.gpus > 0:
            worker_gpu_map = {w: gpus[:args.gpus] for w, gpus in worker_gpu_map.items()}
        # Topology-aware slot building per worker
        gpu_slots = []
        for w in sorted(worker_gpu_map):
            worker_gpus = worker_gpu_map[w]
            n_slots = len(worker_gpus) // gpus_per_run
            if n_slots == 0:
                sweep_print(f"  {_YELLOW}WARN{_RESET}  {w}: only {len(worker_gpus)} GPU(s), "
                            f"need {gpus_per_run} per run — skipping")
                continue
            groups = _best_gpu_groups(worker_gpus, gpus_per_run, n_slots, worker=w)
            gpu_slots.extend((w, grp) for grp in groups)
        if not gpu_slots:
            sweep_print(f"{_RED}Error: no workers with enough GPUs for GPUS_PER_RUN={gpus_per_run}{_RESET}")
            sys.exit(1)
        # Show discovered topology
        worker_slot_counts = Counter(w for w, _ in gpu_slots)
        for w, cnt in sorted(worker_slot_counts.items()):
            total_gpus = cnt * gpus_per_run
            slot_word = "slot" if cnt == 1 else "slots"
            sweep_print(f"  {_GREEN}OK{_RESET}    {w}: {cnt} {slot_word} ({total_gpus} GPUs)")
        sweep_print(f"Exp server: {exp_server}")
        return gpu_slots, exp_server
    else:
        # Local mode
        visible = _visible_devices()
        if not visible:
            visible = [0]
        if args.gpus is None:
            num_gpus = gpus_per_run   # default: one slot
        elif args.gpus == 0:
            num_gpus = len(visible)   # all visible
        elif args.gpus > len(visible):
            sweep_print(f"Error: requested {args.gpus} GPUs but only {len(visible)} visible")
            sys.exit(1)
        else:
            num_gpus = args.gpus
        if num_gpus < gpus_per_run:
            sweep_print(f"Error: need at least {gpus_per_run} GPUs (GPUS_PER_RUN) "
                        f"but only {num_gpus} requested")
            sys.exit(1)
        if num_gpus % gpus_per_run != 0:
            sweep_print(f"Warning: {num_gpus} GPUs is not a multiple of "
                        f"GPUS_PER_RUN={gpus_per_run}; "
                        f"using {(num_gpus // gpus_per_run) * gpus_per_run}")
        n_slots = num_gpus // gpus_per_run
        usable = n_slots * gpus_per_run
        groups = _best_gpu_groups(visible[:usable], gpus_per_run, n_slots)
        return [(None, grp) for grp in groups], exp_server


def main():
    global _log_file

    # Shebang mode: first arg is a .py sweep file
    argv = sys.argv[1:]
    sweep_name = options = command = exclude_fn = sweeps = None
    extra_flags = []
    abbrev = None
    gpus_per_run = 1
    if argv and argv[0].endswith(".py") and os.path.isfile(argv[0]):
        info = load_sweep_file(argv[0])
        sweep_name, options, command, exclude_fn, extra_flags, abbrev = (
            info["name"], info["options"], info["command"],
            info["exclude"], info["extra_flags"], info.get("abbrev"))
        gpus_per_run = info.get("gpus_per_run", 1)
        validate_options(options)
        argv = argv[1:]

    parser = argparse.ArgumentParser(
        description="Run experiment sweeps",
        epilog="Extra args after -- are passed to every training run.")
    if sweep_name is None:
        sweeps = load_sweeps()
        parser.add_argument("--sweep", required=True, choices=sorted(sweeps),
                            help="Sweep name")
    parser.add_argument("--output_dir", default=os.path.join(os.getcwd(), "outputs", "sweeps"),
                        help="Output directory")
    parser.add_argument("--experiment", default=None,
                        help="Experiment name (default: <sweep>_<timestamp>)")
    parser.add_argument("--gpus", "-g", type=int, nargs="?", const=0, default=None,
                        help="Total GPUs to use (0 = all visible; default = GPUS_PER_RUN or 1)")
    parser.add_argument("--jobs-per-gpu", "-j", type=int, default=1,
                        help="Concurrent jobs per GPU (default: 1)")
    parser.add_argument("--workers", default=None,
                        help="SSH targets for remote dispatch (comma-separated or @file)")
    parser.add_argument("--remote-dir", default=None,
                        help="Repo path on workers (default: same as local)")
    parser.add_argument("--exp-server", default=None,
                        help="Exp tracking server host:port (default: auto-detect for remote workers)")
    parser.add_argument("--sync-artifacts", action="store_true",
                        help="Rsync dump_folder back from workers after each job")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs already recorded as completed in sweep_status.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without execution")

    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    if sweep_name is None:
        info = sweeps[args.sweep]
        sweep_name, options, command = args.sweep, info["options"], info["command"]
        exclude_fn  = info.get("exclude")
        extra_flags = info.get("extra_flags", [])
        abbrev      = info.get("abbrev")
        gpus_per_run = info.get("gpus_per_run", 1)
        validate_options(options)

    # GPU setup: build list of (worker, gpu_id) slots
    remote_dir = args.remote_dir or os.getcwd()
    exp_server = args.exp_server
    gpu_slots, exp_server = _setup_gpu_slots(args, gpus_per_run, exp_server)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment = args.experiment or f"{sweep_name}_{timestamp}"
    output_dir = os.path.abspath(args.output_dir)
    exp_dir = os.path.join(output_dir, experiment)
    os.makedirs(exp_dir, exist_ok=True)

    if not args.dry_run:
        _log_file = open(os.path.join(exp_dir, "sweep.log"), "w")

    all_variations = generate_variations(sweep_name, options, exclude_fn, extra_flags)
    expected = count_expected(options)

    # Resume: skip variations that already completed in a prior run
    prior_status = {}
    resumed_count = 0
    if args.resume:
        prior_status = _load_sweep_status(exp_dir)
        if prior_status:
            completed = {name for name, s in prior_status.items() if s.get("status") == "ok"}
            original_n = len(all_variations)
            all_variations = [v for v in all_variations if v["name"] not in completed]
            resumed_count = original_n - len(all_variations)
            if resumed_count:
                sweep_print(f"Resume: skipping {resumed_count} already-completed runs")

    variations = all_variations
    n = len(variations)

    num_total_slots = len(gpu_slots) * args.jobs_per_gpu

    # Header
    sweep_print(f"Command: {' '.join(command)}")
    if expected < n:
        sweep_print(f"Sweep: {sweep_name} ({expected} expected, {n} worst case)")
    else:
        sweep_print(f"Sweep: {sweep_name} ({n} runs)")
    sweep_print(f"Experiment: {experiment}")
    if args.workers:
        sweep_print(f"GPU slots: {len(gpu_slots)}, jobs/GPU: {args.jobs_per_gpu},"
                    f" workers: {num_total_slots}")
    else:
        sweep_print(f"GPUs: {len(gpu_slots)}, jobs/GPU: {args.jobs_per_gpu},"
                    f" workers: {num_total_slots}")
    if gpus_per_run > 1:
        sweep_print(f"GPUs/run: {gpus_per_run}")
    if extra:
        sweep_print(f"Extra overrides: {' '.join(extra)}")
    if not args.dry_run:
        sweep_print(f"Sweep log: {os.path.join(exp_dir, 'sweep.log')}")
    sweep_print(f"Output: {output_dir}\n")

    # List all variations with colored flags
    for var in variations:
        colored = []
        for ci, (key, val) in enumerate(var["combo"].items()):
            flags = var["effective_options"].get(key, {}).get("_flags", {}).get(val, [])
            if flags:
                color = _DIM_COLORS[ci % len(_DIM_COLORS)]
                colored.append(f"{color}{' '.join(flags)}{_RESET}")
        sweep_print(f"{_GREEN}{var['name']}{_RESET}: {' '.join(colored)}")
        sweep_print(f"{' '.join(list(command) + var['overrides'] + list(extra))}\n")

    if args.dry_run:
        sweep_print(f"\n{'=' * 80}")
        if expected < n:
            sweep_print(f"DRY RUN — {expected} expected ({n} worst case)")
        else:
            sweep_print(f"DRY RUN — {n} runs would be executed")
        sweep_print(f"{'=' * 80}")
        return

    # Write sweep_manifest.json before dispatching any jobs
    _write_manifest(exp_dir, experiment, variations)

    run_desc = f"{expected} runs, {n} possible" if expected < n else f"{n} runs"
    sweep_print(f"\n{'=' * 80}")
    sweep_print(f"Starting sweep - ({run_desc})")
    sweep_print(f"{'=' * 80}\n")

    results, skipped, elapsed = run_sweep(
        variations, command, output_dir, experiment, expected, extra,
        gpu_slots, args.jobs_per_gpu, remote_dir=remote_dir,
        exp_server=exp_server, sync_artifacts=args.sync_artifacts,
        exp_dir=exp_dir, abbrev=abbrev)

    has_failures = print_summary(results, skipped, elapsed, abbrev=abbrev)
    sweep_print(f"\nOutput: {output_dir}")
    if _log_file:
        _log_file.close()
        print(f"Sweep log: {os.path.join(exp_dir, 'sweep.log')}")

    if has_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
