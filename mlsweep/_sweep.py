"""Sweep math: loading, variation generation, manifest I/O."""

import importlib.util
import itertools
import json
import math
import os
import random
import re
import shlex
import sys
import types
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import Any

from mlsweep._shared import _val_sort_key, validate_artifact_id, validate_run_id

# Metadata keys in a dimension spec (no dot prefix). Dot-prefixed keys are subdimensions.
_METADATA_KEYS = {"values", "flags", "name", "singular", "monotonic",
                  "distribution", "min", "max", "samples"}

# Accepted values for the top-level GOAL / OPTIMIZE goal.  A dict is used so the
# valid choices have a single definition shared by validation and the docs.
_GOALS: dict[str, str] = {"minimize": "minimize", "maximize": "maximize"}


# ── Sweep loading ──────────────────────────────────────────────────────────────


def _parse_flag_list(f: str | list[str] | None, ctx: str) -> list[str]:
    """Normalize a flags field to a list of CLI strings (subdims/fixed dims only)."""
    if f is None:
        return []
    if isinstance(f, str):
        return [f]
    if isinstance(f, list):
        return f
    raise ValueError(f"{ctx}: flags must be str or list, got {type(f).__name__}")


def _load_module(path: str | Path) -> types.ModuleType:
    path = Path(path)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Every top-level name a sweep file may set.  ``mlsweep lint`` flags a name
# close to one of these, since a misspelled setting is otherwise ignored.
SWEEP_SETTINGS = frozenset({
    "COMMAND", "OPTIONS", "EXCLUDE", "EXTRA_FLAGS", "GPUS_PER_RUN", "NODES_PER_RUN", "RUN_FROM",
    "SET_DIST_ENV", "METRIC", "GOAL", "OPTIMIZE", "SETUP_COMMAND", "UPLOAD_EXCLUDE",
    "MAX_RUNTIME", "PRUNE", "METRIC_CHECK", "DESCRIPTION",
})

# The keys each dict setting accepts.
OPTIMIZE_KEYS = frozenset({"method", "metric", "goal", "budget", "n_initial", "max_failures"})
PRUNE_KEYS = frozenset({"metric", "goal", "strategy", "min_steps", "every", "reduction_factor", "n_warmup"})
PRUNE_STRATEGIES = ("median", "asha")

# How many finished runs without the declared metric pause a sweep, by default.
DEFAULT_METRIC_CHECK_RUNS = 2


def load_sweep_file(path: str | Path) -> dict[str, Any]:
    """Load a single sweep .py file, returning a sweep-info dict."""
    return sweep_info(_load_module(path), path)


def sweep_info(mod: types.ModuleType, path: str | Path) -> dict[str, Any]:
    """The sweep-info dict of an already loaded sweep module.  Raises ValueError."""
    if not hasattr(mod, "OPTIONS"):
        raise ValueError(f"{path}: OPTIONS is required (a dict of dimensions, or {{}} for one run)")
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
    nodes_per_run = getattr(mod, "NODES_PER_RUN", 1)
    if not isinstance(nodes_per_run, int) or nodes_per_run < 1:
        raise ValueError(f"{path}: NODES_PER_RUN must be a positive integer, got {nodes_per_run!r}")
    run_from = getattr(mod, "RUN_FROM", None)
    if run_from is not None and not isinstance(run_from, str):
        raise ValueError(f"{path}: RUN_FROM must be a str, got {type(run_from).__name__}")
    set_dist_env = getattr(mod, "SET_DIST_ENV", False)
    if not isinstance(set_dist_env, bool):
        raise ValueError(f"{path}: SET_DIST_ENV must be a bool, got {type(set_dist_env).__name__}")
    metric = getattr(mod, "METRIC", None)
    if metric is not None and not isinstance(metric, str):
        raise ValueError(f"{path}: METRIC must be a str, got {type(metric).__name__}")
    goal = getattr(mod, "GOAL", None)
    if goal is not None and goal not in _GOALS:
        raise ValueError(
            f"{path}: GOAL must be one of {sorted(_GOALS)}, got {goal!r}"
        )

    optimize = getattr(mod, "OPTIMIZE", None)
    method = "grid"
    if optimize is not None:
        if not isinstance(optimize, dict):
            raise ValueError(f"{path}: OPTIMIZE must be a dict")
        unknown = sorted(set(optimize) - OPTIMIZE_KEYS)
        if unknown:
            raise ValueError(f"{path}: OPTIMIZE has unknown key(s) {unknown}; it takes {sorted(OPTIMIZE_KEYS)}")
        method = optimize.get("method", "grid")
        if method not in ("grid", "bayes"):
            raise ValueError(f"{path}: OPTIMIZE method must be 'grid' or 'bayes'")
        if method == "bayes":
            validate_optimize_config(optimize, str(path))
        # OPTIMIZE wins over the top-level METRIC/GOAL when both are present.
        metric = optimize.get("metric", metric)
        goal = optimize.get("goal", goal)

    setup_command = getattr(mod, "SETUP_COMMAND", None)
    if setup_command is not None and not isinstance(setup_command, str):
        raise ValueError(f"{path}: SETUP_COMMAND must be a str, got {type(setup_command).__name__}")
    upload_exclude = getattr(mod, "UPLOAD_EXCLUDE", [])
    if isinstance(upload_exclude, str):
        upload_exclude = [upload_exclude]
    if not isinstance(upload_exclude, (list, tuple)) or not all(isinstance(p, str) and p for p in upload_exclude):
        raise ValueError(f"{path}: UPLOAD_EXCLUDE must be a list of non-empty str patterns")

    description = getattr(mod, "DESCRIPTION", None)
    if description is not None and not isinstance(description, str):
        raise ValueError(f"{path}: DESCRIPTION must be a str, got {type(description).__name__}")

    controls = build_controls(
        max_runtime=getattr(mod, "MAX_RUNTIME", None),
        prune=getattr(mod, "PRUNE", None),
        metric_check=getattr(mod, "METRIC_CHECK", True),
        metric=metric, goal=goal, where=str(path),
    )

    return {
        "name": Path(path).stem,
        "options": mod.OPTIONS,
        "command": command,
        "exclude": getattr(mod, "EXCLUDE", None),
        "extra_flags": getattr(mod, "EXTRA_FLAGS", []),
        "gpus_per_run": gpus_per_run,
        "nodes_per_run": nodes_per_run,
        "run_from": run_from,
        "set_dist_env": set_dist_env,
        "method": method,
        "optimize": optimize,
        "metric": metric,
        "goal": goal,
        "setup_command": setup_command,
        "upload_exclude": list(upload_exclude),
        "controls": controls,
        "description": (description or "").strip() or None,
    }


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


# ── Run controls: MAX_RUNTIME, PRUNE, METRIC_CHECK ─────────────────────────────
#
# The manager enforces these while a sweep runs, so they are stored on the
# experiment as one JSON object, ``controls``:
#
#   max_runtime   seconds a run may run before it is stopped and failed
#   prune         {metric, goal, strategy, min_steps, every, reduction_factor, n_warmup}
#   metric_check  {metric, runs}: pause the sweep when its first ``runs``
#                 finished runs all logged no ``metric``
#
# The manager also keeps ``note`` in it: why it paused the sweep.

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([dhms])")
_DURATION_UNITS = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}


def parse_duration(value: Any, where: str = "duration") -> float:
    """Seconds in *value*: a number of seconds, or a string such as "90m" or "1h30m"."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        parts = _DURATION_RE.findall(text)
        if not parts or _DURATION_RE.sub("", text).strip():
            raise ValueError(f"{where}: {value!r} is not a duration (use seconds, or e.g. '90m', '1h30m')")
        seconds = sum(float(n) * _DURATION_UNITS[u] for n, u in parts)
    else:
        raise ValueError(f"{where} must be a number of seconds or a string like '2h'")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{where} must be positive")
    return seconds


def format_duration(seconds: float) -> str:
    """*seconds* as e.g. "2h", "1h30m" or "45s"."""
    s = int(round(seconds))
    out = ""
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= size:
            out += f"{s // size}{unit}"
            s %= size
    return out + (f"{s}s" if s or not out else "")


def _check_prune(prune: Any, where: str, metric: str | None, goal: str | None) -> dict[str, Any]:
    if not isinstance(prune, dict):
        raise ValueError(f"{where}: PRUNE must be a dict")
    unknown = sorted(set(prune) - PRUNE_KEYS)
    if unknown:
        raise ValueError(f"{where}: PRUNE has unknown key(s) {unknown}; it takes {sorted(PRUNE_KEYS)}")
    p_metric = prune.get("metric", metric)
    p_goal = prune.get("goal", goal)
    if not isinstance(p_metric, str) or not p_metric:
        raise ValueError(f"{where}: PRUNE needs a metric (its own 'metric', or the sweep's METRIC)")
    if p_goal not in _GOALS:
        raise ValueError(f"{where}: PRUNE needs a goal of 'minimize' or 'maximize' (its own 'goal', or GOAL)")
    strategy = prune.get("strategy", "median")
    if strategy not in PRUNE_STRATEGIES:
        raise ValueError(f"{where}: PRUNE strategy must be one of {list(PRUNE_STRATEGIES)}")
    if not _positive_int(prune.get("min_steps")):
        raise ValueError(f"{where}: PRUNE needs 'min_steps', a positive int (the first step runs are compared at)")
    out: dict[str, Any] = {"metric": p_metric, "goal": p_goal, "strategy": strategy,
                           "min_steps": prune["min_steps"]}
    every = prune.get("every", prune["min_steps"])
    if not _positive_int(every):
        raise ValueError(f"{where}: PRUNE 'every' must be a positive int")
    rf = prune.get("reduction_factor", 3)
    if not isinstance(rf, int) or isinstance(rf, bool) or rf < 2:
        raise ValueError(f"{where}: PRUNE 'reduction_factor' must be an int >= 2")
    n_warmup = prune.get("n_warmup", 5)
    if not isinstance(n_warmup, int) or isinstance(n_warmup, bool) or n_warmup < 1:
        raise ValueError(f"{where}: PRUNE 'n_warmup' must be a positive int")
    if strategy == "median":
        if "reduction_factor" in prune:
            raise ValueError(f"{where}: PRUNE 'reduction_factor' is for strategy 'asha'")
        out["every"] = every
    else:
        if "every" in prune:
            raise ValueError(f"{where}: PRUNE 'every' is for strategy 'median'; asha compares at "
                             "min_steps * reduction_factor**k")
        out["reduction_factor"] = rf
    out["n_warmup"] = n_warmup
    return out


def build_controls(
    *, max_runtime: Any = None, prune: Any = None, metric_check: Any = True,
    metric: str | None = None, goal: str | None = None, where: str = "sweep",
) -> dict[str, Any] | None:
    """The ``controls`` object for a sweep's settings, or None when it has none.  Raises ValueError."""
    out: dict[str, Any] = {}
    if max_runtime is not None:
        out["max_runtime"] = parse_duration(max_runtime, f"{where}: MAX_RUNTIME")
    if prune is not None:
        out["prune"] = _check_prune(prune, where, metric, goal)
    if metric_check is not True and metric_check is not False and not _positive_int(metric_check):
        raise ValueError(f"{where}: METRIC_CHECK must be True, False, or a positive int (runs to judge by)")
    if metric_check is not False and metric:
        runs = DEFAULT_METRIC_CHECK_RUNS if metric_check is True else metric_check
        out["metric_check"] = {"metric": metric, "runs": runs}
    return out or None


def validate_controls(raw: Any) -> dict[str, Any] | None:
    """Check a submitted ``controls`` object.  Returns it normalized; raises ValueError."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("'controls' must be an object")
    unknown = sorted(set(raw) - {"max_runtime", "prune", "metric_check"})
    if unknown:
        raise ValueError(f"controls has unknown key(s) {unknown}")
    check = raw.get("metric_check")
    if check is not None and (not isinstance(check, dict) or not isinstance(check.get("metric"), str)
                              or not _positive_int(check.get("runs"))):
        raise ValueError("controls.metric_check must be {metric: str, runs: positive int}")
    prune = raw.get("prune")
    return build_controls(
        max_runtime=raw.get("max_runtime"),
        prune=prune, metric_check=check["runs"] if check else False,
        metric=check["metric"] if check else None,
        goal=prune.get("goal") if isinstance(prune, dict) else None, where="controls",
    )


def prune_rungs(prune: dict[str, Any]) -> Generator[int, None, None]:
    """The steps a PRUNE spec compares runs at, in order (an endless generator)."""
    step = prune["min_steps"]
    while True:
        yield step
        if prune["strategy"] == "asha":
            step *= prune["reduction_factor"]
        else:
            step += prune["every"]


def validate_optimize_config(optimize: dict[str, Any], where: str = "OPTIMIZE") -> None:
    """Check the fields a Bayesian ``OPTIMIZE`` dict declares.  Raises ValueError."""
    for req in ("metric", "goal", "budget"):
        if req not in optimize:
            raise ValueError(f"{where}: OPTIMIZE requires '{req}' when method='bayes'")
    if not isinstance(optimize["metric"], str) or not optimize["metric"]:
        raise ValueError(f"{where}: OPTIMIZE metric must be a non-empty str")
    if optimize["goal"] not in _GOALS:
        raise ValueError(f"{where}: OPTIMIZE goal must be 'minimize' or 'maximize'")
    for field in ("budget", "n_initial", "max_failures"):
        value = optimize.get(field)
        if (field == "budget" or value is not None) and not _positive_int(value):
            raise ValueError(f"{where}: OPTIMIZE {field} must be a positive int")


def _sample_distribution(dist: str, lo: float, hi: float, n: int) -> list[Any]:
    """Sample n values from the specified distribution over [lo, hi]."""
    if dist == "log_uniform":
        log_lo, log_hi = math.log(lo), math.log(hi)
        return [math.exp(random.uniform(log_lo, log_hi)) for _ in range(n)]
    elif dist == "uniform":
        return [random.uniform(lo, hi) for _ in range(n)]
    else:  # int_uniform
        return [random.randint(int(lo), int(hi)) for _ in range(n)]


def validate_options(
    options: dict[str, Any],
    _ancestor_keys: frozenset[str] | None = None,
    method: str = "grid",
) -> None:
    """Validate and normalize OPTIONS dict.

    Each key in options must start with '.' to identify it as a dimension.
    Within a dim spec, dot-prefixed keys are subdimensions; non-dot keys are metadata.

    Four dimension types (determined by content):
      Value dim       — has 'values'; sweeps over that list with per-value flags.
      Subdim          — no 'values', has dot-prefixed subdim keys; each subdim is a
                        mutually-exclusive branch (the dim's implicit "values").
      Continuous dim  — has 'distribution'; samples from a continuous range.
                        In grid mode, pre-samples N values ('samples' required).
                        In bayes mode, optuna samples at runtime ('samples' forbidden).
      Fixed dim       — no 'values', no subdims; flags are always appended (one combo).

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

        flags = opt.get("flags")
        has_values = "values" in opt
        has_dict_flags = isinstance(flags, dict)
        has_subdims = bool(subdim_keys)
        has_distribution = "distribution" in opt

        if has_distribution and has_subdims:
            raise ValueError(f"Dimension {key!r} cannot have both 'distribution' and subdimensions")
        if has_distribution and has_values:
            raise ValueError(f"Dimension {key!r} cannot have both 'distribution' and 'values'")

        if has_distribution:
            dist = opt["distribution"]
            if dist not in ("uniform", "log_uniform", "int_uniform"):
                raise ValueError(
                    f"Dimension {key!r} distribution must be "
                    f"'uniform', 'log_uniform', or 'int_uniform'"
                )
            if "min" not in opt or "max" not in opt:
                raise ValueError(f"Dimension {key!r} with distribution requires 'min' and 'max'")
            if float(opt["min"]) >= float(opt["max"]):
                raise ValueError(f"Dimension {key!r}: min must be < max")

            if method == "grid":
                n_samples = opt.get("samples")
                if not isinstance(n_samples, int) or n_samples < 1:
                    raise ValueError(
                        f"Dimension {key!r}: 'samples' (int > 0) required in grid mode"
                    )
                flags_spec = opt.get("flags")
                if flags_spec is not None and not isinstance(flags_spec, str):
                    raise ValueError(
                        f"Dimension {key!r}: continuous dim flags must be a str"
                    )
                values = _sample_distribution(dist, float(opt["min"]), float(opt["max"]), n_samples)
                if flags_spec is not None:
                    flags_dict: dict[Any, list[str]] = {v: [flags_spec, str(v)] for v in values}
                else:
                    flags_dict = {v: [] for v in values}
                opt["_values"] = values
                opt["_flags"] = flags_dict
                opt["_sub_opts_map"] = {}
            else:  # bayes
                if "samples" in opt:
                    raise ValueError(f"Dimension {key!r}: 'samples' is not used in bayes mode")
                flags_spec = opt.get("flags")
                if flags_spec is not None and not isinstance(flags_spec, str):
                    raise ValueError(
                        f"Dimension {key!r}: continuous dim flags must be a str"
                    )
                opt["_type"] = "continuous"
                opt["_values"] = []
                opt["_flags"] = {}
                opt["_sub_opts_map"] = {}
            continue  # skip standard value/subdim/fixed processing

        if (has_values or has_dict_flags) and has_subdims:
            raise ValueError(
                f"Dimension {key!r} has both 'values' and subdimensions {subdim_keys!r}. "
                f"Use 'values' for a value dim or dot-prefixed subdim keys for a subdim, not both."
            )

        if has_values or has_dict_flags:
            # VALUE DIM — values explicit, or inferred from flags dict keys
            values = opt["values"] if has_values else list(flags.keys())
            if flags is None:
                raise ValueError(
                    f"Dimension {key!r}: flags=None is not supported. "
                    f"Use 'name': None to omit the dim from the run name, "
                    f"or supply flags."
                )
            elif isinstance(flags, str):
                flags_dict = {v: [flags, str(v)] for v in values}
            elif isinstance(flags, dict):
                for fv, ftokens in flags.items():
                    if not isinstance(ftokens, list):
                        raise ValueError(
                            f"Dimension {key!r} flags dict value for {fv!r} must be a list, "
                            f"got {type(ftokens).__name__!r}"
                        )
                flags_dict = dict(flags)
            else:
                raise ValueError(f"Dimension {key!r} flags must be str or dict, got {type(flags).__name__}")
            if has_values:
                for v in values:
                    if v not in flags_dict:
                        raise ValueError(f"Dimension {key!r} missing flags for value {v!r}")
            _vals = list(values)
            if m == "decreasing":
                _vals = list(reversed(_vals))
            opt["_values"] = _vals
            opt["_flags"] = flags_dict
            opt["_sub_opts_map"] = {}

        elif has_subdims:
            # SUBDIM — subdim keys are the mutually-exclusive branches
            subdim_values, flags_dict, sub_opts_map = [], {}, {}
            for sdkey in subdim_keys:
                sdspec = opt[sdkey]
                val = sdkey[1:]  # branch value name = subdim key without leading dot
                subdim_values.append(val)
                flags_dict[val] = _parse_flag_list(sdspec.get("flags"), f"Subdim {sdkey!r} in {key!r}")
                child_subdims = {k: v for k, v in sdspec.items() if k.startswith(".")}
                if child_subdims:
                    sub_opts_map[val] = child_subdims
                    for bk in child_subdims:
                        if bk in _ancestor_keys or bk in current_keys:
                            raise ValueError(
                                f"Subdim key {bk!r} (under {sdkey!r} of {key!r}) "
                                f"collides with ancestor or sibling dim"
                            )
                    validate_options(child_subdims, _ancestor_keys | current_keys, method=method)
            opt["_values"] = subdim_values
            opt["_flags"] = flags_dict
            opt["_sub_opts_map"] = sub_opts_map

        else:
            # FIXED DIM — no values, no subdims; flags always appended
            f_list = _parse_flag_list(opt.get("flags"), f"Fixed dim {key!r}")
            opt["_values"] = [None]
            opt["_flags"] = {None: f_list}
            opt["_sub_opts_map"] = {}


# ── Variation generation ───────────────────────────────────────────────────────


def _make_part(nm: str | None, val: Any) -> str | None:
    """Build a name part string from dim name and value, or None if no name/value."""
    if nm is None:
        return None
    if val is True or (isinstance(val, str) and val.lower() == "true"):
        return f"{nm}T"
    if val is False or (isinstance(val, str) and val.lower() == "false"):
        return f"{nm}F"
    return f"{nm}{val}"


def _flatten_tokens(tokens: list[Any]) -> list[str]:
    """Flatten a name_tokens list to a plain list of strings."""
    parts = []
    for tok in tokens:
        if isinstance(tok, list):
            parts.extend(tok)
        else:
            parts.append(tok)
    return parts


def _build_level_tokens(all_keys: list[str], vals: Any, options: dict[str, Any], contributing_keys: Any = (), child_tokens: Any = ()) -> list[Any]:
    """Build name tokens for one level of the expansion tree.

    contributing_keys: keys whose selected branch has child sub-dims (child_tokens
                       are attributed to them, dotted onto this level's name part).
    child_tokens:      name tokens from the recursive child expansion.
    """
    tokens: list[Any] = []
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


def _expand_tree(options: dict[str, Any], combo_so_far: dict[str, Any], effective_so_far: dict[str, Any]) -> Generator[tuple[dict[str, Any], dict[str, Any], list[Any]], None, None]:
    """Recursively expand an options tree, yielding (combo, effective_options, name_tokens).

    options: dict with dot-prefixed dimension keys (e.g. {".treatment": {...}}).
    combo keys and effective_options keys use dim names WITHOUT the leading dot.

    name_tokens is a list of:
      - str       — a simple name part (from a non-branching dim)
      - list[str] — a dot group: [parent_part, *child_parts], joined with '.'

    Non-singular (lex) dims vary fastest in sorted order.
    Singular (diag) dims vary slowest in diagonal order.
    Subdims expand their selected branch's child dims as additional dims.
    """
    lex_keys = [k for k in options if not options[k].get("singular")]
    diag_keys = [k for k in options if options[k].get("singular")]
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


def generate_variations(sweep_name: str, options: dict[str, Any], exclude_fn: Callable[[dict[str, Any]], bool] | None = None, extra_flags: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Generate all config variations using tree expansion.

    Singular dims vary slowest (diagonal order — advances all singular dims
    at roughly the same rate).  Non-singular dims vary fastest (lex order —
    interleaves treatments for better parallel probing).

    Subdims expand their selected branch's child dims as additional dims.
    Run names use '.' to signal subdim ancestry and '_' to separate peer dims.
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


def count_expected(options: dict[str, Any]) -> int:
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
            subdim_sum = sum(count_expected(sub) for sub in sub_opts_map.values())
            n *= (n_vals - len(sub_opts_map)) + subdim_sum
    return n


# ── Skip logic ─────────────────────────────────────────────────────────────────


def should_skip(combo: dict[str, Any], failed: list[dict[str, Any]], succeeded: list[dict[str, Any]], options: dict[str, Any]) -> bool:
    """Check monotonic (skip worse on failure) and singular (skip others on success).

    Both combo and options use stripped keys (no dot prefix).  To check many
    combos against the same results, build one ``SkipIndex`` instead.
    """
    return SkipIndex(failed, succeeded, options).skips(combo)


def _freeze(v: Any) -> Any:
    """A hashable stand-in for a JSON value that is equal exactly when the value is."""
    if isinstance(v, dict):
        return ("__dict__", tuple(sorted((k, _freeze(x)) for k, x in v.items())))
    if isinstance(v, list):
        return ("__list__", tuple(_freeze(x) for x in v))
    return v


class SkipIndex:
    """The monotonic and singular skip rules over a set of finished combos.

    Monotonic: if a value at position fi in ``_values`` fails, skip any candidate
    at position ci >= fi whose other dims all match.  For "decreasing",
    ``_values`` is reversed during validate_options so that conservative values
    come first; fi <= ci then correctly skips more aggressive candidates after a
    conservative failure.

    Singular: once a value succeeds, skip the other values of that dim whose
    non-singular dims match (multiple singular dims resolve independently).

    The results are grouped by the dims that must match, so ``skips`` costs a
    few lookups however many results there are.
    """

    def __init__(self, failed: list[dict[str, Any]], succeeded: list[dict[str, Any]], options: dict[str, Any]) -> None:
        self._options = options
        self._monotonic = [k for k, o in options.items() if o.get("monotonic")]
        self._singular = [k for k, o in options.items() if o.get("singular")]
        self._others = {k: [d for d in options if d != k] for k in self._monotonic}
        nonsingular = [d for d in options if not options[d].get("singular")]
        self._others.update({k: [d for d in nonsingular if d != k] for k in self._singular})
        # (dim, other dims' values) → lowest failing position / set of succeeding values
        self._first_fail: dict[tuple[str, tuple[Any, ...]], int] = {}
        self._succeeded: dict[tuple[str, tuple[Any, ...]], set[Any]] = {}
        for fc in failed:
            for key in self._monotonic:
                fi = self._position(key, fc)
                if fi is not None:
                    group = (key, self._key(key, fc))
                    self._first_fail[group] = min(fi, self._first_fail.get(group, fi))
        for sc in succeeded:
            for key in self._singular:
                self._succeeded.setdefault((key, self._key(key, sc)), set()).add(_freeze(sc.get(key)))

    def _key(self, dim: str, combo: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(_freeze(combo.get(d)) for d in self._others[dim])

    def _position(self, dim: str, combo: dict[str, Any]) -> int | None:
        try:
            return int(self._options[dim]["_values"].index(combo[dim]))
        except (ValueError, TypeError, KeyError):
            return None

    def skips(self, combo: dict[str, Any]) -> bool:
        for key in self._monotonic:
            fi = self._first_fail.get((key, self._key(key, combo)))
            if fi is not None:
                ci = self._position(key, combo)
                if ci is not None and fi <= ci:
                    return True
        for key in self._singular:
            values = self._succeeded.get((key, self._key(key, combo)))
            if values and (len(values) > 1 or _freeze(combo.get(key)) not in values):
                return True
        return False


# ── Display helpers ────────────────────────────────────────────────────────────


# ── Manifest & status helpers ──────────────────────────────────────────────────


def _manifest_dims_from_variations(variations: list[dict[str, Any]]) -> tuple[dict[str, list[Any]], dict[str, dict[str, Any]]]:
    """Extract dims and sub_dims from a list of variations.

    Returns (dims, sub_dims_fmt) where:
      dims:          {dim_name: [sorted values]}
      sub_dims_fmt:  {dim_name: {"parentDim": ..., "parentValue": ...}}
    """
    dim_values: dict[str, set[Any]] = {}
    all_names: set[str] = set()
    names_with: dict[str, set[str]] = {}

    for var in variations:
        combo = var["combo"]
        all_names.add(var["name"])
        for k, v in combo.items():
            dim_values.setdefault(k, set()).add(v)
            names_with.setdefault(k, set()).add(var["name"])

    dims = {k: sorted(vs, key=_val_sort_key) for k, vs in dim_values.items()}

    # Detect subdims: dims that only appear when a parent has a specific value
    sub_dims_fmt: dict[str, dict[str, Any]] = {}
    for dim in dims:
        if names_with.get(dim) == all_names:
            continue  # universal dim
        for parent_dim in dims:
            if parent_dim == dim:
                continue
            for parent_val in dims[parent_dim]:
                names_with_parent = {
                    var["name"] for var in variations
                    if var["combo"].get(parent_dim) == parent_val
                }
                if names_with_parent == names_with.get(dim, set()):
                    sub_dims_fmt[dim] = {
                        "parentDim": parent_dim,
                        "parentValue": parent_val,
                    }
                    break
            if dim in sub_dims_fmt:
                break

    return dims, sub_dims_fmt


def _write_manifest(exp_dir: str, experiment: str, variations: list[dict[str, Any]], note: str | None = None) -> None:
    """Write sweep_manifest.json with one run entry per variation."""
    dims, sub_dims = _manifest_dims_from_variations(variations)
    manifest = {
        "experiment": experiment,
        "dims": dims,
        "subDims": sub_dims,
        "runs": [{"name": v["name"], "hash": v["name"], "combo": v["combo"]} for v in variations],
        "metricNames": [],  # populated dynamically from metrics files
    }
    if note:
        manifest["note"] = note
    path = os.path.join(exp_dir, "sweep_manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


# ── Bayesian sweep specs ───────────────────────────────────────────────────────
#
# The manager runs a Bayesian sweep on its own, so it stores everything needed
# to make more runs: an ``optimize`` spec on the experiment.  The spec is JSON.
# The fields set when the sweep is submitted:
#
#   metric, goal, budget, n_initial, max_failures   what to optimize, how far
#   sweep_name, options, extra_flags                what a trial looks like
#   job                                             build_job_payloads' arguments
#
# The manager also keeps these in it:
#
#   finishing            start no new trials, complete once the running ones end
#   failures_cleared_at  failures before this time don't count as a failure streak
#   note                 why the manager paused or completed the sweep


def options_to_wire(options: dict[str, Any]) -> dict[str, Any]:
    """OPTIONS as JSON values, without the fields ``validate_options`` synthesizes.

    A dict of flags may have keys JSON can't (64, True), so it goes as pairs.
    Raises ValueError if anything else in OPTIONS would not survive JSON.
    """
    def encode(opts: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, opt in opts.items():
            spec: dict[str, Any] = {}
            for k, v in opt.items():
                if k.startswith("."):
                    spec.update(encode({k: v}))
                elif k == "flags" and isinstance(v, dict):
                    spec[k] = {"pairs": [[fk, list(tokens)] for fk, tokens in v.items()]}
                elif k == "values":
                    spec[k] = list(v)
                elif not k.startswith("_"):
                    spec[k] = v
            out[key] = spec
        return out

    wire = encode(options)
    try:
        same = json.loads(json.dumps(wire, allow_nan=False)) == wire
    except (TypeError, ValueError):
        same = False
    if not same:
        raise ValueError("a Bayesian sweep's OPTIONS must hold only plain JSON values "
                         "(str, number, bool, None, list, dict), since the manager keeps them")
    return wire


def options_from_wire(wire: dict[str, Any]) -> dict[str, Any]:
    """Invert ``options_to_wire``.  The result still needs ``validate_options``."""
    out: dict[str, Any] = {}
    for key, spec in wire.items():
        opt = dict(spec)
        for k, v in spec.items():
            if k.startswith("."):
                opt[k] = options_from_wire({k: v})[k]
            elif k == "flags" and isinstance(v, dict):
                opt[k] = {fk: tokens for fk, tokens in v["pairs"]}
        out[key] = opt
    return out


def singular_dims(options: dict[str, Any]) -> list[str]:
    """Names of the top-level singular dims of validated *options*."""
    return [k[1:] for k, v in options.items() if v.get("singular")]


def lex_key(combo: dict[str, Any], singular: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """What identifies a Bayesian trial: *combo* without its singular dims."""
    return tuple((k, str(combo[k])) for k in sorted(combo) if k not in singular)


def _default_n_initial(options: dict[str, Any]) -> int:
    n_dims = sum(1 for v in options.values() if not v.get("singular") and v.get("_values") != [None])
    return max(8, 2 * n_dims)


def _check_job_template(job: Any) -> dict[str, Any]:
    if not isinstance(job, dict):
        raise ValueError("optimize.job must be an object")
    command = job.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(c, str) for c in command):
        raise ValueError("optimize.job.command must be a non-empty list of strings")
    overrides = job.get("cli_overrides", [])
    if not isinstance(overrides, list) or not all(isinstance(c, str) for c in overrides):
        raise ValueError("optimize.job.cli_overrides must be a list of strings")
    numbers = {"gpus_per_run": (1, 0), "nodes_per_run": (1, 1), "priority": (0, None), "max_retries": (2, 0)}
    out: dict[str, Any] = {"command": command, "cli_overrides": overrides}
    for field, (default, low) in numbers.items():
        value = job.get(field, default)
        if not isinstance(value, int) or isinstance(value, bool) or (low is not None and value < low):
            raise ValueError(f"optimize.job.{field} must be an int" + (f" >= {low}" if low is not None else ""))
        out[field] = value
    for field in ("artifact_id", "run_from", "setup_command"):
        value = job.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"optimize.job.{field} must be a string")
        out[field] = value
    if out["artifact_id"] is not None:
        validate_artifact_id(out["artifact_id"])
    if not isinstance(job.get("set_dist_env", False), bool):
        raise ValueError("optimize.job.set_dist_env must be a bool")
    out["set_dist_env"] = job.get("set_dist_env", False)
    return out


def validate_optimize_spec(raw: Any) -> dict[str, Any]:
    """Check a submitted ``optimize`` spec.  Returns it with defaults filled in.

    Only the fields a sweep declares are kept.  Raises ValueError.
    """
    if not isinstance(raw, dict):
        raise ValueError("'optimize' must be an object")
    if raw.get("method", "bayes") != "bayes":
        raise ValueError("optimize.method must be 'bayes'")
    validate_optimize_config(raw, "optimize")
    name = raw.get("sweep_name")
    if not isinstance(name, str):
        raise ValueError("optimize.sweep_name must be a string")
    validate_run_id(f"{name}_bayes_0001")
    extra_flags = raw.get("extra_flags", [])
    if not isinstance(extra_flags, list) or not all(isinstance(f, str) for f in extra_flags):
        raise ValueError("optimize.extra_flags must be a list of strings")
    wire = raw.get("options")
    if not isinstance(wire, dict):
        raise ValueError("optimize.options must be an object")
    try:
        options = options_from_wire(wire)
        validate_options(options, method="bayes")
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"optimize.options is malformed: {exc!r}") from exc
    for key, opt in options.items():
        if opt.get("singular") and opt.get("_type") == "continuous":
            raise ValueError(f"Dimension {key!r}: a singular dim needs a list of 'values', not a distribution")
    n_initial = raw.get("n_initial") or _default_n_initial(options)
    return {
        "method": "bayes",
        "metric": raw["metric"],
        "goal": raw["goal"],
        "budget": raw["budget"],
        "n_initial": n_initial,
        "max_failures": raw.get("max_failures") or max(3, n_initial),
        "sweep_name": name,
        "options": wire,
        "extra_flags": extra_flags,
        "job": _check_job_template(raw.get("job")),
    }


def spec_singular_dims(spec: dict[str, Any]) -> list[str]:
    """The singular dims of a stored ``optimize`` spec."""
    return [k[1:] for k, v in spec["options"].items() if v.get("singular")]


# ── Job payloads and results ───────────────────────────────────────────────────


def build_job_payloads(
    variations: list[dict[str, Any]],
    experiment_id: str,
    artifact_id: str | None,
    command: list[str],
    cli_overrides: list[str],
    gpus_per_run: int,
    nodes_per_run: int,
    set_dist_env: bool,
    run_from: str | None,
    priority: int,
    max_retries: int,
    setup_command: str | None = None,
) -> list[dict[str, Any]]:
    """Convert sweep variations into job payloads for the manager API.

    Each payload is a dict with keys matching the JobRecord fields expected
    by POST /api/jobs/bulk.  A run's command is ``COMMAND + EXTRA_FLAGS +
    OPTIONS flags + -- overrides``; each variation's ``overrides`` already
    starts with EXTRA_FLAGS.
    """
    jobs = []
    for var in variations:
        full_command = list(command) + var["overrides"] + list(cli_overrides)
        env_dict: dict[str, str] = {}
        tag_parts = [f"{k}={v}" for k, v in var["combo"].items() if v is not None]
        if tag_parts:
            env_dict["EXP_TAGS"] = ",".join(tag_parts)

        job = {
            "run_id": var["name"],
            "experiment_id": experiment_id,
            "priority": priority,
            "command": full_command,
            "combo": var["combo"],
            "env": env_dict,
            "status": "pending",
            "gpus_per_run": gpus_per_run,
            "nodes_per_run": nodes_per_run,
            "set_dist_env": set_dist_env,
            "run_from": run_from,
            "artifact_id": artifact_id,
            "max_retries": max_retries,
            "return_files": [],  # could configure via sweep file
        }
        if setup_command:
            job["setup_command"] = setup_command
        jobs.append(job)
    return jobs


def metric_values(metrics: list[dict[str, Any]] | None, metric: str) -> list[float]:
    """Finite values of *metric* across metric rows, in step order."""
    return [
        float(r[metric])
        for r in (metrics or [])
        if isinstance(r, dict) and isinstance(r.get(metric), (int, float))
        and math.isfinite(float(r[metric]))
    ]


def best_metric(metrics: list[dict[str, Any]] | None, metric: str, goal: str) -> float | None:
    """Best finite value of *metric* under *goal*, or None if it was never logged."""
    vals = metric_values(metrics, metric)
    if not vals:
        return None
    return min(vals) if goal == "minimize" else max(vals)
