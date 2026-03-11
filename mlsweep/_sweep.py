"""Sweep math: loading, variation generation, manifest/status I/O.

Extracted verbatim from run_sweep.py. No logic changes.
"""

import importlib.util
import itertools
import json
import os
import shlex
import sys
import threading
import types
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import Any

from mlsweep._shared import _val_sort_key

# Metadata keys in a dimension spec (no dot prefix). Dot-prefixed keys are subdimensions.
_METADATA_KEYS = {"values", "flags", "name", "singular", "monotonic",
                  "distribution", "min", "max", "samples"}


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


def load_sweep_file(path: str | Path) -> dict[str, Any]:
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
    run_from = getattr(mod, "RUN_FROM", None)
    if run_from is not None and not isinstance(run_from, str):
        raise ValueError(f"{path}: RUN_FROM must be a str, got {type(run_from).__name__}")
    return {
        "name": Path(path).stem,
        "options": mod.OPTIONS,
        "command": command,
        "exclude": getattr(mod, "EXCLUDE", None),
        "extra_flags": getattr(mod, "EXTRA_FLAGS", []),
        "gpus_per_run": gpus_per_run,
        "run_from": run_from,
    }


def load_sweeps() -> dict[str, dict[str, Any]]:
    """Import all sweep files from sweeps/ directory."""
    return {
        f.stem: load_sweep_file(f)
        for f in sorted((Path(os.getcwd()) / "sweeps").glob("[!_]*.py"))
    }


def validate_options(options: dict[str, Any], _ancestor_keys: frozenset[str] | None = None) -> None:
    """Validate and normalize OPTIONS dict.

    Each key in options must start with '.' to identify it as a dimension.
    Within a dim spec, dot-prefixed keys are subdimensions; non-dot keys are metadata.

    Three dimension types (determined by content):
      Value dim  — has 'values'; sweeps over that list with per-value flags.
      Subdim     — no 'values', has dot-prefixed subdim keys; each subdim is a
                   mutually-exclusive branch (the dim's implicit "values").
      Fixed dim  — no 'values', no subdims; flags are always appended (one combo).

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

        if (has_values or has_dict_flags) and has_subdims:
            raise ValueError(
                f"Dimension {key!r} has both 'values' and subdimensions {subdim_keys!r}. "
                f"Use 'values' for a value dim or dot-prefixed subdim keys for a subdim, not both."
            )

        if has_values or has_dict_flags:
            # VALUE DIM — values explicit, or inferred from flags dict keys
            values = opt["values"] if has_values else list(flags.keys())
            if flags is None:
                flags_dict: dict[Any, list[str]] = {v: [] for v in values}
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
                    validate_options(child_subdims, _ancestor_keys | current_keys)
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


def _treatment_key(combo: dict[str, Any], options: dict[str, Any]) -> tuple[Any, ...]:
    """Non-singular dims identify a treatment. Both combo and options use stripped keys (no dot)."""
    return tuple(combo[k] for k in sorted(options) if not options[k].get("singular"))


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

    Both combo and options use stripped keys (no dot prefix).

    Monotonic skip rule: if a value at position fi in _values fails, skip any candidate
    at position ci >= fi. For "decreasing", _values is reversed during validate_options
    so that conservative values come first; fi <= ci then correctly skips more aggressive
    candidates after a conservative failure.
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
            if fi <= ci:
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


# ── Display helpers ────────────────────────────────────────────────────────────


def _singular_desc(combo: dict[str, Any], options: dict[str, Any]) -> str:
    """Short description of singular dim values, e.g. 'local_batch_size=64, ac=full'.

    Both combo and options use stripped keys (no dot prefix).
    """
    return ", ".join(
        f"{k}={combo[k]}"
        for k in sorted(options) if options[k].get("singular")
    )


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
    """Write the initial sweep_manifest.json before any jobs are dispatched."""
    dims, sub_dims = _manifest_dims_from_variations(variations)
    manifest = {
        "experiment": experiment,
        "dims": dims,
        "subDims": sub_dims,
        "runs": [],         # populated as jobs are dispatched
        "metricNames": [],  # populated dynamically from metrics files
    }
    if note:
        manifest["note"] = note
    path = os.path.join(exp_dir, "sweep_manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


_manifest_lock = threading.Lock()
_status_lock = threading.Lock()


def _append_manifest_run(exp_dir: str, var: dict[str, Any]) -> None:
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


def _load_sweep_status(exp_dir: str) -> dict[str, Any]:
    """Load sweep_status.json if present. Returns {} if missing or corrupt."""
    path = os.path.join(exp_dir, "sweep_status.json")
    try:
        with open(path) as f:
            return json.load(f)  # type: ignore[no-any-return]
    except (OSError, json.JSONDecodeError):
        return {}


def _update_sweep_status(exp_dir: str, run_name: str, status: str,
                          elapsed: float, combo: dict[str, Any]) -> None:
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
