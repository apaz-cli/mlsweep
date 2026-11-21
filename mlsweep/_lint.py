"""``mlsweep lint``: find sweep-file mistakes that would otherwise pass silently.

``load_sweep_file`` rejects what it can't use, but it ignores names it doesn't
know, so ``OPTIMZE = {...}`` submits a grid sweep with no optimizer.  The lint
checks here catch that kind of mistake, plus flags two dims both pass, flag
values no run can use, and settings that contradict each other.

An error stops ``mlsweep run``; a warning is printed and the sweep goes ahead.
"""

from __future__ import annotations

import collections
import difflib
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mlsweep._sweep import (
    OPTIMIZE_KEYS,
    PRUNE_KEYS,
    SWEEP_SETTINGS,
    _load_module,
    count_expected,
    generate_variations,
    sweep_info,
    validate_options,
)

Severity = Literal["error", "warning"]

# More runs than this in a grid gets a warning: it is usually a dim too many.
LARGE_GRID = 2000


@dataclass
class Finding:
    severity: Severity
    message: str

    def __str__(self) -> str:
        return f"{self.severity}: {self.message}"


def _did_you_mean(name: str, known: frozenset[str] | set[str]) -> str | None:
    match = difflib.get_close_matches(name, sorted(known), n=1, cutoff=0.75)
    return match[0] if match else None


def _misspelled_settings(mod: types.ModuleType) -> list[Finding]:
    """Upper-case names that look like a misspelled sweep setting.

    Other upper-case names are left alone, since sweep files often keep their
    own constants (``BASE_LR = 3e-4``).  A leading underscore marks a name as
    the file's own, so it is never flagged.
    """
    out = []
    for name in vars(mod):
        if name.startswith("_") or not name.isupper() or name in SWEEP_SETTINGS:
            continue
        guess = _did_you_mean(name, SWEEP_SETTINGS)
        if guess is not None:
            out.append(Finding("error", f"{name} is not a sweep setting, so it is ignored. Did you mean "
                                        f"{guess}? (Name your own constants _{name} to silence this.)"))
    return out


def _unknown_keys(setting: str, value: Any, known: frozenset[str]) -> list[Finding]:
    if not isinstance(value, dict):
        return []
    out = []
    for key in sorted(set(value) - known):
        guess = _did_you_mean(str(key), known)
        hint = f" Did you mean {guess!r}?" if guess else f" It takes {sorted(known)}."
        out.append(Finding("error", f"{setting} has unknown key {key!r}.{hint}"))
    return out


def _walk_dims(options: dict[str, Any], path: str = "") -> list[tuple[str, dict[str, Any]]]:
    """(display path, spec) of every dim in an OPTIONS tree, subdim children included."""
    out = []
    for key, spec in options.items():
        if not isinstance(spec, dict):
            continue
        here = f"{path}{key}"
        out.append((here, spec))
        for sub_key, sub in spec.items():
            if sub_key.startswith(".") and isinstance(sub, dict):
                out += _walk_dims({k: v for k, v in sub.items() if k.startswith(".")}, f"{here}{sub_key}")
    return out


def _dim_findings(options: dict[str, Any]) -> list[Finding]:
    """Per-dim checks on the raw OPTIONS, before validation fills in defaults."""
    out = []
    for path, spec in _walk_dims(options):
        flags, values = spec.get("flags"), spec.get("values")
        if isinstance(flags, dict) and values is not None:
            unused = [k for k in flags if k not in list(values)]
            if unused:
                out.append(Finding("warning", f"dim {path}: flags given for {unused}, which are not in its "
                                              "values, so no run uses them"))
        if values is not None and len(list(values)) != len(set(map(repr, values))):
            out.append(Finding("error", f"dim {path}: values repeat, so runs would repeat"))
        if spec.get("monotonic") and values is not None and len(list(values)) < 2:
            out.append(Finding("warning", f"dim {path}: monotonic needs at least two values to skip anything"))
    return out


def _flag_tokens(overrides: list[str]) -> list[str]:
    """The option names in a run's flags (``--lr`` of ``--lr 0.1`` or ``--lr=0.1``)."""
    return [tok.split("=", 1)[0] for tok in overrides
            if tok.startswith("-") and len(tok) > 1 and not _is_number(tok)]


def _is_number(tok: str) -> bool:
    try:
        float(tok)
    except ValueError:
        return False
    return True


def _variation_findings(variations: list[dict[str, Any]]) -> list[Finding]:
    out = []
    names = collections.Counter(v["name"] for v in variations)
    dups = sorted(n for n, c in names.items() if c > 1)
    if dups:
        out.append(Finding("error", f"{len(dups)} run name(s) collide, e.g. {dups[0]}. Give each dim a "
                                    "'name' (or distinct values) so every run name is unique."))
    repeated: dict[str, str] = {}
    for var in variations:
        counts = collections.Counter(_flag_tokens(var["overrides"]))
        for flag, n in counts.items():
            if n > 1 and flag not in repeated:
                repeated[flag] = var["name"]
    for flag, run in sorted(repeated.items()):
        out.append(Finding("warning", f"{flag} is passed more than once (e.g. run {run}); two dims, or a dim "
                                      "and EXTRA_FLAGS, set it, and usually only the last one counts"))
    return out


def lint_module(mod: types.ModuleType, path: str | Path) -> list[Finding]:
    """Every finding for a loaded sweep module, errors first."""
    findings = _misspelled_settings(mod)
    key_errors = (_unknown_keys("OPTIMIZE", getattr(mod, "OPTIMIZE", None), OPTIMIZE_KEYS)
                  + _unknown_keys("PRUNE", getattr(mod, "PRUNE", None), PRUNE_KEYS))
    if key_errors:
        return _sorted(findings + key_errors)  # sweep_info would only repeat them
    try:
        info = sweep_info(mod, path)
    except ValueError as exc:
        return _sorted(findings + [Finding("error", str(exc).removeprefix(f"{path}: "))])

    optimize = info["optimize"] or {}
    if info["method"] == "grid" and "budget" in optimize:
        findings.append(Finding("error", "OPTIMIZE has a budget but no method='bayes', so it runs as a "
                                         "grid and the budget is ignored"))
    if info["metric"] is None and info["goal"] is not None:
        findings.append(Finding("warning", "GOAL is set without METRIC, so nothing is ranked by it"))
    if info["metric"] is not None and info["goal"] is None:
        findings.append(Finding("warning", f"METRIC {info['metric']!r} has no GOAL; `mlsweep best` assumes "
                                           "minimize"))

    options = info["options"]
    if not isinstance(options, dict):
        return _sorted(findings + [Finding("error", "OPTIONS must be a dict")])
    findings += _dim_findings(options)
    try:
        validate_options(options, method=info["method"])
    except (ValueError, TypeError, AttributeError) as exc:
        return _sorted(findings + [Finding("error", f"OPTIONS: {exc}")])

    if info["method"] == "grid":
        expected = count_expected(options)
        if expected > LARGE_GRID:
            findings.append(Finding("warning", f"the grid has {expected} runs"))
        else:
            try:
                variations = generate_variations(info["name"], options, info["exclude"], info["extra_flags"])
            except Exception as exc:  # EXCLUDE is user code
                return _sorted(findings + [Finding("error", f"generating runs failed: {exc!r}")])
            if not variations:
                findings.append(Finding("error", "the sweep has no runs (EXCLUDE removes every combo)"))
            findings += _variation_findings(variations)
    return _sorted(findings)


def _sorted(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: f.severity != "error")


def lint_file(path: str | Path) -> list[Finding]:
    """Every finding for the sweep file at *path*."""
    try:
        mod = _load_module(path)
    except Exception as exc:
        return [Finding("error", f"cannot load: {exc!r}")]
    return lint_module(mod, path)
