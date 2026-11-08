"""Data-backed blocks in a note: charts, tables, and evidence links.

A note asks the manager for data with a fenced block:

```mlsweep-chart
experiments: baseline, warmup
metric: val_loss
split_by: lr
band: minmax
```

The spec is a small ``key: value`` format (JSON is accepted too), parsed here
so the dashboard, the lint and the command line all agree on it.  The manager
evaluates it against its metrics database, which makes the block a live query
rather than a pasted image.

An inline ``[0.42](mlsweep:EXP/RUN/val_loss@20000)`` ties a number in prose to
the metric it came from.  ``resolve_evidence`` fetches that value so the tab
can check the number and offer the curve.

Building a series or a table is pure once the rows are loaded, so the builders
can be tested without a manager.  Only ``select_runs``, ``load_rows`` and
``resolve_evidence`` touch the database.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from aiosqlite import Connection

from mlsweep._manager_db import (
    JobRecord,
    experiments_for_run,
    get_job,
    list_experiment_meta,
    list_jobs_by_experiment,
    metric_rows_by_experiment,
    metric_rows_for_run,
    parse_metric_rows,
)

MAX_SPEC_BYTES = 64 * 1024
MAX_RUNS = 500
DEFAULT_CHART_RUNS = 60
DEFAULT_TABLE_RUNS = 300
DEFAULT_POINTS = 400
MAX_POINTS = 2000
MAX_EVIDENCE_REFS = 500

CHART_KEYS = {
    "experiments", "runs", "metric", "x", "x_min", "x_max", "split_by", "color_by",
    "band", "agg", "smooth", "log_x", "log_y", "points", "limit", "where", "caption", "title",
}
TABLE_KEYS = {
    "experiments", "runs", "metric", "metrics", "columns", "agg", "sort", "order",
    "limit", "where", "caption", "title",
}
_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_EVIDENCE_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")
_EVIDENCE_AT_RE = re.compile(r"^[A-Za-z0-9_.+%\-]{1,32}$")
_EVIDENCE_METRIC_RE = re.compile(r"^[^\s@#|\[\]()]{1,256}$")
_FIG_ID_RE = re.compile(r"\s*\{#([A-Za-z][\w:.\-]*)\}\s*$")


class SpecError(ValueError):
    """A block spec the manager cannot evaluate."""


@dataclass
class RunRef:
    """One run a block selected, with what a chart needs to label it."""

    experiment: str
    run_id: str
    status: str
    combo: dict[str, Any]
    label: str | None = None
    experiment_name: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.experiment, self.run_id)


# ── spec parsing ───────────────────────────────────────────────────────────────


def _split_list(value: str) -> list[str]:
    """Split ``a, b, "c, d"`` into items, honouring quotes and brackets."""
    items: list[str] = []
    buf: list[str] = []
    quote = ""
    depth = 0
    for ch in value:
        if quote:
            if ch == quote:
                quote = ""
            else:
                buf.append(ch)
        elif ch in "\"'":
            quote = ch
        elif ch in "[(":
            depth += 1
            buf.append(ch)
        elif ch in "])":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            item = "".join(buf).strip()
            if item:
                items.append(item)
            buf = []
        else:
            buf.append(ch)
    item = "".join(buf).strip()
    if item:
        items.append(item)
    return items


def _scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    low = value.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if _NUM_RE.match(value):
        return float(value) if any(c in value for c in ".eE") else int(value)
    if value.startswith("[") and value.endswith("]"):
        return [_scalar(v) for v in _split_list(value[1:-1])]
    if value.startswith("{") and value.endswith("}"):
        return _where(value[1:-1])
    return value


def _where(value: str) -> dict[str, Any]:
    """``lr=0.001, optimizer=muon`` (or JSON, or ``{lr: 0.001}``) as a filter dict."""
    value = value.strip()
    if value.startswith("{"):
        try:
            data = json.loads(value)
        except ValueError:
            inner = value[1:-1] if value.endswith("}") else value[1:]
            if ":" not in inner or "=" in inner:
                raise SpecError("where: expected key=value or valid JSON") from None
            out = {}
            for part in _split_list(inner):
                key, sep, val = part.partition(":")
                if not sep:
                    raise SpecError(f"where: expected key: value, got {part!r}")
                out[key.strip().strip("\"'")] = _scalar(val)
            return out
        if not isinstance(data, dict):
            raise SpecError("where: expected a mapping")
        return {str(k): v for k, v in data.items()}
    out = {}
    for part in _split_list(value):
        key, sep, val = part.partition("=")
        if not sep:
            raise SpecError(f"where: expected key=value, got {part!r}")
        out[key.strip()] = _scalar(val)
    return out


def _as_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = _split_list(str(value))
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            out.append(text)
    if not out:
        raise SpecError(f"{name}: expected at least one value")
    return out


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SpecError(f"{name}: expected a number") from None


def _as_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise SpecError(f"{name}: expected an integer") from None
    if n < minimum or n > maximum:
        raise SpecError(f"{name}: must be between {minimum} and {maximum}")
    return n


def parse_spec(raw: str) -> dict[str, Any]:
    """A block body as a dict.  Raises :class:`SpecError` on anything unusable."""
    if not isinstance(raw, str):
        raise SpecError("spec must be text")
    if len(raw.encode("utf-8", "replace")) > MAX_SPEC_BYTES:
        raise SpecError("spec is too large")
    text = raw.strip()
    if not text:
        raise SpecError("the block is empty")
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise SpecError(f"invalid JSON: {exc}") from None
        if not isinstance(data, dict):
            raise SpecError("the spec must be a mapping")
        return {str(k).strip().lower().replace("-", "_"): v for k, v in data.items()}
    if text.startswith("```") or text.startswith("~~~"):
        raise SpecError("write the keys directly, without a nested code fence")
    spec: dict[str, Any] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.partition(":")
        if not sep:
            raise SpecError(f"line {lineno}: expected `key: value`, got {line!r}")
        spec[key.strip().lower().replace("-", "_")] = _scalar(val)
    if not spec:
        raise SpecError("the block has no keys")
    return spec


def normalize_spec(raw: str | dict[str, Any], kind: str) -> tuple[dict[str, Any], list[str]]:
    """Validate and fill in defaults for a ``chart`` or ``table`` spec."""
    if isinstance(raw, str):
        spec = parse_spec(raw)
    elif isinstance(raw, dict):
        spec = {str(k).strip().lower().replace("-", "_"): v for k, v in raw.items()}
    else:
        raise SpecError("spec must be text or a mapping")
    allowed = CHART_KEYS if kind == "chart" else TABLE_KEYS
    warnings = [f"unknown key `{k}`" for k in spec if k not in allowed]
    out: dict[str, Any] = {}
    out["experiments"] = _as_list(spec.get("experiments"), "experiments")
    out["runs"] = _as_list(spec.get("runs"), "runs")
    if not out["experiments"] and not out["runs"]:
        raise SpecError("name `experiments` or `runs`")
    if "where" in spec:
        where = spec["where"]
        if isinstance(where, str):
            where = _where(where)
        if not isinstance(where, dict):
            raise SpecError("where: expected a mapping")
        out["where"] = {str(k).strip(): v for k, v in where.items()}
    if "metric" in spec:
        out["metric"] = str(spec["metric"]).strip()
        if not out["metric"]:
            raise SpecError("metric: expected a name")
    if kind == "chart":
        if not out.get("metric"):
            raise SpecError("chart needs `metric`")
        out["x"] = str(spec.get("x") or "step").strip()
        if not out["x"]:
            raise SpecError("x: expected a name")
        split = spec.get("split_by", spec.get("color_by"))
        out["split_by"] = str(split).strip() if split not in (None, "") else None
        band = str(spec.get("band") or ("minmax" if out["split_by"] else "none")).lower()
        if band not in ("none", "minmax", "std"):
            raise SpecError("band: expected none, minmax or std")
        out["band"] = band
        agg = str(spec.get("agg") or "mean").lower()
        if agg not in ("mean", "median"):
            raise SpecError("agg: expected mean or median")
        out["agg"] = agg
        smooth = _as_float(spec.get("smooth", 0) or 0, "smooth")
        if not 0 <= smooth < 1:
            raise SpecError("smooth: expected 0 ≤ smooth < 1")
        out["smooth"] = smooth
        out["log_x"] = _as_bool(spec.get("log_x", False))
        out["log_y"] = _as_bool(spec.get("log_y", False))
        for key in ("x_min", "x_max"):
            if spec.get(key) not in (None, ""):
                out[key] = _as_float(spec[key], key)
        out["points"] = _as_int(spec.get("points", DEFAULT_POINTS), "points",
                               minimum=2, maximum=MAX_POINTS)
        out["limit"] = _as_int(spec.get("limit", DEFAULT_CHART_RUNS), "limit",
                               minimum=1, maximum=MAX_RUNS)
    else:
        metrics = spec.get("metrics")
        if metrics is None:
            metrics = out.get("metric")
        out["metrics"] = _as_list(metrics, "metrics") if metrics not in (None, "") else []
        out["columns"] = _as_list(spec.get("columns"), "columns") if spec.get("columns") not in (None, "") else []
        agg = str(spec.get("agg") or "final").lower()
        if agg in ("last", "latest"):
            agg = "final"
        if agg not in ("final", "min", "max", "mean"):
            raise SpecError("agg: expected final, min, max or mean")
        out["agg"] = agg
        out["sort"] = str(spec["sort"]).strip() if spec.get("sort") not in (None, "") else None
        order = str(spec.get("order") or "asc").lower()
        if order not in ("asc", "desc"):
            raise SpecError("order: expected asc or desc")
        out["order"] = order
        out["limit"] = _as_int(spec.get("limit", DEFAULT_TABLE_RUNS), "limit",
                               minimum=1, maximum=MAX_RUNS)
    caption = spec.get("caption")
    if caption is not None:
        caption = str(caption).strip()
        m = _FIG_ID_RE.search(caption)
        if m:
            out["id"] = m.group(1)
            caption = caption[:m.start()].rstrip()
        if caption:
            out["caption"] = caption
    if spec.get("title") not in (None, ""):
        out["title"] = str(spec["title"]).strip()
    return out, warnings


# ── run selection ─────────────────────────────────────────────────────────────


def _combo(job: JobRecord) -> dict[str, Any]:
    try:
        data = json.loads(job.combo or "{}")
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _same(value: Any, wanted: Any) -> bool:
    if isinstance(value, bool) or isinstance(wanted, bool):
        return bool(value) == bool(wanted)
    if isinstance(value, (int, float)) and isinstance(wanted, (int, float)):
        return float(value) == float(wanted)
    return str(value) == str(wanted)


def _matches(ref: RunRef, where: dict[str, Any]) -> bool:
    return all(_same(ref.combo.get(k), v) for k, v in where.items())


async def select_runs(
    db: Connection,
    spec: dict[str, Any],
    warnings: list[str],
) -> list[RunRef]:
    """The runs a spec names, after ``where`` and ``limit``."""
    jobs: list[JobRecord] = []
    wanted_runs = spec.get("runs") or []
    experiments = spec.get("experiments") or []
    meta = await list_experiment_meta(db)

    if experiments:
        for exp in experiments:
            if exp not in meta:
                warnings.append(f"unknown experiment `{exp}`")
                continue
            jobs.extend(await list_jobs_by_experiment(db, exp))
        if wanted_runs:
            wanted: set[tuple[str, str]] = set()
            for ref in wanted_runs:
                if "/" in ref:
                    exp, run = ref.split("/", 1)
                    wanted.add((exp, run))
                else:
                    wanted.update((exp, ref) for exp in experiments)
            known = {(j.experiment_id, j.run_id) for j in jobs}
            for exp, run in sorted(wanted - known):
                warnings.append(f"`{exp}/{run}` does not exist")
            jobs = [j for j in jobs if (j.experiment_id, j.run_id) in wanted]
    else:
        seen: set[tuple[str, str]] = set()
        for ref in wanted_runs:
            if "/" in ref:
                exp, run = ref.split("/", 1)
                candidates = [exp]
            else:
                run = ref
                candidates = await experiments_for_run(db, run)
                if not candidates:
                    warnings.append(f"no experiment has a run `{run}`")
                    continue
                if len(candidates) > 1:
                    warnings.append(
                        f"run `{run}` is in {len(candidates)} experiments; write `experiment/{run}`")
            for exp in candidates:
                if (exp, run) in seen:
                    continue
                seen.add((exp, run))
                job = await get_job(db, run, exp)
                if job is None:
                    warnings.append(f"`{exp}/{run}` does not exist")
                    continue
                jobs.append(job)

    where = spec.get("where") or {}
    refs = [RunRef(j.experiment_id, j.run_id, j.status, _combo(j), j.label,
                   str(meta.get(j.experiment_id, {}).get("name") or "") or None)
            for j in jobs]
    refs = [r for r in refs if _matches(r, where)]
    if where and not refs:
        pretty = ", ".join(f"{k}={v}" for k, v in where.items())
        warnings.append(f"no runs match {pretty}")
    refs.sort(key=lambda r: (_natural_key(r.experiment), _natural_key(r.run_id)))
    limit = int(spec.get("limit") or DEFAULT_CHART_RUNS)
    if len(refs) > limit:
        warnings.append(f"showing {limit} of {len(refs)} runs; raise `limit` or narrow the spec")
        refs = refs[:limit]
    if not refs:
        warnings.append("no matching runs")
    return refs


async def load_rows(
    db: Connection,
    runs: Sequence[RunRef],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Parsed metric rows for each selected run, keyed by (experiment, run)."""
    by_exp: dict[str, dict[str, list[tuple[int, str | bytes]]]] = {}
    for ref in runs:
        if ref.experiment not in by_exp:
            by_exp[ref.experiment] = await metric_rows_by_experiment(db, ref.experiment)
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for ref in runs:
        packed = by_exp.get(ref.experiment, {}).get(ref.run_id, [])
        out[ref.key] = parse_metric_rows(packed)
    return out


# ── numbers ──────────────────────────────────────────────────────────────────


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        try:
            out = float(str(value).strip().replace(",", ""))
        except (ValueError, TypeError):
            return None
    return out if math.isfinite(out) else None


def _extract(
    rows: Iterable[dict[str, Any]], x_key: str, metric: str,
    x_min: float | None, x_max: float | None,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        y = _number(row.get(metric))
        if y is None:
            continue
        x = _number(row.get(x_key))
        if x is None:
            x = _number(row.get("step"))
        if x is None:
            continue
        if x_min is not None and x < x_min:
            continue
        if x_max is not None and x > x_max:
            continue
        points.append((x, y))
    points.sort(key=lambda p: p[0])
    return points


def _ema(values: Sequence[float], alpha: float) -> list[float]:
    if alpha <= 0:
        return list(values)
    out: list[float] = []
    s: float | None = None
    for v in values:
        s = v if s is None else alpha * s + (1 - alpha) * v
        out.append(s)
    return out


def lttb_indices(xs: Sequence[float], ys: Sequence[float], threshold: int) -> list[int]:
    """Indices of a largest-triangle-three-buckets downsample, first and last kept."""
    n = len(xs)
    if threshold >= n or n <= 2:
        return list(range(n))
    if threshold < 3:
        return sorted({0, n - 1})
    sampled = [0]
    every = (n - 2) / (threshold - 2)
    a = 0
    for i in range(threshold - 2):
        start = int(i * every) + 1
        end = min(int((i + 1) * every) + 1, n - 1)
        avg_end = min(int((i + 2) * every) + 1, n)
        if end <= start:
            end = min(start + 1, n - 1)
        if avg_end <= end:
            avg_end = min(end + 1, n)
        avg_x = sum(xs[end:avg_end]) / max(1, avg_end - end)
        avg_y = sum(ys[end:avg_end]) / max(1, avg_end - end)
        best_area, best = -1.0, start
        for idx in range(start, end):
            area = abs((xs[a] - avg_x) * (ys[idx] - ys[a])
                       - (xs[a] - xs[idx]) * (avg_y - ys[a]))
            if area > best_area:
                best_area, best = area, idx
        sampled.append(best)
        a = best
    sampled.append(n - 1)
    return sampled


def _downsample(xs: list[float], ys: list[float], extra: list[list[float]], points: int) -> dict[str, list[float]]:
    idx = lttb_indices(xs, ys, points)
    out = {"x": [xs[i] for i in idx], "y": [ys[i] for i in idx]}
    names = ("lo", "hi")
    for name, values in zip(names, extra):
        if values:
            out[name] = [values[i] for i in idx]
    return out


def _stats(values: Sequence[float], agg: str) -> float:
    if agg == "median":
        ordered = sorted(values)
        mid = len(ordered) // 2
        return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return sum(values) / len(values)


def _band(values: Sequence[float], how: str, mean: float) -> tuple[float, float]:
    if how == "std":
        var = sum((v - mean) ** 2 for v in values) / len(values)
        sd = math.sqrt(var)
        return mean - sd, mean + sd
    return min(values), max(values)


# ── chart ────────────────────────────────────────────────────────────────────


def _experiment_names(runs: Sequence[RunRef]) -> dict[str, str]:
    """Experiment ID -> the short name, disambiguated only when two share one."""
    names: dict[str, str] = {}
    for ref in runs:
        names.setdefault(ref.experiment, ref.experiment_name or ref.experiment)
    counts = Counter(names.values())
    out: dict[str, str] = {}
    for exp, name in names.items():
        if counts[name] > 1:
            suffix = exp.rsplit("_", 1)[-1]
            out[exp] = f"{name}#{suffix}" if suffix and suffix != name else exp
        else:
            out[exp] = name
    return out


def _run_names(runs: Sequence[RunRef], experiments: dict[str, str]) -> dict[tuple[str, str], str]:
    """Short run names: the display name, else the ID without its experiment prefix."""
    out: dict[tuple[str, str], str] = {}
    for ref in runs:
        if ref.label:
            out[ref.key] = ref.label
            continue
        name = ref.run_id
        prefixes = [ref.experiment_name, experiments.get(ref.experiment), ref.experiment]
        for prefix in prefixes:
            if prefix and name.startswith(prefix + "_") and len(name) > len(prefix) + 1:
                name = name[len(prefix) + 1:]
                break
        out[ref.key] = name
    counts = Counter(out.values())
    for key, name in out.items():
        if counts[name] > 1:  # keep short names unambiguous; the full ID is in the tooltip
            out[key] = key[1]
    return out


def _run_label(ref: RunRef, run_names: dict[tuple[str, str], str],
               experiments: dict[str, str], single_experiment: bool) -> str:
    name = run_names.get(ref.key, ref.run_id)
    return name if single_experiment else f"{experiments.get(ref.experiment, ref.experiment)}/{name}"


def _natural_key(text: str) -> tuple[tuple[int, Any], ...]:
    """Sort ``depth=5m`` before ``depth=10m`` (and keep run names in that order)."""
    parts = re.split(r"(\d+(?:\.\d+)?)", text)
    out: list[tuple[int, Any]] = []
    for part in parts:
        if not part:
            continue
        out.append((0, float(part)) if _NUM_RE.match(part) else (1, part.lower()))
    return tuple(out)


def _short_names(names: Sequence[str], limit: int = 3) -> str:
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" +{len(names) - limit} more"


_SPLIT_LABELS = {"_experiment": "experiment", "_run": "run", "_status": "status"}


def _block_title(
    spec: dict[str, Any],
    runs: Sequence[RunRef],
    metrics: Sequence[str],
    kind: str,
    experiments: dict[str, str],
) -> str:
    """A header for what is plotted; the legend names the individual series.

    Experiment names are added to the header unless the legend already shows
    them (grouping by ``_experiment``).  Run names always stay in the legend, so
    ``loss · exp`` plus a legend of runs, not ``loss · exp · run_a, run_b``.
    """
    if spec.get("title"):
        return str(spec["title"])
    split = spec.get("split_by") if kind == "chart" else None
    if kind == "chart":
        head = str(spec.get("metric") or "Chart")
        if split:
            head = f"{head} by {_SPLIT_LABELS.get(split, split)}"
    else:
        head = ", ".join(metrics) if metrics else "Table"
    # Add the experiment only when the legend will not already name it:
    # `_experiment` groups show experiment names, and without a split two
    # experiments produce `exp/run` series names.
    selected = {r.experiment for r in runs}
    show_scope = split != "_experiment" and not (not split and len(selected) > 1)
    scope = ""
    if show_scope:
        ids = [str(e) for e in spec["experiments"]] if spec.get("experiments") else sorted(selected)
        scope = _short_names([experiments.get(e, e) for e in ids])
    return " · ".join(part for part in (head, scope) if part)


def assemble_chart(
    spec: dict[str, Any],
    runs: Sequence[RunRef],
    rows: dict[tuple[str, str], list[dict[str, Any]]],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """The payload the notes tab plots, from already-loaded rows."""
    warnings = warnings if warnings is not None else []
    metric = spec["metric"]
    x_key = spec.get("x") or "step"
    x_min, x_max = spec.get("x_min"), spec.get("x_max")
    points = int(spec.get("points") or DEFAULT_POINTS)
    smooth = float(spec.get("smooth") or 0)
    split = spec.get("split_by")
    band = spec.get("band") or "none"
    agg = spec.get("agg") or "mean"
    single_exp = len({r.experiment for r in runs}) <= 1
    experiment_names = _experiment_names(runs)
    run_names = _run_names(runs, experiment_names)

    prepared: list[tuple[RunRef, list[float], list[float]]] = []
    missing = 0
    for ref in runs:
        pts = _extract(rows.get(ref.key, []), x_key, metric, x_min, x_max)
        if not pts:
            missing += 1
            continue
        xs = [p[0] for p in pts]
        ys = _ema([p[1] for p in pts], smooth)
        prepared.append((ref, xs, ys))
    if missing:
        warnings.append(f"{missing} run(s) logged no numeric `{metric}`")
    if not prepared:
        warnings.append(f"no numeric `{metric}` logged for the selected runs")
        return {"kind": "chart", "spec": spec, "metric": metric, "x": x_key,
                "title": _block_title(spec, runs, [metric], "chart", experiment_names),
                "series": [], "warnings": warnings}

    series: list[dict[str, Any]] = []

    def group_display(key: str) -> str:
        return experiment_names.get(key, key) if split in ("experiment", "_experiment") else key

    if split:
        groups: dict[str, list[tuple[RunRef, list[float], list[float]]]] = {}
        for item in prepared:
            ref = item[0]
            if split in ("run", "_run"):
                key = f"{experiment_names.get(ref.experiment, ref.experiment)}/{run_names[ref.key]}"
            elif split in ("status", "_status"):
                key = ref.status
            elif split in ("experiment", "_experiment"):
                key = ref.experiment
            else:
                value = ref.combo.get(split)
                key = str(value) if value is not None else "—"
            groups.setdefault(key, []).append(item)
        for key in sorted(groups, key=lambda k: _natural_key(group_display(k))):
            members = groups[key]
            grid: dict[float, list[float]] = {}
            for _, xs, ys in members:
                for x, y in zip(xs, ys):
                    grid.setdefault(x, []).append(y)
            xs = sorted(grid)
            means = [_stats(grid[x], agg) for x in xs]
            lo: list[float] = []
            hi: list[float] = []
            if band != "none" and len(members) > 1:
                for x, mean in zip(xs, means):
                    a, b = _band(grid[x], band, mean)
                    lo.append(a)
                    hi.append(b)
            ysm = means  # each run is already smoothed
            data = _downsample(xs, ysm, [lo, hi], points)
            final = ysm[-1] if ysm else None
            label = group_display(key)
            first = members[0][0]
            if split in ("_experiment", "_run", "_status"):
                shown = label
                full_name = key if split == "_experiment" else f"{first.experiment}/{first.run_id}"
            else:
                shown = f"{split}={label}"
                full_name = shown
            series.append({
                "name": shown,
                "full_name": full_name,
                "experiment": first.experiment if len({m[0].experiment for m in members}) == 1 else "",
                "run": "",
                "status": "",
                "params": {split: label},
                "runs": len(members),
                "x": data["x"], "y": data["y"],
                **({"lo": data["lo"], "hi": data["hi"]} if "lo" in data else {}),
                "final": final,
            })
    else:
        for ref, xs, ys in prepared:
            data = _downsample(xs, ys, [], points)
            series.append({
                "name": _run_label(ref, run_names, experiment_names, single_exp),
                "full_name": f"{ref.experiment}/{ref.run_id}",
                "experiment": ref.experiment,
                "run": ref.run_id,
                "status": ref.status,
                "params": ref.combo,
                "runs": 1,
                "x": data["x"], "y": data["y"],
                "final": ys[-1] if ys else None,
            })

    return {"kind": "chart", "spec": spec, "metric": metric, "x": x_key,
            "title": _block_title(spec, runs, [metric], "chart", experiment_names),
            "series": series, "warnings": warnings}


async def build_chart(db: Connection, raw: str | dict[str, Any]) -> dict[str, Any]:
    """Parse a chart spec, load its runs and return the plot payload."""
    spec, warnings = normalize_spec(raw, "chart")
    runs = await select_runs(db, spec, warnings)
    rows = await load_rows(db, runs) if runs else {}
    return assemble_chart(spec, runs, rows, warnings)


# ── table ────────────────────────────────────────────────────────────────────


def _summary(rows: Sequence[dict[str, Any]], metric: str, agg: str) -> dict[str, Any] | None:
    values = [(int(row["step"]), v) for row in rows
              if (v := _number(row.get(metric))) is not None]
    if not values:
        return None
    values.sort(key=lambda p: p[0])
    nums = [v for _, v in values]
    out: dict[str, Any] = {"final": nums[-1], "step": values[-1][0],
                           "min": min(nums), "max": max(nums),
                           "mean": sum(nums) / len(nums)}
    if agg in out:
        out["value"] = out[agg]
    return out


def assemble_table(
    spec: dict[str, Any],
    runs: Sequence[RunRef],
    rows: dict[tuple[str, str], list[dict[str, Any]]],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Rows and columns for a leaderboard-style table."""
    warnings = warnings if warnings is not None else []
    experiment_names = _experiment_names(runs)
    run_names = _run_names(runs, experiment_names)
    metrics = list(spec.get("metrics") or [])
    if not metrics and spec.get("metric"):
        metrics = [spec["metric"]]
    if not metrics:
        seen: list[str] = []
        for ref in runs:
            for row in rows.get(ref.key, []):
                for key, value in row.items():
                    if key != "step" and _number(value) is not None and key not in seen:
                        seen.append(key)
        metrics = seen[:12]
        if len(seen) > 12:
            warnings.append(f"showing 12 of {len(seen)} metrics; list `metrics` to choose")
    if not metrics:
        warnings.append("no numeric metrics logged for the selected runs")

    columns = list(spec.get("columns") or [])
    if not columns:
        seen_cols: list[str] = []
        for ref in runs:
            for key in ref.combo:
                if key not in seen_cols:
                    seen_cols.append(key)
        columns = sorted(seen_cols, key=lambda s: s.lower())

    agg = spec.get("agg") or "final"
    out_rows: list[dict[str, Any]] = []
    dropped = 0
    for ref in runs:
        summaries = {m: _summary(rows.get(ref.key, []), m, agg) for m in metrics}
        if not any(summaries.values()):
            dropped += 1
            continue
        label = run_names.get(ref.key, ref.run_id)
        out_rows.append({
            "run": ref.run_id,
            "label": label,
            "named": bool(ref.label),
            "experiment": ref.experiment,
            "experiment_name": experiment_names.get(ref.experiment, ref.experiment),
            "status": ref.status,
            "params": {k: ref.combo.get(k) for k in columns},
            "metrics": summaries,
        })

    sort = spec.get("sort")
    order = spec.get("order") or "asc"
    if sort:
        def sort_value(row: dict[str, Any]) -> Any:
            if sort in row["metrics"]:
                summary = row["metrics"].get(sort) or {}
                return summary.get("value", summary.get("final"))
            return row["params"].get(sort)

        def sort_key(row: dict[str, Any]) -> tuple[int, Any]:
            value = sort_value(row)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return (0, float(value))
            return (1, str(value))

        present = [r for r in out_rows if sort_value(r) is not None]
        absent = [r for r in out_rows if sort_value(r) is None]
        present.sort(key=sort_key, reverse=(order == "desc"))
        out_rows = present + absent
    if dropped:
        warnings.append(f"{dropped} run(s) logged none of the selected metrics")
    if not out_rows:
        warnings.append("no rows to show")

    title = _block_title(spec, runs, metrics, "table", experiment_names)
    if out_rows:
        title += f" · {len(out_rows)} run{'s' if len(out_rows) != 1 else ''}"

    # If the runs are named after one of the parameter columns (and that value
    # is unique), the run column repeats the parameter.  Drop it and link the
    # parameter to the run instead; the full run ID stays in the tooltip.
    run_link_column: str | None = None
    if spec.get("columns") and len(out_rows) > 1:
        for c in columns:
            values = [str(row["params"].get(c)) if row["params"].get(c) is not None else ""
                      for row in out_rows]
            if (all(values) and len(set(values)) == len(values)
                    and all(not row["named"] and row["run"].endswith(v)
                            for row, v in zip(out_rows, values))):
                run_link_column = c
                break
    shown_columns = ([{"key": "run", "kind": "run"}] if run_link_column is None else []) + \
                    [{"key": c, "kind": "param"} for c in columns] + \
                    [{"key": m, "kind": "metric"} for m in metrics]
    return {
        "kind": "table", "spec": spec, "warnings": warnings,
        "title": title,
        "run_link_column": run_link_column,
        "columns": shown_columns,
        "rows": out_rows,
    }


async def build_table(db: Connection, raw: str | dict[str, Any]) -> dict[str, Any]:
    """Parse a table spec, load its runs and return columns and rows."""
    spec, warnings = normalize_spec(raw, "table")
    runs = await select_runs(db, spec, warnings)
    rows = await load_rows(db, runs) if runs else {}
    return assemble_table(spec, runs, rows, warnings)


# ── evidence links ───────────────────────────────────────────────────────────


def parse_evidence_ref(ref: str) -> tuple[str, str, str, str]:
    """``EXP/RUN/METRIC[@STEP]`` (an optional ``mlsweep:`` prefix) as its parts.

    The metric may itself contain ``/`` (``train/loss``): the first two segments
    are the experiment and the run, everything else is the metric.
    """
    text = ref.strip()
    if text.startswith("mlsweep:"):
        text = text[len("mlsweep:"):]
    at = "final"
    if "@" in text:
        text, at = text.rsplit("@", 1)
        at = at.strip().lower()
        if not _EVIDENCE_AT_RE.match(at):
            raise SpecError(f"bad reference `@{at}`")
    parts = text.split("/")
    if len(parts) < 3:
        raise SpecError("expected experiment/run/metric@step")
    experiment, run, metric = parts[0], parts[1], "/".join(parts[2:])
    if not _EVIDENCE_PART_RE.match(experiment) or not _EVIDENCE_PART_RE.match(run):
        raise SpecError("bad experiment or run ID")
    if not _EVIDENCE_METRIC_RE.match(metric):
        raise SpecError("bad metric name")
    return experiment, run, metric, at


def pick_evidence(rows: Sequence[dict[str, Any]], metric: str, at: str) -> dict[str, Any]:
    """The value a reference points at, and how closely it matched."""
    values = [(int(row["step"]), v) for row in rows
              if (v := _number(row.get(metric))) is not None]
    if not values:
        return {"ok": False, "error": f"no numeric `{metric}` logged"}
    values.sort(key=lambda p: p[0])
    nums = [v for _, v in values]
    if at in ("final", "last", "latest"):
        return {"ok": True, "value": nums[-1], "step": values[-1][0], "exact": True}
    if at in ("min", "max", "mean"):
        value = {"min": min(nums), "max": max(nums), "mean": sum(nums) / len(nums)}[at]
        return {"ok": True, "value": value, "step": values[-1][0], "exact": True}
    if at.endswith("%"):
        # A percentile of the run's logged values: 50% is the median.
        try:
            q = float(at[:-1])
        except ValueError:
            return {"ok": False, "error": f"bad reference `@{at}`"}
        if not 0 <= q <= 100:
            return {"ok": False, "error": "percentile must be 0–100"}
        pos = (len(nums) - 1) * q / 100
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        value = nums[lo] if lo == hi else nums[lo] + (nums[hi] - nums[lo]) * (pos - lo)
        return {"ok": True, "value": value, "step": values[-1][0], "exact": True}
    try:
        wanted = float(at)
    except ValueError:
        return {"ok": False, "error": f"bad reference `@{at}`"}
    exact = next((v for s, v in values if s == wanted), None)
    if exact is not None:
        return {"ok": True, "value": exact, "step": int(wanted), "exact": True}
    below = [(s, v) for s, v in values if s <= wanted]
    step, value = below[-1] if below else values[0]
    return {"ok": True, "value": value, "step": step, "exact": False}


async def resolve_evidence(db: Connection, ref: str) -> dict[str, Any]:
    """Check one ``mlsweep:`` reference against the logged metrics."""
    try:
        experiment, run, metric, at = parse_evidence_ref(ref)
    except SpecError as exc:
        return {"ok": False, "ref": ref, "error": str(exc)}
    job = await get_job(db, run, experiment)
    if job is None:
        return {"ok": False, "ref": ref, "error": f"no run `{run}` in `{experiment}`"}
    rows = parse_metric_rows(await metric_rows_for_run(db, run, experiment))
    picked = pick_evidence(rows, metric, at)
    return {"ref": ref, "experiment": experiment, "run": run, "metric": metric,
            "at": at, "status": job.status, **picked}


async def resolve_evidence_many(db: Connection, refs: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Resolve references in one batch, loading each run's rows only once."""
    out: dict[str, dict[str, Any]] = {}
    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for ref in refs:  # the HTTP route applies MAX_EVIDENCE_REFS; lint may pass more
        try:
            experiment, run, metric, at = parse_evidence_ref(ref)
        except SpecError as exc:
            out[ref] = {"ok": False, "ref": ref, "error": str(exc)}
            continue
        key = (experiment, run)
        if key not in cache:
            cache[key] = parse_metric_rows(await metric_rows_for_run(db, run, experiment))
        job = await get_job(db, run, experiment)
        if job is None:
            out[ref] = {"ok": False, "ref": ref, "error": f"no run `{run}` in `{experiment}`"}
            continue
        picked = pick_evidence(cache[key], metric, at)
        out[ref] = {"ref": ref, "experiment": experiment, "run": run, "metric": metric,
                    "at": at, "status": job.status, **picked}
    return out
