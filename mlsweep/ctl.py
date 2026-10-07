"""Control-plane subcommands for ``mlsweep``: inspect and manage runs.

These are the lifecycle verbs (ls, logs, cancel, retry, resume, stop, pause,
unpause), run naming (rename), the result-ranking command (best), and campaign
management (campaign). They are thin HTTP clients over the manager API and share helpers
with ``mlsweep.run_sweep``.

Every command works in one campaign (``--campaign``, ``$MLSWEEP_CAMPAIGN``, or
the default) unless given ``--all-campaigns``.  A command naming an experiment
from another campaign exits 1 with a hint.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from mlsweep._shared import _BLUE, _BOLD, _BRIGHT_BLUE, _CYAN, _DIM, _GREEN, _MAGENTA, _RED, _RESET, _YELLOW
from mlsweep._shared import DEFAULT_CAMPAIGN, NON_SCHEDULABLE_EXPERIMENT_STATUSES, validate_campaign
from mlsweep.run_sweep import (
    _add_manager_args,
    _leaderboard_header,
    _leaderboard_row,
    _campaign_argv,
    _combo_str,
    _manager_token,
    _parse_combo,
    _resolve_campaign,
    _wait_until_settled,
    build_leaderboard,
    manager_cancel_job,
    manager_get_experiment_summary,
    manager_get_job_logs,
    manager_get_job_metrics,
    manager_list_campaigns,
    manager_list_experiment_jobs,
    manager_list_experiments,
    manager_move_experiment,
    manager_retry_job,
    manager_set_experiment_status,
    manager_set_job_label,
    print_leaderboard,
    require_campaign,
    resolve_ranking,
    sweep_print,
)


def _select_jobs(
    jobs: list[dict[str, Any]],
    run_ids: list[str],
    statuses: list[str],
) -> list[dict[str, Any]]:
    """Select jobs matching explicit run IDs and/or statuses."""
    ids = set(run_ids)
    return [
        j for j in jobs
        if (ids and j.get("run_id") in ids) or (statuses and j.get("status") in statuses)
    ]


def _report(ok: object, msg: str) -> None:
    sweep_print(f"  {'OK' if ok else _RED + 'FAIL' + _RESET}  {msg}")


_STATUS_COLORS = {
    "done": _GREEN,
    "failed": _RED,
    "running": _CYAN,
    "dispatched": _YELLOW,
    "pending": _YELLOW,
    "cancelled": _YELLOW,
}


def _counts_str(job_counts: dict[str, int] | None) -> str:
    """One-line job tally, e.g. ``3 done / 0 fail / 1 run / 2 pend``."""
    c = job_counts or {}
    return (f"{c.get('done', 0)} done / {c.get('failed', 0)} fail / "
            f"{c.get('running', 0)} run / {c.get('pending', 0)} pend")


def _apply(
    targets: list[dict[str, Any]],
    fn: Callable[..., object],
    verb: str,
    manager: str,
    token: str,
    experiment: str,
    dry_run: bool,
    campaign: str | None = None,
) -> int:
    """Run a per-job action (cancel/retry) over *targets*. Returns # of failures.

    *fn* prints its own FAIL line with the manager's reason.
    """
    failures = 0
    for j in targets:
        run_id = j.get("run_id", "")
        if dry_run:
            sweep_print(f"  would {verb} {run_id}")
        else:
            if fn(manager, token, run_id, experiment, campaign=campaign):
                sweep_print(f"  OK  {verb} {run_id}")
            else:
                failures += 1
    return failures


def _retry(
    targets: list[dict[str, Any]],
    manager: str,
    token: str,
    experiment: str,
    dry_run: bool,
    campaign: str | None = None,
) -> int:
    """``_apply`` with retry, then warn if the re-queued runs are held. Returns # of failures."""
    failures = _apply(targets, manager_retry_job, "retry", manager, token, experiment,
                      dry_run, campaign)
    if not dry_run and failures < len(targets):
        _warn_if_held(manager, token, experiment, campaign)
    return failures


def _warn_if_held(manager: str, token: str, experiment: str, campaign: str | None) -> None:
    """Re-queued runs never dispatch while the experiment is paused or stopped. Say so."""
    summary = manager_get_experiment_summary(manager, token, experiment, quiet=True,
                                             campaign=campaign)
    status = (summary or {}).get("status")
    if status in NON_SCHEDULABLE_EXPERIMENT_STATUSES:
        sweep_print(f"  {_YELLOW}Note{_RESET}: {experiment} is {status}, so re-queued runs will "
                    f"not start. Run `mlsweep unpause {experiment}` to dispatch them.")


def _common_parser(prog: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    _add_manager_args(parser)
    return parser


def _connect(args: argparse.Namespace, experiment: str | None = None) -> tuple[str, str, str | None]:
    """``(manager, token, campaign)`` for parsed *args*.

    With *experiment*, first exit with a hint if it is in another campaign.
    """
    manager, token = _manager_token(args)
    campaign = _resolve_campaign(args)
    if experiment:
        require_campaign(manager, token, experiment, campaign)
    return manager, token, campaign


# ── ls ─────────────────────────────────────────────────────────────────────────
#
# Markdown-shaped output meant to read well for people and for LLMs: one
# heading per status, most urgent first, then one bullet per experiment or run
# whose numbers all carry their labels.  Only the oldest settled entries
# (done experiments and runs) are cut, and --all lifts that cut.  A finished
# experiment reads "done" like a finished run, though the manager (and --json)
# calls it "completed".

_LS_LIMIT = 20

# Section order: what needs attention first.  Unknown statuses go last.
_EXP_SECTIONS = ("running", "paused", "aborted", "failed", "done")
_RUN_SECTIONS = ("running", "dispatched", "pending", "failed", "cancelled", "xfailed", "done")
_COUNT_ORDER = ("done", "failed", "xfailed", "cancelled", "running", "dispatched", "pending")

_SECTION_COLORS = {
    **_STATUS_COLORS,
    "aborted": _RED,
    "paused": _YELLOW,
    "xfailed": _YELLOW,
}


# One color per role, shared by both views.  Status words, counts and section
# headings take their status color from _SECTION_COLORS; everything else is one
# of these.
_C_ID = _BRIGHT_BLUE   # campaign, experiment and run IDs
_C_NAME = _MAGENTA     # names you chose: dim names, run labels, notes
_C_TIME = _BLUE        # ages and durations
_C_AUX = _DIM          # secondary detail: prose around values, workers, hints
_C_WARN = _YELLOW      # needs a look: retries, stalls, nothing active
_C_CMD = _BOLD         # commands to paste


def _paint(text: object, *colors: object) -> str:
    """*text* wrapped in *colors*.  Plain unless --color is on."""
    return "".join(str(c) for c in colors) + f"{text}{_RESET}"


def _parse_time(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        t = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _dur(seconds: float) -> str:
    """Compact duration: ``45s``, ``3m02s``, ``2h14m``, ``10d``."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{s % 3600 // 60:02d}m"
    return f"{s // 86400}d{s % 86400 // 3600}h" if s < 7 * 86400 else f"{s // 86400}d"


def _ago(raw: Any) -> str | None:
    t = _parse_time(raw)
    return f"{_dur((datetime.now(timezone.utc) - t).total_seconds())} ago" if t else None


def _tally(counts: dict[str, int]) -> str:
    """``6 runs, 5 done, 1 failed``: every count labeled, zeros left out."""
    total = counts.get("total", sum(counts.get(k, 0) for k in _COUNT_ORDER))
    parts = [_paint(f"{total} run" + ("" if total == 1 else "s"), _BOLD)]
    parts += [_paint(f"{counts[k]} {k}", _SECTION_COLORS.get(k, "")) for k in _COUNT_ORDER if counts.get(k)]
    return ", ".join(parts)


def _exp_counts(e: dict[str, Any]) -> dict[str, int]:
    return e.get("job_counts") or {}


def _exp_status(status: str | None) -> str:
    """An experiment status as ls shows it: ``completed`` reads ``done``, like a run."""
    return "done" if status == "completed" else status or "unknown"


def _exp_section(e: dict[str, Any]) -> str:
    """The heading an experiment goes under.  A done one whose every run failed is failed."""
    status = _exp_status(e.get("status"))
    c = _exp_counts(e)
    if status == "done" and c.get("failed") and not c.get("done") and not c.get("xfailed"):
        return "failed"
    return status


def _heading(level: int, text: str, status: str) -> str:
    return _paint(f"{'#' * level} {text}", _BOLD, _SECTION_COLORS.get(status, ""))


def _sections(
    items: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], str],
    order: tuple[str, ...],
) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    names = [s for s in order if s in groups] + sorted(s for s in groups if s not in order)
    return [(s, groups[s]) for s in names]


def _print_sections(
    sections: list[tuple[str, list[dict[str, Any]]]],
    bullet: Callable[[dict[str, Any]], list[str]],
    noun: str,
    which: str,
    show_all: bool,
    level: int,
) -> None:
    for status, group in sections:
        shown = group if show_all or status != "done" else group[:_LS_LIMIT]
        count = f"{len(group)}" if len(shown) == len(group) else f"{len(group)}, {which} {len(shown)} shown"
        sweep_print("")
        sweep_print(_heading(level, f"{status.capitalize()} ({count})", status))
        for it in shown:
            for line in bullet(it):
                sweep_print(line)
        if len(shown) < len(group):
            sweep_print(_paint(f"- {len(group) - len(shown)} more {status} {noun} not shown. Pass ", _C_AUX)
                        + _paint("--all", _C_CMD) + _paint(" to list them.", _C_AUX))


def _exp_bullet(e: dict[str, Any]) -> list[str]:
    status = e.get("status") or "unknown"
    c = _exp_counts(e)
    text = f"- {_paint(e.get('experiment_id'), _C_ID)}: {_tally(c)}."
    ago = _ago(e.get("submit_time"))
    if ago:
        text += _paint(" Submitted ", _C_AUX) + _paint(ago, _C_TIME) + _paint(".", _C_AUX)
    if status == "running" and not any(c.get(k) for k in ("running", "dispatched", "pending")):
        text += _paint(" No runs are active.", _C_WARN)
    if e.get("note"):
        text += f" {_paint('Note:', _C_NAME)} {e['note']}"
    return [text]


def _varying_dims(jobs: list[dict[str, Any]]) -> tuple[dict[str, list[Any]], set[str]]:
    """Every dim with its distinct values in first-seen order, and the dims that vary."""
    values: dict[str, list[Any]] = {}
    for j in jobs:
        for k, v in (_parse_combo(j.get("combo")) or {}).items():
            seen = values.setdefault(k, [])
            if v not in seen:
                seen.append(v)
    varying = {k for k, vs in values.items() if len(vs) > 1}
    return values, varying if len(jobs) > 1 else set(values)


_DIM_VALUES_SHOWN = 8


def _dim_values(values: list[Any]) -> str:
    """A dim's values, sorted when numeric; a long list becomes a count and range."""
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    if numeric:
        values = sorted(values)
    if len(values) <= _DIM_VALUES_SHOWN:
        return ", ".join(str(v) for v in values)
    if numeric:
        return f"{len(values)} values from {values[0]:.4g} to {values[-1]:.4g}"
    return f"{len(values)} values"


def _logs_hint(run_id: str, experiment: str, campaign: str | None) -> str:
    scope = " --all-campaigns" if campaign is None else (
        f" --campaign {campaign}" if campaign != DEFAULT_CAMPAIGN else "")
    return (f"  {_paint('Logs:', _C_AUX)} "
            + _paint(f"mlsweep logs {run_id} --experiment {experiment}{scope} --tail 50", _C_CMD))


def _run_bullet(
    j: dict[str, Any],
    varying: set[str],
    stalled: set[str],
    experiment: str,
    campaign: str | None,
) -> list[str]:
    status = j.get("status") or "unknown"
    run_id = j.get("run_id", "?")
    head = f"- {_paint(run_id, _C_ID)}" + (
        f" ({_paint(j['label'], _C_NAME)})" if j.get("label") else "")
    combo = _parse_combo(j.get("combo")) or {}
    dims = " ".join(f"{_paint(k, _C_NAME)}={v}" for k, v in combo.items() if k in varying)
    facts: list[str] = []
    took = _paint(_dur(j["elapsed"]), _C_TIME) if j.get("elapsed") is not None else None
    if status == "running":
        start = _parse_time(j.get("start_time"))
        if start:
            since = _dur((datetime.now(timezone.utc) - start).total_seconds())
            facts.append(f"Running for {_paint(since, _C_TIME)}")
        if run_id in stalled:
            facts.append(_paint("stalled", _BOLD, _C_WARN))
    elif status in ("failed", "cancelled", "xfailed") and j.get("exit_code") is not None:
        code = _paint(f"Exit {j['exit_code']}", _BOLD, _SECTION_COLORS.get(status, ""))
        facts.append(code + (f" after {took}" if took else ""))
    elif took:
        facts.append(took)
    if status != "done" and (j.get("attempt") or 0) > 1:
        facts.append(_paint(f"attempt {j['attempt']}", _C_WARN))
    on = _paint(f" on {j['worker_id']}", _C_AUX) if (
        status in ("running", "dispatched", "failed") and j.get("worker_id")) else ""
    text = head + ":" if dims or facts or on else head
    if dims:
        text += f" {dims}."
    if facts or on:
        text += " " + (", ".join(facts) or "Placed") + on + "."
    lines = [text]
    if status == "failed":
        lines.append(_logs_hint(run_id, experiment, campaign))
    return lines


def _ls_experiments(exps: list[dict[str, Any]], campaign: str | None, show_all: bool) -> None:
    if campaign is not None:
        by_campaign = [(campaign, exps)]
    else:
        names = sorted({e.get("campaign") or DEFAULT_CAMPAIGN for e in exps})
        by_campaign = [(c, [e for e in exps if (e.get("campaign") or DEFAULT_CAMPAIGN) == c])
                       for c in names]
        sweep_print(_paint(f"# All campaigns: {len(exps)} experiments in {len(names)} campaigns", _BOLD))
    for i, (c, group) in enumerate(by_campaign):
        level = 1 if campaign is not None else 2
        if i or campaign is None:
            sweep_print("")
        n = len(group)
        sweep_print(_paint(f"{'#' * level} Campaign ", _BOLD) + _paint(c, _BOLD, _C_ID)
                    + _paint(f": {n} experiment{'' if n == 1 else 's'}", _BOLD))
        _print_sections(_sections(group, _exp_section, _EXP_SECTIONS), _exp_bullet,
                        "experiments", "newest", show_all, level + 1)
    if exps:
        sweep_print("")
        sweep_print(_paint("Run ", _C_AUX) + _paint("mlsweep ls <experiment>", _C_CMD)
                    + _paint(" to list its runs.", _C_AUX))


def _ls_runs(
    experiment: str,
    jobs: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    campaign: str | None,
    show_all: bool,
) -> None:
    counts = dict(Counter(j.get("status", "unknown") for j in jobs), total=len(jobs))
    summary = summary or {}
    status = _exp_status(summary["status"]) if summary.get("status") else None
    sweep_print(_paint("# Experiment ", _BOLD) + _paint(experiment, _BOLD, _C_ID) + ": "
                + (f"{_paint(status, _BOLD, _SECTION_COLORS.get(status, ''))}, " if status else "")
                + _tally(counts))
    values, varying = _varying_dims(jobs)
    if values:
        sweep_print(_paint("Dims:", _BOLD) + " " + ", ".join(
            f"{_paint(k, _C_NAME)} ({_dim_values(vs)})" for k, vs in values.items()))
    if summary.get("note"):
        sweep_print(f"{_paint('Note:', _C_NAME)} {summary['note']}")
    stalled = set(summary.get("stalled_runs") or [])
    _print_sections(
        _sections(jobs, lambda j: j.get("status") or "unknown", _RUN_SECTIONS),
        lambda j: _run_bullet(j, varying, stalled, experiment, campaign),
        "runs", "first", show_all, 2,
    )


def ls_cmd(argv: list[str]) -> None:
    parser = _common_parser("mlsweep ls", "List experiments, or the runs within one experiment.")
    parser.add_argument("experiment", nargs="?", help="Experiment ID (omit to list experiments)")
    parser.add_argument("--status", default=None, help="Filter by status")
    parser.add_argument("--all", action="store_true",
                        help=f"List every done experiment or run "
                             f"(by default only the newest {_LS_LIMIT} are shown)")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--with-dims", action="store_true",
                        help="With --json: add a parsed 'dims' object for each run")
    args = parser.parse_args(argv)
    manager, token, campaign = _connect(args, args.experiment)

    if args.experiment:
        jobs = manager_list_experiment_jobs(manager, token, args.experiment, status_filter=args.status,
                                            campaign=campaign) or []
        if args.json:
            if args.with_dims:
                jobs = [dict(j, dims=_parse_combo(j["combo"]) or {}) for j in jobs]
            print(json.dumps(jobs, indent=2))
            return
        summary = manager_get_experiment_summary(manager, token, args.experiment, quiet=True,
                                                 campaign=campaign)
        _ls_runs(args.experiment, jobs, summary, campaign, args.all)
        return

    # Experiments are listed as "done", so accept that for the manager's "completed".
    status_filter = "completed" if args.status == "done" else args.status
    exps = manager_list_experiments(manager, token, status_filter=status_filter, campaign=campaign)
    if exps is None:
        sweep_print(f"{_RED}FAIL{_RESET}  list experiments")
        sys.exit(1)
    if args.json:
        print(json.dumps(exps, indent=2))
        return
    _ls_experiments(exps, campaign, args.all)


# ── logs ───────────────────────────────────────────────────────────────────────


def logs_cmd(argv: list[str]) -> None:
    parser = _common_parser("mlsweep logs", "Print a run's training log.")
    parser.add_argument("run", help="Run ID")
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    parser.add_argument("--tail", type=int, default=None, help="Show only the last N lines")
    parser.add_argument("--follow", action="store_true", help="Follow new output")
    args = parser.parse_args(argv)
    manager, token, campaign = _connect(args, args.experiment)

    text = manager_get_job_logs(manager, token, args.experiment, args.run, campaign=campaign)
    if text is None:
        sweep_print(f"{_RED}No log found{_RESET} for {args.run} in {args.experiment}")
        sys.exit(1)

    lines = text.splitlines()
    if args.tail:
        lines = lines[-args.tail:]
    print("\n".join(lines))

    if args.follow:
        seen = len(text)
        try:
            while True:
                time.sleep(2)
                newer = manager_get_job_logs(manager, token, args.experiment, args.run, campaign=campaign)
                if newer and len(newer) > seen:
                    print(newer[seen:], end="")
                    seen = len(newer)
        except KeyboardInterrupt:
            sweep_print("\n  interrupted.")


# ── metrics ────────────────────────────────────────────────────────────────────
#
# A read-only view of what runs already logged.  Everything is fetched and
# reshaped per invocation; nothing is stored or cached.


def _sort_key(v: str) -> tuple[int, Any]:
    try:
        return (0, float(v))
    except ValueError:
        return (1, v)


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    print("  ".join(f"{_BOLD}{_CYAN}{h.rjust(w)}{_RESET}" for h, w in zip(headers, widths)))
    for r in rows:
        print("  ".join(c.rjust(w) for c, w in zip(r, widths)))


def select_metrics(rows: list[dict[str, Any]], pattern: re.Pattern[str] | None) -> list[dict[str, Any]]:
    """Keep only keys matching *pattern* (plus ``step``); drop steps with none left."""
    out = []
    for row in rows:
        kept = {k: v for k, v in row.items()
                if k != "step" and (pattern is None or pattern.search(k))}
        if kept:
            out.append({"step": row.get("step"), **kept})
    return out


def _pivot_run(
    rows: list[dict[str, Any]],
    pattern: re.Pattern[str] | None,
    step: int | None,
) -> tuple[dict[tuple[str, ...], Any], Any]:
    """``({(captures…): value}, max_step)`` for one run.

    Each key contributes its own latest value, not the value at the latest step
    that happened to have any matching key.  A key contributes only when every
    capture group matched, so a 2-group pattern never mixes 1- and 2-tuples.
    A None *pattern* takes every key whole, as a 1-tuple.
    *step* caps the steps considered (latest at or before it; the latest
    overall if None).  ``max_step`` is the largest step among the chosen keys,
    or None when nothing matched.
    """
    out: dict[tuple[str, ...], Any] = {}
    last_step: dict[tuple[str, ...], Any] = {}
    for row in rows:
        s = row["step"]
        if not isinstance(s, (int, float)):
            continue
        if step is not None and s > step:
            continue
        for k, v in row.items():
            if k == "step":
                continue
            if pattern is None:
                key: tuple[str, ...] = (k,)
            else:
                m = pattern.search(k)
                if not m or any(g is None for g in m.groups()):
                    continue
                key = tuple(m.groups())
            if key not in last_step or s >= last_step[key]:
                last_step[key] = s
                out[key] = v
    return out, (max(last_step.values()) if last_step else None)


def pivot_metrics(rows: list[dict[str, Any]], pattern: re.Pattern[str] | None,
                  step: int | None) -> tuple[int | None, dict[str, Any]]:
    """Backward-compatible single-group pivot: ``(step, {capture: value})``."""
    vals, max_step = _pivot_run(rows, pattern, step)
    return max_step, {g[0]: v for g, v in vals.items()}


def _dim_columns(jobs: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Ordered dim names and ``run_id → combo`` parsed from a jobs list."""
    dims: list[str] = []
    combos: dict[str, dict[str, Any]] = {}
    for j in jobs:
        rid = j["run_id"]
        combo = _parse_combo(j["combo"]) or {}
        combos[rid] = combo
        for k in combo:
            if k not in dims:
                dims.append(k)
    return dims, combos


def _csv_cell(v: Any) -> str:
    """CSV cell for a dim or metric value (None → empty, not ``-``)."""
    return "" if v is None else str(v)


def metrics_cmd(argv: list[str]) -> None:
    parser = _common_parser(
        "mlsweep metrics",
        "Print what runs logged, filtered and reshaped on demand (nothing is stored).",
    )
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    parser.add_argument("runs", nargs="*", help="Run IDs (default: every run in the experiment)")
    parser.add_argument("--keys", default=None, help="Regex selecting metric keys")
    shape = parser.add_mutually_exclusive_group()
    shape.add_argument("--pivot", action="store_true",
                       help="Reshape keys via --keys' capture groups: one group makes one "
                            "column per run; two groups make a row×column grid per run")
    shape.add_argument("--last", action="store_true",
                       help="One row per run: each selected key's own latest value")
    parser.add_argument("--step", type=int, default=None,
                        help="With --pivot: use steps at or before this (default: latest)")
    parser.add_argument("--tail", type=int, default=None,
                        help="Steps per run in the table view (default 10; 0 = all)")
    parser.add_argument("--with-dims", action="store_true",
                        help="Add each sweep dimension as a column, from the submitted combo")
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="Emit the selected metrics as JSON")
    fmt.add_argument("--csv", action="store_true", help="Emit CSV (long, or wide with --last)")
    args = parser.parse_args(argv)
    if args.tail is not None and (args.json or args.csv or args.last or args.pivot):
        parser.error("--tail only applies to the table view. For one row per run "
                     "(each key's latest value), use --last")
    tail = 10 if args.tail is None else args.tail
    manager, token, campaign = _connect(args, args.experiment)

    try:
        pattern = re.compile(args.keys) if args.keys else None
    except re.error as e:
        sweep_print(f"{_RED}Bad --keys regex{_RESET}: {e}")
        sys.exit(2)

    jobs: list[dict[str, Any]] = []
    if not args.runs or args.with_dims:
        jobs = manager_list_experiment_jobs(manager, token, args.experiment, campaign=campaign) or []
    dims, combo_by_run = _dim_columns(jobs)
    if not args.with_dims:
        dims = []  # so every dim column below comes out empty
    run_ids = list(args.runs) or [j["run_id"] for j in jobs]

    # combo_by_run covers every run when --with-dims is on (it fetched all jobs).
    def dim_cells(rid: str, fmt: Callable[[Any], str]) -> list[str]:
        return [fmt(combo_by_run[rid].get(d)) for d in dims]

    def with_dims(rid: str, data: Any) -> Any:
        return {"dims": combo_by_run[rid], "metrics": data} if args.with_dims else data

    def run_label(rid: str) -> str:
        return f"{rid} ({_combo_str(combo_by_run[rid])})" if args.with_dims else rid

    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched = pool.map(lambda rid: manager_get_job_metrics(manager, token, args.experiment, rid,
                                                               campaign=campaign),
                           run_ids)
        per_run = {rid: select_metrics(rows or [], pattern) for rid, rows in zip(run_ids, fetched)}
    per_run = {rid: rows for rid, rows in per_run.items() if rows}
    if not per_run:
        sweep_print(f"{_RED}No metrics{_RESET} in {args.experiment}"
                    + (f" matching {args.keys!r}" if args.keys else ""))
        sys.exit(1)

    # ── --last: wide, one row per run ────────────────────────────────────────
    if args.last:
        wide = {rid: pivot_metrics(rows, None, None)[1] for rid, rows in per_run.items()}
        keys = sorted({k for vals in wide.values() for k in vals})
        if args.json:
            print(json.dumps({rid: with_dims(rid, vals) for rid, vals in wide.items()}, indent=2))
            return
        if args.csv:
            w = csv.writer(sys.stdout)
            w.writerow(["run"] + dims + keys)
            for rid, vals in wide.items():
                w.writerow([rid] + dim_cells(rid, _csv_cell) + [_csv_cell(vals.get(k)) for k in keys])
            return
        _print_table(["run"] + dims + keys,
                     [[rid] + dim_cells(rid, _fmt) + [_fmt(vals.get(k)) for k in keys]
                      for rid, vals in wide.items()])
        return

    # ── --pivot: reshape key captures ────────────────────────────────────────
    if args.pivot:
        if pattern is None or pattern.groups < 1:
            sweep_print(f"{_RED}--pivot needs --keys with a capture group{_RESET}, "
                        "e.g. --keys 'val/nll@r(\\d+)'")
            sys.exit(2)
        if pattern.groups > 2:
            sweep_print(f"{_RED}--pivot supports at most two capture groups{_RESET}")
            sys.exit(2)

        if pattern.groups == 1:
            cols = []
            for rid, rows in per_run.items():
                pstep, vals = pivot_metrics(rows, pattern, args.step)
                if vals:
                    cols.append((f"{run_label(rid)} @{pstep}", vals))
            if not cols:
                sweep_print(f"{_YELLOW}No pivot values{_RESET} in {args.experiment}")
                return
            xs = sorted({x for _, vals in cols for x in vals}, key=_sort_key)
            _print_table(["x"] + [c for c, _ in cols],
                         [[x] + [_fmt(vals.get(x)) for _, vals in cols] for x in xs])
            return

        # two capture groups: one row×column grid per run
        for rid, rows in per_run.items():
            grid, _ = _pivot_run(rows, pattern, args.step)
            if not grid:
                continue
            xs = sorted({g[0] for g in grid}, key=_sort_key)
            ys = sorted({g[1] for g in grid}, key=_sort_key)
            print(f"{_BOLD}{_CYAN}== {run_label(rid)}{_RESET}")
            _print_table(["x\\y"] + ys,
                         [[x] + [_fmt(grid.get((x, y))) for y in ys] for x in xs])
            print()
        return

    # ── long JSON / CSV ──────────────────────────────────────────────────────
    if args.json:
        print(json.dumps({rid: with_dims(rid, rows) for rid, rows in per_run.items()}, indent=2))
        return
    if args.csv:
        w = csv.writer(sys.stdout)
        w.writerow(["run"] + dims + ["step", "key", "value"])
        for rid, rows in per_run.items():
            cells = dim_cells(rid, _csv_cell)
            for row in rows:
                for k, v in row.items():
                    if k != "step":
                        w.writerow([rid] + cells + [row["step"], k, _csv_cell(v)])
        return

    # ── table view ───────────────────────────────────────────────────────────
    for rid, rows in per_run.items():
        keys = sorted({k for r in rows for k in r if k != "step"})
        shown = rows[-tail:] if tail else rows
        print(f"{_BOLD}{_CYAN}== {run_label(rid)}{_RESET}  ({len(rows)} steps)")
        _print_table(["step"] + keys, [[_fmt(r["step"])] + [_fmt(r.get(k)) for k in keys] for r in shown])
        print()


# ── cancel / retry ─────────────────────────────────────────────────────────────


_SELECTABLE_STATUSES = ("failed", "cancelled", "running", "pending")


def _cancel_retry(argv: list[str], *, retry: bool) -> None:
    verb = "retry" if retry else "cancel"
    parser = _common_parser(
        f"mlsweep {verb}",
        f"{'Re-queue' if retry else 'Cancel'} runs in an experiment.",
    )
    parser.add_argument("experiment", help="Experiment ID")
    parser.add_argument("runs", nargs="*", help="Run IDs to act on")
    for s in _SELECTABLE_STATUSES:
        parser.add_argument(f"--{s}", action="store_true", help=f"Select all {s} runs")
    parser.add_argument("--all", action="store_true", help="Select every run")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation for --all")
    args = parser.parse_args(argv)
    manager, token, campaign = _connect(args, args.experiment)

    jobs = manager_list_experiment_jobs(manager, token, args.experiment, campaign=campaign) or []
    statuses = [s for s in _SELECTABLE_STATUSES if getattr(args, s)]
    targets = jobs if args.all else _select_jobs(jobs, args.runs, statuses)

    if not targets:
        sweep_print("  No matching runs.")
        return

    if args.all and not args.yes:
        sweep_print(f"{_RED}Refusing to {verb} all {len(targets)} runs without --yes.{_RESET}")
        sys.exit(1)

    if retry:
        failures = _retry(targets, manager, token, args.experiment, args.dry_run, campaign)
    else:
        failures = _apply(targets, manager_cancel_job, verb, manager, token, args.experiment,
                          args.dry_run, campaign)
    if failures:
        sys.exit(1)


def cancel_cmd(argv: list[str]) -> None:
    _cancel_retry(argv, retry=False)


def retry_cmd(argv: list[str]) -> None:
    _cancel_retry(argv, retry=True)


# ── rename ─────────────────────────────────────────────────────────────────────


def rename_cmd(argv: list[str]) -> None:
    parser = _common_parser(
        "mlsweep rename",
        "Give a run a display name, shown next to its run ID in ls, best, and the "
        "dashboard. The run ID itself never changes.",
    )
    parser.add_argument("experiment", help="Experiment ID")
    parser.add_argument("run", help="Run ID")
    parser.add_argument("name", nargs="?", default=None, help="New display name")
    parser.add_argument("--clear", action="store_true", help="Remove the display name")
    args = parser.parse_args(argv)
    name = (args.name or "").strip()
    if args.clear == bool(name):
        parser.error("give a non-empty NAME or --clear, not both")
    manager, token, campaign = _connect(args, args.experiment)

    r = manager_set_job_label(manager, token, args.run, args.experiment, name or None,
                              campaign=campaign)
    _report(r, f"cleared the name of {args.run}" if args.clear else f"renamed {args.run} to {name!r}")
    if not r:
        sys.exit(1)


# ── stop / pause / unpause ─────────────────────────────────────────────────────


def _set_status_cmd(argv: list[str], name: str, description: str, status: str, done: str,
                    *, confirm: bool = False) -> None:
    parser = _common_parser(f"mlsweep {name}", description)
    parser.add_argument("experiment", help="Experiment ID")
    if confirm:
        parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args(argv)
    if confirm and not args.yes:
        sweep_print(f"{_RED}Aborting a sweep is destructive. Pass --yes to confirm.{_RESET}")
        sys.exit(1)
    manager, token, campaign = _connect(args, args.experiment)
    r = manager_set_experiment_status(manager, token, args.experiment, status, campaign=campaign)
    _report(r, f"{done} {args.experiment}")


def stop_cmd(argv: list[str]) -> None:
    _set_status_cmd(argv, "stop", "Abort an experiment (cancels in-flight + stops dispatch).",
                    "aborted", "stopped", confirm=True)


def pause_cmd(argv: list[str]) -> None:
    _set_status_cmd(argv, "pause", "Pause an experiment (stops dispatching new jobs).", "paused", "paused")


def unpause_cmd(argv: list[str]) -> None:
    _set_status_cmd(argv, "unpause", "Resume dispatching a paused experiment.", "running", "resumed")


# ── wait ───────────────────────────────────────────────────────────────────────


# Exit codes for `mlsweep wait` (documented in its --help).
WAIT_EXIT_SETTLED = 0   # settled cleanly, or the requested condition was met
WAIT_EXIT_FAILURE = 1   # at least one run failed
WAIT_EXIT_TIMEOUT = 2   # --timeout elapsed before the condition was met
WAIT_EXIT_STALLED = 3   # a running run made no progress for --stalled-after seconds

_WAIT_ACTIVE_STATUSES = ("pending", "dispatched", "running")


def _wait_failed(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runs that finished unsuccessfully."""
    return [j for j in jobs if j.get("status") == "failed"]


def _wait_stalled(
    jobs: list[dict[str, Any]], stalled_after: float,
) -> list[dict[str, Any]]:
    """Running runs that have not logged progress within *stalled_after* seconds."""
    out = []
    for j in jobs:
        if j.get("status") != "running":
            continue
        stall = j.get("stall_seconds")
        if isinstance(stall, (int, float)):
            # The user's --stalled-after wins over the manager's default flag.
            if stall >= stalled_after:
                out.append(j)
        elif j.get("stalled"):
            out.append(j)
    return out


def wait_cmd(argv: list[str]) -> None:
    """Wait for an experiment to finish, fail, or stall.

    Exit codes are part of the interface so scripts and agents can branch on
    them without parsing text:

    \b
      0  settled cleanly (no failed runs)
      1  at least one run failed
      2  --timeout elapsed first
      3  --until stalled and a running run stopped making progress
    """
    parser = argparse.ArgumentParser(
        prog="mlsweep wait",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Wait until an experiment settles, fails, or stalls.\n\n"
            "Exit codes:\n"
            "  0  settled cleanly\n"
            "  1  at least one run failed\n"
            "  2  timed out (--timeout)\n"
            "  3  a running run stalled (--until stalled)"
        ),
    )
    _add_manager_args(parser)
    parser.add_argument("experiment", help="Experiment ID")
    parser.add_argument(
        "--until", choices=("done", "any-failure", "stalled"), default="done",
        help="Condition to wait for (default: done = no active runs)",
    )
    parser.add_argument(
        "--timeout", type=float, default=0, metavar="N",
        help="Seconds to wait before exiting 2 (0 = forever)",
    )
    parser.add_argument(
        "--interval", type=float, default=10, metavar="S",
        help="Seconds between polls (default: 10)",
    )
    parser.add_argument(
        "--stalled-after", type=float, default=900, metavar="S",
        help="Seconds without progress before a running run counts as stalled "
             "for --until stalled (default: 900)",
    )
    args = parser.parse_args(argv)
    manager, token, campaign = _connect(args, args.experiment)

    start = time.monotonic()
    while True:
        jobs = manager_list_experiment_jobs(manager, token, args.experiment, campaign=campaign)
        if jobs is None:
            sweep_print(f"{_RED}FAIL{_RESET}  Cannot list jobs for {args.experiment}")
            sys.exit(WAIT_EXIT_FAILURE)

        failed = _wait_failed(jobs)
        active = [j for j in jobs if j.get("status") in _WAIT_ACTIVE_STATUSES]

        if args.until == "any-failure" and failed:
            names = ", ".join(j.get("run_id", "?") for j in failed)
            sweep_print(f"{_RED}FAILURE{_RESET}  {len(failed)} run(s) failed: {names}")
            sys.exit(WAIT_EXIT_FAILURE)

        if not active:
            # Settled: nothing pending/dispatched/running.
            if failed:
                names = ", ".join(j.get("run_id", "?") for j in failed)
                sweep_print(f"{_RED}FAILURE{_RESET}  {args.experiment} settled with "
                            f"{len(failed)} failed run(s): {names}")
                sys.exit(WAIT_EXIT_FAILURE)
            sweep_print(f"{_GREEN}DONE{_RESET}  {args.experiment} settled cleanly.")
            sys.exit(WAIT_EXIT_SETTLED)

        if args.until == "stalled":
            stalled = _wait_stalled(jobs, args.stalled_after)
            if stalled:
                names = ", ".join(j.get("run_id", "?") for j in stalled)
                sweep_print(f"{_YELLOW}STALLED{_RESET}  {len(stalled)} running run(s) made no "
                            f"progress in {args.stalled_after:g}s: {names}")
                sys.exit(WAIT_EXIT_STALLED)

        if args.timeout and time.monotonic() - start >= args.timeout:
            sweep_print(f"{_YELLOW}TIMEOUT{_RESET}  {args.experiment} still has "
                        f"{len(active)} active run(s) after {args.timeout:g}s.")
            sys.exit(WAIT_EXIT_TIMEOUT)

        time.sleep(args.interval)


# ── resume ─────────────────────────────────────────────────────────────────────


def resume_cmd(argv: list[str]) -> None:
    parser = _common_parser("mlsweep resume", "Continue an experiment.")
    parser.add_argument("experiment", help="Experiment ID")
    parser.add_argument("--sweep", default=None, help="Sweep file (required to continue a Bayesian sweep)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    args = parser.parse_args(argv)
    manager, token, campaign = _connect(args, args.experiment)

    if args.sweep:
        from mlsweep._sweep import load_sweep_file
        info = load_sweep_file(args.sweep)
        if info.get("method") == "bayes":
            # Reuse the submitter's bayes resume path.
            from mlsweep.cli import _forward
            from mlsweep.run_sweep import main as _run_main
            fwd = [args.sweep, "--resume", args.experiment, "--manager", manager,
                   *_campaign_argv(campaign)]
            if args.token:
                fwd += ["--token", args.token]
            _forward("mlsweep run", _run_main, fwd)
            return

    # Grid (or no sweep file): re-queue failed / cancelled runs.
    jobs = manager_list_experiment_jobs(manager, token, args.experiment, campaign=campaign) or []
    targets = [j for j in jobs if j.get("status") in ("failed", "cancelled")]
    if not targets:
        sweep_print("  Nothing to resume, no failed or cancelled jobs.")
        return
    failures = _retry(targets, manager, token, args.experiment, args.dry_run, campaign)
    if failures:
        sys.exit(1)


# ── best ───────────────────────────────────────────────────────────────────────


def _best_by(rows: list[dict[str, Any]], dims: list[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    """The best ranked row for each combination of *dims*' values.

    *rows* are already best-first.  Runs with no value, or missing a dim, are skipped.
    """
    best: dict[tuple[str, ...], dict[str, Any]] = {}
    for r in rows:
        combo = r["combo"] or {}
        if r["value"] is None or not all(d in combo for d in dims):
            continue
        best.setdefault(tuple(_fmt(combo[d]) for d in dims), r)
    return best


def _print_grouped_leaderboard(
    rows: list[dict[str, Any]], group_by: str, metric: str, goal: str, top: int,
) -> None:
    """Print the best run for each value of *group_by* (rows are already best-first)."""
    groups = {k[0]: r for k, r in _best_by(rows, [group_by]).items()}
    if not groups:
        sweep_print(f"  {_YELLOW}(no completed runs with a value for dim {group_by!r}){_RESET}")
        return
    shown = list(groups.items())
    if top:
        shown = shown[:top]
    _leaderboard_header(metric, goal, f"best per {_GREEN}{group_by}{_RESET} ({len(groups)} groups)")
    for i, (key, r) in enumerate(shown, 1):
        _leaderboard_row(i, r, f"{_GREEN}{group_by}={key}{_RESET}  ")


def _print_metric_table(
    rows: list[dict[str, Any]], spec: str, metric: str, goal: str,
) -> None:
    """Print a 2-D grid of *metric* over two dims, best value per cell."""
    parts = [d.strip() for d in spec.split(",")]
    if len(parts) != 2 or not all(parts):
        sweep_print(f"{_RED}--table needs exactly two dims, e.g. --table est,lr{_RESET}")
        sys.exit(2)
    d1, d2 = parts
    best = _best_by(rows, parts)
    if not best:
        sweep_print(f"  {_YELLOW}(no completed runs with values for {d1!r}/{d2!r}){_RESET}")
        return
    xs = sorted({k[0] for k in best}, key=_sort_key)
    ys = sorted({k[1] for k in best}, key=_sort_key)
    sweep_print(f"\n{_BOLD}{_CYAN}{metric} by {d1} × {d2}{_RESET} ({goal})")
    _print_table([f"{d1}\\{d2}"] + ys,
                 [[x] + [_fmt(best[x, y]["value"] if (x, y) in best else None) for y in ys]
                  for x in xs])


def best_cmd(argv: list[str]) -> None:
    parser = _common_parser("mlsweep best", "Show the best runs of an experiment by metric.")
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    parser.add_argument("--metric", default=None,
                        help="Metric to rank by (default: experiment's metric, else loss)")
    parser.add_argument("--goal", default=None, choices=["minimize", "maximize"],
                        help="Rank direction (default: experiment's goal, else minimize)")
    parser.add_argument("--top", type=int, default=10, help="Show top N runs (0 = all)")
    parser.add_argument("--group-by", default=None, metavar="DIM",
                        help="Show the best run for each value of this dimension")
    parser.add_argument("--table", default=None, metavar="D1,D2",
                        help="2-D grid of the metric over two dimensions")
    parser.add_argument("--with-dims", action="store_true",
                        help="With --json: add each sweep dimension as a top-level key")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--wait", action="store_true", help="Block until the experiment settles")
    parser.add_argument("--wait-interval", type=int, default=10, help="Seconds between --wait polls")
    args = parser.parse_args(argv)
    manager, token, campaign = _connect(args, args.experiment)

    had_failure = False
    if args.wait:
        had_failure = _wait_until_settled(manager, token, args.experiment, args.wait_interval,
                                          campaign=campaign)

    metric, goal = resolve_ranking(manager, token, args.experiment, args.metric, args.goal,
                                   campaign=campaign)
    rows = build_leaderboard(manager, token, args.experiment, metric, goal, campaign=campaign)

    if args.json:
        if args.with_dims:
            rows = [{**r, **(r["combo"] or {})} for r in rows]
        print(json.dumps(rows, indent=2))
    elif args.group_by:
        _print_grouped_leaderboard(rows, args.group_by, metric, goal, args.top)
    elif args.table:
        _print_metric_table(rows, args.table, metric, goal)
    else:
        print_leaderboard(rows, metric, goal, args.top)
    if had_failure:
        sys.exit(1)


# ── campaign ───────────────────────────────────────────────────────────────────


def campaign_cmd(argv: list[str]) -> None:
    """``mlsweep campaign [ls]`` lists campaigns; ``campaign move EXP NAME`` re-files one."""
    parser = argparse.ArgumentParser(
        prog="mlsweep campaign",
        description="List campaigns, or move an experiment to another campaign.",
    )
    sub = parser.add_subparsers(dest="action")
    ls = sub.add_parser("ls", help="List campaigns with experiment and job counts (the default action)")
    _add_manager_args(ls)
    ls.add_argument("--json", action="store_true", help="Emit JSON")
    move = sub.add_parser("move", help="Move an experiment, with all its runs, to another campaign")
    _add_manager_args(move)
    move.add_argument("experiment", help="Experiment ID")
    move.add_argument("target", help="Campaign to move it to (created if new)")
    args = parser.parse_args(argv if argv and argv[0] in ("ls", "move", "-h", "--help") else ["ls", *argv])

    if args.action == "move":
        try:
            validate_campaign(args.target)
        except ValueError as e:
            sweep_print(f"{_RED}Error: {e}{_RESET}")
            sys.exit(1)
        manager, token, campaign = _connect(args, args.experiment)
        r = manager_move_experiment(manager, token, args.experiment, args.target, campaign=campaign)
        _report(r, f"moved {args.experiment} to campaign {args.target}")
        if not r:
            sys.exit(1)
        return

    manager, token = _manager_token(args)
    current = _resolve_campaign(args)
    camps = manager_list_campaigns(manager, token)
    if camps is None:
        sweep_print(f"{_RED}FAIL{_RESET}  list campaigns")
        sys.exit(1)
    if args.json:
        print(json.dumps(camps, indent=2))
        return
    sweep_print(f"{_BOLD}{_CYAN}{len(camps)}{_RESET} campaigns:")
    for c in camps:
        name = c.get("campaign", "?")
        mark = "*" if name == current else " "
        n = c.get("experiments", 0)
        sweep_print(f"  {mark} {_GREEN}{name}{_RESET}  {n} experiment{'' if n == 1 else 's'}, "
                    f"{_counts_str(c.get('job_counts'))}")
    if current is not None:
        sweep_print("  (* = current campaign, from --campaign, $MLSWEEP_CAMPAIGN, or the default)")
