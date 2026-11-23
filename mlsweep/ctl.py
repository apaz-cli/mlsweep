"""Control-plane subcommands for ``mlsweep``: inspect and manage runs.

These are the lifecycle verbs (ls, logs, cancel, retry, resume, stop, pause,
unpause), run naming (rename), the result-ranking command (best), and campaign
management (campaign). They are thin HTTP clients over the manager API and share helpers
with ``mlsweep.run_sweep``.

Every command works in one campaign (``--campaign``, a context's campaign, or
``$MLSWEEP_CAMPAIGN``, one of which is required) unless given ``--all-campaigns``.  A command naming an experiment
from another campaign exits 1 with a hint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, NoReturn

from mlsweep._jsonout import emit, emit_line
from mlsweep._shared import _BLUE, _BOLD, _BRIGHT_BLUE, _CYAN, _DIM, _GREEN, _MAGENTA, _RED, _RESET, _YELLOW
from mlsweep._shared import NON_SCHEDULABLE_EXPERIMENT_STATUSES, validate_campaign
from mlsweep.run_sweep import (
    _add_manager_args,
    _fmt_bytes,
    _http_request,
    _manager_url,
    _report_failure,
    _with_campaign,
    _leaderboard_header,
    _leaderboard_row,
    _campaign_argv,
    _combo_str,
    _manager_token,
    _parse_combo,
    _resolve_campaign,
    _selected_campaign,
    _wait_until_settled,
    _get_experiment,
    manager_note,
    build_leaderboard,
    manager_cancel_job,
    manager_get_experiment_summary,
    manager_get_job_logs,
    awaits_more_runs,
    manager_get_job_metrics,
    metric_keys,
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


def _jobs_or_exit(manager: str, token: str, experiment: str, campaign: str | None) -> list[dict[str, Any]]:
    """The experiment's jobs.  A failed listing has been reported; do not mistake it for no jobs."""
    jobs = manager_list_experiment_jobs(manager, token, experiment, campaign=campaign)
    if jobs is None:
        sys.exit(1)
    return jobs


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
) -> list[dict[str, Any]]:
    """Run a per-job action (cancel/retry) over *targets*.

    Returns ``{"run_id", "ok"}`` per run (``ok`` is None for a dry run).  *fn*
    prints its own FAIL line with the manager's reason.
    """
    results: list[dict[str, Any]] = []
    for j in targets:
        run_id = j.get("run_id", "")
        if dry_run:
            sweep_print(f"  would {verb} {run_id}")
            results.append({"run_id": run_id, "ok": None})
        else:
            ok = bool(fn(manager, token, run_id, experiment, campaign=campaign))
            if ok:
                sweep_print(f"  OK  {verb} {run_id}")
            results.append({"run_id": run_id, "ok": ok})
    return results


def _failed(results: list[dict[str, Any]]) -> int:
    return sum(1 for r in results if r["ok"] is False)


def _action_result(action: str, experiment: str, results: list[dict[str, Any]], dry_run: bool) -> dict[str, Any]:
    """What a per-run action prints with --json."""
    return {"experiment_id": experiment, "action": action, "dry_run": dry_run, "runs": results}


def _retry(
    targets: list[dict[str, Any]],
    manager: str,
    token: str,
    experiment: str,
    dry_run: bool,
    campaign: str | None = None,
) -> list[dict[str, Any]]:
    """``_apply`` with retry, then warn if the re-queued runs are held."""
    results = _apply(targets, manager_retry_job, "retry", manager, token, experiment,
                     dry_run, campaign)
    if not dry_run and _failed(results) < len(targets):
        _warn_if_held(manager, token, experiment, campaign)
    return results


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
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Section order: what needs attention first.  Unknown statuses go last.
_EXP_SECTIONS = ("running", "paused", "aborted", "failed", "done")
_RUN_SECTIONS = ("running", "dispatched", "pending", "failed", "cancelled", "xfailed", "pruned", "done")
_COUNT_ORDER = ("done", "pruned", "failed", "xfailed", "cancelled", "running", "dispatched", "pending")

_SECTION_COLORS = {
    **_STATUS_COLORS,
    "aborted": _RED,
    "paused": _YELLOW,
    "xfailed": _YELLOW,
    "pruned": _DIM,
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


def _per_step(seconds: float) -> str:
    """``2.31 s/step``, or ``13 steps/s`` for fast steps."""
    if seconds <= 0:
        return "-"
    if seconds < 0.5:
        return f"{1 / seconds:.3g} steps/s"
    return f"{seconds:.3g} s/step"


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
    if status == "done" and c.get("failed") and not c.get("done") and not c.get("xfailed") \
            and not c.get("pruned"):
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
    scope = " --all-campaigns" if campaign is None else f" --campaign {campaign}"
    return (f"  {_paint('Logs:', _C_AUX)} "
            + _paint(f"mlsweep logs {experiment} {run_id}{scope} --tail 50", _C_CMD))


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
        if j.get("last_step") is not None:
            progress = f"step {_paint(j['last_step'], _C_TIME)}"
            avg = j.get("avg_step_seconds")
            if avg is not None:
                progress += f" at {_paint(_per_step(avg), _C_TIME)} on average"
            facts.append(progress)
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
    if j.get("reason"):
        lines.append(f"  {_paint(j['reason'], _C_WARN if status != 'pruned' else _C_AUX)}")
    if status == "failed":
        lines.append(_logs_hint(run_id, experiment, campaign))
    return lines


def _ls_experiments(exps: list[dict[str, Any]], campaign: str | None, show_all: bool) -> None:
    if campaign is not None:
        by_campaign = [(campaign, exps)]
    else:
        names = sorted({e["campaign"] for e in exps})
        by_campaign = [(c, [e for e in exps if e["campaign"] == c])
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
        listed = manager_list_experiment_jobs(manager, token, args.experiment, status_filter=args.status,
                                              campaign=campaign)
        if listed is None:
            sys.exit(1)  # reported; no answer is not "no runs"
        jobs = listed
        if args.json:
            if args.with_dims:
                jobs = [dict(j, dims=_parse_combo(j["combo"]) or {}) for j in jobs]
            emit(jobs)
            return
        summary = manager_get_experiment_summary(manager, token, args.experiment, quiet=True,
                                                 campaign=campaign)
        _ls_runs(args.experiment, jobs, summary, campaign, args.all)
        return

    # Experiments are listed as "done", so accept that for the manager's "completed".
    status_filter = "completed" if args.status == "done" else args.status
    exps = manager_list_experiments(manager, token, status_filter=status_filter, campaign=campaign)
    if exps is None:
        sys.exit(1)
    if args.json:
        emit(exps)
        return
    _ls_experiments(exps, campaign, args.all)


# ── logs ───────────────────────────────────────────────────────────────────────


def logs_cmd(argv: list[str]) -> None:
    parser = _common_parser(
        "mlsweep logs",
        "Print a run's training log: `mlsweep logs EXP RUN`, or `mlsweep logs RUN --experiment EXP`.")
    parser.add_argument("ids", nargs="+", metavar="[EXP] RUN", help="Experiment ID and run ID")
    parser.add_argument("--experiment", help="Experiment ID (instead of giving it first)")
    parser.add_argument("--tail", type=int, default=None, help="Show only the last N lines")
    parser.add_argument("--follow", action="store_true", help="Follow new output")
    parser.add_argument("--json", action="store_true", help="Print the log as JSON (not with --follow)")
    args = parser.parse_args(argv)
    if len(args.ids) == 2 and not args.experiment:
        args.experiment, args.run = args.ids
    elif len(args.ids) == 1 and args.experiment:
        args.run = args.ids[0]
    else:
        parser.error("give EXP RUN, or RUN --experiment EXP")
    if args.json and args.follow:
        parser.error("--follow streams text; it can't be used with --json")
    manager, token, campaign = _connect(args, args.experiment)

    text = manager_get_job_logs(manager, token, args.experiment, args.run, campaign=campaign)
    if text is None:
        sweep_print(f"{_RED}No log found{_RESET} for {args.run} in {args.experiment}")
        sys.exit(1)

    lines = text.splitlines()
    if args.tail:
        lines = lines[-args.tail:]
    if args.json:
        emit({"experiment_id": args.experiment, "run_id": args.run, "lines": lines})
        return
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
    fmt.add_argument("--json", action="store_true",
                     help='Emit JSON, always {"experiment_id", "runs": [{"run_id", "dims", ...}]}: each run has '
                          '"rows" (every step), "last" ({key: latest value}) with --last, or "pivot" with --pivot')
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
    if not args.runs or args.with_dims or args.json:
        jobs = _jobs_or_exit(manager, token, args.experiment, campaign)
    dims, combo_by_run = _dim_columns(jobs)
    if not args.with_dims:
        dims = []  # so every dim column below comes out empty
    run_ids = list(args.runs) or [j["run_id"] for j in jobs]

    # combo_by_run covers every run when --with-dims is on (it fetched all jobs).
    def dim_cells(rid: str, fmt: Callable[[Any], str]) -> list[str]:
        return [fmt(combo_by_run[rid].get(d)) for d in dims]

    def emit_runs(field_name: str, data: dict[str, Any]) -> None:
        """The one --json shape, whatever the view."""
        emit({"experiment_id": args.experiment,
              "runs": [{"run_id": rid, "dims": combo_by_run.get(rid, {}), field_name: value}
                       for rid, value in data.items()]})

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
            emit_runs("last", wide)
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

        if args.json:
            pivots: dict[str, Any] = {}
            for rid, rows in per_run.items():
                if pattern.groups == 1:
                    pstep, vals = pivot_metrics(rows, pattern, args.step)
                    if vals:
                        pivots[rid] = {"step": pstep, "values": vals}
                else:
                    grid, _ = _pivot_run(rows, pattern, args.step)
                    if grid:
                        nested: dict[str, dict[str, Any]] = {}
                        for (x, y), v in grid.items():
                            nested.setdefault(str(x), {})[str(y)] = v
                        pivots[rid] = {"values": nested}
            emit_runs("pivot", pivots)
            return

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
        emit_runs("rows", per_run)
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
# What each selector picks.  A dispatched run is on its worker, still setting up,
# so --running takes it too, and --active is everything not finished.
_SELECTOR_STATUSES = {
    "failed": ("failed",), "cancelled": ("cancelled",), "running": ("running", "dispatched"),
    "pending": ("pending",), "active": ("pending", "dispatched", "running"),
}


def _cancel_retry(argv: list[str], *, retry: bool) -> None:
    verb = "retry" if retry else "cancel"
    parser = _common_parser(
        f"mlsweep {verb}",
        "Re-queue runs in an experiment. They rerun the code submitted with the experiment, so "
        "after fixing a bug submit a new sweep with `mlsweep run`." if retry
        else "Cancel runs in an experiment.",
    )
    parser.add_argument("experiment", help="Experiment ID")
    parser.add_argument("runs", nargs="*", help="Run IDs to act on")
    for s in _SELECTABLE_STATUSES:
        what = "running runs, and dispatched ones still setting up" if s == "running" else f"{s} runs"
        parser.add_argument(f"--{s}", action="store_true", help=f"Select all {what}")
    parser.add_argument("--active", action="store_true",
                        help="Select every unfinished run: pending, dispatched and running")
    parser.add_argument("--all", action="store_true", help="Select every run")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation for --all")
    parser.add_argument("--json", action="store_true", help="Print what happened to each run as JSON")
    args = parser.parse_args(argv)
    manager, token, campaign = _connect(args, args.experiment)

    jobs = _jobs_or_exit(manager, token, args.experiment, campaign)
    statuses = [st for sel, sts in _SELECTOR_STATUSES.items() if getattr(args, sel) for st in sts]
    targets = jobs if args.all else _select_jobs(jobs, args.runs, statuses)

    if not targets:
        sweep_print("  No matching runs.")
        if args.json:
            emit(_action_result(verb, args.experiment, [], args.dry_run))
        return

    if args.all and not args.yes:
        sweep_print(f"{_RED}Refusing to {verb} all {len(targets)} runs without --yes.{_RESET}")
        sys.exit(1)

    if retry:
        results = _retry(targets, manager, token, args.experiment, args.dry_run, campaign)
        sweep_print("  These rerun the code submitted with the experiment. "
                    "To run changed code, submit with `mlsweep run`.")
    else:
        results = _apply(targets, manager_cancel_job, verb, manager, token, args.experiment,
                         args.dry_run, campaign)
        if awaits_more_runs(manager, token, args.experiment, campaign):
            sweep_print(f"  If this is a Bayesian sweep, the manager will choose new trials to replace "
                        f"these. To end it, use `mlsweep stop {args.experiment}`, or `mlsweep optimize "
                        f"{args.experiment} --finish` to let running runs end.")
    if args.json:
        emit(_action_result(verb, args.experiment, results, args.dry_run))
    if _failed(results):
        sys.exit(1)


def cancel_cmd(argv: list[str]) -> None:
    _cancel_retry(argv, retry=False)


def retry_cmd(argv: list[str]) -> None:
    _cancel_retry(argv, retry=True)


# ── reschedule ─────────────────────────────────────────────────────────────────


def reschedule_cmd(argv: list[str]) -> None:
    parser = _common_parser(
        "mlsweep reschedule",
        "Move a run. A running run is stopped and queued again without spending a retry; it "
        "restarts from scratch unless the script resumes from a checkpoint. With --to it starts "
        "next on that worker (and GPUs) even if they are at their jobs-per-GPU limit.",
    )
    parser.add_argument("experiment", help="Experiment ID")
    parser.add_argument("run", help="Run ID")
    parser.add_argument("--to", metavar="WORKER", help="Worker ID to start it on (see `mlsweep workers`)")
    parser.add_argument("--gpus", help="GPU indices on that worker, e.g. 2,3 (default: its least busy)")
    parser.add_argument("--json", action="store_true", help="Print the manager's answer as JSON")
    args = parser.parse_args(argv)
    if args.gpus and not args.to:
        parser.error("--gpus needs --to")
    manager, token, campaign = _connect(args, args.experiment)
    body: dict[str, Any] = {}
    if args.to:
        body = {"worker_id": args.to, **({"gpus": _parse_ints(args.gpus)} if args.gpus else {})}
    url = _manager_url(manager, _with_campaign(
        f"/api/jobs/{args.run}/reschedule?experiment_id={args.experiment}", campaign))
    status, resp = _http_request("POST", url, token, json_data=body)
    ok = status == 200 and isinstance(resp, dict)
    where = ""
    if ok and resp.get("pinned"):
        where = f" to {resp['pinned']['worker_id']} GPU {','.join(map(str, resp['pinned']['gpus']))}"
    _report(ok, f"{args.run}: {'requeued' if ok and resp.get('requeued') else 'queued'}{where}")
    if not ok:
        _fail("reschedule", status, resp)
    if args.json:
        emit({"experiment_id": args.experiment, "run_id": args.run, **resp})


# ── rename ─────────────────────────────────────────────────────────────────────


def rename_cmd(argv: list[str]) -> None:
    parser = _common_parser(
        "mlsweep rename",
        "Give a run a display name, shown next to its run ID in ls, best, and the "
        "dashboard. The run ID itself never changes.",
    )
    parser.add_argument("experiment", help="Experiment ID")
    parser.add_argument("run", nargs="?", help="Run ID")
    parser.add_argument("name", nargs="?", default=None, help="New display name")
    parser.add_argument("--clear", action="store_true", help="Remove the display name")
    parser.add_argument("--experiment-name", metavar="NAME",
                        help="Rename the experiment itself (its ID never changes)")
    parser.add_argument("--json", action="store_true", help="Print the new name as JSON")
    args = parser.parse_args(argv)
    if args.experiment_name is not None:
        if args.run or args.clear:
            parser.error("--experiment-name takes no RUN, NAME or --clear")
        new = args.experiment_name.strip()
        if not new:
            parser.error("--experiment-name must not be empty")
        manager, token, campaign = _connect(args, args.experiment)
        status, resp = _http_request(
            "PUT", _manager_url(manager, _with_campaign(f"/api/experiments/{args.experiment}/name", campaign)),
            token, json_data={"name": new})
        _report(status == 200, f"renamed experiment {args.experiment} to {new!r}")
        if status != 200:
            _fail("rename", status, resp)
        if args.json:
            emit({"experiment_id": args.experiment, "name": new})
        return
    if not args.run:
        parser.error("give RUN and NAME (or --clear), or --experiment-name NAME")
    name = (args.name or "").strip()
    if args.clear == bool(name):
        parser.error("give a non-empty NAME or --clear, not both")
    manager, token, campaign = _connect(args, args.experiment)

    r = manager_set_job_label(manager, token, args.run, args.experiment, name or None,
                              campaign=campaign)
    _report(r, f"cleared the name of {args.run}" if args.clear else f"renamed {args.run} to {name!r}")
    if not r:
        sys.exit(1)
    if args.json:
        emit({"experiment_id": args.experiment, "run_id": args.run, "label": name or None})


# ── stop / pause / unpause ─────────────────────────────────────────────────────


def _set_status_cmd(argv: list[str], name: str, description: str, status: str, done: str,
                    *, confirm: bool = False) -> None:
    parser = _common_parser(f"mlsweep {name}", description)
    parser.add_argument("experiment", help="Experiment ID")
    if confirm:
        parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--json", action="store_true", help="Print the experiment's new status as JSON")
    args = parser.parse_args(argv)
    if confirm and not args.yes:
        sweep_print(f"{_RED}Aborting a sweep is destructive. Pass --yes to confirm.{_RESET}")
        sys.exit(1)
    manager, token, campaign = _connect(args, args.experiment)
    r = manager_set_experiment_status(manager, token, args.experiment, status, campaign=campaign)
    _report(r, f"{done} {args.experiment}")
    if not r:
        sys.exit(1)
    if args.json:
        emit({"experiment_id": args.experiment, "status": status})


def stop_cmd(argv: list[str]) -> None:
    _set_status_cmd(argv, "stop", "Abort an experiment (cancels every unfinished run).",
                    "aborted", "stopped", confirm=True)


def pause_cmd(argv: list[str]) -> None:
    _set_status_cmd(argv, "pause", "Pause an experiment (stops dispatching new jobs).", "paused", "paused")


def unpause_cmd(argv: list[str]) -> None:
    _set_status_cmd(argv, "unpause", "Resume dispatching a paused experiment.", "running", "resumed")


# ── optimize ───────────────────────────────────────────────────────────────────


def _print_optimizer(view: dict[str, Any]) -> None:
    told, budget = view.get("told"), view.get("budget")
    sweep_print(f"  {_BOLD}{view.get('experiment_id')}{_RESET}: {view.get('status')}"
                + (", finishing (no new trials)" if view.get("finishing") else ""))
    sweep_print(f"  metric:     {view.get('metric')} ({view.get('goal')})")
    pruned = view.get("pruned") or 0
    sweep_print(f"  budget:     {told} of {budget} trials have results"
                + (f", plus {pruned} pruned" if pruned else ""))
    sweep_print(f"  trials:     {view.get('running')} running, {view.get('failed')} failed, "
                f"{view.get('abandoned')} cancelled")
    sweep_print(f"  at once:    {view.get('parallel')} (set with --max-concurrent)")
    sweep_print(f"  failures:   {view.get('failure_streak')} in a row, pauses at {view.get('max_failures')}")
    best = view.get("best")
    if isinstance(best, dict):
        sweep_print(f"  best:       {best.get('value')} {_GREEN}{best.get('run_id')}{_RESET}  "
                    f"{_combo_str(best.get('combo'))}")
    if view.get("note"):
        sweep_print(f"  {_YELLOW}note:{_RESET}       {view.get('note')}")


def optimize_cmd(argv: list[str]) -> None:
    parser = _common_parser(
        "mlsweep optimize",
        "Show or adjust a Bayesian sweep. The manager chooses its runs. Use pause, unpause and stop "
        "to control it, and cancel to drop one trial, which the manager replaces.",
    )
    parser.add_argument("experiment", help="Experiment ID")
    parser.add_argument("--budget", default=None, metavar="N|+N",
                        help="How many trials need results, or +N for that many more. "
                             "Undoes --finish, and a completed sweep with budget left runs again.")
    parser.add_argument("--finish", action="store_true",
                        help="Start no new trials, and complete once the running ones end")
    parser.add_argument("--max-concurrent", type=int, default=None, metavar="K",
                        help="How many trials and jobs run at once (0 = the default, n_initial)")
    parser.add_argument("--max-failures", type=int, default=None, metavar="N",
                        help="Failed trials in a row before the sweep pauses")
    parser.add_argument("--json", action="store_true", help="Print the status as JSON")
    args = parser.parse_args(argv)
    manager, token, campaign = _connect(args, args.experiment)
    url = _manager_url(manager, _with_campaign(f"/api/experiments/{args.experiment}/optimize", campaign))

    status, view = _http_request("GET", url, token)
    if status != 200 or not isinstance(view, dict):
        _fail("optimize", status, view)

    body: dict[str, Any] = {}
    if args.budget is not None:
        try:
            body["budget"] = int(args.budget) + (int(view["budget"]) if args.budget.startswith("+") else 0)
        except ValueError:
            parser.error(f"--budget must be N or +N, not {args.budget!r}")
    if args.finish:
        if args.budget is not None:
            parser.error("--budget undoes --finish, so use one of them")
        body["finish"] = True
    if args.max_failures is not None:
        body["max_failures"] = args.max_failures
    if args.max_concurrent is not None:
        cap_url = _with_campaign(f"/api/experiments/{args.experiment}/max_concurrent", campaign)
        status, resp = _http_request("PUT", _manager_url(manager, cap_url), token,
                                     json_data={"max_concurrent": args.max_concurrent})
        if status != 200:
            _fail("optimize", status, resp)
    if body:
        status, view = _http_request("PATCH", url, token, json_data=body)
        if status != 200 or not isinstance(view, dict):
            _fail("optimize", status, view)
    elif args.max_concurrent is not None:
        status, view = _http_request("GET", url, token)
        if status != 200 or not isinstance(view, dict):
            _fail("optimize", status, view)

    if args.json:
        emit(view)
    else:
        _print_optimizer(view)


# ── controls ───────────────────────────────────────────────────────────────────


def controls_cmd(argv: list[str]) -> None:
    from mlsweep.run_sweep import describe_controls

    parser = _common_parser(
        "mlsweep controls",
        "Show or change an experiment's run controls (MAX_RUNTIME, PRUNE, METRIC_CHECK) while it runs.",
    )
    parser.add_argument("experiment", help="Experiment ID")
    parser.add_argument("--max-runtime", default=None, metavar="DURATION|off",
                        help="Stop and fail runs that run longer, e.g. 4h or 90m; off removes it. "
                             "Applies to runs already running.")
    parser.add_argument("--prune", default=None, metavar="JSON|off",
                        help='A PRUNE dict as JSON, e.g. \'{"strategy": "asha", "min_steps": 500}\', or off')
    parser.add_argument("--metric-check", default=None, metavar="N|on|off",
                        help="Pause if the first N finished runs never log the metric (on = 2)")
    parser.add_argument("--json", action="store_true", help="Print the controls as JSON")
    args = parser.parse_args(argv)
    manager, token, campaign = _connect(args, args.experiment)
    url = _manager_url(manager, _with_campaign(f"/api/experiments/{args.experiment}/controls", campaign))

    body: dict[str, Any] = {}
    if args.max_runtime is not None:
        body["max_runtime"] = None if args.max_runtime == "off" else args.max_runtime
    if args.prune is not None:
        if args.prune == "off":
            body["prune"] = None
        else:
            try:
                body["prune"] = json.loads(args.prune)
            except ValueError:
                parser.error(f"--prune must be JSON or off, not {args.prune!r}")
    if args.metric_check is not None:
        mc = args.metric_check
        if mc in ("on", "off"):
            body["metric_check"] = mc == "on"
        elif mc.isdigit() and int(mc) > 0:
            body["metric_check"] = int(mc)
        else:
            parser.error("--metric-check must be a positive number, on or off")

    status, view = _http_request("PATCH" if body else "GET", url, token, json_data=body or None)
    if status != 200 or not isinstance(view, dict):
        _fail("controls", status, view)
    if args.json:
        emit(view)
        return
    lines = describe_controls(view)
    sweep_print("\n".join(f"  {line}" for line in lines) if lines else "  no run controls")
    if view.get("note"):
        sweep_print(f"  {_paint(view['note'], _C_WARN)}")


# ── wait ───────────────────────────────────────────────────────────────────────


# Exit codes for `mlsweep wait` (documented in its --help).
WAIT_EXIT_SETTLED = 0   # settled cleanly, or the requested condition was met
WAIT_EXIT_FAILURE = 1   # at least one run failed
WAIT_EXIT_TIMEOUT = 2   # --timeout elapsed before the condition was met
WAIT_EXIT_STALLED = 3   # a running run made no progress for --stalled-after seconds
WAIT_EXIT_UNKNOWN = 4   # no answer: manager unreachable, unknown experiment, or no runs
WAIT_EXIT_CANCELLED = 5 # settled with no failures, but runs were cancelled or the sweep stopped

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
      0  every run finished successfully
      1  at least one run failed, or the manager paused the sweep (METRIC_CHECK)
      2  --timeout elapsed first, and runs are still active
      3  --until stalled and a running run stopped making progress
      4  no answer: manager unreachable, unknown experiment, or no runs
      5  no failures, but some runs were cancelled or the sweep was stopped
    """
    parser = argparse.ArgumentParser(
        prog="mlsweep wait",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Wait until an experiment settles, fails, or stalls.\n\n"
            "Exit codes:\n"
            "  0  every run finished successfully\n"
            "  1  at least one run failed, or the manager paused the sweep (METRIC_CHECK)\n"
            "  2  timed out (--timeout), runs still active; waiting again is safe\n"
            "  3  a running run stalled (--until stalled)\n"
            "  4  no answer: manager unreachable, unknown experiment, or no runs\n"
            "  5  no failures, but some runs were cancelled or the sweep was stopped"
        ),
    )
    _add_manager_args(parser)
    parser.add_argument("experiment", help="Experiment ID")
    parser.add_argument(
        "--until", choices=("done", "any-failure", "any-done", "stalled"), default="done",
        help="Condition to wait for (default: done = no active runs). any-done returns as soon as "
             "any run finishes, however it ended; pass --since to pick up where the last call left off",
    )
    parser.add_argument(
        "--since", default=None, metavar="T",
        help="With --until any-done: runs that finished after T count (epoch seconds, or an ISO time; "
             "default: now). The JSON result's 'cursor' is the T for the next call",
    )
    parser.add_argument("--json", action="store_true",
                        help="Print the verdict as JSON: outcome, exit code, the runs it is about, and counts")
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
    try:
        since = _parse_since(args.since) if args.since is not None else time.time()
    except ValueError:
        parser.error(f"--since must be epoch seconds or an ISO time, not {args.since!r}")
    while True:
        jobs = manager_list_experiment_jobs(manager, token, args.experiment, campaign=campaign)
        verdict = wait_outcome(args.experiment, jobs, args.until, args.stalled_after,
                               lambda: _get_experiment(manager, token, args.experiment, campaign), since=since)
        if verdict is None and args.timeout and time.monotonic() - start >= args.timeout:
            active = [j for j in jobs or [] if j.get("status") in _WAIT_ACTIVE_STATUSES]
            if active:
                message = (f"{_YELLOW}TIMEOUT{_RESET}  {args.experiment} still has "
                           f"{len(active)} active run(s) after {args.timeout:g}s.")
            else:
                message = (f"{_YELLOW}TIMEOUT{_RESET}  {args.experiment} has no active runs right now, "
                           f"but the manager still expects more. `mlsweep optimize "
                           f"{args.experiment}` shows where a Bayesian sweep stands.")
            verdict = WaitVerdict(WAIT_EXIT_TIMEOUT, message, "timeout", active)
        if verdict is not None:
            sweep_print(verdict.message)
            if args.json:
                emit(verdict.to_json(args.experiment, jobs, since))
            sys.exit(verdict.code)
        time.sleep(args.interval)


def _parse_since(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        t = datetime.fromisoformat(raw)
        return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp()


def _finish_epoch(job: dict[str, Any]) -> float | None:
    t = _parse_time(job.get("finish_time"))
    return t.timestamp() if t else None


@dataclass
class WaitVerdict:
    """What one poll of ``mlsweep wait`` decided: exit with *code*, saying *message*.

    *outcome* names it for --json (settled, failure, cancelled, stopped, paused,
    stalled, run_finished, timeout, unknown), and *runs* are the runs it is about:
    the failed, stalled or newly finished ones.
    """

    code: int
    message: str
    outcome: str
    runs: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self, experiment: str, jobs: list[dict[str, Any]] | None, since: float) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for j in jobs or []:
            counts[str(j.get("status"))] = counts.get(str(j.get("status")), 0) + 1
        out: dict[str, Any] = {
            # No answer (manager unreachable, unknown experiment) is a failure like any other.
            **({"error": _ANSI_RE.sub("", self.message).replace("FAIL  ", "")} if self.outcome == "unknown" else {}),
            "experiment_id": experiment, "outcome": self.outcome, "exit_code": self.code,
            "message": _ANSI_RE.sub("", self.message),
            "runs": [{k: j.get(k) for k in ("run_id", "status", "reason", "exit_code", "finish_time")}
                     for j in self.runs],
            "counts": counts,
        }
        # The --since of the next any-done call: the newest finish seen, so nothing is missed or repeated.
        finished = [t for j in jobs or [] if (t := _finish_epoch(j)) is not None]
        out["cursor"] = max([since, *finished])
        return out


def wait_outcome(
    experiment: str,
    jobs: list[dict[str, Any]] | None,
    until: str,
    stalled_after: float,
    get_experiment: Callable[[], dict[str, Any]],
    since: float = 0.0,
) -> WaitVerdict | None:
    """What ``mlsweep wait`` does after one poll, or None to keep waiting.

    *jobs* is the experiment's runs (None if the manager did not answer), and
    *get_experiment* fetches the experiment itself, only when it is needed.
    For ``any-done``, runs that finished after *since* (epoch seconds) count.
    """
    if jobs is None:
        return WaitVerdict(WAIT_EXIT_UNKNOWN, f"{_RED}FAIL{_RESET}  Cannot list jobs for {experiment}", "unknown")
    if not jobs:
        return WaitVerdict(WAIT_EXIT_UNKNOWN, f"{_RED}FAIL{_RESET}  {experiment} has no runs. "
                                              "Check the ID with `mlsweep ls`.", "unknown")

    failed = _wait_failed(jobs)
    active = [j for j in jobs if j.get("status") in _WAIT_ACTIVE_STATUSES]

    if until == "any-failure" and failed:
        names = ", ".join(j.get("run_id", "?") for j in failed)
        return WaitVerdict(WAIT_EXIT_FAILURE, f"{_RED}FAILURE{_RESET}  {len(failed)} run(s) failed: {names}",
                           "failure", failed)

    if until == "any-done":
        newly = sorted((j for j in jobs if j.get("status") not in _WAIT_ACTIVE_STATUSES
                        and (t := _finish_epoch(j)) is not None and t > since), key=lambda j: _finish_epoch(j) or 0)
        if newly:
            names = ", ".join(f"{j.get('run_id', '?')} ({j.get('status')})" for j in newly)
            return WaitVerdict(WAIT_EXIT_SETTLED, f"{_GREEN}FINISHED{_RESET}  {len(newly)} run(s): {names}",
                               "run_finished", newly)

    if active and all(j.get("status") == "pending" for j in active):
        exp = get_experiment()
        # Runs added after a `stop`, or left by a stop from an older version, never start.
        if exp.get("status") == "aborted":
            return WaitVerdict(WAIT_EXIT_FAILURE if failed else WAIT_EXIT_CANCELLED,
                               f"{_YELLOW}STOPPED{_RESET}  {experiment} was stopped, leaving "
                               f"{len(active)} run(s) that will not start.", "stopped", failed)
        # The manager paused it (e.g. METRIC_CHECK); it waits for someone to look.
        note = manager_note(exp)
        if exp.get("status") == "paused" and note:
            return WaitVerdict(WAIT_EXIT_FAILURE, f"{_RED}PAUSED{_RESET}  {experiment}: {note}", "paused")

    if not active:
        exp = get_experiment()
        # A Bayesian sweep adds runs as results come in, so a gap isn't the end.
        if exp.get("status") == "running" and (exp.get("optimize") or exp.get("expected_jobs")):
            return None
        # Settled: nothing pending/dispatched/running.
        note = manager_note(exp)
        lines = [f"{_YELLOW}NOTE{_RESET}  {note}"] if note else []
        about: list[dict[str, Any]] = []
        if failed:
            names = ", ".join(j.get("run_id", "?") for j in failed)
            code, outcome, about = WAIT_EXIT_FAILURE, "failure", failed
            lines.append(f"{_RED}FAILURE{_RESET}  {experiment} settled with {len(failed)} failed run(s): {names}")
        elif cancelled := [j for j in jobs if j.get("status") == "cancelled"]:
            code, outcome, about = WAIT_EXIT_CANCELLED, "cancelled", cancelled
            lines.append(f"{_YELLOW}CANCELLED{_RESET}  {experiment} settled with "
                         f"{len(cancelled)} of {len(jobs)} run(s) cancelled.")
        else:
            code, outcome = WAIT_EXIT_SETTLED, "settled"
            pruned = sum(1 for j in jobs if j.get("status") == "pruned")
            lines.append(f"{_GREEN}DONE{_RESET}  {experiment} settled cleanly"
                         + (f" ({pruned} of {len(jobs)} run(s) pruned)." if pruned else "."))
        return WaitVerdict(code, "\n".join(lines), outcome, about)

    if until == "stalled":
        stalled = _wait_stalled(jobs, stalled_after)
        if stalled:
            names = ", ".join(j.get("run_id", "?") for j in stalled)
            return WaitVerdict(WAIT_EXIT_STALLED, f"{_YELLOW}STALLED{_RESET}  {len(stalled)} running run(s) "
                                                  f"made no progress in {stalled_after:g}s: {names}",
                               "stalled", stalled)
    return None


# ── resume ─────────────────────────────────────────────────────────────────────


def resume_cmd(argv: list[str]) -> None:
    parser = _common_parser(
        "mlsweep resume",
        "Re-queue an experiment's failed and cancelled runs. A Bayesian sweep instead goes back "
        "to running and the manager chooses new trials. It does not undo `pause`; use `unpause` for that.")
    parser.add_argument("experiment", help="Experiment ID")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--json", action="store_true",
                        help="Print what happened as JSON (a Bayesian sweep's has no runs, and its new status)")
    args = parser.parse_args(argv)
    manager, token, campaign = _connect(args, args.experiment)

    if _get_experiment(manager, token, args.experiment, campaign).get("optimize"):
        if args.dry_run:
            sweep_print(f"  Would set {args.experiment} running, so the manager chooses new trials.")
        else:
            r = manager_set_experiment_status(manager, token, args.experiment, "running", campaign=campaign)
            _report(r, f"resumed {args.experiment}; the manager chooses new trials")
            if not r:
                sys.exit(1)
        if args.json:
            emit({**_action_result("resume", args.experiment, [], args.dry_run), "status": "running"})
        return

    jobs = _jobs_or_exit(manager, token, args.experiment, campaign)
    targets = [j for j in jobs if j.get("status") in ("failed", "cancelled")]
    results: list[dict[str, Any]] = []
    if not targets:
        sweep_print("  Nothing to resume, no failed or cancelled jobs.")
    else:
        results = _retry(targets, manager, token, args.experiment, args.dry_run, campaign)
    if args.json:
        emit(_action_result("resume", args.experiment, results, args.dry_run))
    if _failed(results):
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
    jobs = _jobs_or_exit(manager, token, args.experiment, campaign)
    rows = build_leaderboard(manager, token, args.experiment, metric, goal, jobs=jobs, campaign=campaign)

    if args.json:
        if args.with_dims:
            rows = [{**r, **(r["combo"] or {})} for r in rows]
        emit(rows)
    elif args.group_by:
        _print_grouped_leaderboard(rows, args.group_by, metric, goal, args.top)
    elif args.table:
        _print_metric_table(rows, args.table, metric, goal)
    else:
        print_leaderboard(rows, metric, goal, args.top)
    done = [r["run_id"] for r in rows if r["status"] == "done"]
    if not args.json and done and all(r["value"] is None for r in rows):
        logged = metric_keys(manager, token, args.experiment, done[0], campaign=campaign)
        print(f"Error: no finished run logged {metric!r}. {done[0]} logged: "
              f"{', '.join(logged) or 'no metrics'}. Pass --metric.", file=sys.stderr)
        sys.exit(1)
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
    move.add_argument("--json", action="store_true", help="Print the result as JSON")
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
        if args.json:
            emit({"experiment_id": args.experiment, "campaign": args.target})
        return

    manager, token = _manager_token(args)
    current = _selected_campaign(args)
    camps = manager_list_campaigns(manager, token)
    if camps is None:
        sweep_print(f"{_RED}FAIL{_RESET}  list campaigns")
        sys.exit(1)
    if args.json:
        emit(camps)
        return
    sweep_print(f"{_BOLD}{_CYAN}{len(camps)}{_RESET} campaigns:")
    for c in camps:
        name = c.get("campaign", "?")
        mark = "*" if name == current else " "
        n = c.get("experiments", 0)
        sweep_print(f"  {mark} {_GREEN}{name}{_RESET}  {n} experiment{'' if n == 1 else 's'}, "
                    f"{_counts_str(c.get('job_counts'))}")
    if current is not None:
        sweep_print("  (* = current campaign, from --campaign, a context, or $MLSWEEP_CAMPAIGN)")


# ── move / purge / gc ──────────────────────────────────────────────────────────


def _select_runs(manager: str, token: str, experiment: str, campaign: str | None,
                 run_ids: list[str], statuses: list[str]) -> list[str]:
    jobs = manager_list_experiment_jobs(manager, token, experiment, campaign=campaign)
    if jobs is None:
        sys.exit(1)
    known = {j.get("run_id") for j in jobs}
    missing = [r for r in run_ids if r not in known]
    if missing:
        sweep_print(f"{_RED}Error: no such run(s) in {experiment}: {', '.join(missing)}{_RESET}")
        sys.exit(1)
    return [j["run_id"] for j in _select_jobs(jobs, run_ids, statuses)]


def _fail(verb: str, status: int, resp: Any) -> NoReturn:
    """Print the server's per-run ``problems`` (or the generic failure), then exit 1."""
    problems = (resp.get("problems") if isinstance(resp, dict) else None) or []
    for p in problems:
        _report(False, p)
    if not problems:
        _report_failure(verb, status, resp)
    sys.exit(1)


def _add_run_flags(parser: argparse.ArgumentParser, flags: tuple[str, ...], help_text: str) -> None:
    """Add one flag per status in *flags*, selecting the runs with that status."""
    for st in flags:
        parser.add_argument(f"--{st}", action="store_true", help=help_text.format(st))


def _chosen_runs(
    parser: argparse.ArgumentParser, args: argparse.Namespace, flags: tuple[str, ...],
) -> tuple[str, str, str | None, list[str]]:
    """``(manager, token, campaign, run_ids)`` for the runs a command names or selects."""
    statuses = [s for s in flags if getattr(args, s)]
    if not args.runs and not statuses:
        parser.error("name runs, or select them with " + "/".join(f"--{s}" for s in flags))
    manager, token, campaign = _connect(args, args.experiment)
    return manager, token, campaign, _select_runs(manager, token, args.experiment, campaign, args.runs, statuses)


def _post_runs(
    verb: str, args: argparse.Namespace, manager: str, token: str, campaign: str | None, body: dict[str, Any],
) -> Any:
    """POST to the experiment's ``jobs/<verb>`` route; print JSON if asked, exit 1 on failure."""
    status, resp = _http_request(
        "POST",
        _manager_url(manager, _with_campaign(f"/api/experiments/{args.experiment}/jobs/{verb}", campaign)),
        token, json_data=body,
    )
    if args.json:
        emit(resp)
    if status != 200:
        _fail(verb, status, resp)
    return resp


def move_cmd(argv: list[str]) -> None:
    parser = _common_parser(
        "mlsweep move",
        "Move finished runs to another experiment, with their metrics, logs and files. "
        "Runs still pending or running must be cancelled first.",
    )
    parser.add_argument("experiment", help="Experiment the runs are in")
    parser.add_argument("runs", nargs="*", help="Run IDs to move")
    parser.add_argument("--to", required=True, metavar="EXPERIMENT", help="Experiment to move them to")
    flags = ("done", "failed", "cancelled", "xfailed")
    _add_run_flags(parser, flags, "also every {} run")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    manager, token, campaign, run_ids = _chosen_runs(parser, args, flags)
    if not run_ids:
        sweep_print("No runs selected.")
        return
    resp = _post_runs("move", args, manager, token, campaign, {"run_ids": run_ids, "to": args.to})
    if not args.json:
        _report(True, f"moved {len(resp['moved'])} run(s) from {args.experiment} to {args.to}")
        for rid in resp.get("files_not_moved", []):
            _report(False, f"{rid}: its files could not be moved (see the manager log)")


def purge_cmd(argv: list[str]) -> None:
    parser = _common_parser(
        "mlsweep purge",
        "Delete finished runs for good: their records, metrics, logs and files. "
        "Use it to clear out failed or cancelled attempts.",
    )
    parser.add_argument("experiment", help="Experiment ID")
    parser.add_argument("runs", nargs="*", help="Run IDs to delete")
    flags = ("failed", "cancelled", "xfailed", "done")
    _add_run_flags(parser, flags, "every {} run")
    parser.add_argument("--yes", action="store_true", help="Actually delete (otherwise list what would go)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    manager, token, campaign, run_ids = _chosen_runs(parser, args, flags)
    if not run_ids:
        sweep_print("No runs selected.")
        return
    if not args.yes:
        sweep_print(f"Would delete {len(run_ids)} run(s) from {args.experiment}:")
        for rid in run_ids:
            sweep_print(f"  {rid}")
        sweep_print(f"{_YELLOW}Nothing deleted. Pass --yes to delete them (this cannot be undone).{_RESET}")
        return
    resp = _post_runs("purge", args, manager, token, campaign, {"run_ids": run_ids})
    if not args.json:
        _report(True, f"deleted {len(resp['purged'])} run(s), freed {_fmt_bytes(resp.get('bytes_freed', 0))}")


def gc_cmd(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="mlsweep gc",
        description="Find what nothing refers to any more: files of deleted experiments and runs, "
                    "database rows of deleted runs, code archives no run uses, and (with "
                    "--downloads) local downloads of experiments the manager no longer has. "
                    "Lists them; --yes deletes them.",
    )
    _add_manager_args(parser)
    parser.add_argument("--yes", action="store_true", help="Delete what is listed")
    parser.add_argument("--vacuum", action="store_true", help="Also compact the manager's database file")
    parser.add_argument("--downloads", action="store_true",
                        help="Also check this machine's ~/.mlsweep/downloads (from `mlsweep fetch`)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    manager, token = _manager_token(args)
    status, plan = _http_request("POST", _manager_url(manager, "/api/gc"), token,
                                 json_data={"dry_run": not args.yes, "vacuum": args.vacuum and args.yes},
                                 timeout=3600)
    if status != 200 or not isinstance(plan, dict):
        _fail("gc", status, plan)

    local: list[tuple[str, int]] = []
    if args.downloads:
        from mlsweep._shared import _dir_size, _mlsweep_dir
        exps = manager_list_experiments(manager, token)
        if exps is None:
            sys.exit(1)
        known = {e["experiment_id"] for e in exps}
        root = _mlsweep_dir() / "downloads"
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if child.is_dir() and not child.is_symlink() and child.name not in known:
                    local.append((str(child), _dir_size(child)))
        if args.yes:
            import shutil
            for path, _ in local:
                shutil.rmtree(path, ignore_errors=True)
        plan["local_downloads"] = [{"path": p, "bytes": b} for p, b in local]

    if args.json:
        emit(plan)
        return
    db = plan["database"]
    verb = "Deleted" if args.yes else "Would delete"
    total = plan["bytes"] + sum(b for _, b in local)
    rows = sum(db.values())
    sweep_print(f"{_BOLD}{verb}{_RESET} {_fmt_bytes(total)}:")
    for d in plan["directories"]:
        sweep_print(f"  {d['path']:<70} {_fmt_bytes(d['bytes']):>9}  ({d['reason']})")
    for f in plan["artifact_files"]:
        sweep_print(f"  {f['path'][:70]:<70} {_fmt_bytes(f['bytes']):>9}  ({f['reason']})")
    for p, b in local:
        sweep_print(f"  {p:<70} {_fmt_bytes(b):>9}  (download of a deleted experiment)")
    objects = plan.get("objects") or {"count": 0, "bytes": 0}
    if objects["count"]:
        sweep_print(f"  {'store/objects: ' + str(objects['count']) + ' stored files':<70} "
                    f"{_fmt_bytes(objects['bytes']):>9}  (no code snapshot lists them)")
    sweep_print(f"  database: {db['metric_rows']} metric rows, {db['log_rows']} log rows, "
                f"{db['job_node_rows']} run-node rows, {plan['artifact_rows']} code records")
    if not args.yes:
        if total or rows or plan["artifact_rows"]:
            sweep_print(f"{_YELLOW}Nothing deleted. Pass --yes to delete these "
                        f"(add --vacuum to also shrink the database file).{_RESET}")
        else:
            sweep_print("Nothing to collect.")
    elif plan.get("vacuumed"):
        sweep_print("Database compacted.")


# ── code: the code snapshots experiments ran ───────────────────────────────────


_CODE_HELP = """\
mlsweep code EXP                     what code EXP's runs used, and where it came from
mlsweep code EXP PATH                print one file as EXP ran it
mlsweep code EXP PATH --lines 40-60  numbered lines, with their note reference (-n: the file, paged)
mlsweep code mlsweep:code/EXP/PATH#L40-60   the same, from a note's link
mlsweep code search PATTERN [EXP...] lines that match, in EXP or across the campaign (-E, -i, -C N)
mlsweep code EXP --files [PREFIX]    list the snapshot's files
mlsweep code diff A B [--path P]     what changed between two experiments' code (--stat for a summary)
mlsweep code checkout EXP DIR        write EXP's code out to DIR

EXP is an experiment ID or a snapshot ID (snap:..., or a unique prefix of 8+ hex digits).
When an experiment's runs used more than one snapshot, --snapshot picks one; the default
is the latest submitted."""


def _code_get(manager: str, token: str, path: str, *, raw: bool = False, timeout: int = 120) -> Any:
    """GET *path*; exit with the manager's error on failure.  *raw* returns bytes."""
    status, resp = _http_request("GET", _manager_url(manager, path), token, timeout=timeout, raw=raw)
    if status != 200:
        _fail(f"GET {path}", status, resp)
    return resp


def _die(msg: str) -> NoReturn:
    sweep_print(f"{_RED}Error: {msg}{_RESET}")
    sys.exit(1)


def _experiment_code(manager: str, token: str, exp: str, campaign: str | None) -> dict[str, Any]:
    require_campaign(manager, token, exp, campaign)
    return _code_get(manager, token, _with_campaign(f"/api/experiments/{exp}/code", campaign))  # type: ignore[no-any-return]


def _pick_snapshot(info: dict[str, Any], wanted: str | None) -> str:
    """The snapshot of an experiment to show: *wanted*, else the latest submitted, else the most used."""
    versions = info.get("versions") or []
    ids = [v["artifact_id"] for v in versions]
    if wanted:
        matches: list[str] = [i for i in ids if i.startswith(wanted)]
        if len(matches) != 1:
            _die(f"{wanted} is {'ambiguous' if matches else 'not one of'} this experiment's code versions: "
                 + ", ".join(ids))
        return matches[0]
    subs = [s["snapshot"] for s in info.get("submissions") or [] if s.get("snapshot") in ids]
    if subs:
        return subs[-1]  # type: ignore[no-any-return]
    if not ids:
        _die(f"no run of {info.get('experiment_id')} has code recorded")
    return max(versions, key=lambda v: v["runs"])["artifact_id"]  # type: ignore[no-any-return]


def _resolve_code_ref(manager: str, token: str, ref: str, campaign: str | None,
                      snapshot: str | None) -> str:
    """A snapshot ID from an experiment ID or a snapshot ID (or prefix)."""
    from mlsweep._snapshot import is_snapshot_id, short_id
    if is_snapshot_id(ref):
        return ref
    info = _experiment_code(manager, token, ref, campaign)
    sid = _pick_snapshot(info, snapshot)
    if not is_snapshot_id(sid):
        _die(f"{ref} ran from a code archive made before snapshots, which can't be shown ({short_id(sid)}...).")
    return sid


def _snapshot_manifest(manager: str, token: str, sid: str) -> tuple[str, Any]:
    from mlsweep._snapshot import parse_manifest
    resp = _code_get(manager, token, f"/api/snapshots/{sid}")
    return resp["artifact_id"], parse_manifest(resp["manifest"])


def _print_code_info(info: dict[str, Any]) -> None:
    from mlsweep._snapshot import short_id
    versions = info.get("versions") or []
    subs = info.get("submissions") or []
    sweep_print(f"{_BOLD}{info['experiment_id']}{_RESET}")
    if not versions:
        sweep_print("  no run has code recorded")
    for v in versions:
        aid = v["artifact_id"]
        if v.get("kind") == "snapshot":
            size = f"{v['files']} files, {_fmt_bytes(v['size_bytes'])}"
            sweep_print(f"  {_CYAN}{short_id(aid)}{_RESET}  {size}  ·  {v['runs']} run(s)")
        else:
            sweep_print(f"  {short_id(aid)}...  code archive from before snapshots  ·  {v['runs']} run(s)")
    if len(versions) > 1:
        sweep_print(f"  {_YELLOW}Runs of this experiment used {len(versions)} different code versions.{_RESET}")
    for sub in subs:
        meta = sub.get("meta") or {}
        when = (sub.get("submit_time") or "")[:19].replace("T", " ")
        sid = sub.get("snapshot") or "?"
        sweep_print(f"\n  {_BOLD}Submitted{_RESET} {when} UTC from {meta.get('host', '?')}: {short_id(sid)}")
        if meta.get("sweep_file"):
            sweep_print(f"    sweep file  {meta['sweep_file']}"
                        + ("" if meta.get("sweep_file_in_snapshot") else "  (outside the snapshot; source kept)"))
        if meta.get("argv"):
            sweep_print(f"    command     mlsweep run {' '.join(meta['argv'])}")
        vcs = meta.get("vcs")
        if vcs:
            commit = (vcs.get("commit") or "no commit")[:12]
            line = f"    git         {commit} {vcs.get('branch') or '(detached)'}"
            if vcs.get("subject"):
                line += f"  {vcs['subject'][:70]}"
            sweep_print(line)
            if vcs.get("dirty"):
                n = len(vcs.get("changed_files") or [])
                sweep_print(f"                {_YELLOW}{n} file(s) changed since that commit{_RESET} "
                            f"(`mlsweep code {info['experiment_id']} --uncommitted` shows the diff)")
            if vcs.get("untracked_count"):
                sweep_print(f"                {vcs['untracked_count']} file(s) not tracked by git, "
                            f"{_fmt_bytes(vcs.get('untracked_bytes') or 0)}")
            if vcs.get("remote"):
                sweep_print(f"    remote      {vcs['remote']}")


def _code_diff(manager: str, token: str, a: str, b: str, prefix: str | None, stat_only: bool) -> None:
    import difflib

    from mlsweep._snapshot import FileEntry, diff_manifests, looks_binary, short_id
    a_id, ma = _snapshot_manifest(manager, token, a)
    b_id, mb = _snapshot_manifest(manager, token, b)
    d = diff_manifests(ma, mb, prefix)
    sweep_print(f"{_BOLD}{short_id(a_id)} → {short_id(b_id)}{_RESET}: {len(d.changed)} changed, "
                f"{len(d.added)} added, {len(d.removed)} removed" + (f" under {prefix}" if prefix else ""))
    if d.empty:
        return
    if stat_only:
        for old, new in d.changed:
            note = " (mode)" if old.hash == new.hash else ""
            sweep_print(f"  M {new.path}  {_fmt_bytes(old.size)} → {_fmt_bytes(new.size)}{note}")
        for f in d.added:
            sweep_print(f"  {_GREEN}A{_RESET} {f.path}  {_fmt_bytes(f.size)}")
        for f in d.removed:
            sweep_print(f"  {_RED}D{_RESET} {f.path}")
        return

    limit = 2 << 20

    def text(h: str | None, size: int) -> list[str] | None:
        if h is None:
            return []
        if size > limit:
            return None
        data: bytes = _code_get(manager, token, f"/api/objects/{h}", raw=True)
        if looks_binary(data):
            return None
        return data.decode("utf-8", errors="replace").splitlines(keepends=True)

    pairs: list[tuple[str, FileEntry | None, FileEntry | None]] = (
        [(o.path, o, n) for o, n in d.changed] + [(f.path, None, f) for f in d.added]
        + [(f.path, f, None) for f in d.removed])
    for path, was, now in sorted(pairs, key=lambda t: t[0]):
        before = text(was.hash if was else None, was.size if was else 0)
        after = text(now.hash if now else None, now.size if now else 0)
        if before is None or after is None:
            sweep_print(f"Binary or large file {path} differs "
                        f"({_fmt_bytes(was.size) if was else 'absent'} → {_fmt_bytes(now.size) if now else 'absent'})")
            continue
        lines = list(difflib.unified_diff(before, after, fromfile=f"a/{path}" if was else "/dev/null",
                                          tofile=f"b/{path}" if now else "/dev/null"))
        if not lines and was and now:
            sweep_print(f"mode change {path}: {oct(was.mode)} → {oct(now.mode)}")
        for line in lines:
            sys.stdout.write(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")


def _code_checkout(manager: str, token: str, sid: str, dest_dir: str) -> None:
    import tempfile
    from pathlib import Path

    from mlsweep._snapshot import ObjectStore, fetch_object, materialize
    dest = Path(dest_dir)
    if dest.exists() and any(dest.iterdir()):
        _die(f"{dest} is not empty")
    sid, manifest = _snapshot_manifest(manager, token, sid)
    with tempfile.TemporaryDirectory(prefix="mlsweep-code-") as tmp:
        store = ObjectStore(tmp)

        def fetch(h: str) -> None:
            fetch_object(store, _manager_url(manager, f"/api/objects/{h}"), h,
                         headers={"Authorization": f"Bearer {token}"}, timeout=600)

        materialize(manifest, dest, store, fetch=fetch)
    sweep_print(f"Wrote {len(manifest.files)} files ({_fmt_bytes(manifest.total_bytes)}) of {sid} to {dest}")


def _print_uncommitted(manager: str, token: str, exp: str, campaign: str | None, snapshot: str | None) -> None:
    info = _experiment_code(manager, token, exp, campaign)
    sid = _pick_snapshot(info, snapshot)
    subs = [s for s in info.get("submissions") or [] if s.get("snapshot") == sid]
    if not subs:
        _die(f"no submission of {sid[:17]} to {exp} recorded where its code came from")
    full = _code_get(manager, token, f"/api/code/submissions/{subs[-1]['submission_id']}")
    vcs = (full.get("meta") or {}).get("vcs") or {}
    if not vcs.get("diff"):
        sweep_print("No uncommitted changes were recorded." if vcs else "The code was not in a git repository.")
        return
    sys.stdout.write(vcs["diff"] if vcs["diff"].endswith("\n") else vcs["diff"] + "\n")
    if vcs.get("diff_truncated"):
        sweep_print(f"{_YELLOW}(cut at 1 MB when it was recorded){_RESET}")


def code_cmd(argv: list[str]) -> None:
    verb = argv[0] if argv and argv[0] in ("diff", "checkout", "search") else "show"
    parser = argparse.ArgumentParser(prog=f"mlsweep code{'' if verb == 'show' else ' ' + verb}",
                                     description="The code experiments ran: show, print, search, diff, "
                                                 "check out.",
                                     epilog=_CODE_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_manager_args(parser)
    parser.add_argument("--json", action="store_true", help="Print what the manager returned, as JSON")
    if verb == "show":
        parser.add_argument("ref", metavar="EXP",
                            help="Experiment ID, snapshot ID, or a note's mlsweep:code/EXP/PATH#L40-60 link")
        parser.add_argument("path", nargs="?", help="A file to print, as the experiment ran it")
        parser.add_argument("--files", nargs="?", const="", metavar="PREFIX", help="List the snapshot's files")
        parser.add_argument("--snapshot", help="Which of the experiment's code versions (ID or prefix)")
        parser.add_argument("--uncommitted", action="store_true",
                            help="Print the uncommitted changes recorded when the code was submitted")
        parser.add_argument("--lines", metavar="A-B",
                            help="Print these lines, numbered (40, 40-60, or 40- for the rest)")
        parser.add_argument("-n", "--numbered", action="store_true",
                            help="Print the file numbered, within --budget, with the command for the next page")
        parser.add_argument("--budget", type=int, default=2000, metavar="TOKENS",
                            help="With -n or --lines, stop after about this many tokens (default 2000, 0: no limit)")
    elif verb == "search":
        parser.add_argument("pattern", metavar="PATTERN", help="Text to find (a regular expression with -E)")
        parser.add_argument("targets", nargs="*", metavar="EXP",
                            help="Experiments or snapshots to search (default: every experiment of the campaign)")
        parser.add_argument("--path", help="Only files under this path")
        parser.add_argument("-E", "--regex", action="store_true", help="PATTERN is a regular expression")
        parser.add_argument("-i", "--ignore-case", action="store_true")
        parser.add_argument("-C", "--context-lines", dest="context_lines", type=int, default=0, metavar="N",
                            help="Lines around each match (at most 10)")
        parser.add_argument("--limit", type=int, default=50, help="At most this many lines (max 500)")
    elif verb == "diff":
        parser.add_argument("a", metavar="A")
        parser.add_argument("b", metavar="B")
        parser.add_argument("--path", help="Only files under this path")
        parser.add_argument("--stat", action="store_true", help="List changed files instead of the diff")
    else:
        parser.add_argument("ref", metavar="EXP")
        parser.add_argument("dest", metavar="DIR")
        parser.add_argument("--snapshot", help="Which of the experiment's code versions (ID or prefix)")
    args = parser.parse_args(argv[1:] if verb != "show" else argv)
    manager, token = _manager_token(args)
    campaign = _resolve_campaign(args)
    if verb == "show":
        from mlsweep._notes_code import CodeRefError, is_code_ref, parse_code_ref
        if is_code_ref(args.ref):     # a note's link: mlsweep:code/EXP/PATH#L40-60
            try:
                cref = parse_code_ref(args.ref)
            except CodeRefError as exc:
                _die(str(exc))
            args.ref, args.path = cref.target, args.path or cref.path or None
            if cref.start is not None and not args.lines:
                args.lines = f"{cref.start}-{cref.end}" if cref.end else str(cref.start)

    if verb == "search":
        from mlsweep._code_read import search
        from mlsweep._snapshot import is_snapshot_id
        for t in args.targets:
            if not is_snapshot_id(t):
                require_campaign(manager, token, t, campaign)
        search(manager, token, args, campaign)
    elif verb == "diff":
        a = _resolve_code_ref(manager, token, args.a, campaign, None)
        b = _resolve_code_ref(manager, token, args.b, campaign, None)
        _code_diff(manager, token, a, b, args.path, args.stat)
    elif verb == "checkout":
        _code_checkout(manager, token, _resolve_code_ref(manager, token, args.ref, campaign, args.snapshot),
                       args.dest)
    elif args.uncommitted:
        _print_uncommitted(manager, token, args.ref, campaign, args.snapshot)
    elif args.path is not None or args.files is not None:
        sid = _resolve_code_ref(manager, token, args.ref, campaign, args.snapshot)
        if args.path is not None:
            from urllib.parse import quote
            data = _code_get(manager, token, f"/api/snapshots/{sid}/files/{quote(args.path)}", raw=True)
            if args.lines or args.numbered:
                from mlsweep._code_read import parse_line_range, print_lines
                from mlsweep._snapshot import is_snapshot_id, short_id
                try:
                    start, end = parse_line_range(args.lines) if args.lines else (1, None)
                except ValueError as exc:
                    _die(str(exc))
                # Cite the experiment, or the snapshot when one was picked or given.
                cite = short_id(sid) if (args.snapshot or is_snapshot_id(args.ref)) else args.ref
                print_lines(data, cite_target=cite, path=args.path, start=start, end=end,
                            budget=args.budget, again=f"mlsweep code {cite} {quote(args.path, safe='/')}",
                            as_json=args.json)
                return
            sys.stdout.buffer.write(data)
            sys.stdout.flush()
            return
        from mlsweep._snapshot import under_prefix
        sid, manifest = _snapshot_manifest(manager, token, sid)
        for f in manifest.files:
            if under_prefix(f.path, args.files):
                print(f"{f.size:>12}  {oct(f.mode)[2:]}  {f.hash[:12]}  {f.path}")
    elif args.ref.startswith("snap:"):
        resp = _code_get(manager, token, f"/api/snapshots/{args.ref}")
        info = {k: resp[k] for k in ("artifact_id", "files", "size_bytes")}
        sid = info["artifact_id"]
        if args.json:
            emit(info)
        else:
            sweep_print(f"{_BOLD}{sid}{_RESET}  {info['files']} files, {_fmt_bytes(info['size_bytes'])}")
    else:
        info = _experiment_code(manager, token, args.ref, campaign)
        if args.json:
            emit(info)
        else:
            _print_code_info(info)


# ── delete / export / workers / scheduler (what the dashboard can do) ──────────


def delete_cmd(argv: list[str]) -> None:
    parser = _common_parser(
        "mlsweep delete",
        "Delete an experiment and all its runs (in-flight runs are stopped first). "
        "Its files stay on disk until `mlsweep gc --yes`.",
    )
    parser.add_argument("experiment", help="Experiment ID")
    parser.add_argument("--yes", action="store_true", help="Confirm")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")
    args = parser.parse_args(argv)
    if not args.yes:
        sweep_print(f"{_RED}Deleting an experiment removes its runs, metrics and logs. "
                    f"Pass --yes to confirm.{_RESET}")
        sys.exit(1)
    manager, token, campaign = _connect(args, args.experiment)
    status, resp = _http_request(
        "DELETE", _manager_url(manager, _with_campaign(f"/api/experiments/{args.experiment}", campaign)), token)
    _report(status == 200, f"deleted {args.experiment}")
    if status != 200:
        _fail("delete", status, resp)
    if args.json:
        emit({"experiment_id": args.experiment, "deleted": True})


def _export_query(args: argparse.Namespace) -> str:
    from urllib.parse import urlencode
    q: dict[str, str] = {}
    if args.experiment:
        q["experiment_id"] = args.experiment
    else:
        campaign = _resolve_campaign(args)
        if campaign is not None:
            q["campaign"] = campaign
    if args.only_done:
        q["only_done"] = "1"
    return urlencode(q)


def export_cmd(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="mlsweep export",
        description="Export finished runs' metrics: `tensorboard` writes a zip of event files "
                    "here; `wandb` makes the manager upload them to W&B. The scope is one "
                    "experiment (--experiment), the current campaign (--campaign), or everything "
                    "(--all-campaigns).",
    )
    sub = parser.add_subparsers(dest="target", required=True)
    tb = sub.add_parser("tensorboard", help="Download a zip of TensorBoard event files")
    wb = sub.add_parser("wandb", help="Upload to Weights & Biases from the manager")
    cancel = sub.add_parser("cancel", help="Stop a running W&B export")
    _add_manager_args(cancel)
    cancel.add_argument("export_id", help="The ID `export wandb` printed when it started")
    for p in (tb, wb, cancel):
        p.add_argument("--json", action="store_true",
                       help="Print the result as JSON: the zip's path, or the export's status")
    for p in (tb, wb):
        _add_manager_args(p)
        p.add_argument("--experiment", help="Only this experiment")
        p.add_argument("--only-done", action="store_true", help="Only runs that succeeded")
    tb.add_argument("-o", "--output", help="Zip file to write (default: <scope>-tensorboard.zip)")
    wb.add_argument("--project", required=True)
    wb.add_argument("--entity")
    wb.add_argument("--api-key", help="Default: the manager's WANDB_API_KEY or saved login")
    wb.add_argument("--no-wait", action="store_true", help="Start the upload and return")
    args = parser.parse_args(argv)
    manager, token = _manager_token(args)

    if args.target == "cancel":
        status, resp = _http_request("POST", _manager_url(manager, f"/api/exports/{args.export_id}/cancel"),
                                     token)
        ok = status == 200 and isinstance(resp, dict)
        _report(ok, f"W&B export {args.export_id}: cancelling")
        if not ok:
            _fail("cancel export", status, resp)
        if args.json:
            emit(resp)
        return

    query = _export_query(args)

    if args.target == "tensorboard":
        import urllib.request
        label = args.experiment or _resolve_campaign(args) or "mlsweep"
        out = args.output or f"{label}-tensorboard.zip"
        req = urllib.request.Request(_manager_url(manager, f"/api/export/tensorboard.zip?{query}"),
                                     headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp, open(out, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
        except Exception as exc:  # urllib.error.HTTPError or network trouble
            sweep_print(f"  {_RED}FAIL{_RESET}  tensorboard export: {exc}")
            sys.exit(1)
        _report(True, f"wrote {out}  (tensorboard --logdir on the unzipped folder)")
        if args.json:
            emit({"path": os.path.abspath(out), "bytes": os.path.getsize(out)})
        return

    body = {"project": args.project, "entity": args.entity, "api_key": args.api_key}
    status, resp = _http_request("POST", _manager_url(manager, f"/api/export/wandb?{query}"), token,
                                 json_data={k: v for k, v in body.items() if v})
    if status not in (200, 202) or not isinstance(resp, dict):
        _fail("wandb export", status, resp)
    eid = resp["export_id"]
    sweep_print(f"  started W&B export {eid}: {resp['total']} run(s) of {resp['scope_label']} "
                f"to {args.entity + '/' if args.entity else ''}{args.project}")
    if args.no_wait:
        sweep_print(f"  `mlsweep export cancel {eid}` stops it")
        if args.json:
            emit(resp)
        return
    last = -1
    while True:
        time.sleep(2)
        status, st = _http_request("GET", _manager_url(manager, f"/api/exports/{eid}"), token, quiet=True)
        if status != 200 or not isinstance(st, dict):
            continue
        if st["done"] != last:
            last = st["done"]
            sweep_print(f"  {st['done']}/{st['total']} uploaded")
        if st["status"] != "running":
            ok = st["status"] == "done"
            _report(ok, f"W&B export {st['status']}" + (f": {st['error']}" if st.get("error") else ""))
            if args.json:
                emit(st)
            if not ok:
                sys.exit(1)
            return


def _parse_ints(s: str | None) -> list[int] | None:
    if s is None:
        return None
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except ValueError:
        sweep_print(f"{_RED}Error: expected comma-separated GPU indices, got {s!r}{_RESET}")
        sys.exit(2)


def _report_request(verb: str, success: str, method: str, url: str, token: str, **kwargs: Any) -> bool:
    """Make a request and report how it went; True if the manager answered 200."""
    status, resp = _http_request(method, url, token, **kwargs)
    _report(status == 200, success)
    if status != 200:
        _report_failure(verb, status, resp)
    return status == 200


def workers_cmd(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="mlsweep workers",
        description="List, connect, disconnect, update and tune workers (the System page's controls).",
    )
    sub = parser.add_subparsers(dest="action")
    ls = sub.add_parser("ls", help="List workers (the default)")
    add = sub.add_parser("add", help="Connect a worker over SSH (mlsweep is installed there for you)")
    rm = sub.add_parser("rm", help="Disconnect and forget a worker")
    st = sub.add_parser("set", help="Change a live worker's jobs per GPU or its GPUs")
    up = sub.add_parser("update", help="Reinstall this manager's mlsweep on a worker and restart it "
                                       "(its in-flight runs are requeued)")
    for p in (ls, add, rm, st, up):
        _add_manager_args(p)
        p.add_argument("--json", action="store_true", help="Print the result as JSON")
    add.add_argument("host", help="SSH destination, e.g. user@host")
    add.add_argument("--remote-dir", default="", help="Project dir on the worker (optional)")
    add.add_argument("--ssh-key", help="SSH private key")
    add.add_argument("--venv", help="Existing venv for the worker binary (advanced)")
    add.add_argument("--devices", help="GPU indices to use, e.g. 0,1 (default: all)")
    rm.add_argument("worker_id")
    up.add_argument("worker_id")
    st.add_argument("worker_id")
    st.add_argument("--jobs", type=int, help="Max concurrent jobs per GPU (0 = unlimited)")
    st.add_argument("--add-devices", help="GPU indices to start using, e.g. 2,3")
    st.add_argument("--remove-devices", help="GPU indices to stop using; their runs are re-queued")
    st.add_argument("--drain-devices",
                    help="GPU indices to take out of the pool while their runs finish")
    args = parser.parse_args(argv if argv and argv[0] in ("ls", "add", "rm", "set", "update", "-h", "--help")
                             else ["ls", *argv])
    manager, token = _manager_token(args)

    if args.action == "add":
        body = {"host": args.host, "remote_dir": args.remote_dir, "ssh_key": args.ssh_key,
                "venv": args.venv, "devices": _parse_ints(args.devices)}
        status, resp = _http_request("POST", _manager_url(manager, "/api/workers"), token,
                                     json_data={k: v for k, v in body.items() if v not in (None, "")},
                                     timeout=600)
        ok = status in (200, 201) and isinstance(resp, dict)
        _report(ok, f"connected {resp.get('worker_id', args.host) if ok else args.host}")
        if not ok:
            _fail("add worker", status, resp)
        if args.json:
            emit(resp)
        return
    if args.action == "rm":
        if not _report_request("remove worker", f"removed {args.worker_id}", "DELETE",
                               _manager_url(manager, f"/api/workers/{args.worker_id}"), token):
            sys.exit(1)
        if args.json:
            emit({"worker_id": args.worker_id, "removed": True})
        return
    if args.action == "update":
        status, resp = _http_request("POST", _manager_url(manager, f"/api/workers/{args.worker_id}/update"),
                                     token)
        ok = status == 202
        _report(ok, f"{args.worker_id}: updating (`mlsweep workers` shows when it reconnects)")
        if not ok:
            _fail("update worker", status, resp)
        if args.json:
            emit(resp if isinstance(resp, dict) else {"worker_id": args.worker_id, "status": "updating"})
        return
    if args.action == "set":
        if args.jobs is None and not (args.add_devices or args.remove_devices or args.drain_devices):
            parser.error("give --jobs, --add-devices, --remove-devices or --drain-devices")
        ok = True
        if args.jobs is not None:
            ok &= _report_request("set concurrency", f"{args.worker_id}: {args.jobs or 'unlimited'} job(s) per GPU",
                                  "PATCH", _manager_url(manager, f"/api/workers/{args.worker_id}/concurrency"),
                                  token, json_data={"max_jobs_per_gpu": args.jobs})
        if args.add_devices or args.remove_devices or args.drain_devices:
            body = {"add": _parse_ints(args.add_devices) or [], "remove": _parse_ints(args.remove_devices) or [],
                    "drain": _parse_ints(args.drain_devices) or []}
            ok &= _report_request("set devices", f"{args.worker_id}: devices updated",
                                  "PATCH", _manager_url(manager, f"/api/workers/{args.worker_id}/devices"),
                                  token, json_data=body)
        if not ok:
            sys.exit(1)
        if args.json:
            status, workers = _http_request("GET", _manager_url(manager, "/api/workers"), token)
            if status != 200 or not isinstance(workers, list):
                _fail("list workers", status, workers)
            emit(next((w for w in workers if w.get("worker_id") == args.worker_id),
                      {"worker_id": args.worker_id}))
        return

    status, workers = _http_request("GET", _manager_url(manager, "/api/workers"), token)
    if status != 200 or not isinstance(workers, list):
        _fail("list workers", status, workers)
    if args.json:
        emit(workers)
        return
    if not workers:
        sweep_print("No workers.")
        return
    for w in workers:
        st_ = w.get("status", "?")
        color = _GREEN if st_ == "connected" else _YELLOW if st_ in ("connecting", "launching") else _DIM
        devs = w.get("devices") or w.get("gpus") or ""
        sweep_print(f"  {color}{st_:<11}{_RESET} {w.get('worker_id')}  {_DIM}{w.get('host', '')}"
                    f"{'  gpus ' + str(devs) if devs else ''}"
                    f"{'  ' + str(w.get('last_error')) if w.get('last_error') else ''}{_RESET}")


def scheduler_cmd(argv: list[str]) -> None:
    parser = _common_parser(
        "mlsweep scheduler",
        "Show or set GPU placement: pack fills busy GPUs first (keeps whole GPUs and machines "
        "free for big jobs); spread puts each job on the least busy GPU.",
    )
    parser.add_argument("--placement", choices=("pack", "spread"))
    parser.add_argument("--json", action="store_true", help="Print the scheduler settings as JSON")
    args = parser.parse_args(argv)
    manager, token = _manager_token(args)
    if args.placement:
        status, resp = _http_request("PATCH", _manager_url(manager, "/api/scheduler"), token,
                                     json_data={"placement": args.placement})
    else:
        status, resp = _http_request("GET", _manager_url(manager, "/api/scheduler"), token)
    if status != 200 or not isinstance(resp, dict):
        _fail("scheduler", status, resp)
    if args.json:
        emit(resp)
        return
    sweep_print(f"  placement: {resp.get('placement')}")
