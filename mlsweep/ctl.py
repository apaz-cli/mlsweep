"""Control-plane subcommands for ``mlsweep``: inspect and manage runs.

These are the lifecycle verbs (ls, logs, cancel, retry, resume, stop, pause,
unpause) plus the result-ranking command (best). They are thin HTTP clients over
the manager API and share helpers with ``mlsweep.run_sweep``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mlsweep._shared import _BOLD, _CYAN, _GREEN, _RED, _RESET, _YELLOW
from mlsweep.run_sweep import (
    _add_manager_args,
    _combo_str,
    _manager_token,
    _parse_combo,
    _wait_until_settled,
    build_leaderboard,
    manager_cancel_job,
    manager_get_job_logs,
    manager_get_job_metrics,
    manager_list_experiment_jobs,
    manager_list_experiments,
    manager_retry_job,
    manager_set_experiment_status,
    print_leaderboard,
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


def _color_status(text: str, status: str) -> str:
    """Wrap an already-width-formatted status field in its status color."""
    color = _STATUS_COLORS.get(status)
    return f"{color}{text}{_RESET}" if color else text


def _apply(
    targets: list[dict[str, Any]],
    fn: Callable[[str, str, str, str], object],
    verb: str,
    manager: str,
    token: str,
    experiment: str,
    dry_run: bool,
) -> int:
    """Run a per-job action (cancel/retry) over *targets*. Returns # of failures."""
    failures = 0
    for j in targets:
        run_id = j.get("run_id", "")
        if dry_run:
            sweep_print(f"  would {verb} {run_id}")
        else:
            ok = fn(manager, token, run_id, experiment)
            _report(ok, f"{verb} {run_id}")
            if not ok:
                failures += 1
    return failures


def _common_parser(prog: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    _add_manager_args(parser)
    return parser


# ── ls ─────────────────────────────────────────────────────────────────────────


def ls_cmd(argv: list[str]) -> None:
    parser = _common_parser("mlsweep ls", "List experiments, or the runs within one experiment.")
    parser.add_argument("experiment", nargs="?", help="Experiment ID (omit to list experiments)")
    parser.add_argument("--status", default=None, help="Filter by status")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)
    manager, token = _manager_token(args)

    if args.experiment:
        jobs = manager_list_experiment_jobs(manager, token, args.experiment, status_filter=args.status) or []
        if args.json:
            print(json.dumps(jobs, indent=2))
            return
        if not jobs:
            sweep_print("  No jobs found.")
            return
        sweep_print(f"{_BOLD}{_CYAN}{args.experiment}{_RESET} — {len(jobs)} runs:")
        for j in jobs:
            combo_s = _combo_str(_parse_combo(j.get("combo")))
            status = j.get("status", "?")
            sweep_print(f"  {_color_status(f'{status:>11}', status)}  "
                        f"{_GREEN}{j.get('run_id')}{_RESET}  {combo_s}")
        return

    exps = manager_list_experiments(manager, token, status_filter=args.status)
    if exps is None:
        sweep_print(f"{_RED}FAIL{_RESET}  list experiments")
        sys.exit(1)
    if args.json:
        print(json.dumps(exps, indent=2))
        return
    sweep_print(f"{_BOLD}{_CYAN}{len(exps)}{_RESET} experiments:")
    for e in exps:
        c = e.get("job_counts") or {}
        counts = f"{c.get('done',0)} done / {c.get('failed',0)} fail / {c.get('running',0)} run / {c.get('pending',0)} pend"
        name = e.get("name") or ""
        note = f"  # {e.get('note')}" if e.get("note") else ""
        status = e.get("status", "?")
        sweep_print(f"  {_color_status(f'{status:>10}', status)}  "
                    f"{_GREEN}{e.get('experiment_id')}{_RESET}  {name}{note}")
        sweep_print(f"             {counts}")


# ── logs ───────────────────────────────────────────────────────────────────────


def logs_cmd(argv: list[str]) -> None:
    parser = _common_parser("mlsweep logs", "Print a run's training log.")
    parser.add_argument("run", help="Run ID")
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    parser.add_argument("--tail", type=int, default=None, help="Show only the last N lines")
    parser.add_argument("--follow", action="store_true", help="Follow new output")
    args = parser.parse_args(argv)
    manager, token = _manager_token(args)

    text = manager_get_job_logs(manager, token, args.experiment, args.run)
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
                newer = manager_get_job_logs(manager, token, args.experiment, args.run)
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


def pivot_metrics(rows: list[dict[str, Any]], pattern: re.Pattern[str],
                  step: int | None) -> tuple[int | None, dict[str, Any]]:
    """For a pattern with one capture group, return ``(step, {capture: value})`` from the
    latest logged step at or before *step* (the latest overall if None) that has any
    matching key.  Turns flat keys like ``val/nll@r16`` into a curve over the capture."""
    chosen = None
    for row in sorted(rows, key=lambda r: r.get("step") or 0):
        if step is not None and (row.get("step") or 0) > step:
            break
        if any(k != "step" and pattern.search(k) for k in row):
            chosen = row
    if chosen is None:
        return None, {}
    vals = {}
    for k, v in chosen.items():
        m = pattern.search(k) if k != "step" else None
        if m:
            vals[m.group(1)] = v
    return chosen.get("step"), vals


def metrics_cmd(argv: list[str]) -> None:
    parser = _common_parser(
        "mlsweep metrics",
        "Print what runs logged, filtered and reshaped on demand (nothing is stored).",
    )
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    parser.add_argument("runs", nargs="*", help="Run IDs (default: every run in the experiment)")
    parser.add_argument("--keys", default=None, help="Regex selecting metric keys")
    parser.add_argument("--pivot", action="store_true",
                        help="Turn keys into rows using --keys' first capture group, e.g. "
                             "--keys 'val/nll@r(\\d+)' --pivot prints value vs. r, one column "
                             "per run (a curve from flat keys)")
    parser.add_argument("--step", type=int, default=None,
                        help="With --pivot: use this step (latest at or before it; default: latest)")
    parser.add_argument("--tail", type=int, default=10,
                        help="Steps per run in the table view (default 10; 0 = all)")
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="Emit the selected metrics as JSON")
    fmt.add_argument("--csv", action="store_true", help="Emit long-format CSV: run,step,key,value")
    args = parser.parse_args(argv)
    manager, token = _manager_token(args)

    try:
        pattern = re.compile(args.keys) if args.keys else None
    except re.error as e:
        sweep_print(f"{_RED}Bad --keys regex{_RESET}: {e}")
        sys.exit(2)

    run_ids = list(args.runs)
    if not run_ids:
        jobs = manager_list_experiment_jobs(manager, token, args.experiment) or []
        run_ids = [j["run_id"] for j in jobs]
    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched = pool.map(lambda rid: manager_get_job_metrics(manager, token, args.experiment, rid),
                           run_ids)
        per_run = {rid: select_metrics(rows or [], pattern) for rid, rows in zip(run_ids, fetched)}
    per_run = {rid: rows for rid, rows in per_run.items() if rows}
    if not per_run:
        sweep_print(f"{_RED}No metrics{_RESET} in {args.experiment}"
                    + (f" matching {args.keys!r}" if args.keys else ""))
        sys.exit(1)

    if args.json:
        print(json.dumps(per_run, indent=2))
        return
    if args.csv:
        w = csv.writer(sys.stdout)
        w.writerow(["run", "step", "key", "value"])
        for rid, rows in per_run.items():
            for row in rows:
                for k, v in row.items():
                    if k != "step":
                        w.writerow([rid, row["step"], k, v])
        return

    if args.pivot:
        if pattern is None or pattern.groups < 1:
            sweep_print(f"{_RED}--pivot needs --keys with a capture group{_RESET}, "
                        "e.g. --keys 'val/nll@r(\\d+)'")
            sys.exit(2)
        cols = []
        for rid, rows in per_run.items():
            step, vals = pivot_metrics(rows, pattern, args.step)
            if vals:
                cols.append((f"{rid} @{step}", vals))
        xs = sorted({x for _, vals in cols for x in vals}, key=_sort_key)
        _print_table(["x"] + [c for c, _ in cols],
                     [[x] + [_fmt(vals.get(x)) for _, vals in cols] for x in xs])
        return

    for rid, rows in per_run.items():
        keys = sorted({k for r in rows for k in r if k != "step"})
        shown = rows[-args.tail:] if args.tail else rows
        print(f"{_BOLD}{_CYAN}== {rid}{_RESET}  ({len(rows)} steps)")
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
    manager, token = _manager_token(args)

    jobs = manager_list_experiment_jobs(manager, token, args.experiment) or []
    statuses = [s for s in _SELECTABLE_STATUSES if getattr(args, s)]
    targets = jobs if args.all else _select_jobs(jobs, args.runs, statuses)

    if not targets:
        sweep_print("  No matching runs.")
        return

    if args.all and not args.yes:
        sweep_print(f"{_RED}Refusing to {verb} all {len(targets)} runs without --yes.{_RESET}")
        sys.exit(1)

    fn = manager_retry_job if retry else manager_cancel_job
    failures = _apply(targets, fn, verb, manager, token, args.experiment, args.dry_run)
    if failures:
        sys.exit(1)


def cancel_cmd(argv: list[str]) -> None:
    _cancel_retry(argv, retry=False)


def retry_cmd(argv: list[str]) -> None:
    _cancel_retry(argv, retry=True)


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
    manager, token = _manager_token(args)
    r = manager_set_experiment_status(manager, token, args.experiment, status)
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
    manager, token = _manager_token(args)

    start = time.monotonic()
    while True:
        jobs = manager_list_experiment_jobs(manager, token, args.experiment)
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
    manager, token = _manager_token(args)

    if args.sweep:
        from mlsweep._sweep import load_sweep_file
        info = load_sweep_file(args.sweep)
        if info.get("method") == "bayes":
            # Reuse the submitter's bayes resume path.
            from mlsweep.cli import _forward
            from mlsweep.run_sweep import main as _run_main
            fwd = [args.sweep, "--resume", args.experiment, "--manager", manager]
            if args.token:
                fwd += ["--token", args.token]
            _forward("mlsweep run", _run_main, fwd)
            return

    # Grid (or no sweep file): re-queue failed / cancelled runs.
    jobs = manager_list_experiment_jobs(manager, token, args.experiment) or []
    targets = [j for j in jobs if j.get("status") in ("failed", "cancelled")]
    if not targets:
        sweep_print("  Nothing to resume, no failed or cancelled jobs.")
        return
    failures = _apply(targets, manager_retry_job, "retry", manager, token, args.experiment, args.dry_run)
    if failures:
        sys.exit(1)


# ── best ───────────────────────────────────────────────────────────────────────


def best_cmd(argv: list[str]) -> None:
    parser = _common_parser("mlsweep best", "Show the best runs of an experiment by metric.")
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    parser.add_argument("--metric", default=None,
                        help="Metric to rank by (default: experiment's metric, else loss)")
    parser.add_argument("--goal", default=None, choices=["minimize", "maximize"],
                        help="Rank direction (default: experiment's goal, else minimize)")
    parser.add_argument("--top", type=int, default=10, help="Show top N runs (0 = all)")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--wait", action="store_true", help="Block until the experiment settles")
    parser.add_argument("--wait-interval", type=int, default=10, help="Seconds between --wait polls")
    args = parser.parse_args(argv)
    manager, token = _manager_token(args)

    had_failure = False
    if args.wait:
        had_failure = _wait_until_settled(manager, token, args.experiment, args.wait_interval)

    metric, goal = resolve_ranking(manager, token, args.experiment, args.metric, args.goal)
    rows = build_leaderboard(manager, token, args.experiment, metric, goal)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_leaderboard(rows, metric, goal, args.top)
    if had_failure:
        sys.exit(1)
