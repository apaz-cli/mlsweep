"""Control-plane subcommands for ``mlsweep``: inspect and manage runs.

These are the lifecycle verbs (ls, logs, cancel, retry, resume, stop, pause,
unpause) plus the result-ranking command (best). They are thin HTTP clients over
the manager API and share helpers with ``mlsweep.run_sweep``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from typing import Any

from mlsweep._shared import _GREEN, _RED, _RESET
from mlsweep.run_sweep import (
    _add_manager_args,
    _combo_str,
    _parse_combo,
    _require_token,
    _wait_until_settled,
    build_leaderboard,
    manager_cancel_job,
    manager_get_job_logs,
    manager_list_experiment_jobs,
    manager_list_experiments,
    manager_retry_job,
    manager_set_experiment_status,
    print_leaderboard,
    sweep_print,
)


def _manager_token(args: argparse.Namespace) -> tuple[str, str]:
    """Return (manager, token) resolved from parsed args."""
    return args.manager.rstrip("/"), _require_token(args.token)


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
        sweep_print(f"{args.experiment} — {len(jobs)} runs:")
        for j in jobs:
            combo_s = _combo_str(_parse_combo(j.get("combo")))
            sweep_print(f"  {j.get('status','?'):>11}  {_GREEN}{j.get('run_id')}{_RESET}  {combo_s}")
        return

    exps = manager_list_experiments(manager, token, status_filter=args.status)
    if exps is None:
        sweep_print(f"{_RED}FAIL{_RESET}  list experiments")
        sys.exit(1)
    if args.json:
        print(json.dumps(exps, indent=2))
        return
    sweep_print(f"{len(exps)} experiments:")
    for e in exps:
        c = e.get("job_counts") or {}
        counts = f"{c.get('done',0)} done / {c.get('failed',0)} fail / {c.get('running',0)} run / {c.get('pending',0)} pend"
        name = e.get("name") or ""
        note = f"  # {e.get('note')}" if e.get("note") else ""
        sweep_print(f"  {e.get('status','?'):>10}  {_GREEN}{e.get('experiment_id')}{_RESET}  {name}{note}")
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
    parser.add_argument("--metric", default="loss", help="Metric to rank by")
    parser.add_argument("--goal", default="minimize", choices=["minimize", "maximize"], help="Rank direction")
    parser.add_argument("--top", type=int, default=10, help="Show top N runs (0 = all)")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--wait", action="store_true", help="Block until the experiment settles")
    parser.add_argument("--wait-interval", type=int, default=10, help="Seconds between --wait polls")
    args = parser.parse_args(argv)
    manager, token = _manager_token(args)

    if args.wait:
        _wait_until_settled(manager, token, args.experiment, args.wait_interval)

    rows = build_leaderboard(manager, token, args.experiment, args.metric, args.goal)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print_leaderboard(rows, args.metric, args.goal, args.top)
