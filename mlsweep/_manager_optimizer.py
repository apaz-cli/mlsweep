"""The manager's side of a Bayesian sweep: adding trials as results come in.

An experiment with an ``optimize`` spec (see ``_sweep.py``) has its runs chosen
here instead of by whoever submitted it.  Each scheduler pass calls
``advance_optimizers_locked``, which compares the experiment's jobs with its spec
and acts on the difference.  No optimizer state outlives the pass except a cache
of metric values, so there's nothing to restore after a restart, and a retried,
moved or purged run just changes what the next pass sees.

A trial is one combo of the non-singular dims, and may have several runs (one per
singular probe).  Its result is the best logged value of the metric in the first
of its runs that logged one.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import aiosqlite

from mlsweep._manager_db import (
    ExperimentRecord,
    get_metrics_for_run,
    list_optimizer_experiments,
    optimizer_job_rows,
)
from mlsweep._manager_state import ManagerState
from mlsweep._shared import _RED, _RESET, _YELLOW
from mlsweep._sweep import best_metric, build_job_payloads, lex_key, spec_singular_dims, validate_optimize_spec

_ACTIVE = frozenset({"pending", "dispatched", "running"})


@dataclass
class Trial:
    combo: dict[str, Any]
    value: float | None = None
    best_run: str | None = None
    active: bool = False  # a run of it is pending or running
    all_cancelled: bool = True
    finished_at: float = 0.0

    @property
    def outcome(self) -> str:
        if self.value is not None:
            return "complete"
        if self.active:
            return "running"
        return "abandoned" if self.all_cancelled else "failed"


@dataclass
class Progress:
    """Where an experiment's trials stand, read from its jobs."""

    trials: list[Trial] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    any_active: bool = False  # includes other probes of a trial that already has a result

    def count(self, outcome: str) -> int:
        return sum(t.outcome == outcome for t in self.trials)

    def failure_streak(self, cleared_at: float) -> int:
        """Failed trials since the last result (or since *cleared_at*), by finish time."""
        streak = 0
        finished = [t for t in self.trials if t.outcome in ("complete", "failed")]
        for t in sorted(finished, key=lambda t: t.finished_at, reverse=True):
            if t.outcome == "complete" or t.finished_at <= cleared_at:
                break
            streak += 1
        return streak

    def next_number(self) -> int:
        numbers = [int(m.group(1)) for m in map(_RUN_NUMBER.search, self.run_ids) if m]
        return max(numbers, default=0) + 1


_RUN_NUMBER = re.compile(r"_bayes_(\d+)$")


async def _run_value(
    db: aiosqlite.Connection, state: ManagerState, exp_id: str, spec: dict[str, Any], row: Any,
) -> float | None:
    """A finished run's metric value.  Cached, since it can't change until the next attempt."""
    key = (row["job_key"], row["attempt"])
    if key not in state.run_values:
        metrics = await get_metrics_for_run(db, row["run_id"], exp_id)
        state.run_values[key] = best_metric(metrics, spec["metric"], spec["goal"])
    return state.run_values[key]


async def read_progress(
    db: aiosqlite.Connection, state: ManagerState, experiment_id: str, spec: dict[str, Any],
) -> Progress:
    """Group an experiment's runs into trials, in the order the trials were made."""
    singular = spec_singular_dims(spec)
    trials: dict[Any, Trial] = {}
    progress = Progress()
    for row in await optimizer_job_rows(db, experiment_id):
        combo = json.loads(row["combo"])
        trial = trials.setdefault(lex_key(combo, singular), Trial(combo))
        progress.run_ids.append(row["run_id"])
        status = row["status"]
        trial.all_cancelled = trial.all_cancelled and status == "cancelled"
        trial.finished_at = max(trial.finished_at, row["finish_time"] or 0.0)
        if status in _ACTIVE:
            trial.active = True
            progress.any_active = True
        elif status == "done" and trial.value is None:
            trial.value = await _run_value(db, state, experiment_id, spec, row)
            trial.best_run = row["run_id"] if trial.value is not None else None
    progress.trials = list(trials.values())
    return progress


def status_view(exp: ExperimentRecord, spec: dict[str, Any], progress: Progress) -> dict[str, Any]:
    """What ``GET /api/experiments/{id}/optimize`` returns."""
    complete = [t for t in progress.trials if t.value is not None]
    pick = min if spec["goal"] == "minimize" else max
    best = pick(complete, key=lambda t: t.value) if complete else None  # type: ignore[arg-type,return-value]
    return {
        "experiment_id": exp.experiment_id,
        "status": exp.status,
        "metric": spec["metric"],
        "goal": spec["goal"],
        "budget": spec["budget"],
        "told": len(complete),
        "running": progress.count("running"),
        "failed": progress.count("failed"),
        "abandoned": progress.count("abandoned"),
        "parallel": exp.max_concurrent or spec["n_initial"],
        "n_initial": spec["n_initial"],
        "max_failures": spec["max_failures"],
        "failure_streak": progress.failure_streak(spec.get("failures_cleared_at", 0.0)),
        "finishing": bool(spec.get("finishing")),
        "note": spec.get("note"),
        "best": best and {"run_id": best.best_run, "value": best.value, "combo": best.combo},
    }


def check_submitted_spec(raw: Any) -> dict[str, Any]:
    """Validate a submitted spec, and draw one trial to prove optuna accepts its dims.

    Raises ValueError (or whatever optuna raises).  Imports optuna, so call it off
    the event loop.
    """
    from mlsweep._bayes import suggest

    spec = validate_optimize_spec(raw)
    suggest(spec, [], 1)
    return spec


async def _set_status_locked(
    state: ManagerState, exp: ExperimentRecord, spec: dict[str, Any], status: Any, note: str | None,
) -> None:
    spec["note"] = note
    await state.db_writer.update_experiment_optimize(exp.experiment_id, spec)
    await state.db_writer.update_experiment_status(exp.experiment_id, status)


async def advance_optimizer_locked(
    db: aiosqlite.Connection, state: ManagerState, exp: ExperimentRecord,
) -> None:
    """Bring one running experiment up to date: add trials, pause it, or complete it."""
    # In a thread: a first import of optuna takes a while, and the event loop has workers to serve.
    bayes: Any = await asyncio.to_thread(importlib.import_module, "mlsweep._bayes")

    assert exp.optimize is not None
    spec: dict[str, Any] = json.loads(exp.optimize)
    progress = await read_progress(db, state, exp.experiment_id, spec)
    told, running = progress.count("complete"), progress.count("running")

    want = 0
    if not spec.get("finishing"):
        parallel = exp.max_concurrent or spec["n_initial"]
        want = min(parallel - running, spec["budget"] - told - running)

    if want > 0:
        streak = progress.failure_streak(spec.get("failures_cleared_at", 0.0))
        if streak >= spec["max_failures"]:
            await _give_up_locked(db, state, exp, spec, f"{streak} trials in a row failed or logged no "
                                                        f"{spec['metric']!r}. Unpause to try again.")
            return

    added = False
    note = spec.get("note")
    if want > 0:
        history = [bayes.PastTrial(t.combo, t.outcome, t.value) for t in progress.trials]
        variations = await asyncio.to_thread(bayes.suggest, spec, history, want, progress.next_number())
        if variations:
            await state.db_writer.insert_jobs_bulk(
                build_job_payloads(variations, exp.experiment_id, **spec["job"]))
            added = True
        elif not running:
            note = f"search space used up after {len(progress.trials)} trials"

    if not added and not progress.any_active:
        await _set_status_locked(state, exp, spec, "completed", note)
        state.broadcast(exp.experiment_id, {
            "type": "experiment_done", "experiment_id": exp.experiment_id, "submitted_count": told,
        })


async def _give_up_locked(
    db: aiosqlite.Connection, state: ManagerState, exp: ExperimentRecord, spec: dict[str, Any], why: str,
) -> None:
    """Pause a sweep that isn't working.  Runs already going may finish.

    Its queued runs are cancelled, since they'd likely fail the same way and they
    could never start while it's paused, which would leave ``mlsweep wait`` hanging.
    Unpausing chooses new trials in their place.
    """
    print(f"  {_YELLOW}PAUSE{_RESET} {exp.experiment_id}: {why}")
    await _set_status_locked(state, exp, spec, "paused", why)
    queued = [(exp.experiment_id, r["run_id"]) for r in await optimizer_job_rows(db, exp.experiment_id)
              if r["status"] == "pending"]
    for job in await state.db_writer.cancel_jobs(queued):
        state.broadcast(exp.experiment_id, {
            "type": "job_done", "run_id": job.run_id, "status": "cancelled", "success": False,
        })
    state.broadcast(exp.experiment_id, {"type": "status_updated", "status": "paused", "note": why})


async def advance_optimizers_locked(db: aiosqlite.Connection, state: ManagerState) -> None:
    """Advance every running Bayesian experiment.  One that errors is paused, not retried forever."""
    for exp in await list_optimizer_experiments(db):
        try:
            await advance_optimizer_locked(db, state, exp)
        except Exception as e:
            print(f"  {_RED}ERROR{_RESET} optimizer of {exp.experiment_id} failed: {e!r}")
            traceback.print_exc()
            assert exp.optimize is not None
            await _give_up_locked(db, state, exp, json.loads(exp.optimize), f"optimizer error: {e!r}")


def cleared_for_running(spec: dict[str, Any]) -> dict[str, Any]:
    """*spec* for an experiment someone set running again: earlier failures are forgiven."""
    return {**spec, "failures_cleared_at": time.time(), "note": None}
