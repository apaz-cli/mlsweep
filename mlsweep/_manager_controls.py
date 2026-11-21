"""The manager's run controls: MAX_RUNTIME, PRUNE and METRIC_CHECK.

A sweep declares them (see ``build_controls`` in ``_sweep.py``) and they are
stored on the experiment as ``controls``.  The scheduler task reloads them
each pass into ``state.controls``, so a metric arriving from a worker can be
checked against them without a database read.  This module decides; the
actions (stopping a run, pausing a sweep) are in ``_manager_workers``.

Pruning compares runs at fixed steps, called rungs.  At each rung a run's
value is the best value of the metric it logged up to the first step at or
past the rung.  Once ``n_warmup`` other runs have a value at a rung, a run
reaching it is stopped if it is worse than their median (``median``), or not
in the best ``1/reduction_factor`` of them and itself (``asha``).  The rung
values live in memory and are rebuilt from the stored metrics after a restart.
"""

from __future__ import annotations

import difflib
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from mlsweep._sweep import prune_rungs

# Rungs past this many are never reached in practice; the bound keeps a
# runaway step counter from looping.
_MAX_RUNGS = 10_000


@dataclass
class _Run:
    attempt: int
    best: float | None = None
    rungs: Any = None  # iterator over the rungs still ahead of this run
    next_rung: int | None = None
    stopped: bool = False


@dataclass
class PruneTracker:
    """Rung values of one experiment's runs."""

    spec: dict[str, Any]
    loaded: bool = False
    # Metrics that arrived while the stored ones were being read: (run_id, attempt, step, value).
    buffered: list[tuple[str, int, int, float]] = field(default_factory=list)
    runs: dict[str, _Run] = field(default_factory=dict)
    values: dict[int, dict[str, float]] = field(default_factory=dict)  # rung step → run_id → value

    def _better(self, a: float, b: float) -> bool:
        return a < b if self.spec["goal"] == "minimize" else a > b

    def _start(self, run_id: str, attempt: int) -> _Run:
        """Track *run_id* afresh: a new attempt forgets what the earlier one logged."""
        for by_run in self.values.values():
            by_run.pop(run_id, None)
        rungs = prune_rungs(self.spec)
        run = _Run(attempt=attempt, rungs=rungs, next_rung=next(rungs))
        self.runs[run_id] = run
        return run

    def observe(self, run_id: str, attempt: int, step: int, value: float, *, judge: bool = True) -> str | None:
        """Record *value* of the metric at *step*.  Returns why to stop the run, or None.

        With *judge* False the value is recorded but the run is never stopped
        (used for finished runs read back after a restart).
        """
        run = self.runs.get(run_id)
        if run is None or run.attempt != attempt:
            run = self._start(run_id, attempt)
        if run.stopped:
            return None
        if run.best is None or self._better(value, run.best):
            run.best = value
        reason = None
        n = 0
        while run.next_rung is not None and step >= run.next_rung and n < _MAX_RUNGS:
            rung = run.next_rung
            others = [v for rid, v in self.values.get(rung, {}).items() if rid != run_id]
            self.values.setdefault(rung, {})[run_id] = run.best
            run.next_rung = next(run.rungs)
            n += 1
            if judge and reason is None:
                reason = self._judge(rung, run.best, others)
        if reason is not None:
            run.stopped = True
        return reason

    def _judge(self, rung: int, value: float, others: list[float]) -> str | None:
        spec = self.spec
        if len(others) < spec["n_warmup"]:
            return None
        metric = spec["metric"]
        if spec["strategy"] == "median":
            median = statistics.median(others)
            if self._better(median, value):
                return (f"pruned at step {rung}: best {metric} {value:.6g} is worse than the median "
                        f"{median:.6g} of {len(others)} other runs there")
            return None
        rf = spec["reduction_factor"]
        population = sorted([*others, value], reverse=spec["goal"] == "maximize")
        keep = max(1, len(population) // rf)
        cutoff = population[keep - 1]
        if self._better(cutoff, value):
            return (f"pruned at step {rung}: best {metric} {value:.6g} is outside the best {keep} of "
                    f"{len(population)} runs there (cutoff {cutoff:.6g})")
        return None


def metric_value(data: dict[str, Any], metric: str) -> float | None:
    """*metric* in one metric row, if it is there and finite."""
    v = data.get(metric)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def missing_metric_reason(metric: str, logged: set[str]) -> str:
    """The ``reason`` of a finished run that never logged *metric*."""
    from mlsweep._manager_db import MISSING_METRIC_REASON

    keys = sorted(k for k in logged if k not in ("step", "_step_seconds"))
    if not keys:
        return f"{MISSING_METRIC_REASON} {metric!r}: the run logged no metrics at all"
    close = difflib.get_close_matches(metric, keys, n=1, cutoff=0.5)
    shown = ", ".join(keys[:8]) + (f" and {len(keys) - 8} more" if len(keys) > 8 else "")
    hint = f"; did you mean {close[0]!r}?" if close else ""
    return f"{MISSING_METRIC_REASON} {metric!r}: the run logged {shown}{hint}"
