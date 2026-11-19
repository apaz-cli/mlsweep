"""Training script metrics logger for mlsweep.

Usage::

    from mlsweep.logger import MLSweepLogger

    with MLSweepLogger() as logger:
        for step in range(1, 1001):
            logger.log({"loss": 0.5, "acc": 0.9}, step=step)

    # Without context manager:
    logger = MLSweepLogger()
    logger.log({"loss": 0.5})   # step auto-increments; logger.step becomes 1
    logger.close()

Steps never go backwards, and a metric is logged at most once per step:
``log()`` raises ``ValueError`` rather than overwrite an earlier value.
Several calls at the same step are fine when their keys differ
(``log({"loss": l}, step=s)`` then ``log({"val_loss": v}, step=s)``).

All methods are silent no-ops when ``MLSWEEP_WORKER_SOCKET`` is not set, or when
``MLSWEEP_GPU_RANK`` / ``MLSWEEP_NODE_RANK`` indicate a non-lead rank (see
``rank_zero_only`` parameter).
"""

import json
import operator
import os
import socket
import threading
import time
from typing import Any


def _as_step(step: Any) -> int:
    """*step* as an int.  Accepts numpy/torch integers and whole floats (5.0)."""
    if isinstance(step, bool):
        raise TypeError("step must be an integer, got bool")
    try:
        return operator.index(step)
    except TypeError:
        if isinstance(step, float) and step.is_integer():
            return int(step)
        raise TypeError(f"step must be an integer, got {step!r}") from None




class MLSweepLogger:
    """Metrics logger for mlsweep training scripts.

    Communicates with the worker daemon via a unix socket.  All operations
    are no-ops when ``MLSWEEP_WORKER_SOCKET`` is not set in the environment.

    Attributes:
        step: Current training step.  Updated by every ``log()`` call.
        step_seconds: Wall seconds per step over the last step advance, or
            ``None`` until the step has advanced once.  Sent with each metric
            so the dashboard can show a step timer.
    """

    def __init__(
        self,
        *,
        hparams: dict[str, Any] | None = None,
        run_name: str | None = None,
        rank_zero_only: bool = True,
    ) -> None:
        """Create a logger.

        Args:
            hparams: Hyperparameter dict stored for reference (currently unused
                     in the worker protocol; reserved for future use).
            run_name: Override for the run name.  Defaults to
                      ``MLSWEEP_RUN_NAME`` env var.
            rank_zero_only: When ``True`` (default), all operations are no-ops
                            unless both ``MLSWEEP_GPU_RANK`` and
                            ``MLSWEEP_NODE_RANK`` are unset or ``"0"``.  Set to
                            ``False`` to log from every rank.
        """
        self._run_id: str = run_name if run_name is not None else os.environ.get("MLSWEEP_RUN_NAME", "")
        self._experiment: str = os.environ.get("EXP_EXPERIMENT", "")
        _is_lead = (
            os.environ.get("MLSWEEP_GPU_RANK", "0") == "0"
            and os.environ.get("MLSWEEP_NODE_RANK", "0") == "0"
        )
        self._sock_path: str | None = (
            os.environ.get("MLSWEEP_WORKER_SOCKET") if (not rank_zero_only or _is_lead) else None
        )
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self.step: int = 0
        self._logged_any = False
        self._step_keys: set[str] = set()   # keys already logged at self.step
        self.step_seconds: float | None = None
        self._step_clock: float | None = None  # monotonic time self.step was first logged

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_sock(self) -> socket.socket | None:
        if not self._sock_path:
            return None
        if self._sock is None:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self._sock_path)
                self._sock = s
            except OSError:
                return None
        return self._sock

    def _send(self, msg: dict[str, Any]) -> None:
        sock = self._get_sock()
        if sock is None:
            return
        line = json.dumps(msg) + "\n"
        with self._lock:
            try:
                sock.sendall(line.encode())
            except OSError:
                self._sock = None  # will reconnect on next call

    # ── Public API ─────────────────────────────────────────────────────────────

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a metrics dict for the given step.

        Args:
            metrics: Mapping of metric name to scalar value.
            step:    Training step.  If omitted, ``self.step`` is incremented
                     by 1 and the new value is used.

        Raises:
            ValueError: *step* is lower than the last logged step, or a key in
                *metrics* was already logged at *step*.  Either would
                overwrite a value already recorded.
        """
        prev_step = self.step if self._logged_any else None
        if step is not None:
            step = _as_step(step)
            if self._logged_any and step < self.step:
                raise ValueError(
                    f"step {step} is before the last logged step {self.step}; "
                    "steps must not go backwards (that would overwrite logged metrics)")
            if self._logged_any and step == self.step:
                again = sorted(self._step_keys.intersection(metrics))
                if again:
                    raise ValueError(
                        f"{', '.join(map(repr, again))} already logged at step {step}; "
                        "log each metric once per step")
            else:
                self._step_keys = set()
            self.step = step
        else:
            self.step += 1
            self._step_keys = set()
        self._step_keys.update(metrics)
        self._time_step(prev_step)
        self._logged_any = True
        self._send({
            "type": "metric",
            "run_id": self._run_id,
            "experiment": self._experiment,
            "step": self.step,
            "data": metrics,
            "step_seconds": self.step_seconds,
        })

    def _time_step(self, prev_step: int | None) -> None:
        """Update step_seconds when this log() moved to a new step."""
        if prev_step is not None and self.step == prev_step:
            return
        now = time.monotonic()
        if prev_step is not None and self._step_clock is not None:
            self.step_seconds = (now - self._step_clock) / (self.step - prev_step)
        self._step_clock = now

    def sync(self) -> None:
        """Fire-and-forget: signal the worker to rsync artifacts to the controller."""
        self._send({"type": "sync", "run_id": self._run_id, "experiment": self._experiment})

    def close(self) -> None:
        """Close the unix socket connection."""
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def __enter__(self) -> "MLSweepLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
