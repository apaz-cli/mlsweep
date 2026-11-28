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

A list (or tuple, or array) is logged as one metric per element, with the
index after a slash: ``log({"moe_load": [0.2, 0.5]})`` logs ``moe_load/0`` and
``moe_load/1``.  A nested dict is flattened the same way (``{"val": {"nll": x}}``
logs ``val/nll``), and numpy or torch scalars become plain numbers.

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
from typing import Any, Callable


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




def _flatten(metrics: dict[str, Any]) -> dict[str, Any]:
    """*metrics* with lists, tuples, arrays and nested dicts spread into ``key/i`` and ``key/sub``."""
    out: dict[str, Any] = {}

    def add(key: str, value: Any) -> None:
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            value = value.tolist()  # numpy / torch: arrays to lists, scalars to numbers
        elif hasattr(value, "item") and not isinstance(value, (int, float, bool)):
            value = value.item()
        if isinstance(value, dict):
            for k, v in value.items():
                add(f"{key}/{k}", v)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                add(f"{key}/{i}", v)
        else:
            out[key] = value

    for key, value in metrics.items():
        add(str(key), value)
    return out


class MLSweepLogger:
    """Metrics logger for mlsweep training scripts.

    Communicates with the worker daemon via a unix socket.  All operations
    are no-ops when ``MLSWEEP_WORKER_SOCKET`` is not set in the environment.

    Attributes:
        step: Current training step.  Updated by every ``log()`` call.
        step_seconds: Wall seconds per step over the last step advance, or
            ``None`` until the step has advanced once.  Sent with each metric
            so the dashboard can show a step timer.
        total_steps: The step the run ends at, if known.  Sent with each
            metric, so ``mlsweep ls`` and ``mlsweep status`` can say when the
            run will finish.  Set it here or with the constructor argument.
    """

    def __init__(
        self,
        *,
        hparams: dict[str, Any] | None = None,
        run_name: str | None = None,
        rank_zero_only: bool = True,
        socket_path: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        total_steps: int | None = None,
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
            socket_path: The worker's socket.  Defaults to
                         ``MLSWEEP_WORKER_SOCKET``.
            clock: Seconds, for ``step_seconds``.  Defaults to ``time.monotonic``.
            total_steps: The last step the run will log, for an estimate of
                         when it finishes.  ``None`` (default) means unknown.
        """
        self._run_id: str = run_name if run_name is not None else os.environ.get("MLSWEEP_RUN_NAME", "")
        self._experiment: str = os.environ.get("EXP_EXPERIMENT", "")
        _is_lead = (
            os.environ.get("MLSWEEP_GPU_RANK", "0") == "0"
            and os.environ.get("MLSWEEP_NODE_RANK", "0") == "0"
        )
        sock_path = socket_path if socket_path is not None else os.environ.get("MLSWEEP_WORKER_SOCKET")
        self._sock_path: str | None = sock_path if (not rank_zero_only or _is_lead) else None
        self._clock = clock
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self.step: int = 0
        self._logged_any = False
        self._step_keys: set[str] = set()   # keys already logged at self.step
        self.step_seconds: float | None = None
        self.total_steps: int | None = total_steps
        self._last_heartbeat: float | None = None
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
            metrics: Mapping of metric name to value: a number, or a list or
                     dict of them, which is logged one key per element
                     (``name/0``, ``name/sub``).
            step:    Training step.  If omitted, ``self.step`` is incremented
                     by 1 and the new value is used.

        Raises:
            ValueError: *step* is lower than the last logged step, or a key in
                *metrics* was already logged at *step*.  Either would
                overwrite a value already recorded.
        """
        metrics = _flatten(metrics)
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
            **({"total_steps": int(self.total_steps)} if self.total_steps is not None else {}),
        })

    def _time_step(self, prev_step: int | None) -> None:
        """Update step_seconds when this log() moved to a new step."""
        if prev_step is not None and self.step == prev_step:
            return
        now = self._clock()
        if prev_step is not None and self._step_clock is not None:
            self.step_seconds = (now - self._step_clock) / (self.step - prev_step)
        self._step_clock = now

    def heartbeat(self) -> None:
        """Say the script is still working, during a stretch that logs nothing.

        A run that logs no metric and prints nothing for ``--stalled-after``
        seconds counts as stalled for ``mlsweep wait --until stalled``.  Call
        this inside a long quiet phase (an evaluation pass, a checkpoint, a
        branch that doesn't log) so it doesn't.  Cheap to call often: at most
        one message goes out every 10 seconds.
        """
        now = self._clock()
        if self._last_heartbeat is not None and now - self._last_heartbeat < 10.0:
            return
        self._last_heartbeat = now
        self._send({"type": "heartbeat", "run_id": self._run_id, "experiment": self._experiment})

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
