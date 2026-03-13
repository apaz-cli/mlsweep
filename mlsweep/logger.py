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

All methods are silent no-ops when ``MLSWEEP_WORKER_SOCKET`` is not set, or when
``MLSWEEP_GPU_RANK`` / ``MLSWEEP_NODE_RANK`` indicate a non-lead rank (see
``rank_zero_only`` parameter).
"""

import json
import os
import socket
import threading
from typing import Any




class MLSweepLogger:
    """Metrics logger for mlsweep training scripts.

    Communicates with the worker daemon via a unix socket.  All operations
    are no-ops when ``MLSWEEP_WORKER_SOCKET`` is not set in the environment.

    Attributes:
        step: Current training step.  Updated by every ``log()`` call.
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
        self._run_id: str = run_name or os.environ.get("MLSWEEP_RUN_NAME", "")
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
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1
        self._send({
            "type": "metric",
            "run_id": self._run_id,
            "step": self.step,
            "data": metrics,
        })

    def sync(self) -> None:
        """Fire-and-forget: signal the worker to rsync artifacts to the controller."""
        self._send({"type": "sync", "run_id": self._run_id})

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
