"""Weights & Biases writer.

Design note: inverted ownership
--------------------------------
The typical wandb pattern has training scripts call ``wandb.init()`` directly.
Here that is inverted: training scripts only call ``mlsweep.logger``, which is a
thin backend-agnostic shim that sends metric events over a socket to the
controller. The controller owns all wandb I/O, which keeps the training script
free of any wandb dependency and allows backends to be configured or swapped at
the call site without touching training code.

The cost is that all ``wandb.init()`` and ``run.finish()`` calls funnel through
the controller process. To prevent them from serializing on the event loop, each
run gets its own background thread. The thread does ``wandb.init()``, drains a
``queue.Queue`` of metric and finish events, then exits. ``on_sweep_end()`` joins
all threads, so the process does not exit until every run has finished uploading.
"""

import os
import queue
import re
import threading
import uuid
from typing import Any

from mlsweep._writers import RunWriter


def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "-", s)[:64]


def _import_wandb() -> Any:
    try:
        import wandb
        return wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is required. Install with: pip install 'mlsweep[wandb]'"
        ) from exc


class WandbRunWriter:
    def __init__(self, wandb: Any, init_kwargs: dict[str, Any]) -> None:
        self._q: queue.Queue[Any] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, args=(wandb, init_kwargs), daemon=True
        )
        self._thread.start()

    def _run(self, wandb: Any, init_kwargs: dict[str, Any]) -> None:
        run = wandb.init(**init_kwargs)
        while True:
            item = self._q.get()  # blocks until work arrives, yields scheduler
            kind: str = item[0]
            if kind == "metric":
                run.log(item[2], step=item[1])
            elif kind == "finish":
                run.summary["elapsed"] = item[2]
                run.finish(exit_code=0 if item[1] == "ok" else 1)
                break

    def on_metric(self, step: int, data: dict[str, Any]) -> None:
        self._q.put(("metric", step, data))

    def on_finish(self, status: str, elapsed: float) -> None:
        self._q.put(("finish", status, elapsed))

    def join(self) -> None:
        self._thread.join()


class WandbWriterFactory:
    def __init__(
        self,
        project: str,
        entity: str | None = None,
        mode: str = "online",
        resume: str | None = None,
    ) -> None:
        self._wandb: Any = _import_wandb()
        self._project = project
        self._entity = entity
        self._mode = mode
        self._resume = resume
        self._experiment = ""
        self._writers: list[WandbRunWriter] = []
        api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")
        self._wandb.login(key=api_key, relogin=False)

    def on_sweep_start(self, experiment: str, dims: list[str], runs: list[str]) -> None:
        self._experiment = experiment

    def make(self, run_id: str, combo: dict[str, Any], output_run_dir: str) -> RunWriter:
        if self._resume is not None:
            wid = _slugify(f"{self._experiment}--{run_id}")
            resume = self._resume
        else:
            wid = uuid.uuid4().hex[:8]
            resume = None
        kwargs: dict[str, Any] = dict(
            project=self._project,
            entity=self._entity,
            id=wid,
            name=run_id,
            group=self._experiment,
            config=combo,
            mode=self._mode,
            reinit="create_new",
        )
        if resume is not None:
            kwargs["resume"] = resume
        writer = WandbRunWriter(self._wandb, kwargs)
        self._writers.append(writer)
        return writer

    def on_sweep_end(self) -> None:
        for w in self._writers:
            w.join()
        self._writers.clear()
