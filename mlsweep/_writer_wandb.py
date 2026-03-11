"""Weights & Biases writer."""

import os
import re
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
    def __init__(self, run: Any) -> None:
        self._run = run

    def on_metric(self, step: int, data: dict[str, Any]) -> None:
        self._run.log(data, step=step)

    def on_finish(self, status: str, elapsed: float) -> None:
        self._run.summary["elapsed"] = elapsed
        self._run.finish(exit_code=0 if status == "ok" else 1)


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
        api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")
        self._wandb.login(key=api_key, relogin=False)

    def on_sweep_start(self, experiment: str, dims: list[str], runs: list[str]) -> None:
        self._experiment = experiment

    def make(self, run_id: str, combo: dict[str, Any], output_run_dir: str) -> RunWriter:
        if self._resume is not None:
            wid = _slugify(f"{self._experiment}--{run_id}")
            resume = self._resume
        else:
            # Generate a fresh ID explicitly so WANDB_RUN_ID env var cannot
            # override us and cause all concurrent runs to share one run.
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
        run: Any = self._wandb.init(**kwargs)
        return WandbRunWriter(run)

    def on_sweep_end(self) -> None:
        pass
