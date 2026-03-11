"""RunWriter protocol and built-in writer implementations."""

import json
import os
from typing import Any, IO, Protocol


class RunWriter(Protocol):
    def on_metric(self, step: int, data: dict[str, Any]) -> None: ...
    def on_finish(self, status: str, elapsed: float) -> None: ...


class WriterFactory(Protocol):
    def on_sweep_start(self, experiment: str, dims: list[str], runs: list[str]) -> None: ...
    def make(self, run_id: str, combo: dict[str, Any], output_run_dir: str) -> RunWriter: ...
    def on_sweep_end(self) -> None: ...


# ── MlsweepWriter ─────────────────────────────────────────────────────────────


class MlsweepRunWriter:
    def __init__(self, path: str) -> None:
        self._fh: IO[str] = open(path, "w", buffering=1)

    def on_metric(self, step: int, data: dict[str, Any]) -> None:
        record: dict[str, Any] = {"step": step, **data}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def on_finish(self, status: str, elapsed: float) -> None:
        self._fh.close()


class MlsweepWriterFactory:
    def on_sweep_start(self, experiment: str, dims: list[str], runs: list[str]) -> None:
        pass

    def make(self, run_id: str, combo: dict[str, Any], output_run_dir: str) -> RunWriter:
        path = os.path.join(output_run_dir, "metrics.jsonl")
        return MlsweepRunWriter(path)

    def on_sweep_end(self) -> None:
        pass


# ── MultiWriter ────────────────────────────────────────────────────────────────


class MultiRunWriter:
    def __init__(self, writers: list[RunWriter]) -> None:
        self._writers = writers

    def on_metric(self, step: int, data: dict[str, Any]) -> None:
        for w in self._writers:
            w.on_metric(step, data)

    def on_finish(self, status: str, elapsed: float) -> None:
        for w in self._writers:
            w.on_finish(status, elapsed)


class MultiWriterFactory:
    def __init__(self, factories: list[WriterFactory]) -> None:
        self._factories = factories

    def on_sweep_start(self, experiment: str, dims: list[str], runs: list[str]) -> None:
        for f in self._factories:
            f.on_sweep_start(experiment, dims, runs)

    def make(self, run_id: str, combo: dict[str, Any], output_run_dir: str) -> RunWriter:
        writers = [f.make(run_id, combo, output_run_dir) for f in self._factories]
        return MultiRunWriter(writers)

    def on_sweep_end(self) -> None:
        for f in self._factories:
            f.on_sweep_end()
