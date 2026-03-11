"""TensorBoard writer."""

import os
from typing import Any

from mlsweep._writers import RunWriter


class _StandaloneSummaryWriter:
    """Minimal SummaryWriter using the standalone tensorboard package's protos."""

    def __init__(self, log_dir: str) -> None:
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        import tensorboard.compat.proto.event_pb2 as event_pb2
        import tensorboard.compat.proto.summary_pb2 as summary_pb2
        import time

        os.makedirs(log_dir, exist_ok=True)
        self._writer = EventFileWriter(log_dir)
        self._event_pb2 = event_pb2
        self._summary_pb2 = summary_pb2
        self._time = time

    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0) -> None:
        summary = self._summary_pb2.Summary()
        v = summary.value.add()
        v.tag = tag
        v.simple_value = float(scalar_value)
        event = self._event_pb2.Event(
            summary=summary, step=global_step, wall_time=self._time.time()
        )
        self._writer.add_event(event)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


def _import_summary_writer() -> Any:
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter
    except ImportError:
        pass
    try:
        from tensorboardX import SummaryWriter
        return SummaryWriter
    except ImportError:
        pass
    try:
        from tensorboard.summary.writer.event_file_writer import EventFileWriter  # noqa: F401
        return _StandaloneSummaryWriter
    except ImportError:
        pass
    raise ImportError(
        "TensorBoard writing requires torch, tensorboardX, or the standalone "
        "tensorboard package. Install with: pip install 'mlsweep[tensorboard]'."
    )


class TensorBoardRunWriter:
    def __init__(self, writer: Any) -> None:
        self._writer = writer

    def on_metric(self, step: int, data: dict[str, Any]) -> None:
        for k, v in data.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(k, v, global_step=step)

    def on_finish(self, status: str, elapsed: float) -> None:
        self._writer.close()


class TensorBoardWriterFactory:
    def __init__(self, tb_dir: str) -> None:
        self._tb_dir = tb_dir
        self._SummaryWriter: Any = _import_summary_writer()
        self._experiment = ""

    def on_sweep_start(self, experiment: str, dims: list[str], runs: list[str]) -> None:
        self._experiment = experiment

    def make(self, run_id: str, combo: dict[str, Any], output_run_dir: str) -> RunWriter:
        log_dir = os.path.join(self._tb_dir, self._experiment, run_id)
        writer: Any = self._SummaryWriter(log_dir=log_dir)
        return TensorBoardRunWriter(writer)

    def on_sweep_end(self) -> None:
        pass
