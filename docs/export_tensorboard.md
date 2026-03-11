# TensorBoard Export Adapter

This document is a self-contained implementation guide for an adapter that converts mlsweep
experiment data into TensorBoard event files. No other context is required.

---

## What TensorBoard Expects

TensorBoard reads binary **event files** (`.tfevents.*`) from a directory tree. Each
subdirectory it finds is treated as a separate run. You point TensorBoard at a root log
directory and it scans recursively:

```
logdir/
  run-a/
    events.out.tfevents.1720000000.hostname.12345.0
  run-b/
    events.out.tfevents.1720000001.hostname.12346.0
```

```bash
tensorboard --logdir logdir/
```

Event files are written using either:

- `torch.utils.tensorboard.SummaryWriter` (requires PyTorch; most common in ML projects)
- `tensorflow.summary.create_file_writer()` (requires TensorFlow)
- The standalone `tensorboard` package's `SummaryWriter` (same API as PyTorch's, no PyTorch required)

The standalone `tensorboard` package is the right minimal dependency here since neither
PyTorch nor TensorFlow should be required by an export adapter. Install with:
`pip install tensorboard`.

---

## Writing Scalars

```python
from torch.utils.tensorboard import SummaryWriter
# or: from tensorboard.summary.writer.event_file_writer import EventFileWriter (lower-level)

writer = SummaryWriter(log_dir="logdir/run-a")
writer.add_scalar("loss", 0.42, global_step=100)
writer.add_scalar("accuracy", 0.91, global_step=100)
writer.close()
```

Key points:
- `add_scalar(tag, scalar_value, global_step)` — `tag` is the metric name, `global_step` is the step integer.
- Each call to `add_scalar` also records a wall-clock timestamp automatically.
- Call `writer.close()` (or use it as a context manager) to flush and finalize the file.
- The file is named with the current Unix timestamp and hostname at creation time.

---

## HParams Plugin (Multi-Run Comparison)

TensorBoard's **HParams plugin** enables a dedicated dashboard for comparing hyperparameter
configurations across runs. It requires an additional write step using the
`tensorboard.plugins.hparams` API.

### Step 1: Write the experiment config (once, in the root logdir)

```python
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf  # or: import torch; use SummaryWriter

with tf.summary.create_file_writer("logdir").as_default():
    hp.hparams_config(
        hparams=[
            hp.HParam("lr", hp.Discrete([1e-4, 3e-4, 1e-3])),
            hp.HParam("batch_size", hp.Discrete([32, 64, 128])),
        ],
        metrics=[
            hp.Metric("loss", display_name="Loss"),
            hp.Metric("accuracy", display_name="Accuracy"),
        ],
    )
```

This step is **optional** but enables proper axis types and filtering in the HParams
dashboard. Without it, TensorBoard will still show the HParams tab but with less metadata.

### Step 2: Write per-run hparams and metrics

```python
def write_run(run_dir, hparams_dict, metrics_steps):
    with tf.summary.create_file_writer(run_dir).as_default():
        # Log hyperparameters for this run
        hp.hparams(hparams_dict)
        # Log scalar metrics
        for step, metrics in metrics_steps:
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=step)
```

The `hp.hparams()` call and `tf.summary.scalar()` calls must be in the same writer scope.
The HParams dashboard reads the `hp.hparams()` call to know this run's configuration.

**Constraint:** HParam values must be scalar — `int`, `float`, `str`, or `bool`. Lists,
dicts, and `None` are not supported. The mlsweep combo values are typically floats, ints,
or strings (dimension values), so this is usually not a problem, but any non-scalar combo
value will need to be stringified.

---

## mlsweep Data → TensorBoard Mapping

### Source data (mlsweep)

```
outputs/sweeps/{experiment}/
  sweep_manifest.json          # dims, runs list, metricNames
  sweep_status.json            # per-run: status, elapsed, combo
  {run_id}/
    metrics.jsonl              # {"step": N, "loss": v, ...} one per line
    training.log               # stdout/stderr text
    artifacts/                 # arbitrary files
```

**`sweep_manifest.json` structure:**
```json
{
  "experiment": "my_sweep_20250310_094530",
  "dims": {"lr": [1e-4, 3e-4, 1e-3], "batch_size": [32, 64, 128]},
  "runs": [{"name": "my_sweep_lr1e-4_bs32", "combo": {"lr": 0.0001, "batch_size": 32}}],
  "metricNames": ["loss", "accuracy"]
}
```

**`metrics.jsonl` structure:**
```
{"step": 1, "loss": 0.91, "accuracy": 0.43}
{"step": 2, "loss": 0.88, "accuracy": 0.47}
```

### Mapping

| mlsweep concept | TensorBoard concept |
|-----------------|---------------------|
| `experiment` | Root `logdir` name (e.g. `tb_logs/{experiment}/`) |
| `run_id` | Subdirectory under logdir (e.g. `tb_logs/{experiment}/{run_id}/`) |
| `combo` dict | `hp.hparams({HP_OBJ: value, ...})` per run |
| `dims` dict | `hp.HParam(name, hp.Discrete(values))` list for experiment config |
| `metricNames` list | `hp.Metric(name)` list for experiment config |
| `step` field in metrics.jsonl | `global_step` in `add_scalar` / `tf.summary.scalar` |
| metric name (e.g. `"loss"`) | `tag` in `add_scalar` / `name` in `tf.summary.scalar` |
| metric value | `scalar_value` / `data` |
| `elapsed` (from sweep_status) | No direct mapping; could be logged as a summary scalar |
| `training.log` | No native TensorBoard concept; skip or upload as a text file artifact |
| `artifacts/` | No native TensorBoard concept; skip |

---

## What Matches Well

- **Step numbers:** mlsweep's `step` field maps exactly to TensorBoard's `global_step`.
  Both are plain integers and have the same semantics.
- **Metric names:** mlsweep metric names are arbitrary strings; TensorBoard tags are
  arbitrary strings. Direct 1:1 mapping with no transformation needed.
- **Hyperparameters:** mlsweep's `combo` dict (e.g. `{"lr": 0.001, "batch_size": 32}`) is
  a flat dict of scalar values — exactly what the HParams plugin expects.
- **Multiple runs:** mlsweep's one-directory-per-run structure maps cleanly to TensorBoard's
  one-subdirectory-per-run model.
- **Sweep as experiment:** mlsweep's `dims` dict (all hyperparameter names and their value
  ranges) maps directly to `hp.HParam(name, hp.Discrete(values))` for the experiment config.

---

## What Doesn't Match / Gaps

### 1. Wall-clock timestamps are unavailable

TensorBoard event files record wall-clock time per event. The writer sets this automatically
to `time.time()` at the moment `add_scalar()` is called.

mlsweep only stores `elapsed` (total run duration, seconds), not per-step timestamps or an
absolute run start time. The adapter will write all historical events with wall-clock times
reflecting when the export ran, not when training happened. This means:

- The "time" x-axis view in TensorBoard will be meaningless (all events appear simultaneous).
- The "step" x-axis view is unaffected and is the primary view.

**Workaround if desired:** Use `walltime` parameter in PyTorch's SummaryWriter
(`add_scalar(tag, value, step, walltime=computed_time)`) to reconstruct approximate timestamps
by distributing `elapsed` proportionally across steps. This is an approximation and can be
skipped for a first implementation.

### 2. HParam values must be scalar

The HParams plugin does not support list or dict values. mlsweep combo values are always
scalar (int, float, or string) from dimension value lists, so this shouldn't be a problem
in practice. However, the adapter should stringify any non-scalar combo value defensively:
`str(v) if not isinstance(v, (int, float, bool, str)) else v`.

### 3. No native "failed run" concept

TensorBoard has no concept of a failed run. Runs are just directories with event files —
there is no status field. The adapter can either:
- Skip failed runs entirely (simplest).
- Export them anyway and include a summary scalar like `tf.summary.scalar("run_failed", 1, step=0)`.

### 4. TensorBoard cannot display training logs

mlsweep captures full stdout/stderr in `training.log`. TensorBoard has no equivalent.
The text can be logged with `writer.add_text("training_log", content, step=0)` as a fallback,
but this is awkward for large logs. The simplest approach: ignore `training.log` in the export.

### 5. Dependency choice

`torch.utils.tensorboard.SummaryWriter` requires PyTorch. If the project already depends on
PyTorch, use it. Otherwise, use the standalone `tensorboard` package's writer:

```python
from tensorboard.summary.writer.event_file_writer import EventFileWriter
# or install tensorboardX: pip install tensorboardX
from tensorboardX import SummaryWriter
```

`tensorboardX` is the most self-contained option and has identical API to PyTorch's
SummaryWriter. The `tensorflow` package is not required for writing events.

---

## Suggested Implementation Sketch

```python
# mlsweep/export/tensorboard.py
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter  # or tensorboardX

def export_to_tensorboard(experiment_dir: Path, output_dir: Path) -> None:
    manifest = json.loads((experiment_dir / "sweep_manifest.json").read_text())
    status = json.loads((experiment_dir / "sweep_status.json").read_text())
    experiment = manifest["experiment"]
    root = output_dir / experiment

    # Optional: write hparams_config to root logdir
    # (requires tensorboard.plugins.hparams + tensorflow)

    for run in manifest["runs"]:
        run_id = run["name"]
        combo = run["combo"]
        run_status = status.get(run_id, {})
        metrics_file = experiment_dir / run_id / "metrics.jsonl"
        if not metrics_file.exists():
            continue

        writer = SummaryWriter(log_dir=str(root / run_id))
        try:
            with metrics_file.open() as f:
                for line in f:
                    row = json.loads(line)
                    step = row["step"]
                    for key, val in row.items():
                        if key == "step":
                            continue
                        writer.add_scalar(key, val, global_step=step)
        finally:
            writer.close()
```

---

## TensorBoard Launch

After export:

```bash
tensorboard --logdir tb_logs/my_sweep_20250310_094530/
```

Open `http://localhost:6006`. The **Scalars** tab shows per-run metric curves. The **HParams**
tab (if hparams were written) shows the comparison table and parallel coordinates view.
