# Weights & Biases Export Adapter

This document is a self-contained implementation guide for an adapter that converts mlsweep
experiment data into Weights & Biases (wandb) runs. No other context is required.

---

## How wandb Works

wandb tracks ML experiments through **runs**. A run is the atomic unit: it has a config
(hyperparameters), a history (time-series metrics), a summary (final values), and associated
files. Runs belong to a **project** and optionally a **group**.

You create and populate a run using the Python SDK:

```python
import wandb

run = wandb.init(
    project="my-project",
    name="run-a",               # display name (human-readable)
    id="run-a-unique-id",       # unique identifier for resuming
    group="my-sweep-name",      # groups related runs together
    config={"lr": 0.001, "batch_size": 32},  # hyperparameters
    mode="offline",             # "online", "offline", or "disabled"
)
wandb.log({"loss": 0.42, "accuracy": 0.91}, step=100)
wandb.finish()
```

### Online vs. Offline

- **`mode="online"`**: Metrics stream to wandb servers in real time. Requires an API key
  and internet access. Simplest if the export machine has both.
- **`mode="offline"`**: All data written locally to `./wandb/run-{timestamp}-{id}/` as
  `.wandb` binary files. Sync later with `wandb sync ./wandb/run-*/`.
- **`mode="disabled"`**: No-op; useful for testing.

### Run Grouping

Passing `group="experiment_name"` to `wandb.init()` groups related runs in the UI. The
**group view** aggregates metrics and allows side-by-side comparison of all runs in the
group — this is the natural analog of mlsweep's sweep concept.

Groups are **not** wandb's built-in sweep system (which involves a `wandb agent` running
new training processes). They are simply a UI organizational feature. We use groups, not
wandb sweeps.

### `wandb.log()` Stepping

```python
wandb.log({"loss": 0.5}, step=1)
wandb.log({"loss": 0.4}, step=2)
```

- `step` is an integer. wandb requires `step` to be non-decreasing across calls.
- If `step` is omitted, wandb auto-increments an internal counter.
- Multiple metrics can be logged at the same step in one call or separate calls.
- **You cannot backfill steps retroactively** — once a run is finished (`wandb.finish()`),
  you cannot add data to it via the Python SDK. You must log all data before calling
  `wandb.finish()`.

### `wandb.finish()`

Marks the run as complete. In online mode, flushes all queued data. In offline mode,
finalizes the local `.wandb` file. Takes optional `exit_code` (0 = success, non-zero = failed).

---

## mlsweep Data → wandb Mapping

### Source data (mlsweep)

```
outputs/sweeps/{experiment}/
  sweep_manifest.json          # dims, runs list, metricNames
  sweep_status.json            # per-run: status, elapsed, combo
  {run_id}/
    metrics.jsonl              # {"step": N, "loss": v, ...} one per line
    training.log               # stdout/stderr text
    artifacts/                 # arbitrary files (checkpoints, etc.)
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

| mlsweep concept | wandb concept |
|-----------------|---------------|
| `experiment` name | `group` in `wandb.init()` |
| `run_id` | `name` in `wandb.init()` (display name) |
| `run_id` (slugified) | `id` in `wandb.init()` (unique key, no slashes) |
| `combo` dict | `config` in `wandb.init()` |
| `step` field in metrics.jsonl | `step` in `wandb.log()` |
| metric name/value pairs | dict passed to `wandb.log()` |
| `status == "ok"` | `wandb.finish(exit_code=0)` |
| `status == "failed"` | `wandb.finish(exit_code=1)` |
| `elapsed` (seconds) | `run.summary["elapsed"]` (set via `wandb.summary.update()`) |
| `training.log` | `wandb.save()` to upload as a file, or skip |
| `artifacts/` | `wandb.save()` or wandb Artifacts API |

---

## What Matches Well

- **Config / hyperparameters:** mlsweep's `combo` dict (e.g. `{"lr": 0.001, "batch_size": 32}`)
  is a flat dict of scalar values — exactly what `wandb.init(config=...)` accepts. Config
  values are displayed in the runs table and can be used for filtering, grouping, and
  parallel coordinates plots.

- **Metrics:** mlsweep's `metrics.jsonl` rows map directly to `wandb.log()` calls. The
  `step` field maps to wandb's `step` parameter. All other keys become metric names. No
  transformation needed.

- **Multiple runs as a group:** mlsweep's experiment = a set of runs sharing a sweep name.
  wandb's `group` parameter is exactly this: a label that groups related runs in the UI.
  The group view shows aggregate metrics and a run comparison table with configs.

- **Run status:** `sweep_status.json` records `"ok"` or `"failed"`. This maps to
  `wandb.finish(exit_code=0)` or `wandb.finish(exit_code=1)`, which sets the run's final
  state in the UI (green checkmark vs. red X).

- **Sweep-level metadata:** `metricNames` and `dims` from the manifest can be logged as
  summary metadata on a synthetic "sweep summary" run or simply left to wandb to infer from
  the logged data.

---

## What Doesn't Match / Gaps

### 1. wandb's built-in sweep system is entirely separate

wandb has its own hyperparameter search system (`wandb sweep` + `wandb agent`) that
generates new training processes. mlsweep is its own independent scheduler. The export
adapter does **not** use `wandb.sweep()` or `wandb.agent()` — it just creates plain runs
organized into a group. Runs will appear in the "Runs" tab, not the "Sweeps" tab. This is
intentional and correct for our use case.

### 2. No per-step wall-clock timestamps

wandb records wall-clock time (`_timestamp`) at the moment `wandb.log()` is called.
mlsweep stores only total `elapsed` time per run, not per-step timestamps or absolute start
time. When the adapter replays historical data, all steps will have the wall-clock time of
the export, not the original training time. This makes wandb's time-based axes inaccurate.

**Workaround:** wandb.log() accepts a `commit=True` parameter but no explicit `walltime`
parameter in the standard Python SDK. There is no clean public API to set historical
timestamps. This limitation is inherent to the offline-export approach.

### 3. `wandb.log()` is append-only per run session

All `wandb.log()` calls for a run must happen in a single `wandb.init()` / `wandb.finish()`
session, in non-decreasing step order. You cannot re-open a finished run and add data.
The adapter must read the entire `metrics.jsonl` and replay it in order within a single run
session.

This means the adapter is a one-shot export: run it once per experiment, or once per run.
Re-running the adapter on the same experiment will create duplicate wandb runs (with
different `id`s unless you deterministically derive them and use `resume="must"`).

To make re-runs idempotent, the adapter should derive a deterministic run `id` from
`(project, experiment, run_id)` and use `resume="allow"` so that if the run already exists
in wandb it resumes rather than creating a duplicate.

### 4. API key required for online mode

`mode="online"` requires a valid `WANDB_API_KEY` environment variable or interactive login
(`wandb login`). The export adapter should support `mode="offline"` as a default with
instructions to sync afterward via `wandb sync`.

### 5. `training.log` has no natural wandb equivalent

wandb does not have a native streaming-log concept equivalent to mlsweep's `training.log`.
Options:
- Upload as a file: `wandb.save(str(training_log_path), base_path=str(run_dir))` — viewable
  in the Files tab of the run page.
- Log as a text artifact: create a `wandb.Artifact` of type `"log"`.
- Skip entirely (simplest for a first implementation).

### 6. `artifacts/` directory

mlsweep rsyncs arbitrary files from the worker into `artifacts/`. These could be checkpoints,
outputs, etc. wandb has an Artifacts system for versioned files. For the first implementation,
artifacts can be skipped or uploaded with `wandb.save()` as plain files.

### 7. Run ID must not contain slashes or special characters

mlsweep `run_id` values look like `my_sweep_lr1e-4_bs32` and may contain `-`, `_`, `.`,
`=`. wandb `id` must be alphanumeric with no slashes. The adapter should slugify: replace
`.`, `=`, `/` with `-` or `_`, and truncate to 64 characters if needed.

---

## Suggested Implementation Sketch

```python
# mlsweep/export/wandb_export.py
import json
import re
from pathlib import Path
import wandb

def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", s)[:64]

def export_to_wandb(
    experiment_dir: Path,
    project: str,
    entity: str | None = None,
    mode: str = "offline",
) -> None:
    manifest = json.loads((experiment_dir / "sweep_manifest.json").read_text())
    status = json.loads((experiment_dir / "sweep_status.json").read_text())
    experiment = manifest["experiment"]

    for run in manifest["runs"]:
        run_id = run["name"]
        combo = run["combo"]
        run_status = status.get(run_id, {})
        metrics_file = experiment_dir / run_id / "metrics.jsonl"
        if not metrics_file.exists():
            continue

        exit_code = 0 if run_status.get("status") == "ok" else 1

        wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=run_id,
            id=slugify(f"{experiment}-{run_id}"),
            group=experiment,
            config=combo,
            mode=mode,
            resume="allow",
        )
        try:
            with metrics_file.open() as f:
                for line in f:
                    row = json.loads(line)
                    step = row.pop("step")
                    row.pop("t", None)  # backward compat: skip old "t" field
                    wandb.log(row, step=step)

            if "elapsed" in run_status:
                wandb.summary["elapsed"] = run_status["elapsed"]
        finally:
            wandb.finish(exit_code=exit_code)
```

After exporting in offline mode:

```bash
wandb sync ./wandb/run-*/
```

---

## wandb UI Features Available After Export

Once runs are uploaded (online or synced from offline):

- **Runs table:** All runs listed with config columns (lr, batch_size, etc.) and summary
  metrics. Sortable and filterable.
- **Charts:** Interactive metric curves per run, with run selection and comparison overlays.
- **Group view:** Since all runs share the same `group` (the experiment name), the group
  page shows all runs aggregated and side-by-side.
- **Parallel coordinates:** Requires the "Sweep" tab or a custom panel; available in group
  view. Shows which hyperparameter combinations produced the best metrics.
- **Run comparison:** Select multiple runs and compare their config + metrics directly.

Note: wandb's **Sweeps tab** (with Bayesian optimization, early stopping, etc.) will **not**
be populated — those features require using `wandb.agent()` to run training. Our export
only produces regular runs grouped together, which is sufficient for visualization and
comparison.
