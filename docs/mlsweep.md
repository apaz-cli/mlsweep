# mlsweep skill

Use this skill when a user needs help using the `mlsweep` package — writing sweep files, instrumenting training scripts, running sweeps, or understanding configuration options.

---

## What mlsweep does

mlsweep runs hyperparameter sweeps for ML training. You write a Python config file that declares *what* to vary; mlsweep handles launching, GPU assignment, logging, and visualization.

The core loop:
1. Write a sweep file (`sweeps/my_sweep.py`) declaring which hyperparams to vary.
2. Add a few lines to your training script to log metrics.
3. Run `mlsweep_run sweeps/my_sweep.py -g N` with N GPUs.
4. View results with `mlsweep_viz`.

---

## Minimal working example

**sweeps/my_sweep.py:**
```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]

OPTIONS = {
    ".lr": {
        "values": [1e-4, 3e-4, 1e-3],
        "flags": "--lr",
        "name": "lr",
    },
    ".bs": {
        "values": [32, 64],
        "flags": "--batch_size",
        "name": "bs",
    },
}
```
This produces 6 runs (3 × 2 grid). Each run calls `python train.py` with the appropriate `--lr` and `--batch_size` flags appended.

**train.py (the only change needed):**
```python
from mlsweep.logger import MLSweepLogger

with MLSweepLogger() as logger:
    for step in range(1, num_steps + 1):
        loss = train_step()
        logger.log({"loss": loss, "lr": current_lr}, step=step)
```
`MLSweepLogger` is a no-op when run outside mlsweep, so training scripts work unchanged.

**Run it:**
```bash
mlsweep_run sweeps/my_sweep.py            # 1 GPU, sequential
mlsweep_run sweeps/my_sweep.py -g 4       # 4 GPUs, up to 4 runs in parallel
mlsweep_run sweeps/my_sweep.py -g 4 -j 2  # 4 GPUs, 2 runs per GPU (8 concurrent)
```

**Visualize:**
```bash
mlsweep_viz              # most recent experiment
mlsweep_viz my_exp_name  # specific experiment
```

---

## Key concepts

### OPTIONS keys start with `.`

Every dimension key in `OPTIONS` starts with `.`. The dot distinguishes dimension keys from metadata keys (`values`, `flags`, `name`, etc.).

```python
OPTIONS = {
    ".lr": {                     # dimension key (starts with .)
        "values": [1e-4, 1e-3],
        "flags": "--lr",         # metadata key (no dot)
        "name": "lr",
    },
}
```

### Three dimension types

**Value dim** — sweep over an explicit list:
```python
".lr": {"values": [1e-4, 3e-4, 1e-3], "flags": "--lr", "name": "lr"}
```

**Fixed dim** — always-on flags, no variation:
```python
".dtype": {"flags": ["--dtype", "bfloat16"]}
```

**Subdim** — mutually exclusive branches. Use when some parameters only make sense for certain other parameters:
```python
".optimizer": {
    "name": "opt",
    ".adam": {
        "flags": ["--optimizer", "adam"],
        ".beta1": {"values": [0.85, 0.9, 0.95], "flags": "--beta1", "name": "b1"},
        ".beta2": {"values": [0.9, 0.999],       "flags": "--beta2", "name": "b2"},
    },
    ".muon": {
        "flags": ["--optimizer", "muon"],
        ".lr_scale": {"values": [0.1, 1.0, 10.0], "flags": "--lr_scale", "name": "lrs"},
    },
}
```
This produces:
- 6 Adam runs (3 beta1 × 2 beta2) with flags like `--optimizer adam --beta1 0.9 --beta2 0.999`
- 3 Muon runs with flags like `--optimizer muon --lr_scale 1.0`

Without subdims, a flat product would generate nonsensical combinations — Adam runs with `--lr_scale` and Muon runs with `--beta1`/`--beta2`. `EXCLUDE` could filter them out, but subdims express the structure directly and are easier to extend.

### Run naming

Runs are automatically named `{sweep_name}_{dim1_name}{val1}_{dim2_name}{val2}…`. For example: `my_sweep_lr0.001_bs32`.

Set `"name": None` to omit a dim from the run name (useful for hardware-tuning dims that don't affect results).

### The `flags` field

- `"flags": "--lr"` → appends `["--lr", "0.001"]` for each value.
- `"flags": {"none": ["--ac", "none"], "full": ["--ac", "full"]}` → explicit tokens per value (use when values need different flag structures).

---

## Common patterns

### Find the largest batch size that fits in memory

`singular: True` tries values in order and stops at the first success:

```python
".bs": {
    "values": [512, 256, 128, 64, 32],  # largest first
    "flags": "--training.local_batch_size",
    "name": None,      # hardware detail, omit from run name
    "singular": True,  # commit to first size that works
},
```

In a 3-LR × 5-BS sweep, this generates 15 combinations but *expects* only 3 runs (one per LR). The run count shown will reflect this.

### Share hardware dims across sweep files

```python
# sweeps/_common.py
LOCAL_BATCH_SIZE = {
    "values": [256, 128, 64, 32, 16],
    "flags": "--training.local_batch_size",
    "name": None,
    "singular": True,
}
```

```python
# sweeps/my_sweep.py
from _common import LOCAL_BATCH_SIZE

OPTIONS = {
    ".local_batch_size": LOCAL_BATCH_SIZE,
    ".lr": {"values": [1e-4, 3e-4], "flags": "--lr", "name": "lr"},
}
```

### Skip combinations

```python
def EXCLUDE(combo: dict) -> bool:
    return combo["wd"] == 0.0 and combo["wd_compensation"] is True
```

Prefer subdims when one dim only applies to certain values of another. Use `EXCLUDE` for cross-cutting constraints.

### Pass flags to every run

```python
EXTRA_FLAGS = ["--training.steps", "5000"]
```

Or at the command line (after `--`):
```bash
mlsweep_run sweeps/my_sweep.py -g 4 -- --training.steps 5000
```

### Multi-GPU (DDP) per run

```python
GPUS_PER_RUN = 4
SET_DIST_ENV = True   # auto-sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
```

mlsweep spawns one process per GPU and sets `MLSWEEP_GPU_RANK` (0-based) and `CUDA_VISIBLE_DEVICES`.

With `-g 8` and `GPUS_PER_RUN = 4`, you get 2 concurrent runs, each using 4 GPUs. GPU groups are chosen to maximize NVLink connectivity.

In your training script:
```python
local_rank = int(os.environ["MLSWEEP_GPU_RANK"])
device = torch.device(f"cuda:{local_rank}")
```

With `SET_DIST_ENV = True`, standard `RANK`/`LOCAL_RANK`/`WORLD_SIZE` are set automatically, so frameworks like TorchTitan work without any wrapper script.

### Remote workers

```toml
# workers.toml
[[workers]]
host = "user@host1"
remote_dir = "/home/user/myproject"
gpus = 4
jobs = 2
ssh_key = "~/.ssh/id_ed25519"
venv = "/home/user/myproject/.venv"
```

```bash
mlsweep_run sweeps/my_sweep.py --workers workers.toml
```

Requires passwordless SSH. Test with: `ssh -o BatchMode=yes user@host1 nvidia-smi`.

### Bayesian optimization

```python
OPTIMIZE = {
    "method": "bayes",
    "metric": "val_loss",
    "goal": "minimize",
    "budget": 40,
}

OPTIONS = {
    ".lr": {
        "distribution": "log_uniform",  # sample from a continuous range
        "min": 1e-5,
        "max": 1e-1,
        "flags": "--lr",
        "name": "lr",
    },
    ".wd": {
        "distribution": "log_uniform",
        "min": 1e-6,
        "max": 1e-2,
        "flags": "--wd",
        "name": "wd",
    },
}
```

Requires: `pip install 'mlsweep[bayes]'`

Discrete dims (`"values"`) and subdims work unchanged in bayes mode — optuna treats them as categorical.

---

## What mlsweep passes to your training script

mlsweep launches your `COMMAND` as a subprocess and injects the following environment variables before it starts. Your training script reads them via `os.environ`. None of these are required — `MLSweepLogger` is a no-op if they're absent — but they let you integrate with mlsweep's GPU assignment, checkpointing, and distributed training coordination.

| Variable | When set | Value |
|---|---|---|
| `MLSWEEP_RUN_DIR` | always | Directory to write checkpoints; synced to output dir at run end |
| `MLSWEEP_RUN_NAME` | always | Unique name for this run (e.g. `sweep_lr0.001_bs32`) |
| `MLSWEEP_WORKER_SOCKET` | always | Unix socket used by `MLSweepLogger` to communicate with the worker |
| `CUDA_VISIBLE_DEVICES` | always | Comma-separated GPU IDs assigned to this run (e.g. `0,1,2,3`) |
| `HIP_VISIBLE_DEVICES` | always | Same as `CUDA_VISIBLE_DEVICES` (for AMD ROCm) |
| `MLSWEEP_GPU_RANK` | always | 0-based rank of this process within the run's GPU group |
| `MLSWEEP_NNODES` | multi-node | Total node count for this run |
| `MLSWEEP_NODE_RANK` | multi-node | This node's rank (0-based) |
| `MLSWEEP_MASTER_ADDR` | multi-node | Hostname of rank-0 worker |
| `MLSWEEP_MASTER_PORT` | multi-node | Port for the distributed rendezvous |
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | `SET_DIST_ENV=True` | Standard PyTorch distributed vars, derived from the above |

The per-run hyperparameter flags (from `OPTIONS`) are appended as **CLI arguments**, not env vars. Your script receives them exactly as if you had typed them: `python train.py --lr 0.001 --batch_size 32`.

---

## Checkpoint artifacts

Save checkpoints to `os.environ["MLSWEEP_RUN_DIR"]`. They are rsynced to the output directory at run end. Trigger a mid-training sync:

```python
logger.sync()  # fire-and-forget rsync
```

---

## Output layout

```
outputs/sweeps/
└── my_sweep_20240315_1430/
    ├── sweep_manifest.json   # dims, combos
    ├── sweep_status.json     # per-run status (used by --resume)
    ├── sweep.log             # runner log
    └── my_sweep_lr0.001_bs32/
        ├── metrics.jsonl     # one JSON object per logger.log() call
        └── training.log      # stdout + stderr from training script
```

---

## mlsweep_viz

`mlsweep_viz` starts a local web server and opens an interactive browser UI for exploring loss curves and comparing runs. It reads from `outputs/sweeps/` and live-updates as new runs complete.

```bash
mlsweep_viz                         # open most recent experiment
mlsweep_viz my_sweep_20240315_1430  # open a specific experiment by name
mlsweep_viz --open-browser          # auto-open browser tab (default: prints URL)
mlsweep_viz --port 8080             # use a different port (default: 43801)
mlsweep_viz --dir ./other/path      # look for experiments in a different directory
```

The UI lets you:
- Select which metric to plot on the y-axis.
- Filter runs by dimension value (e.g. show only `lr=0.001` runs).
- Toggle individual runs on/off.
- Switch between linear and log scale.

---

## Useful CLI flags

```bash
mlsweep_run sweep.py --dry-run            # print commands without running
mlsweep_run sweep.py --validate           # check config and list all combos, exit
mlsweep_run sweep.py --resume             # skip already-completed runs
mlsweep_run sweep.py --experiment NAME    # custom experiment name
mlsweep_run sweep.py --output_dir PATH    # custom output directory
mlsweep_run sweep.py --wandb-project P    # stream metrics to W&B
mlsweep_run sweep.py --tensorboard-dir D  # write TensorBoard logs
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `COMMAND is required` | Add `COMMAND = ["python", "train.py"]` |
| `Dimension key '...' must start with '.'` | Add a `.` prefix to all `OPTIONS` keys |
| `has both 'values' and subdimensions` | A dim is either a value list or subdim branches, not both |
| Monotonic/singular not skipping | Only works with `-g 1 -j 1` (sequential); skipping doesn't apply in parallel mode |
| Remote not connecting | Test SSH: `ssh -o BatchMode=yes user@host nvidia-smi` |
| `need at least GPUS_PER_RUN GPUs` | Increase `-g N` or decrease `GPUS_PER_RUN` |

---

## Reference docs

- `docs/sweep_configuration.md` — complete reference for all directives, dim types, flags behavior, and options
- `docs/examples.md` — patterns for DDP, multi-node, TorchTitan, Prime-RL
