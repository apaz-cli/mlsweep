# mlsweep skill

Use this skill when a user needs help using the `mlsweep` package — writing sweep files, instrumenting training scripts, running sweeps, or understanding configuration options.

## Architecture

mlsweep has three components:

**`mlsweep_manager`** is a persistent daemon you start once. It owns the job queue (SQLite), GPU scheduling, artifact storage, and the web dashboard (port 7891 by default). Workers connect to it over TCP; clients submit jobs to it over HTTP.

**`mlsweep_worker`** is a long-lived daemon that runs on each machine with GPUs. Launched automatically by the manager (locally, or over SSH for remote machines). Executes training jobs, streams logs and metrics back to the manager, rsyncs artifacts on completion.

**`mlsweep_run`** is a thin HTTP client. Loads a sweep file, generates the run combinations, and POSTs them to the manager. It does not launch anything itself.

The manager bootstraps mlsweep on remote workers automatically over SSH (builds wheels locally, SCPs them, installs into `/tmp/mlsweep_venv/`). No manual install is needed on workers.

Token auth: the manager generates a token on first startup and saves it to `~/.mlsweep/manager.token`. `mlsweep_run` reads it automatically for local managers. For remote managers pass `--token` or set `MLSWEEP_TOKEN`.

## What was removed in v1.1 (don't suggest these)

- `mlsweep_viz` — gone. Use the manager web dashboard.
- `-g`/`--gpus` flag on `mlsweep_run` — gone. GPU count is configured in the workers file or defaulted by the manager.
- `--workers` flag on `mlsweep_run` — gone. Workers are configured at manager startup (`mlsweep_manager --workers workers.toml`).
- `WorkerPool` / `mlsweep.pool` — gone. The manager HTTP API is the replacement.

## Workflow

```bash
# 1. Start the manager (once, keep it running)
mlsweep_manager                            # local GPUs
mlsweep_manager --workers workers.toml    # remote workers

# 2. Submit a sweep
mlsweep_run sweeps/my_sweep.py --manager http://localhost:7891
mlsweep_run sweeps/my_sweep.py --manager http://localhost:7891 --stream  # live terminal status

# 3. View results in the browser
# URL is printed at manager startup: http://localhost:7891/?token=...
```

## Sweep file format

Every sweep file defines `COMMAND` and `OPTIONS`. Everything else is optional.

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]       # str or list[str]

OPTIONS = { ... }                      # required — see below

# Optional top-level vars:
GPUS_PER_RUN = 1                       # GPUs per run per node (default: 1)
NODES_PER_RUN = 1                      # nodes per run (default: 1)
SET_DIST_ENV = False                   # auto-set RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
RUN_FROM = "/abs/path/or/rel/subdir"   # working directory for each run (default: git root)
EXTRA_FLAGS = ["--seed", "42"]         # appended to every run before OPTIONS flags

def EXCLUDE(combo: dict) -> bool: ...  # return True to drop a combination
OPTIMIZE = { ... }                     # enables Bayesian optimization (see below)
```

Full command for each run: `COMMAND + EXTRA_FLAGS + OPTIONS flags (declaration order) + -- overrides`

## OPTIONS dictionary

Every key in `OPTIONS` starts with `.` (marks it as a dimension). Keys without `.` inside a dim spec are metadata.

### Dimension types

A **value dim** has a `"values"` list and sweeps over it:
```python
".lr": {
    "values": [1e-4, 3e-4, 1e-3],
    "flags": "--lr",    # str shorthand: generates ["--lr", str(v)] for each value
    "name": "lr",       # used in run name; None to omit; defaults to key without "."
}
```

A **fixed dim** has no `"values"` and no subdim keys. Its flags are always appended and it contributes nothing to the run name:
```python
".precision": {"flags": ["--dtype", "bfloat16"]}
```

A **subdim** has no `"values"` but has dot-prefixed child keys. Each child is a mutually exclusive branch with its own flags and optional further dims:
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
# Produces: 6 Adam runs (3×2) + 3 Muon runs = 9 total
# Adam flags: --optimizer adam --beta1 0.9 --beta2 0.999 (etc.)
# Muon flags: --optimizer muon --lr_scale 1.0 (etc.)
```

Use subdims instead of EXCLUDE when a dimension only applies within certain values of another. EXCLUDE is for cross-cutting constraints.

### `flags` field variants

For value dims, `flags` can be a string shorthand or a per-value dict:

```python
# str shorthand — "flags": "--lr" generates ["--lr", str(v)] for each value
# Python True/False become "True"/"False" (capital — works with hydra; use dict form for lowercase)
".lr": {"values": [1e-3, 3e-4], "flags": "--lr"}

# dict — explicit token list per value; "values" is optional (inferred from dict key order)
".ac": {
    "flags": {
        "none": ["--ac.mode", "none"],
        "op":   ["--ac.mode", "selective", "--ac.selective_option", "op"],
        "full": ["--ac.mode", "full"],
    },
}

# dict with explicit values (e.g. to control monotonic trial order independently of dict key order)
".bs": {
    "values": [8, 16, 32, 64],
    "monotonic": "increasing",
    "flags": {"64": ["--bs", "64"], "32": ["--bs", "32"], "16": ["--bs", "16"], "8": ["--bs", "8"]},
}
```

For subdim branches and fixed dims, `flags` is a string (single token) or list of strings (constant token list).

### Metadata keys

| Key | Type | Description |
|-----|------|-------------|
| `"values"` | `list` | Values to sweep. Optional when `flags` is a dict. |
| `"flags"` | see above | CLI flags to emit. |
| `"name"` | `str` or `None` | Prefix in run name. Defaults to dim key without `.`. `None` omits the dim from the name. |
| `"singular"` | `bool` | Commit to the first value that succeeds; skip the rest. Default `False`. |
| `"monotonic"` | `str` | `"increasing"` or `"decreasing"` — stop trying values after the first failure. |
| `"distribution"` | `str` | Bayes mode only: `"log_uniform"`, `"uniform"`, `"int_uniform"`. |
| `"min"` / `"max"` | `float` | Bayes continuous range. |
| `"samples"` | `int` | Grid mode only: pre-sample N values from a continuous distribution. |

### Run naming

`{sweep_name}_{dim1_name}{val1}_{dim2_name}{val2}…`

Subdim segments are dotted onto their parent: `sweep_optmuon.lrs0.1_bs32`. `"name": None` omits a dim. Boolean values abbreviate to `T`/`F`.

## Skipping: singular and monotonic

Both work only in sequential mode (one job at a time per slot). In parallel mode results are recorded but skipping is not applied dynamically.

**`singular: True`** — good for hardware dims (batch size, activation checkpointing) where you just need one value that fits. The first success locks in the value; all other values for that dim are skipped. Singular dims vary slowest in the cartesian product so other dims are explored first.

```python
".bs": {
    "values": [512, 256, 128, 64, 32],  # largest first; stops at first success
    "flags": "--training.local_batch_size",
    "name": None,
    "singular": True,
}
```

**`monotonic`** — good for finding a boundary before committing to any value. A failure at position `i` stops all further trials from that point onward. `"increasing"` tries values in listed order; `"decreasing"` reverses the list first.

```python
".bs": {
    "values": [8, 16, 32, 64],
    "monotonic": "increasing",   # if 32 fails, 64 is skipped
}
```

Don't combine them — `singular` stops on the first success, `monotonic` stops on the first failure; whichever comes first ends the search.

## Multi-GPU and multi-node

```python
GPUS_PER_RUN = 4     # allocate 4 GPUs per run; manager divides available GPUs into groups
NODES_PER_RUN = 2    # span each run across 2 workers (multi-node)
SET_DIST_ENV = True  # auto-set RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
```

With `GPUS_PER_RUN = 4` and 8 available GPUs: 2 concurrent runs, each on 4 GPUs. Groups are chosen to maximise NVLink connectivity.

mlsweep spawns one process per GPU per node. Each process receives:
- `CUDA_VISIBLE_DEVICES` — the assigned GPU IDs for this run
- `MLSWEEP_GPU_RANK` — 0-based local GPU rank within the run's group
- `MLSWEEP_NODE_RANK`, `MLSWEEP_NNODES`, `MLSWEEP_MASTER_ADDR`, `MLSWEEP_MASTER_PORT` — multi-node only

`SET_DIST_ENV = True` derives standard PyTorch distributed vars from the above automatically. Use this for frameworks that expect `RANK`/`LOCAL_RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT` (TorchTitan, torchrun-launched scripts).

```python
# In train.py — without SET_DIST_ENV
local_rank = int(os.environ["MLSWEEP_GPU_RANK"])
device = torch.device(f"cuda:{local_rank}")

# With SET_DIST_ENV = True — standard vars are set, no wrapper needed
dist.init_process_group(backend="nccl")  # reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
```

## Bayesian optimization

Add `OPTIMIZE` to use TPE (optuna) instead of exhaustive grid search:

```python
OPTIMIZE = {
    "method": "bayes",      # required
    "metric": "val_loss",   # metric name logged via MLSweepLogger
    "goal": "minimize",     # "minimize" or "maximize"
    "budget": 40,           # number of successful runs
    "n_initial": 8,         # runs to queue upfront before adaptive sampling starts (default: max(8, n_params*2))
}
```

Continuous dims (bayes only; use `"samples"` for grid mode):
```python
".lr": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-1, "flags": "--lr", "name": "lr"}
".dropout": {"distribution": "uniform", "min": 0.0, "max": 0.5, "flags": "--dropout", "name": "drop"}
".layers": {"distribution": "int_uniform", "min": 2, "max": 8, "flags": "--layers", "name": "L"}
```

Discrete dims and subdims work unchanged in bayes mode (optuna treats them as categorical). Singular dims are invisible to the optimizer; the full singular probe sequence is generated for each optimizer suggestion.

Bayes runs are named `{sweep_name}_bayes_{N:04d}`. Resume a bayes sweep: `mlsweep_run sweep.py --manager ... --resume EXP_ID`.

## Logger

```python
from mlsweep.logger import MLSweepLogger

with MLSweepLogger() as logger:
    for step in range(1, num_steps + 1):
        logger.log({"loss": 0.42, "lr": 1e-3}, step=step)  # step auto-increments if omitted
        logger.sync()  # fire-and-forget rsync of MLSWEEP_RUN_DIR artifacts mid-training
```

`MLSweepLogger` is only active when `MLSWEEP_WORKER_SOCKET` is set, which the worker sets before launching the script. Run the script directly and it's a no-op; no import guard is needed.

If the script doesn't use the logger at all, mlsweep still dispatches the job and captures stdout/stderr to `training.log`. Metrics plots will be empty.

Save checkpoints to `os.environ["MLSWEEP_RUN_DIR"]`. They're rsynced to the experiment output directory at run end and immediately on `logger.sync()`.

## Environment variables injected into each run

| Variable | When set | Value |
|---|---|---|
| `MLSWEEP_RUN_DIR` | always | Write checkpoints here; rsynced to output dir at run end |
| `MLSWEEP_RUN_NAME` | always | Unique run name, e.g. `sweep_lr0.001_bs32` |
| `MLSWEEP_WORKER_SOCKET` | always | Unix socket for `MLSweepLogger` |
| `CUDA_VISIBLE_DEVICES` | always | Assigned GPU IDs, e.g. `0,1,2,3` |
| `HIP_VISIBLE_DEVICES` | always | Same as `CUDA_VISIBLE_DEVICES` (AMD ROCm) |
| `MLSWEEP_GPU_RANK` | always | 0-based local GPU rank within the run's group |
| `MLSWEEP_NNODES` | multi-node | Total node count (`NODES_PER_RUN`) |
| `MLSWEEP_NODE_RANK` | multi-node | This node's 0-based rank |
| `MLSWEEP_MASTER_ADDR` | multi-node | Rank-0 worker hostname |
| `MLSWEEP_MASTER_PORT` | multi-node | Distributed rendezvous port |
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | `SET_DIST_ENV=True` | Standard PyTorch distributed vars |

Per-run hyperparameter flags are appended as CLI arguments to the training command. They are never set as environment variables.

## Remote workers

```toml
[[workers]]
host = "user@host1"              # SSH target (required)
remote_dir = "/path/to/project"  # project root on the remote (required)
ssh_key = "~/.ssh/id_ed25519"
gpus = 4                         # total GPUs to use (default: all visible)
jobs = 2                         # concurrent jobs per GPU slot (default: 1)
devices = [0, 1, 2, 3]           # specific GPU IDs (alternative to gpus)
venv = "/path/to/venv"           # prefer this venv over the auto-bootstrapped one
```

```bash
mlsweep_manager --workers workers.toml
```

The manager bootstraps mlsweep on remote machines automatically; no manual install is needed. Requires passwordless SSH. Test with `ssh -o BatchMode=yes user@host1 nvidia-smi`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `COMMAND is required` | Add `COMMAND = ["python", "train.py"]` to the sweep file |
| `Dimension key must start with '.'` | All `OPTIONS` keys (and subdim keys) need a `.` prefix |
| `has both 'values' and subdimensions` | A dim can have a value list or subdim branches; it cannot have both |
| `--manager URL is required` | Pass `--manager http://host:7891` — `mlsweep_run` is a submission client and needs a running manager |
| Token errors | Check `~/.mlsweep/manager.token` or pass `--token` / set `MLSWEEP_TOKEN` |
| Singular/monotonic not skipping | Only works with one job at a time per slot; no dynamic skipping in parallel mode |
| Remote not connecting | Test SSH: `ssh -o BatchMode=yes user@host nvidia-smi` |
| `need at least GPUS_PER_RUN GPUs` | Worker has fewer GPUs than `GPUS_PER_RUN`; adjust worker config or reduce `GPUS_PER_RUN` |
| No metrics plots | Script must use `MLSweepLogger`; stdout/stderr are still captured without it |

## Reference

- `docs/sweep_configuration.md` — complete reference: all dim types, flags behavior, CLI options, output layout
- `docs/examples.md` — DDP, multi-node, TorchTitan, Prime-RL patterns
