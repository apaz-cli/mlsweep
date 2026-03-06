# Sweep Configuration Guide

## Overview

Sweep definitions are Python files that define a grid of hyperparameter combinations to run. Each sweep file specifies the command to execute and the dimensions to vary across runs.

There are two ways to invoke a sweep:

```bash
# Named mode: load from sweeps/<name>.py in the current directory
mlsweep_run --sweep <name> [options]

# Direct mode: pass the sweep file as the first argument
mlsweep_run sweeps/my_sweep.py [options]
```

## Sweep File Format

Every sweep file must define `COMMAND` and `OPTIONS`, and may optionally define `EXCLUDE`, `EXTRA_FLAGS`, `ABBREV`, `GPUS_PER_RUN`, and `RUN_FROM`:

```python
COMMAND = ["python", "train.py"]
OPTIONS = { ... }

# Optional
GPUS_PER_RUN = 1   # GPUs to allocate per run (default: 1)
RUN_FROM = "/abs/path/to/dir"  # working directory for each run (default: git root)
EXTRA_FLAGS = ["--seed", "42"]
ABBREV = {"local_batch_size": "bs"}

def EXCLUDE(combo: dict) -> bool: ...
```

### `COMMAND`

The base command used to launch each run. Can be a string (parsed with `shlex.split`) or a list of strings.

```python
# List form (preferred):
COMMAND = ["python", "train.py"]

# String form (auto-parsed):
COMMAND = "python train.py --config configs/base.yaml"
```

Each run is launched as `COMMAND + per-variation flags + extra overrides`.

### `OPTIONS` Dictionary

`OPTIONS` is a dictionary where **each key starts with `.`** to mark it as a dimension. Within a dimension spec, keys without `.` are metadata; keys with `.` are sub-dimensions.

Each dimension spec supports these metadata keys:

| Key | Type | Description |
|-----|------|-------------|
| `"values"` | `list` | Explicit list of values to sweep (value dims only). |
| `"flags"` | `str`, `list`, or `dict` | CLI flags to pass for each value (see below). |
| `"name"` | `str` or `None` | Short identifier used in run names. Defaults to the dim key without the leading dot. `None` suppresses the name from the run name entirely. |
| `"monotonic"` | `str` or `None` | `"increasing"`, `"decreasing"`, or `None`. Determines skip direction on failure. |
| `"singular"` | `bool` | If `True`, skip all other values once one succeeds. Default `False`. |

### Dimension Types

There are three dimension types, determined by the content of the spec:

#### Value Dims (explicit sweep)

Has a `"values"` list. Sweeps over those values with per-value CLI flags:

```python
".learning_rate": {
    "values": [1e-4, 3e-4, 1e-3],
    "flags": "--optimizer.lr",
    "name": "lr",
}
```

The `flags` field accepts:
- **`str`** — shorthand: generates `["--flag", str(v)]` for each value `v`
- **`dict`** — explicit per-value mapping: `{v: [args...], ...}`
- **`None`** — no flags added (value affects naming only)

#### Branch Dims (mutually exclusive cases)

Has **no** `"values"`, but has dot-prefixed sub-dim keys. Each sub-dim is a distinct branch; when that branch is selected, its `"flags"` are applied and its own sub-dims expand:

```python
".optimizer": {
    "name": "opt",
    ".adam": {
        "flags": ["--optimizer", "adam"],
    },
    ".muon": {
        "flags": ["--optimizer", "muon"],
        # .lr_scale only expands within the muon branch
        ".lr_scale": {
            "values": [0.1, 0.3, 1.0],
            "flags": "--optimizer.lr_scale",
            "name": "lrs",
        },
    },
}
```

This produces:
- `(optimizer=adam)` — 1 combo, no `lr_scale` dimension
- `(optimizer=muon, lr_scale=*)` — 3 combos

Branch dims support arbitrary nesting depth. Sub-dim keys are validated against ancestor and sibling dim names to prevent collisions.

#### Fixed Dims (constant flags)

Has neither `"values"` nor sub-dim keys. The flags are always appended — this dim contributes one combo and nothing to the run name. Useful for grouping constant arguments.

```python
".precision": {
    "flags": ["--dtype", "bfloat16"],
}
```

### Flags Shorthand

For the common pattern of passing a flag name with each value:

```python
# Verbose (dict):
"flags": {v: ["--optimizer.lr", str(v)] for v in [1e-4, 3e-4, 1e-3]}

# Shorthand (string):
"flags": "--optimizer.lr"
```

The shorthand generates `{v: ["--optimizer.lr", str(v)] for v in values}`.

### `GPUS_PER_RUN`

An optional positive integer specifying how many GPUs to allocate per run. Defaults to `1`.

```python
GPUS_PER_RUN = 4
```

When set, the runner divides the available GPUs into non-overlapping groups of `GPUS_PER_RUN` and assigns one group per concurrent run. `CUDA_VISIBLE_DEVICES` is set to all GPUs in the group (e.g. `0,1,2,3`).

The `-g N` flag still controls the **total** number of GPUs to use. With `GPUS_PER_RUN=4` and `-g 8`, you get 2 parallel slots (2 concurrent runs). `-g 0` uses all visible GPUs, divided into as many slots as fit.

GPU groups are chosen to maximise interconnect quality using `nvidia-smi topo -m`. If topology data is unavailable, groups are assigned sequentially.

See [Multi-GPU Runs (torchrun)](#multi-gpu-runs-torchrun) below for a complete example.

### `RUN_FROM`

An optional string specifying the working directory passed to each local run's subprocess. Defaults to the git root of the directory where `mlsweep_run` is invoked.

```python
RUN_FROM = "/home/user/myproject"
```

This is useful when the sweep file lives outside the project root, or when the training command uses paths relative to a specific directory. Ignored for remote runs (which `cd` to `--remote-dir` on the worker instead).

### `EXTRA_FLAGS`

An optional list of flags appended to **every** run in the sweep, before any per-run overrides. Useful for constants that apply across the whole sweep but aren't part of the default command.

```python
EXTRA_FLAGS = ["--training.steps", "5000", "--checkpoint.enable", "false"]
```

### `ABBREV`

An optional dict mapping dimension names (without the leading dot) to short display labels, used in the per-treatment summary when singular dims are present.

```python
ABBREV = {
    "local_batch_size": "bs",
    "activation_checkpointing": "ac",
}
```

Without `ABBREV`, the summary truncates dim names to 4 characters automatically.

## `EXCLUDE` Predicate (Static Filtering)

A sweep file may optionally define an `EXCLUDE` function to statically filter out combinations before any runs are launched:

```python
def EXCLUDE(combo: dict) -> bool:
    """Return True to exclude this combination from the sweep."""
    return combo["weight_decay"] == 0.0 and combo["wd_correction"] is True
```

The function receives a `combo` dict mapping dimension names (without the leading dot) to their values. If it returns `True`, that combination is dropped entirely — it won't appear in `--dry-run` output and won't count toward the run total.

Use `EXCLUDE` for **simple combo exclusions** — cases where one combination of existing dimensions is logically meaningless. For combinations that might **fail at runtime** (e.g., OOM), use monotonic/singular skipping instead.

**Prefer branch dims over `EXCLUDE`** when an entire dimension only applies to certain values of another dimension. Branch dims express this as tree structure rather than a filter on a flat product.

## Monotonic Skipping (Skip on Failure)

When a dimension has `"monotonic"` set and a run **fails**, the sweep will automatically skip future runs where that dimension's value is worse (larger or smaller, depending on direction) while all other dimensions stay the same.

- **`"increasing"`**: failure likelihood increases with larger values. If a run with `steps=1000` fails, any run with `steps=10000` (and identical other settings) will be skipped.
- **`"decreasing"`**: failure likelihood increases with smaller values. If `local_batch_size=32` OOMs, all larger batch sizes (64, 128, ...) will also OOM and can be skipped.

## Singular Skipping (Skip on Success)

When a dimension has `"singular": True` and a run **succeeds**, the sweep will automatically skip **all other values** in that dimension (with all other dimensions held constant).

- **Use case**: Dimensions where you only need ONE working value, not all of them.
- **Example**: Local batch size tuning — if `local_batch_size=64` succeeds, there's no need to try 32, 16, 8.
- **Dimension ordering**: Singular dimensions vary **slowest** in the cartesian product (diagonal order), so other dimensions are fully explored first. This maximizes parallelism when using `-g`.
- **Expected run count**: The sweep displays "expected" runs treating singular dimensions as contributing only 1 value.

### Combining Monotonic and Singular

These can work together:

```python
".local_batch_size": {
    "values": [512, 256, 128, 64, 32, 16, 8],
    "monotonic": "decreasing",  # if 128 OOMs, skip 256, 512
    "singular": True,            # if 64 succeeds, skip 32, 16, 8
}
```

**Important:** Both monotonic and singular skipping work **only in sequential mode** (`-g 1 -j 1`). In parallel mode, results are recorded but skipping does not happen dynamically because runs are launched concurrently.

## Run Naming

Each run is given a unique name constructed from its dimension values:

```
{sweep_name}_{dim1_name}{value1}_{dim2_name}{value2}_…
```

The `name` field on a dim spec provides the prefix; the value (or branch key) is appended. If `name` is omitted, the dim key without the leading dot is used. If `name` is `None`, that dim contributes nothing to the run name.

Boolean values are abbreviated as `T` (True) and `F` (False).

When branch dims are in use, the branch sub-dim's name is **dotted** onto its parent dimension's name segment; peer dimensions are still separated by `_`:

```
{sweep_name}_opt{optimizer}.lrs{lr_scale}_bs{batch_size}
```

For example:
- Adam run: `my_sweep_optadam_bs32`
- Muon run: `my_sweep_optmuon.lrs0.1_bs32`

The `.` signals "this segment is a branch of the preceding segment." For three-level nesting: `sweep_optmuon.lrs0.1.subX_bs...`

## Invocation

### Named Mode

```bash
mlsweep_run --sweep <name> [options] [-- extra_overrides...]
```

Loads `sweeps/<name>.py` from the current directory. Example:

```bash
mlsweep_run --sweep beta -g 4 -j 2 -- --training.steps 1000
```

### Direct / Shebang Mode

Pass the sweep file path as the first argument:

```bash
mlsweep_run sweeps/beta.py -g 4 -j 2
```

Or make the sweep file executable with a shebang and run it directly:

```bash
chmod +x sweeps/beta.py
./sweeps/beta.py -g 4 -j 2
```

The shebang line to use is:

```python
#!/usr/bin/env mlsweep_run
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--sweep <name>` | Name of the sweep (loads `sweeps/<name>.py`). Required in named mode. |
| `--output_dir <dir>` | Directory where run outputs will be stored. Default: `<git-root>/outputs/sweeps`. |
| `--experiment <name>` | Experiment name. Default: `<sweep_name>_<YYYYMMDD_HHMM>`. |
| `-g [N]`, `--gpus [N]` | GPUs to use. `-g` alone = all visible GPUs. `-g N` = N GPUs. Default: 1. |
| `-j N`, `--jobs-per-gpu N` | Concurrent jobs per GPU. Default: 1. Total workers = GPUs × jobs-per-GPU. |
| `--workers <targets>` | SSH targets for remote dispatch. Comma-separated or `@file` (one host per line). |
| `--remote-dir <path>` | Repo path on remote workers. Default: git root of the local working directory. |
| `--exp-server <host:port>` | Experiment tracking server. Auto-detected in remote mode (local machine IP:53800). |
| `--sync-artifacts` | After each remote job, rsync run outputs back to the local machine. |
| `--resume` | Skip runs already recorded as completed in `sweep_status.json`. |
| `--dry-run` | Print the commands that would be executed without running them. |
| `--` | Pass extra arguments to every training run (e.g., `-- --training.steps 1000`). |

## GPU Management

The script respects `CUDA_VISIBLE_DEVICES`. If the environment variable is set, GPU indices are parsed from it (including ranges like `0-3`). Otherwise, visible GPUs are discovered via `nvidia-smi`. If no GPUs are found, GPU 0 is used as a fallback.

GPUs are allocated to jobs in a round-robin fashion; oversubscription with `-j` is intentional and supported.

## Remote Execution

Use `--workers` to dispatch jobs to remote machines via SSH (passwordless key-based auth required):

```bash
mlsweep_run --sweep beta --workers user@host1,user@host2 --remote-dir /home/user/myproject
```

The runner SSH-discovers available GPUs on each worker, then dispatches jobs across all of them. The `--gpus` flag, when combined with `--workers`, acts as a per-worker GPU limit.

A `@file` syntax is supported for large worker lists:

```bash
mlsweep_run --sweep beta --workers @workers.txt --remote-dir /home/user/myproject
```

`workers.txt` should have one SSH target per line; lines starting with `#` are ignored.

An experiment tracking server is started on the local machine and its address is automatically passed to all workers via `EXP_SERVER`. Override with `--exp-server <host:port>`.

## Environment Variables

The following environment variables are set for each run:

| Variable | Value |
|----------|-------|
| `MLSWEEP_RUN_DIR` | Absolute path to the run's output directory. |
| `MLSWEEP_RUN_NAME` | The run's unique name (e.g. `my_sweep_lr1e-3_bs32`). |
| `EXP_EXPERIMENT` | The experiment name. |
| `EXP_TAGS` | Comma-separated `key=value` pairs for this run's combo (e.g. `lr=0.001,bs=32`). |
| `EXP_SERVER` | HTTP URL of the experiment tracking server, if `--exp-server` is active. |
| `CUDA_VISIBLE_DEVICES` | Comma-separated GPU indices for this run (e.g. `0` or `0,1,2,3` with `GPUS_PER_RUN=4`). |
| `HIP_VISIBLE_DEVICES` | Same as above (for AMD ROCm compatibility). |

Your training script can use these to integrate with the experiment tracker or write outputs to the right place.

## Output Structure

Each run creates a subdirectory:

```
{output_dir}/{experiment_name}/{run_name}/
    training.log       # stdout+stderr of the run
    training.1.log     # if re-run without --resume (increments to avoid overwriting)
```

The experiment directory also contains:

```
{output_dir}/{experiment_name}/
    sweep.log          # plain-text copy of the sweep runner's terminal output
    sweep_manifest.json  # axes, combos, and run list (written before dispatch)
    sweep_status.json    # per-run completion status (used by --resume)
```

## Example Sweep Files

### Basic Learning Rate × Batch Size Sweep

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py", "--config", "configs/base.yaml"]

OPTIONS = {
    ".learning_rate": {
        "values": [1e-4, 3e-4, 1e-3],
        "flags": "--optimizer.lr",
        "name": "lr",
    },
    ".batch_size": {
        "values": [32, 64, 128],
        "flags": "--training.batch_size",
        "name": "bs",
    },
}
```

Produces 9 runs: `sweep_lr1e-4_bs32`, `sweep_lr1e-4_bs64`, ..., `sweep_lr1e-3_bs128`.

### Local Batch Size Tuning with Monotonic + Singular

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]

LOCAL_BATCH_SIZES = [512, 256, 128, 64, 32, 16, 8]

OPTIONS = {
    ".local_batch_size": {
        "values": LOCAL_BATCH_SIZES,
        "flags": "--training.local_batch_size",
        "name": "lbs",
        "monotonic": "decreasing",  # if 128 OOMs, skip 256, 512
        "singular": True,            # if 64 succeeds, skip 32, 16, 8
    },
    ".learning_rate": {
        "values": [1e-4, 3e-4, 1e-3],
        "flags": "--optimizer.lr",
        "name": "lr",
    },
}
```

With 7 batch sizes and 3 learning rates this generates 21 total variations, but expects only **3 runs** (one successful batch size per learning rate) in the best case.

### Optimizer × Batch Size Sweep with Branch Dim

Muon has an additional `lr_scale` sub-dimension that doesn't apply to Adam. Using a branch dim gives 8 runs instead of 12 with a flat `EXCLUDE`:

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]

OPTIONS = {
    ".optimizer": {
        "name": "opt",
        ".adam": {
            "flags": ["--optimizer", "adam"],
        },
        ".muon": {
            "flags": ["--optimizer", "muon"],
            ".lr_scale": {
                "values": [0.1, 1.0, 10.0],
                "flags": "--optimizer.lr_scale",
                "name": "lrs",
            },
        },
    },
    ".batch_size": {
        "values": [32, 64],
        "flags": "--training.batch_size",
        "name": "bs",
    },
}
```

Produces:
- `sweep_optadam_bs32`, `sweep_optadam_bs64` (2 runs)
- `sweep_optmuon.lrs0.1_bs32`, `sweep_optmuon.lrs0.1_bs64`, ... (6 runs)

### Multi-GPU Runs (torchrun)

Use `GPUS_PER_RUN` together with `torchrun` (or any other multi-process launcher) in `COMMAND` to run each sweep variation across multiple GPUs:

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["torchrun", "--nproc_per_node", "4", "train.py"]
GPUS_PER_RUN = 4  # allocate 4 GPUs per run

OPTIONS = {
    ".learning_rate": {
        "values": [1e-4, 3e-4, 1e-3],
        "flags": "--optimizer.lr",
        "name": "lr",
    },
}
```

Run with 8 GPUs to get 2 parallel 4-GPU jobs:

```bash
mlsweep_run sweeps/torchrun_sweep.py -g 8
```

The runner sets `CUDA_VISIBLE_DEVICES=0,1,2,3` for the first slot and `CUDA_VISIBLE_DEVICES=4,5,6,7` for the second. GPU groups are chosen to maximise NVLink connectivity.

## Troubleshooting

- **`COMMAND is required`** — Add `COMMAND = ["python", "train.py"]` to the sweep file.
- **`Dimension key '...' must start with '.'`** — All keys in `OPTIONS` (and in branch dim specs) must begin with `.`. Metadata keys inside a dim spec (`name`, `flags`, `values`, `singular`, `monotonic`) do not.
- **`has both 'values' and subdimensions`** — A dim can be either a value dim (`values` list) or a branch dim (dot-prefixed subdim keys), not both.
- **Monotonic/singular skipping not working** — These only work in sequential mode (`-g 1 -j 1`). In parallel mode results are recorded but skipping is not applied dynamically.
- **Remote jobs not connecting** — Ensure passwordless SSH key-based authentication is configured for each worker. Test with `ssh -o BatchMode=yes <worker> nvidia-smi`.
- **`need at least GPUS_PER_RUN GPUs`** — The requested GPU count (`-g N`) is less than `GPUS_PER_RUN`. Either increase `-g` or reduce `GPUS_PER_RUN`.
- **Remote worker skipped with `need N per run`** — A remote worker has fewer GPUs than `GPUS_PER_RUN`. That worker is skipped; other workers with enough GPUs are still used.

## See Also

- `mlsweep/run_sweep.py` — full implementation and inline documentation.
- `tests/test_sweep.py` — minimal example sweep for the bundled test training script.
