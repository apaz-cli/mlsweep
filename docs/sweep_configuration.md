# Sweep Configuration Guide

## Overview

Sweep definitions are Python files that define hyperparameter combinations to run. Each sweep file specifies the command to execute and the dimensions to vary across runs.

Pass the sweep file as the first argument:

```bash
mlsweep_run sweeps/my_sweep.py [options]
```

## Sweep File Format

Every sweep file must define `COMMAND` and `OPTIONS`, and may optionally define `EXCLUDE`, `EXTRA_FLAGS`, `GPUS_PER_RUN`, and `RUN_FROM`:

```python
COMMAND = ["python", "train.py"]
OPTIONS = { ... }

# Optional
GPUS_PER_RUN = 1  # (default: 1)
RUN_FROM = "/abs/path/to/dir"  # working directory for each run (default: git root)
EXTRA_FLAGS = ["--seed", "42"]

def EXCLUDE(combo: dict) -> bool: ...
```

### `COMMAND`

The base command used to launch each run. Can be a string (parsed with `shlex.split`) or a list of strings.

```python
# List form:
COMMAND = ["python", "train.py"]
# String form:
COMMAND = "python train.py --config configs/base.yaml"
```

Each run is launched as `COMMAND + per-variation flags + extra overrides`.

### `OPTIONS` Dictionary

`OPTIONS` is a dictionary where each key starts with `.` to mark it as a dimension. Within a dimension spec, keys without `.` are metadata and keys with `.` are dimensions or subdimensions.

Each dimension spec supports these metadata keys:

| Key | Type | Description |
|-----|------|-------------|
| `"values"` | `list` | Explicit list of values to sweep (value dims only). Optional when `flags` is a dict — values are inferred from the dict keys in declaration order. |
| `"flags"` | `str` or `dict` (value dims); `str` or `list[str]` (subdims/fixed dims) | CLI flags to pass. Behavior varies by dim type — see [How Flags Are Produced](#how-flags-are-produced). |
| `"name"` | `str` or `None` | Short identifier used in run names. Defaults to the dim key without the leading dot. `None` suppresses the name from the run name entirely. |
| `"monotonic"` | `str` or `None` | `"increasing"` or `"decreasing"`. Controls skip-on-failure direction (see below). |
| `"singular"` | `bool` | If `True`, skip all other values once one succeeds. Default `False`. |

### Dimension Types

There are three dimension types, determined by the content of the spec:

#### Value Dims (explicit sweep)

Has a `"values"` list. Sweeps over those values with per-value CLI flags:

```python
OPTIONS = {
    ".learning_rate": {
        "values": [1e-4, 3e-4, 1e-3],
        "flags": "--optimizer.lr",
        "name": "lr",
    },
}
```

The `flags` field accepts:
- `str` — shorthand: generates `["--flag", str(v)]` for each value `v`
- `dict` — explicit per-value mapping: `{v: [args...], ...}`
- `None` — no flags added (value affects naming only)

#### Subdims (mutually exclusive cases)

Has no `"values"`, but has dot-prefixed subdim keys. Each subdim is a distinct branch; when that branch is selected, its `"flags"` are applied and its own subdims expand:

```python
OPTIONS = {
    ".optimizer": {
        "name": "opt",
        ".adam": {
            "flags": ["--optimizer", "adam"],
        },
        ".muon": {
            "flags": ["--optimizer", "muon"],
            # .lr_scale only expands within the .muon subdim
            ".lr_scale": {
                "values": [0.1, 0.5, 1.0],
                "flags": "--optimizer.lr_scale",
                "name": "lrs",
            },
        },
    },
}
```

This produces runs named as such, with the following flags.
```sh
sweep_optadam:        --optimizer adam
sweep_optmuon.lrs0.1: --optimizer muon --optimizer.lr_scale 0.1
sweep_optmuon.lrs0.5: --optimizer muon --optimizer.lr_scale 0.5
sweep_optmuon.lrs1.0: --optimizer muon --optimizer.lr_scale 1.0
```

Subdims support arbitrary nesting depth. Subdims cannot have the same names as parent or sibling dims.

#### Fixed Dims (constant flags)

Has neither `"values"` nor subdim keys. The flags are always appended — this dim contributes one combo and nothing to the run name. Useful for grouping constant arguments.

```python
".precision": {
    "flags": ["--dtype", "bfloat16"],
}
```

### How Flags Are Produced

The `flags` field works differently depending on the dim type.

#### Value dims

`flags` maps each value to a list of CLI tokens. Three forms are accepted:

| Form | Example | Expands to |
|------|---------|------------|
| `str` | `"--lr"` | `["--lr", str(v)]` for each value `v` |
| `dict` | `{v: [tokens...], ...}` | the token list you supply per value |
| `None` | `None` | no flags (value affects naming only) |

The `str` shorthand is the common case. The `dict` form is used when different values need different flag structures — a different number of tokens, or flags that aren't simply `--key value`. **Dict values must be lists**; passing a string as a dict value is an error caught at validation time.

When `flags` is a dict, `"values"` is optional — the values are inferred from the dict keys in declaration order. Only supply `"values"` explicitly when you need a specific ordering that differs from the dict key order (e.g. for `monotonic`).

```python
# str shorthand — values required; generates ["--lr", "0.001"] etc.
".lr": {"values": [1e-3, 3e-4, 1e-4], "flags": "--lr"}

# dict without values — values inferred as ["none", "op", "full"] from key order
".ac": {
    "flags": {
        "none": ["--ac.mode", "none"],
        "op":   ["--ac.mode", "selective", "--ac.selective_option", "op"],
        "full": ["--ac.mode", "full"],
    },
}

# dict with explicit values — needed when the dict key order doesn't match
# the desired trial order (here: small first, but dict has large first)
".batch_size": {
    "values": [8, 16, 32, 64],
    "monotonic": "increasing",
    "flags": {
        "64": ["--batch-size", "64"],
        "32": ["--batch-size", "32"],
        "16": ["--batch-size", "16"],
        "8":  ["--batch-size", "8"],
    },
}

# None — vary the name only, no flags passed (values required)
".seed": {"values": [1, 2, 3], "flags": None, "name": "seed"}
```

**Boolean values:** The `str` shorthand converts values with `str(v)`, so Python `True` becomes the token `"True"` and `False` becomes `"False"` (capital T/F). This works for frameworks that accept Python-style booleans (e.g. hydra). If your framework expects lowercase `true`/`false`, a different convention, or a flag with no value argument, use the `dict` form or use string values instead of Python booleans:

```python
# hydra-compatible (default str shorthand works):
".use_amp": {"values": [True, False], "flags": "--training.use_amp"}
# → --training.use_amp True  /  --training.use_amp False

# if your framework wants lowercase or special forms, use dict or string values:
".use_amp": {"values": ["true", "false"], "flags": "--training.use_amp"}
".use_amp": {
    "values": [True, False],
    "flags": {True: ["--use_amp"], False: []},
}
```

#### Flag emission order

Flags from OPTIONS dims are emitted in **declaration order** — the order dims appear in the `OPTIONS` dict. This applies within each level of the tree; parent-level dims always come before child dims. `EXTRA_FLAGS` are prepended (appear before per-dim flags), and `--` overrides from the CLI are appended (appear after).

Full command for each run: `COMMAND + EXTRA_FLAGS + OPTIONS flags (declaration order) + -- overrides`

#### Subdims

Each dimension has its own `flags`, which are a fixed list of tokens applied whenever that branch is selected. The form is `str` (a single token) or `list[str]`:

```python
".optimizer": {
    ".adam": {"flags": ["--optimizer", "adam"]},                      # list form
    ".sgd":  {"flags": "--sgd"},                                      # str form (single token)
    ".muon": {"flags": ["--optimizer", "muon"], ".lr_scale": {...}},  # further subdims
}
```

There is no per-value shorthand for subdim branches — because the "values" are the branch names themselves (derived from the dot-key names), they are usually qualitatively distinct and each branch needs its own explicit flag list.

#### Fixed dims

A fixed dim has no values to vary, so `flags` is a single constant list of tokens always appended to every run. The form is `str` (single token) or `list[str]`:

```python
# Always appends ["--dtype", "bfloat16"] to every run
".precision": {"flags": ["--dtype", "bfloat16"]}
```

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
RUN_FROM = "/abs/path/to/dir"   # absolute
RUN_FROM = "subdir"             # relative — resolved from git root
```

This is useful when the sweep file lives outside the project root, or when the training command uses paths relative to a specific directory. Relative paths are resolved against the git root for local runs and against the worker's `remote_dir` for remote runs.

### `EXTRA_FLAGS`

An optional list of flags appended to **every** run in the sweep, before any per-run overrides generated by OPTIONS. Useful for constants that apply across the whole sweep but aren't part of the default command.

```python
EXTRA_FLAGS = ["--training.steps", "5000", "--checkpoint.enable", "false"]
```

## `EXCLUDE` Predicate (Static Filtering)

A sweep file may optionally define an `EXCLUDE` function to statically filter out combinations before any runs are launched:

```python
def EXCLUDE(combo: dict) -> bool:
    """Return True to exclude this combination from the sweep."""

    return combo["weight_decay"] == 0.0 and combo["wd_compensation"] is True
```

Subdims are usually the best approach, but EXCLUDE is useful when you want to sweep over dims where some values don't make sense together.

Prefer subdims over `EXCLUDE` when an entire dimension only applies to certain values of another dimension. Subdims express this as tree structure rather than a filter on a flat product.

## Monotonic Skipping (Skip on Failure)

When a dimension has `"monotonic"` set and a run fails (returns a nonzero exit code), the sweep skips all subsequent values in that dimension's trial order (all other dims held constant). The skip rule is: **if the value at position `i` fails, skip that value and every value at position `> i`.**

`"increasing"` and `"decreasing"` only control the trial order — `"increasing"` tries values in the order listed; `"decreasing"` reverses the list first. The skip rule is identical either way: a failure stops further trials from that point onward.

Use `"increasing"` when you list values in the order you want to try them:

```python
".batch_size": {
    "values": [8, 16, 32, 64],   # tried in this order
    "monotonic": "increasing",   # if 32 OOMs, skip 64
}
```

Use `"decreasing"` when your values are naturally written largest-first and you want to try them smallest-first (the list is reversed internally):

```python
".batch_size": {
    "values": [64, 32, 16, 8],   # reversed internally to [8, 16, 32, 64]
    "monotonic": "decreasing",   # if 32 OOMs, skip 64; same behavior as above
}
```

Both examples above produce identical behavior. The choice is purely about which order is more natural to write.

**Important:** Both monotonic and singular skipping work **only in sequential mode** (`-g 1 -j 1`). In parallel mode, results are recorded but skipping does not happen dynamically because runs are launched concurrently.

## Singular Skipping (Skip on Success)

When a dimension has `"singular": True` and a run **succeeds**, the sweep will automatically skip **all other values** in that dimension (with all other dimensions held constant).

- **Use case**: Dimensions where you only need ONE working value. Values are tried in declaration order; the first success wins and the rest are skipped.
- **Dimension ordering**: Singular dimensions vary **slowest** in the cartesian product (diagonal order), so other dimensions are fully explored first. This maximizes parallelism when using `-g`/`-j`.
- **Expected run count**: The sweep displays "expected" runs treating singular dimensions as contributing only 1 value.

Use `singular` when you want to find the first working value and stop. Put the value you most want to use first:

```python
".batch_size": {
    "values": [64, 32, 16, 8],  # try 64 first; if it works, skip 32/16/8
    "singular": True,
}
```

This is the right pattern for "run the fastest config that fits in memory": OOMs surface in seconds, so starting aggressive means you find and commit to the best option with minimal wasted time.

### Monotonic and Singular together

These address different situations and are **not meant to be combined**:

- **`monotonic`** is for finding a boundary — e.g. discovering which batch sizes fit in memory before committing to any of them. Tries values in order; stops at the first failure.
- **`singular`** is for committing to the first success — e.g. finding the largest batch size that fits and running only that one. Tries values in order; stops at the first success.

If you combine them, `singular` fires on the first success (the first value tried, if it works) and `monotonic` fires on the first failure — whichever comes first ends the search. In practice, `singular` alone is simpler and sufficient for the "find and commit" use case.

## Run Naming

Each run is given a unique name constructed from its dimension values:

```
{sweep_name}_{dim1_name}{value1}_{dim2_name}{value2}_…
```

The `name` field on a dim spec provides the prefix; the value (or subdim key) is appended. If `name` is omitted, the dim key without the leading dot is used. If `name` is `None`, that dim contributes nothing to the run name.

Boolean values are abbreviated as `T` (True) and `F` (False).

When subdims are in use, the subdim's name is **dotted** onto its parent dim's name segment; peer dims are still separated by `_`:

```
{sweep_name}_opt{optimizer}.lrs{lr_scale}_bs{batch_size}
```

For example:
- Adam run: `my_sweep_optadam_bs32`
- Muon run: `my_sweep_optmuon.lrs0.1_bs32`

The `.` signals "this segment is a subdim of the preceding segment." For three-level nesting: `sweep_optmuon.lrs0.1.subX_bs...`

## Invocation

```bash
mlsweep_run sweeps/beta.py [options] [-- extra_overrides...]
```

Example:

```bash
mlsweep_run sweeps/beta.py -g 4 -j 2 -- --training.steps 1000
```

### Shebang Mode

Make the sweep file executable and run it directly:

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
| `--output_dir <dir>` | Directory where run outputs will be stored. Default: `<git-root>/outputs/sweeps`. |
| `--experiment <name>` | Experiment name. Default: `<sweep_name>_<YYYYMMDD_HHMM>`. |
| `-g [N]`, `--gpus [N]` | GPUs to use (local mode only). `-g` alone = all visible GPUs. `-g N` = N GPUs. Default: one slot (`GPUS_PER_RUN` GPUs). Cannot be combined with `--workers`. |
| `-j N`, `--jobs-per-gpu N` | Concurrent jobs per GPU (local mode only). Default: 1. Cannot be combined with `--workers`; set `jobs` per machine in the workers file instead. |
| `--workers <file>` | Path to a TOML workers file for remote dispatch (see [Remote Execution](#remote-execution)). |
| `--resume` | Skip runs already recorded as completed in `sweep_status.json`. |
| `--dry-run` | Print the commands that would be executed without running them. |
| `--validate` | Validate sweep config, print all combinations, and exit (no jobs launched). |
| `--note <text>` | Human-readable note stored in `sweep_manifest.json`. |
| `--max-retries N` | Max retry attempts for orphaned runs (default: 2). |
| `--scratch-dir <dir>` | Worker scratch directory for run buffers (default: `/tmp/mlsweep`). |
| `--` | Pass extra arguments to every training run (e.g., `-- --training.steps 1000`). |

## GPU Management

The controller respects `CUDA_VISIBLE_DEVICES`. If the environment variable is set, GPU indices are parsed from it (including ranges like `0-3`). Otherwise, visible GPUs are discovered via `nvidia-smi` or `amd-smi`. If no GPUs are found, GPU 0 is used as a fallback.

## Remote Execution

Use `--workers` to dispatch jobs to remote machines via SSH (passwordless key-based auth required):

```bash
mlsweep_run sweeps/beta.py --workers workers.toml
```

The workers file is TOML with one `[[workers]]` entry per machine. `remote_dir` is the repo root on that machine — the directory training commands run from.

```toml
[[workers]]
host = "user@host1"
remote_dir = "/home/user/myproject"

[[workers]]
host = "user@host2"
remote_dir = "/home/user/myproject"
gpus = 4
jobs = 2
ssh_key = "~/.ssh/my_key"
venv = "/home/user/myproject"

[[workers]]
host = "user@host3"
remote_dir = "/home/user/myproject"
gpus = 4
devices = [4, 5, 6, 7]
jobs = 2
pass = "hunter2"
venv = "/home/user/myproject/.venv"
```

| Field | Type | Description |
|-------|------|-------------|
| `host` | string | SSH target (required) |
| `remote_dir` | string | Repo root on the remote machine (required) |
| `gpus` | int | Total GPU count to use (default: all visible) |
| `jobs` | int | Concurrent jobs per GPU slot (default: 1) |
| `devices` | list of ints | Specific GPU device IDs to use |
| `ssh_key` | string | Path to SSH identity file (passed as `-i`) |
| `pass` | string | SSH password. Requires `sshpass` to be installed. Falls back to `MLSWEEP_SSH_PASS` env var if omitted. |
| `venv` | string | Path used to locate the `mlsweep_worker` binary. Defaults to `remote_dir`. Accepts a project root (tries `venv/`, `.venv/`), a venv root, a `bin/` dir, an `activate` script, or a python binary — all resolved to the same `bin/` directory. |

The `-g` and `-j` flags cannot be passed on the command line when `--workers` is specified — use the per-machine entries in the workers file instead.

The controller starts a worker process on each machine via SSH (`python -m mlsweep.worker`). The worker ignores SIGHUP so SSH disconnects do not kill it. If the controller exits cleanly it sends a shutdown signal; if the controller crashes, the worker keeps any in-flight runs running to completion and then exits. Run logs and metrics are streamed back to the controller over a persistent TCP connection and written to disk immediately. Artifacts are rsynced at the end of each run.

## Metrics API

Training scripts log metrics via the `MLSweepLogger` class:

```python
from mlsweep.logger import MLSweepLogger

with MLSweepLogger() as logger:
    for step in range(num_steps):
        # step auto-increments from 1:
        logger.log({"loss": 0.42, "lr": 1e-3, "grad_norm": 0.8})

        # Or supply an explicit step:
        logger.log({"loss": 0.40}, step=step)

        # Trigger an artifact sync mid-training (fire-and-forget):
        logger.sync()
```

The context manager calls `close()` on exit. Without it, call `logger.close()` manually when done.

`MLSweepLogger` is a no-op when `MLSWEEP_WORKER_SOCKET` is not set, so training scripts work unchanged outside of mlsweep.

## Environment Variables

The following environment variables are set for each run:

| Variable | Value |
|----------|-------|
| `MLSWEEP_RUN_DIR` | Path to `artifacts/` inside the run's scratch directory. Write checkpoints here; they are rsynced to the output dir at the end of the run. |
| `MLSWEEP_RUN_NAME` | The run's unique name (e.g. `my_sweep_lr1e-3_bs32`). |
| `EXP_EXPERIMENT` | The experiment name. |
| `EXP_TAGS` | Comma-separated `key=value` pairs for this run's combo (e.g. `lr=0.001,bs=32`). |
| `MLSWEEP_WORKER_SOCKET` | Unix socket path for `MLSweepLogger` communication with the worker daemon. |
| `CUDA_VISIBLE_DEVICES` | Comma-separated GPU indices for this run (e.g. `0` or `0,1,2,3` with `GPUS_PER_RUN=4`). |
| `HIP_VISIBLE_DEVICES` | Same as above (for AMD ROCm compatibility). |

Your training script can use these to write outputs to the right place and log metrics.

## Output Structure

Each run creates a subdirectory:

```
{output_dir}/{experiment_name}/{run_name}/
    training.log       # stdout+stderr of the run
    metrics.jsonl      # metrics logged via MLSweepLogger (one JSON object per step)
```

The experiment directory also contains:

```
{output_dir}/{experiment_name}/
    sweep.log          # plain-text copy of the sweep runner's terminal output
    sweep_manifest.json  # dims, combos, and run list (written before dispatch)
    sweep_status.json    # per-run completion status (used by --resume)
```

## Useful Examples

### Reusable Option Dicts

Hardware-limit dimensions (local batch size, activation checkpointing) are often shared across many sweep files. Define them once and spread them in:

```python
# sweeps/_common.py

LOCAL_BATCH_SIZE = {
    "values": [64, 32, 16, 8, 4, 2, 1],  # largest (fastest) first
    "flags": "--training.local_batch_size",
    "name": None,     # omit from run name — it's a hardware detail
    "singular": True,
}

AC_MODE = {
    "values": ["none", "op", "full"],  # aggressive first: none = fastest, most memory
    "flags": {
        "none": ["--activation_checkpoint.mode", "none"],
        "op":   ["--activation_checkpoint.mode", "selective",
                 "--activation_checkpoint.selective_ac_option", "op"],
        "full": ["--activation_checkpoint.mode", "full"],
    },
    "name": None,
    "singular": True,  # tries none first; stops at first success
}

SPEED_OPTIONS = {
    ".local_batch_size": LOCAL_BATCH_SIZE,
    ".ac_mode": AC_MODE,
}
```

```python
# sweeps/paper_repro.py
#!/usr/bin/env mlsweep_run

from _common import SPEED_OPTIONS

OPTIONS = {
    **SPEED_OPTIONS,          # prepend hardware-limit dims (singular, vary slowest)
    ".treatment": {
        "values": ["bf16", "fp8_rtn", "fp8_sr"],
        "flags": "--training.dtype",
        "name": "tmt",
    },
}
```

Because `SPEED_OPTIONS` dims are `singular`, they vary slowest and resolve within the first few runs. The remaining runs use the locked-in batch size and AC mode.

### Local Batch Size Tuning with Singular

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]

OPTIONS = {
    ".local_batch_size": {
        "values": [512, 256, 128, 64, 32, 16, 8],  # largest first
        "flags": "--training.local_batch_size",
        "name": "lbs",
        "singular": True,  # once one size succeeds, skip the rest
    },
    ".learning_rate": {
        "values": [1e-4, 3e-4, 1e-3],
        "flags": "--optimizer.lr",
        "name": "lr",
    },
}
```

With 7 batch sizes and 3 learning rates this generates 21 total variations, but expects only **3 runs** (one successful batch size per learning rate) in the best case.

### Optimizer × Batch Size Sweep with Subdim

Muon has an additional `lr_scale` subdim that doesn't apply to Adam. Using a subdim gives 8 runs instead of 12 with a flat `EXCLUDE`:

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
- **`Dimension key '...' must start with '.'`** — All keys in `OPTIONS` (and in subdim specs) must begin with `.`. Metadata keys inside a dim spec (`name`, `flags`, `values`, `singular`, `monotonic`) do not.
- **`has both 'values' and subdimensions`** — A dim can be either a value dim (`values` list) or a subdim (dot-prefixed subdim keys), not both.
- **Monotonic/singular skipping not working** — These only work in sequential mode (`-g 1 -j 1`). In parallel mode results are recorded but skipping is not applied dynamically.
- **Remote jobs not connecting** — Ensure passwordless SSH key-based authentication is configured for each worker. Test with `ssh -o BatchMode=yes <worker> nvidia-smi`.
- **`need at least GPUS_PER_RUN GPUs`** — The requested GPU count (`-g N`) is less than `GPUS_PER_RUN`. Either increase `-g` or reduce `GPUS_PER_RUN`.
- **Remote worker skipped with `need N per run`** — A remote worker has fewer GPUs than `GPUS_PER_RUN`. That worker is skipped; other workers with enough GPUs are still used.

## See Also

- `mlsweep/run_sweep.py` — full implementation and inline documentation.
- `tests/test_sweep.py` — minimal example sweep for the bundled test training script.
