# Sweep Configuration Guide

## Overview

Sweep definitions are Python files that define hyperparameter combinations to run. Each sweep file specifies the command to execute and the dimensions to vary across runs.

Pass the sweep file as the first argument:

```bash
mlsweep_run sweeps/my_sweep.py [options]
```

Dimensions in `OPTIONS` multiply, so don't enumerate combinations by hand. A control
arm that ignores a dimension goes in a subdim, with that dimension nested under the arms
that use it, so the control runs once. See [Subdims](#subdims-mutually-exclusive-cases)
and, for pairs that don't fit a tree, [`EXCLUDE`](#exclude-predicate-static-filtering).

## Sweep File Format

Every sweep file must define `COMMAND` and `OPTIONS`, and may optionally define `EXCLUDE`, `EXTRA_FLAGS`, `GPUS_PER_RUN`, `NODES_PER_RUN`, `SET_DIST_ENV`, `RUN_FROM`, `UPLOAD_EXCLUDE`, `SETUP_COMMAND`, `DESCRIPTION`, and the [run controls](#run-controls) `MAX_RUNTIME`, `PRUNE` and `METRIC_CHECK`:

```python
COMMAND = ["python", "train.py"]
OPTIONS = { ... }

# Optional
GPUS_PER_RUN = 1      # (default: 1) GPUs per run on each node
NODES_PER_RUN = 1     # (default: 1) worker nodes allocated per run
SET_DIST_ENV = False  # (default: False) auto-set standard dist env vars
RUN_FROM = "/abs/path/to/dir"  # working directory for each run (default: git root)
UPLOAD_EXCLUDE = ["/runs", "*.ckpt"]  # leave these out of the uploaded project
EXTRA_FLAGS = ["--seed", "42"]
SETUP_COMMAND = "pip install -e ."  # shell command run in the workspace before each run
DESCRIPTION = "Does warmup length matter at 3e-4?"  # what the experiment is for

# Optional ranking defaults (stored on the experiment; see below)
METRIC = "val_loss"   # metric name used to rank runs
GOAL = "minimize"     # "minimize" or "maximize"

# Optional run controls, enforced by the manager (see Run Controls)
MAX_RUNTIME = "2h"    # stop and fail a run that runs longer
PRUNE = {"strategy": "asha", "min_steps": 500}  # stop runs that fall behind
METRIC_CHECK = True   # (default) pause if the first runs never log METRIC

def EXCLUDE(combo: dict) -> bool: ...
```

### `DESCRIPTION`

A string saying what the experiment is for. It is stored on the experiment and
shown in the dashboard (on hover over the name on the Experiments page, and under
the name on Results) and in `mlsweep status`. `mlsweep run --description TEXT`
overrides it. An experiment submitted with neither gets a warning, and so does a
sweep file without one in `mlsweep lint`.

Change or remove it later with `mlsweep rename EXP --description TEXT` (or
`--clear-description`), the "Edit description…" item in an experiment's menu on
the Experiments page, or a click on the description on Results. `--append` keeps
the description the experiment already has. Over HTTP it is
`PUT /api/experiments/EXP/description` with `{"description": TEXT}`, where null
or `""` clears it. Experiment JSON calls the field `note`.

### `METRIC` and `GOAL`

Optional top-level strings that record which metric should rank this sweep's
runs and in which direction. They are stored on the experiment when the sweep is
submitted, and `mlsweep fetch` / `mlsweep best` use them as defaults for
`--metric` / `--goal` (an explicit CLI flag still wins; when neither is set the
fallback is `loss` / `minimize`). This makes the ranking decision part of the
sweep definition for grid and Bayesian sweeps alike:

```python
METRIC = "val_loss"
GOAL = "minimize"   # or "maximize"
```

If `OPTIMIZE` is also present it takes precedence: its `metric`/`goal` are used
(and required, for `method="bayes"`). Top-level `METRIC`/`GOAL` are then only a
fallback for sweeps that do not declare `OPTIMIZE`.

Upper-case names that `mlsweep` does not know are ignored, so a sweep file can
keep its own constants (`BASE_LR = 3e-4`). A name that looks like a misspelled
setting, such as `OPTIMZE`, is an error; see [Checking a sweep file](#checking-a-sweep-file).
Start your own constant with `_` if it happens to look like one.

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
```

The `str` shorthand converts values with `str(v)`, so Python `True` becomes the token `"True"` and `False` becomes `"False"` (capital T/F). This works for frameworks that accept Python-style booleans (e.g. hydra). If your framework expects lowercase `true`/`false`, a different convention, or a flag with no value argument, use the `dict` form or use string values instead of Python booleans:

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

Flags from OPTIONS dims are emitted in declaration order, the order dims appear in the `OPTIONS` dict. This applies within each level of the tree; parent-level dims always come before child dims. `EXTRA_FLAGS` are prepended (appear before per-dim flags), and `--` overrides from the CLI are appended (appear after).

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

An optional positive integer specifying how many GPUs to allocate per run on each node. Defaults to `1`.

```python
GPUS_PER_RUN = 4
```

When set, each run is given `GPUS_PER_RUN` GPUs on one worker.  For each run, mlsweep spawns `GPUS_PER_RUN` copies of `COMMAND` — one per GPU — and sets `CUDA_VISIBLE_DEVICES` to the full group and `MLSWEEP_GPU_RANK` to each process's 0-based local rank.  Your training script uses `MLSWEEP_GPU_RANK` as its device index and distributed rank.

The GPUs available to a worker are set via `devices` in workers.toml, or autodetected when running locally. With `GPUS_PER_RUN=4`, 8 available GPUs and the default `jobs = 1`, two runs fit at once.

The manager picks the group when it dispatches the run, from GPUs that still have room under the worker's `jobs` cap. The placement policy (see [Remote Execution](#remote-execution)) decides between busy and idle GPUs. Among equally loaded GPUs, it prefers the ones with the best interconnect according to `nvidia-smi topo -m`. Without topology data it takes them in device order.

See [Multi-GPU Runs](#multi-gpu-runs) below for a complete example.

### `NODES_PER_RUN`

An optional positive integer specifying how many worker nodes to allocate for each run. Defaults to `1`.

```python
NODES_PER_RUN = 2
```

When `NODES_PER_RUN > 1`, the runner reserves one slot (of `GPUS_PER_RUN` GPUs) from each of N distinct workers and launches the run command on all of them simultaneously. The run is treated as complete only when all N nodes have reported back; success requires all nodes to succeed.

The following environment variables are injected into each node's subprocess:

| Variable | Value |
|----------|-------|
| `MLSWEEP_NNODES` | Total number of nodes for this run (equals `NODES_PER_RUN`) |
| `MLSWEEP_NODE_RANK` | This node's rank (0-based) |
| `MLSWEEP_MASTER_ADDR` | Hostname of the rank-0 worker (SSH host with `user@` prefix stripped) |
| `MLSWEEP_MASTER_PORT` | Port for the distributed rendezvous (cycles through 29500–29599) |

Your training command can use these to build the `torchrun` invocation directly:

```python
NODES_PER_RUN = 2
GPUS_PER_RUN = 8

COMMAND = [
    "bash", "-c",
    "torchrun "
    "--nproc_per_node $MLSWEEP_NNODES "  # intentional: env var expanded by shell
    "--nnodes $MLSWEEP_NNODES "
    "--node_rank $MLSWEEP_NODE_RANK "
    "--master_addr $MLSWEEP_MASTER_ADDR "
    "--master_port $MLSWEEP_MASTER_PORT "
    "-m torchtitan.train \"$@\"",
    "--",  # $0 for bash -c
]
```

Or more readably:

```python
NODES_PER_RUN = 2
GPUS_PER_RUN = 8
COMMAND = ["bash", "-c", """
    torchrun \
      --nproc_per_node $GPUS_PER_NODE \
      --nnodes $MLSWEEP_NNODES \
      --node_rank $MLSWEEP_NODE_RANK \
      --master_addr $MLSWEEP_MASTER_ADDR \
      --master_port $MLSWEEP_MASTER_PORT \
      -m torchtitan.train "$@"
""", "--"]
```

`NODES_PER_RUN > 1` requires at least `NODES_PER_RUN` connected workers. Logs and metrics are streamed from all nodes, but only artifacts from rank 0's scratch directory are rsynced to the output directory. If your training script writes artifacts on all ranks, use a shared filesystem or handle aggregation in the script itself.

### `SET_DIST_ENV`

An optional boolean (default `False`). When `True`, mlsweep automatically sets the standard distributed training environment variables for each spawned process, derived from the `MLSWEEP_*` vars:

| Variable | Single-node value | Multi-node value |
|----------|------------------|-----------------|
| `RANK` | `MLSWEEP_GPU_RANK` (== global rank for single-node) | `MLSWEEP_NODE_RANK * GPUS_PER_RUN + MLSWEEP_GPU_RANK` |
| `LOCAL_RANK` | `MLSWEEP_GPU_RANK` | `MLSWEEP_GPU_RANK` |
| `WORLD_SIZE` | `GPUS_PER_RUN` | `MLSWEEP_NNODES * GPUS_PER_RUN` |
| `MASTER_ADDR` | `localhost` | `MLSWEEP_MASTER_ADDR` |
| `MASTER_PORT` | hash of run name (range 20000–29999) | `MLSWEEP_MASTER_PORT` |

The hash-derived `MASTER_PORT` for single-node runs ensures that concurrent runs on the same machine each get a distinct port automatically.

`SET_DIST_ENV = True` is the easiest way to use frameworks that expect standard PyTorch distributed env vars (e.g. TorchTitan, torchrun-launched scripts) without writing a `bash -c` wrapper in `COMMAND`.

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "-m", "torchtitan.train", "--module", "llama3", "--config", "llama3_8b"]
GPUS_PER_RUN = 8
SET_DIST_ENV = True

OPTIONS = {
    ".lr": {
        "values": [3e-4, 8e-4, 2e-3],
        "flags": "--optimizer.lr",
        "name": "lr",
    },
}
```

### `UPLOAD_EXCLUDE`

Patterns for files and directories to leave out of the project upload, on top of the
built-in list (VCS dirs, caches, `outputs`, venvs and so on). A pattern without `/`
matches a file or directory name at any depth (`*.ckpt`, `wandb`). A pattern with `/`
matches the path from the project root (`/runs` is only the top-level `runs`,
`data/*.bin` only those files). `mlsweep run --exclude PATTERN` adds more for one
submission; the run prints what it leaves out.

The manager stores each file's content once and takes only what it doesn't already have,
up to 512 MB of new files per submission. See "Code snapshots" in `mlsweep docs mlsweep`.

```python
UPLOAD_EXCLUDE = ["/runs", "/swarm_run", "*.ckpt"]
```

### `RUN_FROM`

An optional string specifying the working directory passed to each local run's subprocess. Defaults to the git root of the directory where `mlsweep_run` is invoked.

```python
RUN_FROM = "/abs/path/to/dir"   # absolute
RUN_FROM = "subdir"             # relative — resolved from git root
```

This is useful when the sweep file lives outside the project root, or when the training command uses paths relative to a specific directory. Relative paths are resolved against the copy of your project that `mlsweep_run` uploads to each worker. If a job has no uploaded project, they are resolved against the worker's `remote_dir` instead. Absolute paths are used as-is.

### `EXTRA_FLAGS`

An optional list of flags appended to every run in the sweep, before any per-run overrides generated by OPTIONS. Useful for constants that apply across the whole sweep but aren't part of the default command.

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

The manager applies both rules whenever a run finishes. Jobs still pending that a rule makes unnecessary are marked `xfailed` and never run. Jobs already running are left to finish, so the fewer that run at once (`--max-concurrent`, or fewer GPUs), the more skipping saves. Skipping applies to grid sweeps. See [Singular dims in bayes mode](#singular-dims-in-bayes-mode) for Bayesian sweeps.

## Singular Skipping (Skip on Success)

When a dimension has `"singular": True` and a run succeeds, the sweep will automatically skip all other values in that dimension (with all other dimensions held constant). Values are tried in declaration order; the first success locks in the value and the rest are skipped. Singular dimensions vary slowest in the cartesian product so other dimensions are fully explored first. The sweep displays "expected" runs treating singular dimensions as contributing only 1 value.

Use `singular` when you want to find the first working value and stop. Put the value you most want to use first:

```python
".batch_size": {
    "values": [64, 32, 16, 8],  # try 64 first; if it works, skip 32/16/8
    "singular": True,
}
```

This is the right pattern for "run the fastest config that fits in memory": OOMs surface in seconds, so starting aggressive means you find and commit to the best option with minimal wasted time.

### Monotonic and Singular together

These address different situations and are not meant to be combined:

`monotonic` is for finding a boundary — e.g. discovering which batch sizes fit in memory before committing to any of them. It stops at the first failure. `singular` is for committing to the first success — e.g. finding the largest batch size that fits and running only that one. It stops at the first success.

If you combine them, `singular` fires on the first success (the first value tried, if it works) and `monotonic` fires on the first failure — whichever comes first ends the search. In practice, `singular` alone is simpler and sufficient for the "find and commit" use case.

## Run Controls

Three settings make the manager act on runs while a sweep runs. They are stored
on the experiment, so they keep working after `mlsweep run` exits or the manager
restarts, and they apply to grid and Bayesian sweeps alike. Each run the manager
stops or flags gets a `reason`, which `mlsweep ls EXP` prints under the run and
the dashboard shows when you hover over its status.

To stop a run, the worker sends it SIGTERM, and SIGKILL 30 seconds later if it is
still going. A script that saves a checkpoint on SIGTERM keeps it: the run's
artifacts, log and metrics are synced to the manager as for a run that exited by
itself, and then its job is marked.

`mlsweep controls EXP` shows them, and changes them while the sweep runs:

```bash
mlsweep controls EXP --max-runtime 4h        # or off; applies to runs already running
mlsweep controls EXP --prune '{"strategy": "asha", "min_steps": 500}'   # or off
mlsweep controls EXP --metric-check 5        # or on, off
```

Over HTTP these are `GET` and `PATCH /api/experiments/{id}/controls`.

### `MAX_RUNTIME`

```python
MAX_RUNTIME = "2h"      # or "90m", "1h30m", "1d", or a number of seconds
```

A run whose process has run longer than this is killed and marked `failed`, with
the reason `timed out: ran longer than MAX_RUNTIME (2h)`. The clock starts when
the process starts, so setup and waiting in the queue don't count. A timed-out
run is a failure like any other: it counts toward monotonic skipping and a
Bayesian sweep's failure streak, and `mlsweep retry` reruns it.

### `PRUNE`

Early stopping: a run that is clearly behind the others is stopped so its GPU goes
to the next run.

```python
PRUNE = {
    "strategy": "median",   # "median" (default) or "asha"
    "min_steps": 500,       # the first step runs are compared at (required)
    "every": 500,           # median: compare again every N steps (default: min_steps)
    "reduction_factor": 3,  # asha: compare at min_steps * 3**k, keep the best 1/3 (default: 3)
    "n_warmup": 5,          # how many other runs must reach a step before anything is pruned there
    "metric": "val_loss",   # default: METRIC (or OPTIMIZE's metric)
    "goal": "minimize",     # default: GOAL
}
```

Runs are compared at fixed steps, called rungs. A run's value at a rung is the best
value of the metric it logged up to the first step at or past the rung, so the
metric only needs to be logged now and then (an eval every 500 steps is fine). Once
`n_warmup` other runs have a value at a rung, a run reaching it is stopped:

- with `median`, if its value is worse than the median of theirs;
- with `asha`, if it is not among the best `1/reduction_factor` of all the runs at that rung, itself included.

A stopped run is marked `pruned`, with a reason such as `pruned at step 1000: best
val_loss 0.92 is worse than the median 0.71 of 6 other runs there`. Its metrics
stay, so its curve still shows in Results. A pruned run did not fail. It
counts as finished for completion and `mlsweep wait`, as a success for skip rules,
and toward a Bayesian sweep's `budget`. Optuna is told the trial was pruned and how
far it got.

The first `n_warmup` runs to reach a rung are never pruned there, since there is
little to compare them with. The rung values are rebuilt from the stored metrics
after a manager restart.

### `METRIC_CHECK`

If the sweep declares a metric (`METRIC` or `OPTIMIZE`'s), the manager checks that
runs actually log it. A run that finishes without logging it, or with every value
NaN, gets a reason naming the keys it did log: `missing metric 'val_loss': the run
logged train_loss, val/loss; did you mean 'val/loss'?`. When the first two finished
runs both lack it, the manager pauses the sweep and keeps its queued runs, so a
misspelled metric doesn't use up the cluster. `mlsweep wait` exits `1` and prints
the reason. Fix `METRIC` or the training script, or `mlsweep unpause EXP` to carry
on as it is. The check fires once per sweep.

```python
METRIC_CHECK = True    # (default) judge by the first 2 finished runs
METRIC_CHECK = 5       # judge by the first 5
METRIC_CHECK = False   # don't check
```

## Bayesian Mode

By default, mlsweep runs an exhaustive grid search over all combinations. Add an `OPTIMIZE` dict to your sweep file to use Bayesian optimization (TPE via optuna) instead, which intelligently samples the hyperparameter space to find good configs faster and supports continuous parameter ranges.

```python
OPTIMIZE = {
    "method": "bayes",       # "grid" (default) or "bayes"
    "metric": "val_loss",    # metric name logged via MLSweepLogger
    "goal": "minimize",      # "minimize" or "maximize"
    "budget": 50,            # how many trials need results
    "n_initial": 5,          # random trials before TPE takes over (default: max(8, n_params*2))
}
```

### `OPTIMIZE` fields

| Key | Required | Description |
|-----|----------|-------------|
| `method` | no | `"grid"` (default) or `"bayes"` |
| `metric` | yes (bayes) | Metric name to optimize (must be logged via `MLSweepLogger`) |
| `goal` | yes (bayes) | `"minimize"` or `"maximize"` |
| `budget` | yes (bayes) | How many trials need results. A trial is one combo of the non-singular dims. Failed trials don't count. |
| `n_initial` | no | Number of random trials before TPE starts using results. Default: `max(8, n_params * 2)`. These count toward `budget`. Also how many trials run at once, unless `--max-concurrent` is set. |
| `max_failures` | no | Failed trials in a row before the manager pauses the sweep. Default: `max(3, n_initial)`. |

### How a Bayesian sweep runs

The manager runs the sweep, so `mlsweep run` submits it and exits. The sweep keeps going if your terminal closes or the manager restarts. `--stream` only watches.

The sweep's options and job settings are stored with the experiment. Each time a result comes in, the manager chooses new trials from all the results so far and queues their runs. It keeps up to `--max-concurrent` trials going, or `n_initial` if that isn't set. Fewer at once means each trial learns from more results. So `--max-concurrent 1` is fully sequential, and a high value is closer to random search.

The sweep completes when `budget` trials have results and no runs are left, or when a small discrete search space has no untried combos left.

A trial fails if its runs all fail, or if they finish without logging `metric`. Its value is the best value of `metric` the run logged. After `max_failures` failed trials in a row, the manager pauses the sweep and cancels its queued runs, so a broken script or a misspelled metric can't use the cluster forever. `mlsweep optimize EXP` shows the reason. Fix what you can and `mlsweep unpause EXP` to try again. To run changed code, submit a new sweep.

OPTIONS values must be plain JSON (strings, numbers, booleans, `None`, lists and dicts), since the manager stores them. `EXCLUDE` is a Python function, so it does not apply to Bayesian sweeps. `--append` is refused for them.

### Controlling a Bayesian sweep

| Do this | With | What happens |
|---------|------|--------------|
| See where it stands | `mlsweep optimize EXP` | Budget used, trials running and failed, the best run. `--json` for scripts. |
| Change the budget | `mlsweep optimize EXP --budget N` or `+N` | Takes effect on the next trial. A completed sweep with budget left runs again. Lowering it leaves existing runs alone. |
| End it gently | `mlsweep optimize EXP --finish` | No new trials. Runs already queued or running end, then the sweep completes. `--budget` undoes this. |
| Change how many run at once | `mlsweep optimize EXP --max-concurrent K` | Raising it starts more trials now. Lowering it lets running runs finish. |
| Hold it | `mlsweep pause EXP`, `mlsweep unpause EXP` | While paused, no new runs start and no new trials are chosen. Running runs finish. |
| Drop one trial | `mlsweep cancel EXP RUN` | The run stops, it doesn't use budget, and the manager chooses another trial. |
| End it now | `mlsweep stop EXP --yes` | Every unfinished run is cancelled. `mlsweep resume EXP` starts it choosing trials again. |

`mlsweep retry EXP RUN` works as usual. The rerun's result replaces the old one when it finishes. Purging or moving a run takes its trial out of the count, so a completed sweep needs `--budget` to replace it.

`mlsweep wait` and `mlsweep fetch --wait` wait until the manager completes, pauses or stops the sweep. The same controls are available over HTTP as `GET` and `PATCH /api/experiments/{id}/optimize`.

### Continuous dimensions

In bayes mode, dimensions can be sampled from a continuous range instead of a fixed list:

```python
OPTIONS = {
    ".lr": {
        "distribution": "log_uniform",  # "uniform" | "log_uniform" | "int_uniform"
        "min": 1e-5,
        "max": 1e-1,
        "flags": "--optimizer.lr",
        "name": "lr",
    },
    ".dropout": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.5,
        "flags": "--model.dropout",
        "name": "drop",
    },
    ".num_layers": {
        "distribution": "int_uniform",
        "min": 2,
        "max": 8,
        "flags": "--model.num_layers",
        "name": "layers",
    },
}
```

| Distribution | Optuna call | Typical use |
|---|---|---|
| `"log_uniform"` | `suggest_float(log=True)` | Learning rate, weight decay |
| `"uniform"` | `suggest_float(log=False)` | Dropout, clip norm |
| `"int_uniform"` | `suggest_int()` | Integer params |

To use a continuous dim in grid mode, add `"samples": N` to pre-sample N values at load time. The dimension then behaves like a discrete value dim. The `"samples"` key is forbidden in bayes mode.

```python
# Grid mode — pre-samples 8 random LR values at load time:
".lr": {
    "distribution": "log_uniform",
    "min": 1e-5,
    "max": 1e-1,
    "samples": 8,
    "flags": "--optimizer.lr",
    "name": "lr",
}
```

### Discrete and subdim dimensions in bayes mode

Existing discrete dims and subdims work unchanged in bayes mode — optuna uses categorical sampling for them:

```python
OPTIONS = {
    ".lr": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-1, "flags": "--lr", "name": "lr"},
    ".optimizer": {
        "name": "opt",
        ".adam": {"flags": ["--optimizer", "adam"]},
        ".muon": {
            "flags": ["--optimizer", "muon"],
            ".lr_scale": {"values": [0.1, 1.0, 10.0], "flags": "--lr_scale", "name": "lrs"},
        },
    },
}
```

Subdims become conditional optuna parameters — when `muon` is selected, `lr_scale` is sampled too.

### Singular dims in bayes mode

Singular dims (hardware constraints like batch size) must list their `values`. The optimizer sees only the other dims. Each trial queues one run per singular value, all at once, and they may run together. A trial's result comes from the first run to log the metric, and the runs that failed are marked `xfailed`. Unlike in a grid sweep, the other probes of a trial are not skipped once one succeeds, so a trial can use up to one run per singular value.

```python
OPTIMIZE = {"method": "bayes", "metric": "val_loss", "goal": "minimize", "budget": 30}

OPTIONS = {
    ".batch_size": {
        "values": [64, 32, 16, 8],
        "flags": "--training.batch_size",
        "name": "bs",
        "singular": True,  # finds largest that fits; invisible to optimizer
    },
    ".lr": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-1, "flags": "--lr", "name": "lr"},
}
```

### Run naming in bayes mode

Bayes runs are named `{sweep_name}_bayes_{N:04d}`, numbered in order across all the experiment's runs, singular probes included. Each run's combo is stored with its job, and `mlsweep ls EXP` shows it.

### Full example

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]

OPTIMIZE = {
    "method": "bayes",
    "metric": "val_loss",
    "goal": "minimize",
    "budget": 40,
    "n_initial": 8,
}

OPTIONS = {
    # Singular dim: finds largest batch size that fits (invisible to optimizer)
    ".batch_size": {
        "values": [128, 64, 32, 16],
        "flags": "--training.batch_size",
        "name": None,
        "singular": True,
    },
    # Continuous dims: optimizer samples from these ranges
    ".lr": {
        "distribution": "log_uniform",
        "min": 1e-5,
        "max": 1e-1,
        "flags": "--optimizer.lr",
        "name": "lr",
    },
    ".wd": {
        "distribution": "log_uniform",
        "min": 1e-6,
        "max": 1e-2,
        "flags": "--optimizer.weight_decay",
        "name": "wd",
    },
    # Discrete dim: categorical sampling
    ".warmup_steps": {
        "values": [100, 500, 1000, 2000],
        "flags": "--scheduler.warmup_steps",
        "name": "wu",
    },
}
```

Run with:
```bash
mlsweep_run sweeps/bayes_sweep.py
```

Workers own the GPUs that jobs run on. Set each machine's GPUs with `devices`
and per-GPU packing with `jobs` in `workers.toml`, or live from the System page.
See [Remote Execution](#remote-execution).

## Run Naming

Each run is given a unique name constructed from its dimension values:

```
{sweep_name}_{dim1_name}{value1}_{dim2_name}{value2}_…
```

The `name` field on a dim spec provides the prefix; the value (or subdim key) is appended. If `name` is omitted, the dim key without the leading dot is used. If `name` is `None`, that dim contributes nothing to the run name.

Boolean values are abbreviated as `T` (True) and `F` (False).

When subdims are in use, the subdim's name is dotted onto its parent dim's name segment; peer dims are still separated by `_`:

```
{sweep_name}_opt{optimizer}.lrs{lr_scale}_bs{batch_size}
```

For example:
- Adam run: `my_sweep_optadam_bs32`
- Muon run: `my_sweep_optmuon.lrs0.1_bs32`

The `.` signals "this segment is a subdim of the preceding segment." For three-level nesting: `sweep_optmuon.lrs0.1.subX_bs...`

## Invocation

`mlsweep_run` submits a sweep to a running `mlsweep_manager` daemon. Start the manager first, then submit:

```bash
mlsweep_manager                                             # start manager (local GPUs)
mlsweep_manager --workers workers.toml                      # start manager (remote workers)
mlsweep_run sweeps/beta.py --manager http://localhost:7891 [options] [-- extra_overrides...]
```

Example:

```bash
mlsweep_run sweeps/beta.py --manager http://localhost:7891 -- --training.steps 1000
```

### Shebang Mode

Make the sweep file executable and run it directly:

```bash
chmod +x sweeps/beta.py
./sweeps/beta.py --manager http://localhost:7891
```

The shebang line to use is:

```python
#!/usr/bin/env mlsweep_run
```

### Subcommands

`mlsweep_run` also supports subcommands for interacting with submitted experiments:

```bash
mlsweep_run fetch --manager http://host:port --experiment EXP_ID            # download + summarize + rank results
mlsweep_run fetch --manager http://host:port --experiment EXP_ID --status done  # filter by status
mlsweep_run fetch --manager http://host:port --experiment EXP_ID --wait     # block until settled, then rank
mlsweep_run watch EXP_ID --manager http://host:port                         # stream live status to terminal
mlsweep_run watch EXP_ID --manager http://host:port --events                # one JSON event per line
```

`fetch --status` accepts: `done`, `failed`, `pending`, `running`, `dispatched`.
`fetch` and `best` rank runs by `--metric` and `--goal`; when those flags are
omitted they default to the experiment's stored `METRIC`/`GOAL` (or `OPTIMIZE`),
falling back to `loss` / `minimize`. Use `mlsweep best --experiment EXP_ID` for
just the leaderboard, or add `--json` to either for machine-readable output.

### Waiting for a sweep

`mlsweep wait EXP` blocks until an experiment reaches a requested state. It is
built for scripts and agents: the result is reported through distinct exit codes
rather than text that must be parsed.

```bash
mlsweep wait EXP --until done            # no pending/dispatched/running jobs
mlsweep wait EXP --until any-failure     # return as soon as any run failed
mlsweep wait EXP --until any-done        # return as soon as any run finished, however it ended
mlsweep wait EXP --until stalled         # return when a running run stops progressing
mlsweep wait EXP --until started         # return once every run has logged a step (a smoke test)
mlsweep wait EXP --until done --timeout 3600 --interval 15
mlsweep wait EXP --until stalled --stalled-after 900
```

| Exit code | Meaning |
|-----------|---------|
| `0` | Every run finished successfully. |
| `1` | At least one run finished as `failed`, or the manager paused the sweep for a [run control](#metric_check). |
| `2` | `--timeout` elapsed with work outstanding (`0` = wait forever). Waiting again is safe. |
| `3` | `--until stalled` and a running run made no log/metric progress for `--stalled-after` seconds (default `900`). A script with long quiet stretches calls `logger.heartbeat()` in them. |
| `4` | No answer: the manager is unreachable, the ID is unknown, or the experiment has no runs. |
| `5` | No failures, but some runs were cancelled or the sweep was stopped, so the results are partial. |

`--until started` exits `0` once every run has logged a metric step or finished, and `1`
as soon as one fails. It waits for pending runs too, so it says a smoke test's runs all got
going. A script that never uses the logger never counts as started; use `--until
any-failure` with a timeout for those.

`--until stalled` also exits when the sweep settles, so it covers `--until done`. Neither notices runs
stuck in `pending`. Pruned runs count as finished; the summary line says how many there were. A running Bayesian sweep is not settled, even with no active runs, because the manager is about to add more.

`--until any-done` reports each run as it finishes. It counts runs that finished
after `--since T` (epoch seconds or an ISO time, default now) and exits `0` naming
them. With `--json` the result has a `cursor`. Pass that as the next call's `--since`
so a run that finishes between two calls is reported once:

```bash
cursor=$(date +%s)
while :; do
  out=$(mlsweep wait EXP --until any-done --since "$cursor" --json)
  cursor=$(jq .cursor <<<"$out")
  jq -r '.runs[] | "\(.run_id) \(.status)"' <<<"$out"
  [ "$(jq -r .outcome <<<"$out")" = run_finished ] || break   # settled, failed, ...
done
```

`wait --json` prints one verdict: `outcome` (`settled`, `failure`, `cancelled`,
`stopped`, `paused`, `stalled`, `run_finished`, `timeout` or `unknown`), `exit_code`,
the `runs` it is about with their status and reason, `counts` by status, and `cursor`.

#### Several experiments at once

`wait` takes more than one experiment ID, or none. With none it watches every
experiment in the campaign that has active runs, and picks up experiments
submitted while it waits, so one loop covers everything an agent has queued:

```bash
mlsweep wait EXP_A EXP_B --until any-done --since "$cursor" --json
mlsweep wait --until any-done --since "$cursor" --json   # the whole campaign
```

The answer is the most urgent one any of them gives. An unknown ID (`4`) comes
first, then a sweep the manager paused (`1`), a failure under `--until any-failure`
(`1`), a stalled run (`3`), and runs that finished (`0`). Finished runs from every
experiment are reported together, so one cursor covers them all. Otherwise
`wait` returns once all of them have settled, with the worst of their results:
failure, then cancelled, then settled. Without IDs, `--until any-failure` counts
only runs that failed after `--since`, so an old failure in a still-running
experiment doesn't end it at once. With nothing active in the campaign it exits
`0` straight away.

The `--json` result has `experiment_ids`, `campaign` when no IDs were given,
`runs` with each run's `experiment_id`, `counts` over all of them, `experiments`
with each one's counts, and `cursor`.

`--interval` controls the poll period (default `10` seconds). `mlsweep fetch
--wait` and `mlsweep best --wait` also exit non-zero when the experiment settles
with failures.

### Comparing a sweep file with an experiment

`mlsweep run SWEEP.py --compare EXP` submits nothing. It matches the file's
runs with EXP's by their full command line, so a renamed run still matches, and prints how
many are already in EXP, how many `--append` would add, and how many runs EXP has that the
file doesn't make. A new run whose name EXP already uses for another command is listed too,
since `--append` would refuse it. Use it to check that a rewritten sweep file still
describes an experiment, or that several files appended to one experiment add up to it.
`--json` gives the lists (`same`, `new`, `clash`, `only_in_experiment`).

```bash
mlsweep run sweeps/scaling.py --compare scaling_20261126_0217_24d4
```

### Checking a sweep file

`mlsweep lint SWEEP.py ...` reports mistakes that would otherwise pass silently.
`mlsweep run` runs the same checks first, refuses a file with errors (even with
`--validate` or `--dry-run`), and prints the warnings.

With no files, `mlsweep lint` checks everything. That means every sweep file in
the project (any `.py` file whose first line is an `mlsweep_run` shebang, skipping
hidden directories and virtualenvs) and every experiment on the manager in all
campaigns. For an experiment, the one check so far is a missing description. A
manager it can't reach is a warning, and the files are still checked. With
`--json`, experiments appear as `experiment:CAMPAIGN/ID` next to the file paths,
and `manager` holds the warning when the manager was not checked.

Errors:

- an upper-case name close to a setting, such as `OPTIMZE` or `GPUS_PER_RUNN`
- an unknown key in `OPTIMIZE` or `PRUNE`, with the likely one
- `OPTIMIZE` with a `budget` but no `method="bayes"`, which would run as a grid
- a dim whose `values` repeat, run names that collide, or no runs at all

Warnings:

- a flag passed twice in one run, by two dims or by a dim and `EXTRA_FLAGS`
- `flags` given for values that are not in the dim's `values`
- `METRIC` without `GOAL`, or `GOAL` without `METRIC`
- a grid of more than 2000 runs
- no `DESCRIPTION` (`mlsweep run` leaves this one out, and warns after submitting
  if the experiment ended up with no description)

`--strict` also exits `1` on warnings, and `--json` prints the findings for scripts.

## Campaigns

A campaign is a named group of experiments, the way an experiment is a group of
runs. Every experiment belongs to exactly one campaign, and there is no default
one. You name it when you submit.

Each `mlsweep` command works inside one campaign, chosen in this order:

1. `--campaign NAME`
2. the campaign of the active context (see Contexts below)
3. the `MLSWEEP_CAMPAIGN` environment variable

A command that needs a campaign and finds none exits 1 and says how to pick one.
`--all-campaigns` (also spelled `--all_campaigns`) lifts the restriction.

```bash
export MLSWEEP_CAMPAIGN=tokenizer_ablation
mlsweep run sweeps/vocab.py --manager http://localhost:7891   # submitted under tokenizer_ablation
mlsweep ls                                                    # tokenizer_ablation's experiments only
mlsweep ls --all-campaigns                                    # every experiment, tagged with its campaign
mlsweep best --experiment EXP_ID --campaign baseline          # reach into another campaign
```

| Command | What the campaign does |
|---------|------------------------|
| `run` | Creates the experiment in the campaign. A new sweep can't use `--all-campaigns`. `--resume` needs the experiment to be in the campaign. |
| `ls`, `status` | List only the campaign's experiments. `ls --all-campaigns` tags each experiment with its campaign. |
| `watch`, `fetch`, `best`, `wait`, `logs`, `metrics`, `cancel`, `retry`, `resume`, `stop`, `pause`, `unpause`, `campaign move` | Refuse an experiment filed elsewhere. They exit 1, change nothing, and name the experiment's campaign. |

A refusal looks like this:

```
Error: experiment vocab_20260930_1412_3f2a is in campaign 'tokenizer_ablation', not 'baseline'.
Pass --campaign tokenizer_ablation or --all-campaigns.
```

Campaign names follow the experiment ID rules: 1 to 128 letters, digits, `_`
or `-`. A campaign exists while it holds an experiment, so there is nothing to
create first.

```bash
mlsweep campaign                       # campaigns with experiment and job counts; * marks the current one
mlsweep campaign move EXP_ID NEW_NAME  # move an experiment, with all its runs, to another campaign
```

Campaigns don't change scheduling or storage. Runs from every campaign share one
queue and the same GPUs, and results stay under
`~/.mlsweep/experiments/<experiment_id>/`.

### Campaigns in the dashboard

The Experiments page has a campaign selector at the top. The Results, Logs and
Artifacts pages have one in the sidebar, above the experiment selector. Each
page lists only the selected campaign's experiments, or every experiment under
"All campaigns". The choice is kept in the page URL (`?campaign=NAME`), carried
by the navigation links, and remembered by the browser. A first visit shows
`default`. A link to one experiment (`?experiment=ID`) switches to that
experiment's campaign. Each row on the Experiments page moves with `Manage` → Move to
campaign.

### Campaigns in the HTTP API

Every endpoint accepts `?campaign=NAME`:

- `GET /api/experiments`, `GET /api/jobs` and `GET /api/jobs/pending` return
  only that campaign's experiments and jobs.
- `POST /api/experiments` creates the experiment in that campaign, as does a
  `campaign` field in the body. Re-creating an existing experiment under another
  campaign returns 409.
- A request that names an experiment from another campaign, in the path, in
  `?experiment_id=`, or in a job body, gets 404 with
  `{"error": ..., "campaign": "<its campaign>"}` and changes nothing. The
  WebSocket stream follows the same rule.

Without the parameter nothing is restricted. `GET /api/campaigns` lists
campaigns with their counts, and `PUT /api/experiments/{id}/campaign` with
`{"campaign": NAME}` moves an experiment.

## Command-Line Options

### `mlsweep`

The unified entry point (`mlsweep --help` prints the workflow, grouped into Run /
Monitor / Control / Campaigns / Docs). Individual binaries remain as aliases.

| Subcommand | Description |
|-----------|-------------|
| `manager` | Start the manager daemon (alias of `mlsweep_manager`). |
| `run` | Submit a sweep (alias of `mlsweep_run`). |
| `worker` | Start a worker daemon (alias of `mlsweep_worker`). |
| `lint [SWEEP.py...]` | Check sweep files, or with none every sweep file and experiment (see [Checking a sweep file](#checking-a-sweep-file)). `--strict`, `--json`. |
| `gen_makefile` | Write a standard `Makefile` (adds `make sweep-*` targets). |
| `watch <exp_id>` | Live terminal status for an experiment (`--events` / `--json` stream one JSON event per line). |
| `fetch` | Download + summarize + rank an experiment's results (leaderboard). |
| `best` | Show the top runs of an experiment by metric. |
| `wait <exp_id>` | Block until an experiment settles, fails, or stalls. Exit codes: `0` clean, `1` failure, `2` timeout, `3` stalled. |
| `status` (`doctor`) | Diagnose manager / token / GPUs / results / disk. |
| `ls [exp_id]` | List experiments, or the runs within one experiment. |
| `logs <exp_id> <run_id>` | Print (and optionally follow) a run's training log. `logs <run_id> --experiment <exp_id>` also works. |
| `metrics` | Print runs' logged metrics as tables, fetched on demand. `--keys REGEX` selects keys; `--pivot` turns keys into rows using the regex's capture group (e.g. `val/nll@r(\d+)` becomes value vs. r, one column per run); `--step`, `--tail` (table view only), `--json`, `--csv`. `--json` always has the shape `{"experiment_id", "runs": [{"run_id", "dims", ...}]}`, where each run carries `rows` (every step), `last` (with `--last`) or `pivot` (with `--pivot`). |
| `cancel <exp_id>` | Cancel runs (`--active` for every unfinished one, `--running` (dispatched ones too) / `--pending` / run IDs / `--all`). |
| `retry <exp_id>` | Re-queue finished runs (`--failed` / `--cancelled` / run IDs) with the code submitted with the experiment. Each gets a fresh `--max-retries` budget. |
| `resume <exp_id>` | Re-queue failed and cancelled runs, with the same code. A Bayesian sweep instead goes back to running, and the manager chooses new trials. It does not undo `pause`. |
| `controls <exp_id>` | Show or change an experiment's [run controls](#run-controls) (`--max-runtime`, `--prune`, `--metric-check`, `--json`). |
| `optimize <exp_id>` | Show or adjust a Bayesian sweep (`--budget N`, `--budget +N`, `--finish`, `--max-concurrent K`, `--max-failures N`, `--json`). See [Controlling a Bayesian sweep](#controlling-a-bayesian-sweep). |
| `stop <exp_id>` | Abort a sweep (requires `--yes`). Every unfinished run is cancelled. |
| `pause <exp_id>` | Stop dispatching new jobs. |
| `unpause <exp_id>` | Resume dispatching. |
| `campaign [ls]` | List campaigns with experiment and job counts (`--json`). |
| `campaign move <exp_id> <name>` | Move an experiment to another campaign. |
| `context [ls\|show\|add\|use\|rm]` | Named client contexts (see [Contexts](#contexts)). |
| `completion bash\|zsh\|fish` | Print a shell completion script. |
| `docs [topic]` | Print the bundled runbook. |
| `version` | Print the mlsweep version. |

Every command defaults `--manager` to `http://localhost:7891` (override with
`--manager` or `MLSWEEP_MANAGER`). `fetch` and `best` rank by `--metric` / `--goal`, defaulting
to the experiment's stored `METRIC`/`GOAL` (or `OPTIMIZE`) and finally
`loss`/`minimize`. `fetch` also takes `--wait` and `--top`.
For a running run, `ls EXP` shows its current step and the average time per step
since the manager started tracking it, e.g. `step 1200 at 2.5 s/step on average`.
`ls EXP --json` has the same as `last_step` and `avg_step_seconds`.

### `--json`

Every command that returns data or changes something takes `--json`, and they all
keep the same contract, so scripts and agents can rely on it:

- stdout holds one JSON value, the result. `watch --events` (alias `--json`) is the
  exception: it streams one JSON object per line.
- Everything meant for people goes to stderr: progress, OK/FAIL lines, warnings.
- A failure prints `{"error": "...", "exit_code": N}` and exits with that `N`.
  Bad arguments, an unreachable manager and an unknown experiment are failures like
  any other. A failure may add fields, e.g. `run --json` with lint errors adds `findings`.
- A command whose result explains a nonzero exit prints that result and no error
  object: `wait` (the verdict) and `lint` (the findings).

Commands that change things report what they changed. `cancel`, `retry` and
`resume` print `{"experiment_id", "action", "dry_run", "runs": [{"run_id", "ok"}]}`.
`stop`, `pause` and `unpause` print the experiment's new status. `run --json`
prints the submitted experiment (`experiment_id`, `campaign`, `runs`, `snapshot`),
or with `--validate` and `--dry-run` the runs it would make.

`cancel`, `retry`, and `resume` exit non-zero if any targeted run fails, so
scripts can detect partial failures.

Every subcommand that talks to the manager takes `--campaign NAME` and
`--all-campaigns` (see [Campaigns](#campaigns)), and `--context NAME`.

### Contexts

A context names a manager, where its token is, and a campaign, so switching
between a local manager and a lab cluster is one flag:

```bash
mlsweep context add lab --manager http://lab-box:7891 --token-file ~/.mlsweep/lab.token --campaign lr-study
mlsweep context add local --manager http://localhost:7891
mlsweep context use lab          # the current context, for every command
mlsweep ls --context local       # or one command at a time (or MLSWEEP_CONTEXT=local)
mlsweep context                  # list them; * marks the current one
mlsweep context show             # the one in force, and whether its token is readable
```

Contexts are kept in `~/.config/mlsweep/config.toml` (`$XDG_CONFIG_HOME` is honored,
and `$MLSWEEP_CONFIG` names another file). A token is never written there.
`--token-file` names a file holding it and `--token-env` an environment variable.

Where a setting comes from, first match wins:

1. `--manager`, `--token`, `--campaign` on the command line.
2. A context named by `--context NAME` or `$MLSWEEP_CONTEXT`.
3. `$MLSWEEP_MANAGER`, `$MLSWEEP_TOKEN`, `$MLSWEEP_CAMPAIGN`.
4. The current context, unless `$MLSWEEP_MANAGER` is set, since its token and campaign belong to its own manager.
5. Otherwise the manager is `http://localhost:7891` and the token is in `$MLSWEEP_DIR/manager.token`. No campaign is filled in here.

What a named context leaves out falls through to the steps below it, never to the
current context's values.

### Shell completion

```bash
eval "$(mlsweep completion bash)"   # in ~/.bashrc
eval "$(mlsweep completion zsh)"    # in ~/.zshrc, after compinit
mlsweep completion fish > ~/.config/fish/completions/mlsweep.fish
```

It completes subcommands, each subcommand's flags, sweep files, context names, and
experiment IDs and campaigns from the manager in force. A manager that is down
only means no IDs are offered.

### `mlsweep_manager`

| Option | Description |
|--------|-------------|
| `--port N` | HTTP/WebSocket port (default: 7891). |
| `--host <name>` | Externally-reachable hostname used to construct artifact download URLs (default: `localhost`). Set this to your server's hostname or IP when clients access the manager from other machines. Does not affect the bind address (always `0.0.0.0`). |
| `--mlsweep-dir <dir>` | State directory for the DB, token file, PID file, and experiment outputs (env: `MLSWEEP_DIR`, default: `~/.mlsweep`). |
| `--workers <file>` | Path to a TOML workers file. If omitted, a local worker using all visible GPUs is started. |
| `--token <tok>` | Auth token. Auto-generated and saved to `<mlsweep-dir>/manager.token` if not provided. |
| `--db <path>` | SQLite database path (env: `MLSWEEP_DB_PATH`, default: `<mlsweep-dir>/manager.db`). |
| `--scratch-dir <dir>` | Where workers keep each run's working files while it runs, unless a workers file entry sets `scratch_dir` (env: `MLSWEEP_SCRATCH_DIR`, default: `/tmp/mlsweep`). |

### `mlsweep_run`

| Option | Description |
|--------|-------------|
| `--manager <url>` | Manager URL (default: `$MLSWEEP_MANAGER`, then `http://localhost:7891`). |
| `--token <tok>` | Auth token. Resolved in order: `--token`, `MLSWEEP_TOKEN`, then `~/.mlsweep/manager.token`. |
| `--output-dir <dir>` | Directory for local submit-side artifacts (submit log + manifest). Default: `~/.mlsweep/submissions`. |
| `--experiment <name>` | Experiment name. Default: `<sweep_name>_<YYYYMMDD_HHMM>_<rand>`. A name already in use is refused unless you pass `--append`. |
| `--append` | Add this sweep's runs to the existing experiment named by `--experiment`. Run names must not clash with its runs. |
| `--campaign <name>` | Campaign to submit under (env: `MLSWEEP_CAMPAIGN`, or the context's campaign). Required. |
| `--all-campaigns` | Refused for a new sweep. |
| `--priority N` | Job priority — higher values run sooner (default: 0). |
| `--stream` | Watch live status in the terminal over the manager's WebSocket event stream. The sweep doesn't depend on it. |
| `--fetch` | Fetch and display results after submission (when `--stream` is not used). |
| `--dry-run` | Print the commands that would be executed without running them. |
| `--validate` | Validate sweep config, print all combinations, and exit (no jobs submitted). |
| `--compare EXP` | Like `--validate`, but compare the runs with EXP's (see [Comparing a sweep file with an experiment](#comparing-a-sweep-file-with-an-experiment)). |
| `--description <text>` | What the experiment is for. Overrides the sweep file's [`DESCRIPTION`](#description). `--note` is the older name and still works. |
| `--max-retries N` | Max automatic retries of a run lost with its worker (default: 2). A run that exits with an error is not retried automatically. `mlsweep retry` is not counted and resets the count. |
| `--setup-command <cmd>` | Shell command executed before training in the worker workspace. |
| `--wandb-project P` | W&B project name (enables W&B logging). |
| `--wandb-entity E` | W&B entity/team. |
| `--tensorboard-dir D` | TensorBoard output directory (enables TensorBoard logging). |
| `--` | Pass extra arguments to every training run (e.g., `-- --training.steps 1000`). |

## GPU Management

The controller respects `CUDA_VISIBLE_DEVICES`. If the environment variable is set, GPU indices are parsed from it (including ranges like `0-3`). Otherwise, visible GPUs are discovered via `nvidia-smi` or `amd-smi`. If no GPUs are found, GPU 0 is used as a fallback.

## Remote Execution

Remote workers are configured when the manager starts. Pass a `--workers` file to `mlsweep_manager`:

```bash
mlsweep_manager --workers workers.toml
```

Then submit sweeps normally:

```bash
mlsweep_run sweeps/beta.py --manager http://localhost:7891
```

The manager bootstraps mlsweep on remote machines automatically over SSH — no manual install required. It builds wheels from the local source at startup, SCPs them to each remote, and installs them into `/tmp/mlsweep_venv/`.

The workers file is TOML with one `[[workers]]` entry per machine. `remote_dir` is the repo root on that machine — the directory training commands run from.

```toml
[[workers]]
host = "user@host1"
remote_dir = "/home/user/myproject"

[[workers]]
host = "user@host2"
remote_dir = "/home/user/myproject"
devices = [0, 1, 2, 3]
jobs = 2
ssh_key = "~/.ssh/my_key"
venv = "/home/user/myproject"

[[workers]]
host = "user@host3"
remote_dir = "/home/user/myproject"
devices = [4, 5, 6, 7]
jobs = 2
pass = "hunter2"
venv = "/home/user/myproject/.venv"
```

| Field | Type | Description |
|-------|------|-------------|
| `host` | string | SSH target (required) |
| `remote_dir` | string | Repo root on the remote machine (required) |
| `jobs` | int | Worker-level cap: max concurrent jobs per GPU on this machine (worker CLI: `-j`). Default 1; set to 0 for unlimited. |
| `devices` | list of ints | Specific GPU device IDs to use (worker CLI: `-g`). Default: all visible. |
| `ssh_key` | string | Path to SSH identity file (passed as `-i`) |
| `pass` | string | SSH password. Requires `sshpass` to be installed. Falls back to `MLSWEEP_SSH_PASS` env var if omitted. |
| `venv` | string | Existing venv to prefer over the auto-bootstrapped one. Accepts a project root (tries `venv/`, `.venv/`), a venv root, a `bin/` dir, an `activate` script, or a python binary. If omitted, the manager bootstraps mlsweep automatically. |
| `port` | int | TCP port the worker listens on (default: 7890; `0` = ephemeral). |
| `scratch_dir` | string | Where the worker keeps each run's working files while it runs (default: the manager's `--scratch-dir`, `/tmp/mlsweep`). A second worker given a directory that a live worker already uses takes a `.worker-<port>` subdirectory of it. |

The manager places jobs in one of two ways, chosen with the Placement toggle on the web UI's System page (or `PATCH /api/scheduler` with `{"placement": "spread"}`) and saved across manager restarts. Pack is the default. It puts each job on the busiest GPU and worker that still have room under `jobs`. With `jobs = 2`, GPU 0 takes two jobs before GPU 1 gets one, which keeps idle GPUs and whole machines free for multi-GPU and multi-node jobs.

Spread puts each job on the least busy GPU, so every GPU gets one job before any gets a second, and work is shared evenly across workers. The cost is that a later job needing a whole machine may have to wait for one to drain.

The manager starts a worker process on each machine via SSH (`python -m mlsweep.worker`). The worker ignores SIGHUP so SSH disconnects do not kill it. If the manager exits cleanly it sends a shutdown signal; if the manager crashes, the worker keeps any in-flight runs running to completion and then exits. Run logs and metrics are streamed back to the manager over a persistent TCP connection and written to disk immediately. Artifacts are rsynced at the end of each run.

### Run environment

Your project's code is uploaded with each sweep, but its Python environment is not: venvs and compiled packages don't carry across machines. The upload leaves out VCS directories, caches, `node_modules`, `outputs`, `.venv`/`venv`/`.env`, `*.egg-info` and `.mlsweep`. If the project root holds a virtualenv, a top-level `build/` is left out too, because it is setuptools output. Without one, `build/` is shipped in case it is source.

Each run gets an environment from the first of:

1. `.venv/` in the run's working directory, then in the worker's `remote_dir`.
2. A default env the worker builds from the project's dependency files, the same ones `pip install .` reads:
   - `pyproject.toml` with a `[project]` table
   - `requirements.txt`
   - `setup.py` / `setup.cfg`

When the dependencies are a plain list (static `[project].dependencies`, or a `requirements.txt` without `-e`, `-r` or local paths), the env holds only those dependencies and is cached per machine at `~/.cache/mlsweep/envs/<hash>/` (or `$MLSWEEP_ENV_CACHE`), keyed on the list, the Python version and the platform. The first run on a machine builds it; concurrent runs wait for that one build, and later runs reuse it until the dependencies change. Your project's own code always comes from the uploaded workspace, which is on `PYTHONPATH`. Anything else (`setup.py`, dynamic dependencies) is `pip install`ed per run into the run's scratch directory.

The build output goes to the run's `training.log`. The env uses the worker's Python (`/tmp/mlsweep_venv`'s interpreter on bootstrapped workers). To turn this off, set `MLSWEEP_AUTO_ENV=0` in the worker's environment; it is also off when the worker itself was started inside another venv or conda env, whose interpreter runs then inherit.

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

A list is logged as one metric per element, with its index after a slash, so
per-layer values reach the manager: `logger.log({"moe_load": [0.2, 0.5]})` logs
`moe_load/0` and `moe_load/1`. A nested dict is spread the same way
(`{"val": {"nll": x}}` logs `val/nll`), and numpy or torch values become plain
numbers. Each resulting key may be logged once per step.

The context manager calls `close()` on exit. Without it, call `logger.close()` manually when done.

`MLSweepLogger` is a no-op when `MLSWEEP_WORKER_SOCKET` is not set, so training scripts work unchanged outside of mlsweep.

## Environment Variables

The worker inherits the shell environment and adds the following variables before launching each `COMMAND` process:

| Variable | Always set? | Value |
|----------|-------------|-------|
| `MLSWEEP_RUN_DIR` | yes | Path to `artifacts/` inside the run's scratch directory. Write checkpoints here; they are rsynced to the output dir at the end of the run. |
| `MLSWEEP_RUN_NAME` | yes | The run's unique name (e.g. `my_sweep_lr1e-3_bs32`). |
| `EXP_EXPERIMENT` | yes | The experiment name. |
| `EXP_TAGS` | when non-empty | Comma-separated `key=value` pairs for this run's combo (e.g. `lr=0.001,bs=32`). |
| `MLSWEEP_WORKER_SOCKET` | yes | Unix socket path for `MLSweepLogger` communication with the worker daemon. |
| `CUDA_VISIBLE_DEVICES` | yes | Comma-separated GPU indices assigned to this run (e.g. `0,1,2,3` with `GPUS_PER_RUN=4`). Same for every process in the group. |
| `HIP_VISIBLE_DEVICES` | yes | Same as `CUDA_VISIBLE_DEVICES` (for AMD ROCm compatibility). |
| `MLSWEEP_GPU_RANK` | yes | This process's 0-based local GPU rank within the group (0 to `GPUS_PER_RUN-1`). Use this as the device index and local distributed rank in your training script. `MLSweepLogger` logs only from rank 0 by default. |
| `MLSWEEP_WORKSPACE` | when `MsgRun.files` is non-empty | Path to the per-run isolated workspace. Set when a file payload was sent with the run (programmatic API only); the command's cwd is also changed to this directory. |
| `MLSWEEP_NNODES` | multi-node only | Total node count for this run (`NODES_PER_RUN`). |
| `MLSWEEP_NODE_RANK` | multi-node only | This node's rank (0-based). |
| `MLSWEEP_MASTER_ADDR` | multi-node only | Rank-0 worker hostname. |
| `MLSWEEP_MASTER_PORT` | multi-node only | Port for the distributed rendezvous (cycles through 29500–29599). |
| `RANK` | when `SET_DIST_ENV=True` | Global distributed rank for this process. |
| `LOCAL_RANK` | when `SET_DIST_ENV=True` | Local rank (same as `MLSWEEP_GPU_RANK`). |
| `WORLD_SIZE` | when `SET_DIST_ENV=True` | Total number of processes across all nodes. |
| `MASTER_ADDR` | when `SET_DIST_ENV=True` | Rendezvous address (`localhost` for single-node). |
| `MASTER_PORT` | when `SET_DIST_ENV=True` | Rendezvous port (hash-derived for single-node; `MLSWEEP_MASTER_PORT` for multi-node). |

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
    sweep_manifest.json  # run list with dimension values (written before dispatch)
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

### Sweep with Subdims

Adam has two subdims (`beta1`, `beta2`); Muon has one (`lr_scale`). The branches don't need to be symmetric. Using subdims gives 24 runs instead of 108 with a flat `EXCLUDE`:

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]

OPTIONS = {
    ".optimizer": {
        "name": "opt",
        ".adam": {
            "flags": ["--optimizer", "adam"],
            ".beta1": {
                "values": [0.85, 0.9, 0.95],
                "flags": "--optimizer.beta1",
                "name": "b1",
            },
            ".beta2": {
                "values": [0.9, 0.95, 0.999],
                "flags": "--optimizer.beta2",
                "name": "b2",
            },
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
- `sweep_optadam.b10.85.b20.9_bs32`, `sweep_optadam.b10.85.b20.9_bs64`, ... (18 runs)
- `sweep_optmuon.lrs0.1_bs32`, `sweep_optmuon.lrs0.1_bs64`, ... (6 runs)

### Multi-GPU Runs

Set `GPUS_PER_RUN` to the number of GPUs each run needs.  mlsweep spawns one copy of `COMMAND` per GPU, passing each process its rank via `MLSWEEP_GPU_RANK` (0-based):

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]
GPUS_PER_RUN = 4

OPTIONS = {
    ".learning_rate": {
        "values": [1e-4, 3e-4, 1e-3],
        "flags": "--optimizer.lr",
        "name": "lr",
    },
}
```

In `train.py`, use `MLSWEEP_GPU_RANK` as the local rank:

```python
import os
import torch

local_rank = int(os.environ["MLSWEEP_GPU_RANK"])
device = torch.device(f"cuda:{local_rank}")
# pass local_rank as rank when initialising your process group
```

With `gpus_per_run = 4` in the sweep file, a worker exposing 8 GPUs runs two of
these jobs in parallel (8 / 4):

```bash
mlsweep_run sweeps/my_sweep.py
```

The worker sets `CUDA_VISIBLE_DEVICES=0,1,2,3` for the first job and `CUDA_VISIBLE_DEVICES=4,5,6,7` for the second; GPU groups are chosen to maximise NVLink connectivity.  `MLSweepLogger` is a no-op on all ranks except `MLSWEEP_GPU_RANK=0`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `COMMAND is required` | Add `COMMAND = ["python", "train.py"]` to the sweep file. |
| `Dimension key '...' must start with '.'` | All keys in `OPTIONS` (and subdim specs) must begin with `.`. Metadata keys (`name`, `flags`, `values`, `singular`, `monotonic`) do not. |
| `has both 'values' and subdimensions` | A dim can have a value list or subdim branches; it cannot have both. |
| Monotonic/singular not skipping | Only jobs still pending when a result arrives are skipped; jobs already running finish. Lower `--max-concurrent` to run fewer at once. |
| Remote jobs not connecting | Ensure passwordless SSH is configured. Test with `ssh -o BatchMode=yes <worker> nvidia-smi`. |
| `need at least GPUS_PER_RUN GPUs` | A worker has fewer GPUs than `GPUS_PER_RUN`. Increase the worker's GPU count or reduce `GPUS_PER_RUN`. |
| Remote worker skipped with `need N per run` | That worker has fewer GPUs than `GPUS_PER_RUN` and is skipped; other workers with enough GPUs are still used. |

## See Also

- `mlsweep/run_sweep.py` — full implementation and inline documentation.
- `tests/test_sweep.py` — minimal example sweep for the bundled test training script.
