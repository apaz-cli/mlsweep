# mlsweep guide

This guide is for using the `mlsweep` package. It covers writing sweep files, instrumenting training scripts, running sweeps, and understanding configuration options.

## Architecture

mlsweep has three components:

**`mlsweep_manager`** is a persistent daemon you start once. It owns the job queue (SQLite), GPU scheduling, artifact storage, and the web dashboard (port 7891 by default). Workers connect to it over TCP; clients submit jobs to it over HTTP.

**`mlsweep_worker`** is a long-lived daemon that runs on each machine with GPUs. Launched automatically by the manager (locally, or over SSH for remote machines). Executes training jobs, streams logs and metrics back to the manager, rsyncs artifacts on completion.

**`mlsweep_run`** is a thin HTTP client. Loads a sweep file, generates the run combinations, and POSTs them to the manager. It does not launch anything itself.

All three are reachable through one command: `mlsweep manager` / `mlsweep run` / `mlsweep worker` (the individual binaries are aliases). `mlsweep` also adds `watch`, `fetch`, `best`, `status`/`doctor`, `ls`, `logs`, `docs`, `gen_makefile`, `campaign`, the control verbs `cancel` / `retry` / `resume` / `stop` / `pause` / `unpause`, `rename` for run display names, and `code` for the code an experiment ran.

The manager bootstraps mlsweep on remote workers automatically over SSH (builds wheels locally, SCPs them, installs into `/tmp/mlsweep_venv/`). No manual install is needed on workers.

That venv is reused only when its version and a fingerprint of its worker code both match the manager's, so a source checkout that changed worker code without a version bump gets reinstalled on the next launch. A worker that is already running keeps its code until it is relaunched. The System page flags a connected worker whose code differs, and a worker on a fixed port that the manager refused for speaking an older protocol. For those it offers an Update worker button, which shuts the worker down and relaunches it (`POST /api/workers/{id}/update`).

Token auth: the manager generates a token on first startup and saves it to `~/.mlsweep/manager.token`. All client commands (`mlsweep run`, `watch`, `fetch`, `status`) resolve it the same way, in order: `--token`, `MLSWEEP_TOKEN`, then `~/.mlsweep/manager.token`.

## What was removed in v1.1 (don't suggest these)

- `mlsweep_viz` is gone. Use the manager web dashboard.
- `-g`/`-j` on `mlsweep_run` are gone from the submitter. They were always per-machine settings, so they now live on the worker, as `mlsweep_worker -g <device-ids> -j <jobs-per-gpu>`, or `devices`/`jobs` in `workers.toml`. The manager passes the `workers.toml` values down to each worker.
- The `--workers` flag on `mlsweep_run` is gone. Workers are configured at manager startup (`mlsweep_manager --workers workers.toml`).
- `WorkerPool` / `mlsweep.pool` is gone. The manager HTTP API is the replacement.

## Workflow

```bash
# 0. (optional) add the standard project Makefile
mlsweep gen_makefile                       # adds make sweep-run / sweep-watch / sweep-fetch / sweep-status

# 1. Start the manager (once, keep it running)
mlsweep manager                            # local GPUs
mlsweep manager --workers workers.toml    # remote workers

# 2. Submit a sweep
mlsweep run sweeps/my_sweep.py --manager http://localhost:7891 --stream
mlsweep run sweeps/my_sweep.py --validate   # or: print all combos without submitting

# 3. Monitor / rank / fetch
mlsweep status                             # manager / token / GPUs / result paths
mlsweep watch EXP_ID                       # live terminal status
mlsweep ls                                 # experiments grouped by status (`mlsweep ls EXP_ID` lists runs, --all lifts the cap of 20 settled)
mlsweep ls --finished-since 2h             # runs that finished lately, across the campaign (failures first)
mlsweep logs EXP_ID RUN_ID                # tail a run's training.log
mlsweep metrics --experiment EXP_ID --keys 'val/nll@r(\d+)' --pivot   # reshape keys (--last, --with-dims, 2-group --pivot, --json, --csv)
mlsweep best --experiment EXP_ID           # top runs by metric (--group-by DIM, --table D1,D2)
mlsweep fetch --experiment EXP_ID --wait   # block until done, then leaderboard + download
                                           # (on the manager's machine: prints its results folder, no copy)
mlsweep sql "SELECT ..."                   # read-only SQL over the manager's database (no query: list tables)

# 4. Control a running sweep
mlsweep cancel EXP_ID --running            # cancel jobs (--pending / --all --yes)
mlsweep retry  EXP_ID --failed             # re-queue failed jobs, with the code they were submitted with
mlsweep resume EXP_ID                      # re-queue failed and cancelled jobs (Bayesian: choose new trials)
mlsweep optimize EXP_ID                    # show or adjust a Bayesian sweep (--budget N|+N, --finish, --max-concurrent K)
mlsweep pause EXP_ID                       # stop dispatching new jobs
mlsweep unpause EXP_ID                     # undo pause
mlsweep stop EXP_ID --yes                  # abort a sweep
mlsweep rename EXP_ID RUN_ID NAME          # display name for a run (--clear removes it)

# 5. View results in the browser
# URL is printed at manager startup: http://localhost:7891/?token=...

# 6. Group experiments into campaigns
export MLSWEEP_CAMPAIGN=my_campaign        # or pass --campaign my_campaign to any command
mlsweep campaign                           # list campaigns
mlsweep campaign move EXP_ID other         # re-file an experiment
mlsweep ls --all-campaigns                 # look across every campaign
```

## Campaigns

A campaign groups experiments the way an experiment groups runs. Every command
works in one campaign, taken from `--campaign NAME`, else the context's campaign,
else `MLSWEEP_CAMPAIGN`. There is no default, so a command that needs a campaign
exits 1 without one. `--all-campaigns` (or `--all_campaigns`) covers every campaign.

- `mlsweep run` submits the new experiment under the current campaign. It
  refuses `--all-campaigns`, because a new experiment needs one campaign.
- `ls` and `status` list only the current campaign's experiments.
- Commands given an experiment ID from a different campaign exit 1 and print
  the campaign it is in, so the fix is to add `--campaign <that one>`.
- The dashboard has a campaign selector at the top of the Experiments page and
  above the experiment selector on the Results, Logs and Artifacts pages.

Campaigns change neither scheduling nor where results are written.

## Sweep file format

Every sweep file defines `COMMAND` and `OPTIONS`. Everything else is optional.

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]       # str or list[str]

OPTIONS = { ... }                      # required, see below

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

A value dim has a `"values"` list and sweeps over it:
```python
".lr": {
    "values": [1e-4, 3e-4, 1e-3],
    "flags": "--lr",    # str shorthand: generates ["--lr", str(v)] for each value
    "name": "lr",       # used in run name; None to omit; defaults to key without "."
}
```

A fixed dim has no `"values"` and no subdim keys. Its flags are always appended and it contributes nothing to the run name:
```python
".precision": {"flags": ["--dtype", "bfloat16"]}
```

A subdim has no `"values"` but has dot-prefixed child keys. Each child is a mutually exclusive branch with its own flags and optional further dims:
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
# str shorthand: "flags": "--lr" generates ["--lr", str(v)] for each value
# Python True/False become "True"/"False" (capitalized, which works with hydra; use the dict form for lowercase)
".lr": {"values": [1e-3, 3e-4], "flags": "--lr"}

# dict: explicit token list per value; "values" is optional (inferred from dict key order)
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
| `"monotonic"` | `str` | `"increasing"` or `"decreasing"`. Stop trying values after the first failure. |
| `"distribution"` | `str` | Bayes mode only: `"log_uniform"`, `"uniform"`, `"int_uniform"`. |
| `"min"` / `"max"` | `float` | Bayes continuous range. |
| `"samples"` | `int` | Grid mode only: pre-sample N values from a continuous distribution. |

### Run naming

`{sweep_name}_{dim1_name}{val1}_{dim2_name}{val2}…`

Subdim segments are dotted onto their parent: `sweep_optmuon.lrs0.1_bs32`. `"name": None` omits a dim. Boolean values abbreviate to `T`/`F`.

A run can also carry a display name. Set it with `mlsweep rename` or the Rename button on the dashboard's Experiments and Logs pages, and remove it with `--clear` or by saving an empty name. The name shows next to the run ID in `ls`, `best`, and every dashboard page. It never replaces the run ID. Directories, `logs`, `cancel` and the API keep using the ID. Run IDs repeat across experiments of the same sweep, so a rename applies to one experiment only. Over HTTP it is `PUT /api/jobs/RUN_ID/label` with `{"experiment_id": ..., "label": NAME}`, where a null or blank label clears it.

## Skipping: singular and monotonic

Both work only in sequential mode (one job at a time per slot). In parallel mode results are recorded but skipping is not applied dynamically.

`singular: True` suits hardware dims (batch size, activation checkpointing) where you just need one value that fits. The first success locks in the value; all other values for that dim are skipped. Singular dims vary slowest in the cartesian product so other dims are explored first.

```python
".bs": {
    "values": [512, 256, 128, 64, 32],  # largest first; stops at first success
    "flags": "--training.local_batch_size",
    "name": None,
    "singular": True,
}
```

`monotonic` suits finding a boundary before committing to any value. A failure at position `i` stops all further trials from that point onward. `"increasing"` tries values in listed order; `"decreasing"` reverses the list first.

```python
".bs": {
    "values": [8, 16, 32, 64],
    "monotonic": "increasing",   # if 32 fails, 64 is skipped
}
```

Don't combine them. `singular` stops on the first success, `monotonic` stops on the first failure; whichever comes first ends the search.

## Multi-GPU and multi-node

```python
GPUS_PER_RUN = 4     # allocate 4 GPUs per run; manager divides available GPUs into groups
NODES_PER_RUN = 2    # span each run across 2 workers (multi-node)
SET_DIST_ENV = True  # auto-set RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
```

With `GPUS_PER_RUN = 4` and 8 available GPUs: 2 concurrent runs, each on 4 GPUs. Groups are chosen to maximise NVLink connectivity.

mlsweep spawns one process per GPU per node. Each process receives:
- `CUDA_VISIBLE_DEVICES`, the assigned GPU IDs for this run
- `MLSWEEP_GPU_RANK`, the 0-based local GPU rank within the run's group
- `MLSWEEP_NODE_RANK`, `MLSWEEP_NNODES`, `MLSWEEP_MASTER_ADDR`, `MLSWEEP_MASTER_PORT`, multi-node only

`SET_DIST_ENV = True` derives standard PyTorch distributed vars from the above automatically. Use this for frameworks that expect `RANK`/`LOCAL_RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT` (TorchTitan, torchrun-launched scripts).

```python
# In train.py, without SET_DIST_ENV
local_rank = int(os.environ["MLSWEEP_GPU_RANK"])
device = torch.device(f"cuda:{local_rank}")

# With SET_DIST_ENV = True the standard vars are set, so no wrapper is needed
dist.init_process_group(backend="nccl")  # reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
```

## Bayesian optimization

Add `OPTIMIZE` to use TPE (optuna) instead of exhaustive grid search:

```python
OPTIMIZE = {
    "method": "bayes",      # required
    "metric": "val_loss",   # metric name logged via MLSweepLogger
    "goal": "minimize",     # "minimize" or "maximize"
    "budget": 40,           # how many trials need results
    "n_initial": 8,         # random trials before adaptive sampling starts (default: max(8, n_params*2))
    "max_failures": 8,      # failed trials in a row before the sweep pauses (default: max(3, n_initial))
}
```

Continuous dims (bayes only; use `"samples"` for grid mode):
```python
".lr": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-1, "flags": "--lr", "name": "lr"}
".dropout": {"distribution": "uniform", "min": 0.0, "max": 0.5, "flags": "--dropout", "name": "drop"}
".layers": {"distribution": "int_uniform", "min": 2, "max": 8, "flags": "--layers", "name": "L"}
```

Discrete dims and subdims work unchanged in bayes mode (optuna treats them as categorical). Singular dims are invisible to the optimizer; the full singular probe sequence is generated for each optimizer suggestion.

Bayes runs are named `{sweep_name}_bayes_{N:04d}`.

The manager runs the sweep, so `mlsweep run` submits it and exits. It chooses new trials from the results so far as each one comes in, and it carries on across a manager restart. It keeps up to `--max-concurrent` trials going, or `n_initial` if that isn't set. After `max_failures` failed trials in a row it pauses the sweep, and `mlsweep optimize EXP_ID` shows why.

| Do this | With | What happens |
|---------|------|--------------|
| See where it stands | `mlsweep optimize EXP` | Budget used, trials running and failed, the best run. `--json` for scripts. |
| Change the budget | `mlsweep optimize EXP --budget N` or `+N` | Takes effect on the next trial. A completed sweep with budget left runs again. Lowering it leaves existing runs alone. |
| End it gently | `mlsweep optimize EXP --finish` | No new trials. Runs already queued or running end, then the sweep completes. `--budget` undoes this. |
| Change how many run at once | `mlsweep optimize EXP --max-concurrent K` | Raising it starts more trials now. Lowering it lets running runs finish. |
| Hold it | `mlsweep pause EXP`, `mlsweep unpause EXP` | While paused, no new runs start and no new trials are chosen. Running runs finish. |
| Drop one trial | `mlsweep cancel EXP RUN` | The run stops, it doesn't use budget, and the manager chooses another trial. |
| End it now | `mlsweep stop EXP --yes` | Every unfinished run is cancelled. `mlsweep resume EXP` starts it choosing trials again. |

## Logger

```python
from mlsweep.logger import MLSweepLogger

with MLSweepLogger(total_steps=num_steps) as logger:
    for step in range(1, num_steps + 1):
        logger.log({"loss": 0.42, "lr": 1e-3}, step=step)  # step auto-increments if omitted
        logger.sync()  # fire-and-forget rsync of MLSWEEP_RUN_DIR artifacts mid-training
```

A run that logs no metric and prints nothing for `--stalled-after` seconds counts as stalled
for `mlsweep wait --until stalled`. A script with a long quiet stretch, such as a long
evaluation or a phase that doesn't log, calls `logger.heartbeat()` inside it. At most one
heartbeat every 10 seconds goes out, however often it's called. A background thread that sends
heartbeats would hide a real hang, so put the call in the work itself.

`total_steps` is optional. It tells the manager where the run ends, so `mlsweep ls EXP`
shows the time each running run has left (`step 500 of 1500 at 2 s/step on average,
about 33m20s left`), and `mlsweep status` says when each GPU's runs end. The estimate is
the average time per step so far. If the script learns the number later, set
`logger.total_steps` before the next `log()`.

Steps never go backwards and each metric is logged at most once per step: `log()` raises `ValueError` instead of silently overwriting a value it already recorded. Several calls at one step are fine when their keys differ, e.g. `log({"loss": l}, step=s)` then `log({"val_loss": v}, step=s)`. This holds when running the script directly too, so the mistake shows up before you launch a sweep.

`MLSweepLogger` is only active when `MLSWEEP_WORKER_SOCKET` is set, which the worker sets before launching the script. Run the script directly and it's a no-op; no import guard is needed.

If the script doesn't use the logger at all, mlsweep still dispatches the job and captures stdout/stderr to `training.log`. Metrics plots will be empty.

Save checkpoints to `os.environ["MLSWEEP_RUN_DIR"]`. They're rsynced to the experiment output directory at run end and immediately on `logger.sync()`.

## Reading metrics

`mlsweep metrics` prints what runs logged, fetched when you ask. Without a view it prints
each run's last steps as a table. The views below give one row per run instead, with the
run's latest step in a `step` column:

```sh
mlsweep metrics --experiment EXP --keys 'loss' --last                    # each key's latest value
mlsweep metrics --experiment EXP --keys 'loss|diverged' --agg final --agg max:diverged
mlsweep metrics --experiment EXP --keys '^val_loss$' --agg min           # best value, with its step
mlsweep metrics --experiment EXP --keys '^val_loss$' --step 500,1000,1500
mlsweep metrics --experiment EXP --keys '^val_loss$' --x tokens --at 1e8,2e8
mlsweep metrics --experiment A,B --keys '^val_loss$' --last              # several experiments
mlsweep metrics --active --keys '^val_loss$' --last                      # those with runs still going
```

`--last` gives each key's latest value. A value logged before the run's latest step shows
that step, as in `4.10 @1000` for a run at step 1250 that last evaluated at 1000. `--agg`
takes `final`, `min`, `max` or `mean` over all of a key's steps. `FN:REGEX` applies FN to
the keys REGEX matches, and the flag repeats, so `max:diverged` says whether a run ever
diverged while the losses stay final. `min` and `max` show the step they came from.
`--step` takes the value at each step listed, or the latest logged before it. `--x KEY --at
V` compares at the first step where KEY reaches V, for comparisons at equal tokens, FLOPs or
seconds rather than at equal steps, and adds a `step@KEY=V` column for that step.

With several experiments, each run is named `EXPERIMENT/RUN`. `--csv` gives the same table
with a `<column> step` column next to each value that has one. `--json` gives each run's
`step`, `values` and the `steps` they came from (and `last` for `--last`), and every run
names its `experiment_id`. `--pivot` reshapes keys by their capture groups, as before, and
`--step` with one step caps the steps it uses.

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
jobs = 2                         # worker-level cap: max concurrent jobs per GPU (0 = unlimited)
devices = [0, 1, 2, 3]           # GPU IDs to use (default: all visible)
venv = "/path/to/venv"           # prefer this venv over the auto-bootstrapped one
scratch_dir = "/scratch/mlsweep" # run working files (default: /tmp/mlsweep)
tunnel = true                    # reach the worker over SSH (see below)
```

```bash
mlsweep_manager --workers workers.toml
```

The manager bootstraps mlsweep on remote machines automatically; no manual install is needed. Requires passwordless SSH. Test with `ssh -o BatchMode=yes user@host1 nvidia-smi`.

The manager then connects to the worker's port directly. A cloud machine whose firewall
lets in only SSH refuses that, and the worker dies with `the worker started and listens on
port N, but the manager could not connect`. For such a machine, set `tunnel = true`, or
add it with `mlsweep workers add user@host --tunnel`. The manager then reaches the worker
through `ssh -L` from a loopback port, so only port 22 has to be open, and the machine's
address can change between restarts as long as `host` (or an entry in `~/.ssh/config`)
follows it. The worker reaches the manager through its own SSH tunnel either way.

When a worker starts, it probes each candidate GPU by creating a real CUDA
context on it. A device can show up in `nvidia-smi` as idle and still be unusable
for compute. This usually happens after a fault that leaves `GPU Recovery Action:
Reset`. The scheduler skips any GPU that fails the probe. The GPU stays in the
worker's pool, and the worker probes it again every 5 minutes. When a probe
passes, for example after a reset, the GPU goes back into scheduling on its own.

Each excluded GPU carries a reason. It is either the failing CUDA call and error
(`cuInit failed: CUDA_ERROR_SYSTEM_NOT_READY (802)`) or `CUDA probe timed out
after 60s`. The worker and manager logs print it. `mlsweep status` shows
`unhealthy=[...]` and the reasons under the worker's line. The System page shows
the reason as a tooltip on the GPU tile. `/api/workers` returns `unhealthy_gpus`
and `unhealthy_reasons`.

A timeout does not mean the GPU is broken. The driver creates contexts one at a
time, so a round of probes takes about as long as all of them run back to back,
and a busy shared host stretches that further. The hello never waits more than
15s for the probe. GPUs still being probed at that point are reported as
`probing` and join scheduling once their result arrives. Pass `--no-device-probe`
to a worker to skip the probe, for example on a host where creating a context is
itself restricted.

## Moving, deleting and cleaning up

```sh
mlsweep move EXP RUN... --to EXP2       # also --done --failed --cancelled --xfailed --pending --running
mlsweep move EXP --running --to EXP2 --create   # make EXP2 first; running runs move when they finish
mlsweep history EXP [RUN]               # what happened to its runs, and who did it
mlsweep purge EXP --failed              # lists what would go; add --yes to delete
mlsweep delete EXP --yes                # the experiment and its runs
mlsweep gc                              # lists what nothing refers to
mlsweep gc --yes [--vacuum]             # deletes it; --vacuum also shrinks the database
mlsweep gc --downloads                  # also this machine's downloads of deleted experiments
```

`move` takes runs with their metrics, logs and files. Finished and pending runs move at
once, and a pending one is then scheduled under its new experiment. A run still running is
marked, keeps running where it is, and moves when it finishes. `--now-only` refuses it
instead. `--create` makes the target if it doesn't exist, in the same campaign, with the
source's metric, goal and run controls. A run whose ID the target already has is refused,
and a marked run whose target has since gained that ID stays put, which its history says.
So splitting an experiment into several is one `move` per new experiment, with no
resubmitting. `purge` removes finished runs for good. `delete` leaves the experiment's
files on disk until `gc --yes`.

`mlsweep history EXP [RUN]` lists what happened to an experiment's runs, oldest first, and
who did it. The source is `manager`, `dashboard`, `cli`, or `api` for any other client. The manager
records each submission, each dispatch with its worker and GPUs, each result, and each
requeue with its reason (a lost worker, removed GPUs, a reschedule). It also records
people's cancels, retries, reschedules, moves, purges, renames, priority changes, pauses,
stops and deletes. A deleted experiment keeps its history, including which runs
were still going when it was deleted. `--json` gives the entries, and `GET
/api/history?experiment_id=EXP[&run_id=RUN]` is the same over HTTP. A client names itself
with the `X-Mlsweep-Client` header (`cli` or `dashboard`).

`gc` finds files of deleted experiments and runs, database rows of deleted runs, code
snapshots no run uses, and stored files that no remaining snapshot lists. Anything used in
the last hour is left alone, in case it's about to be used again. The System page's Storage
cleanup does the same.

Each row on the Experiments page has `View ▾` (Results, Logs, Artifacts, Code, Notes, Export)
and `Manage` (Rename, Move to campaign, Delete). A run's row has the same two: `View ▾`
for its Artifacts and Logs, `Manage` for Rename, Cancel or Retry, Move and Delete. The
page has
Pause, Unpause, Stop, Retry failed, Cancel active and Delete failed above an expanded
experiment's runs. Right-click a row for both menus in one, with Copy ID added. The Code
entry counts the files in the code: the experiment's latest version, or the run's own.

Right-click works the same way across the dashboard. On the System page it covers a worker,
one GPU (drain, remove or return it), a run chip and an in-flight run. Logs and Artifacts
have it on their run lists, Artifacts also on files and compare cards, Traces on traces and
events, and Export on uploads and their runs. On a link, a field or selected text you get
the browser's own menu instead.

The Artifacts page previews images and text files. For model and array files (safetensors,
PyTorch `.pt`/`.pth`/`.ckpt`/`.bin`, TorchScript, pickle, NumPy `.npy`/`.npz`, GGUF and
ONNX) it shows the tensor names, dtypes and shapes grouped by module, along with any
metadata. For a checkpoint it also shows the values around the weights, such as the epoch
and the optimizer settings. The manager reads only headers and pickle structure. It never
imports torch, and a pickle's code is never run. Compare shows the same summary for each
run, so you can see a checkpoint's epoch or step across runs.

## Exporting

```sh
mlsweep export tensorboard [--experiment EXP] [--only-done] [-o FILE.zip]
mlsweep export wandb --project P [--entity E] [--api-key K] [--experiment EXP] [--only-done] [--no-wait]
mlsweep export cancel EXPORT_ID
```

The scope is one experiment, else the current campaign, else everything with
`--all-campaigns`. `wandb` uploads from the manager, for real, using the key given or the
manager's own login; set `WANDB_MODE=offline` where the manager runs to try it safely. An
upload prints its export ID when it starts, and `export cancel` stops it.

## Workers and placement

```sh
mlsweep workers                         # list
mlsweep workers add user@host [--remote-dir D] [--ssh-key K] [--venv V] [--devices 0,1] [--port P] [--tunnel]
mlsweep workers set WORKER --jobs N [--add-devices 2,3] [--remove-devices 0] [--drain-devices 1]
mlsweep workers update WORKER
mlsweep workers rm WORKER                # forget WORKER also drops it from the list
mlsweep scheduler [--placement pack|spread]
mlsweep scheduler --pause [WORKER|all]      # --unpause [WORKER|all] undoes it
mlsweep reschedule EXP RUN [--to WORKER [--gpus 2,3]]
```

A worker's GPUs are each in its pool or out of it, and the scheduler only places runs on
GPUs in the pool. `--drain-devices` takes GPUs out of the pool and lets the runs already on
them finish. `--remove-devices` takes them out and requeues their runs at once, without
spending a retry. `--add-devices` puts GPUs back.

`workers rm` disconnects a worker and takes it out of the pool, in any state, including one
that never started. Its runs are requeued without spending a retry. A manager restart does
not relaunch it. It stays in the list of known workers, so `workers add` or the System
page's Connect dialog can bring it back. `workers forget` also drops it from that list.

`mlsweep scheduler --pause` stops the scheduler from starting new jobs anywhere, and
`--pause WORKER` stops it on one worker. Use it to place runs by hand without the scheduler
filling the GPUs first. Runs already going keep running. A run pinned with
`reschedule --to` waits until its worker is unpaused, and the pin does not expire during the
pause. `--unpause WORKER` lifts one worker's pause. `--unpause` on its own lifts every pause,
the cluster's and each worker's. Pauses are saved across manager restarts. On the System
page, the Pause scheduling button pauses every worker, and a worker's ⋯ menu pauses that
worker alone.

`workers add` waits up to `--wait` seconds (default 300) for the worker to connect, then prints
its GPUs, or the reason it didn't connect and exits 1. `--tunnel` is the SSH connection
described under [Remote workers](#remote-workers), and `--port` fixes the port the worker
listens on.

`mlsweep workers update` installs the manager's mlsweep on a worker that runs older code and
restarts it. Its in-flight runs are requeued without spending a retry.

`mlsweep reschedule` stops a running run and queues it again without spending a retry. The
run starts over, unless the script resumes from a checkpoint in `MLSWEEP_RUN_DIR`. With
`--to` it starts next on that worker, on the given GPUs or the least busy ones, even when
they are at their jobs-per-GPU limit. A pin that has not been used after ten minutes is
dropped. Multi-node runs can be requeued but not pinned.

On the System page each GPU is a tile with its runs on it. The switch on a tile drains the
GPU or puts it back. Click tiles to select several, then drain, remove or return them
together. Drag a run onto another GPU, on any worker, to restart it there, or use the run's
menu.

`pack` fills busy GPUs first, keeping whole GPUs and machines free for big jobs. `spread`
puts each job on the least busy GPU. `mlsweep rename EXP --experiment-name NAME` renames an
experiment; its ID never changes. `--description TEXT` sets its description (the
two can go together), and `--clear-description` removes it.

## Code snapshots

`mlsweep run` sends the project's files to the manager as a snapshot. A snapshot is a list
of every file's path, size, permissions and the sha256 of its content. Its ID
(`snap:3fa9c1e2...`) is the hash of that list, so the same files always give the same ID.
Each file's content is stored once on the manager, under `<mlsweep-dir>/store/objects`,
however many snapshots use it. A submission uploads only the files the manager doesn't
have, and prints how much was new:

```
  OK    Code snap:3fa9c1e2b0d4: 240 files, 501.0 MB
          new: 2 files, 41.0 KB · reused: 238 files, 501.0 MB
          git: 9a5b61a3c0 master +1 uncommitted
          not tracked by git: data/ (450.0 MB)
```

The manager refuses a submission whose new files add up to more than 512 MB. Files it
already has don't count, so a large dataset costs one upload and is reused after that.
Shipping data with the code works, but it's better kept out with `UPLOAD_EXCLUDE` and
read from a path on the workers.

Each worker keeps the files its runs have used in `<scratch>/.objects` and fetches only
what a new snapshot adds. The cache is trimmed to 20 GB after each run's code is written
out, least recently used first. `MLSWEEP_OBJECT_CACHE_GB` on the worker changes that.

The client also records where the code came from: the sweep file, the `mlsweep run`
command line, the host, and, when the project is a git repository, the commit, branch,
remote, whether the tree had uncommitted changes, the diff against the commit, and which
shipped files git doesn't track. None of this is needed to run anything. It's there for
reading an experiment later.

```sh
mlsweep code EXP                        # the code EXP's runs used, and where it came from
mlsweep code EXP yoco/train.py          # one file, as EXP ran it
mlsweep code EXP yoco/train.py --lines 40-60   # numbered, with the note reference (-n: the whole file, paged)
mlsweep code "mlsweep:code/EXP/yoco/train.py#L40-60"   # the same, from a note's link
mlsweep code search warmup [EXP...]     # matching lines, in EXP or across the campaign (-E, -i, -C N, --path)
mlsweep code EXP --files [PREFIX]       # the snapshot's file list
mlsweep code EXP --uncommitted          # the uncommitted changes recorded at submit
mlsweep code diff EXP_A EXP_B           # what changed (--path DIR to narrow, --stat for a summary)
mlsweep code checkout EXP DIR           # write EXP's code out to DIR
```

With `--lines` or `-n`, a file prints with line numbers and stops at about 2000 tokens
(`--budget`). The last lines give the note reference for what was shown and the command for
the next page. Without either, the file prints raw, so `> FILE` saves it as it was. `search`
matches text, or a regular expression with `-E` (given 20 seconds, after which it stops and
keeps what it found), and gives each line with its
`mlsweep:code/EXP/path#L12` reference. With no experiment named it searches the campaign's
experiments, newest first, each as its latest code. A file several of them share is searched
once, and its hits name the newest experiment and count the others. Binary files and files
over 2 MB are skipped.

The dashboard's Code tab shows the same. Pick an experiment and a run, as on Logs, to see
that run's code as a file tree and files with line numbers, and where it came from.
Model and array files (`.pt`, `.safetensors`, `.npy` and the like) get the same summary of
tensors and shapes as on Artifacts, and images are shown. The Wrap box beside a file's path
folds long lines.
Details shows everything recorded at submit, the uncommitted diff included. The version
menu picks any version of the experiment's code. Compare picks another version, a run that
used other code, or any experiment from a list you can search, then marks the tree and
shows each changed file's diff. Right-click a file, a folder, a line number or a run for
the rest: comparing a file with the previous version, links for a note, and downloads. The
download button gives the whole snapshot as a zip. Results, Logs and Artifacts, a run's
details and `View ▾` menu on the Experiments page, and the search palette link to it. Notes can link to a file and lines there, and show code or a diff inline.
See "Code links" in `mlsweep docs notes`.

Where an experiment ID is accepted, a snapshot ID works too, or a unique prefix of one with
at least 8 hex digits. An experiment appended to with different code has more than one
snapshot. `mlsweep code EXP` lists them all, and `--snapshot ID` picks one.

Experiments submitted before snapshots ran from a single tarball per submission. Their
runs still retry from it, but `mlsweep code`, the code page and notes can't show it.

## Notes

Research notes live in `<mlsweep-dir>/notes` and show in the dashboard's Notes tab. See
`docs/notes.md` (`mlsweep docs notes`) for the tab, the markdown it renders, sharing between
machines, and lint.

## SQL

`mlsweep sql "SELECT ..."` runs one query over the manager's database and prints the rows as a
table (`--json` for the data). With no query it lists the tables and their columns. The query
can only read. It runs on a read-only connection that refuses writes, `ATTACH` and settings,
and it stops after 5 seconds (`--timeout`, at most 30) and 200 rows (`--limit`, at most 5000).
Table layouts can change between versions, so prefer a command where one fits. `POST /api/sql`
with `{"sql": ...}` is the same thing over HTTP.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `COMMAND is required` | Add `COMMAND = ["python", "train.py"]` to the sweep file |
| `Dimension key must start with '.'` | All `OPTIONS` keys (and subdim keys) need a `.` prefix |
| `has both 'values' and subdimensions` | A dim can have a value list or subdim branches; it cannot have both |
| Token errors | Check `~/.mlsweep/manager.token` or pass `--token` / set `MLSWEEP_TOKEN` |
| `experiment X is in campaign 'Y', not 'Z'` | Pass `--campaign Y` (or `--all-campaigns`), or set `MLSWEEP_CAMPAIGN=Y` |
| Singular/monotonic not skipping | Only jobs still pending when a result arrives are skipped; lower `--max-concurrent` so fewer run at once |
| Remote not connecting | Test SSH: `ssh -o BatchMode=yes user@host nvidia-smi` |
| `need at least GPUS_PER_RUN GPUs` | Worker has fewer GPUs than `GPUS_PER_RUN`; adjust worker config or reduce `GPUS_PER_RUN` |
| `mlsweep status` shows a worker with fewer GPUs than expected, or `unhealthy=[N]` | GPU `N` failed the CUDA-context probe. Read the reason printed under the worker. A CUDA error (often after `GPU Recovery Action: Reset`) needs a reset (`sudo nvidia-smi --gpu-reset -i N`), or drop the GPU from the worker's `devices` list. A timeout usually means a loaded host, and the next probe 5 minutes later can clear it |
| Jobs die with `CUDA error: CUDA-capable device(s) is/are busy or unavailable` | The device could not create a primary context. Reset it and confirm the worker excludes it (`mlsweep status`) |
| No metrics plots | Script must use `MLSweepLogger`; stdout/stderr are still captured without it |

## Reference

- `docs/sweep_configuration.md` is the complete reference for dim types, flag behavior, CLI options and the output layout
- `docs/examples.md` has worked examples for DDP, multi-node runs, TorchTitan and Prime-RL
- `docs/notes.md` covers the Notes tab, its markdown, sync, the git server and lint
- `mlsweep/docs/SKILL.md` is the agent-facing quick reference. Read it with `mlsweep docs skill`. The README's "Running an agent" section covers installing it and starting an agent
