# Programmatic API (`WorkerPool`)

`WorkerPool` is a lightweight alternative to the sweep runner for callers that
want to submit individual jobs and receive structured results — without sweep
files, variation grids, or logger integration. It uses the same
`mlsweep_worker` backend as `mlsweep_run`.

Typical use case: an orchestrator that generates work dynamically, wants to
send file payloads to workers, and needs the modified files back.

## Basic usage

```python
from mlsweep.pool import WorkerPool, WorkerConfig, RunResult
from mlsweep._shared import MsgRun

pool = WorkerPool([
    WorkerConfig(host="user@gpu-box",
                 remote_dir="/home/user/project",
                 devices=[0, 1, 2, 3],
                 ssh_key="~/.ssh/id_ed25519"),
    WorkerConfig(host=None,  # local
                 remote_dir="/home/user/project",  # same machine
                 devices=[],  # CPU-only
                 ),
])

with pool:
    result: RunResult = pool.run(MsgRun(command=["python", "train.py"]))
    print(result.stdout)
    print("exit code:", result.exit_code)
```

`pool.run()` is a blocking convenience wrapper around `pool.submit()` +
`pool.wait()`. Use `submit` / `wait` directly to overlap dispatch and
collection across multiple concurrent runs.

## `WorkerConfig` fields

| Field | Default | Notes |
|-------|---------|-------|
| `host` | — | `None` for local; `"user@host"` for SSH |
| `remote_dir` | — | Project root on worker; cwd when `files={}` |
| `devices` | `[]` | GPU indices (empty = CPU-only) |
| `gpus_per_run` | `1` | GPUs per job slot |
| `jobs` | `1` | Concurrent jobs per device group |
| `scratch_dir` | `"/tmp/mlsweep"` | mlsweep scratch base on the worker |
| `port` | `7890` | Worker TCP port; `0` = ephemeral (no reuse) |
| `ssh_key` | `None` | Path to SSH identity file |
| `password` | `None` | SSH password (requires `sshpass`) |
| `venv` | `None` | Venv locator — same format as `workers.toml` |

## `RunResult` fields

| Field | Notes |
|-------|-------|
| `run_id` | Echo of the submitted `MsgRun.run_id` |
| `success` | `True` when exit code is 0 |
| `exit_code` | Raw process exit code |
| `elapsed` | Wall-clock seconds on the worker |
| `stdout` | Concatenated rank-0 stdout (no logger required) |
| `files` | `{rel_path: content}` for each `return_files` entry (see below) |

## `MsgRun` fields

Only `command` is required. Everything else has a sensible default; `gpu_ids`,
`remote_dir`, and `scratch` are filled in by the pool before dispatch.

| Field | Default | Notes |
|-------|---------|-------|
| `command` | — | Command to run as a list, e.g. `["python", "train.py"]` |
| `run_id` | auto | Unique ID for this run; auto-generated if not set |
| `experiment` | `"pool"` | Groups output under `outputs/{experiment}/{run_id}/` |
| `env` | `{}` | Extra environment variables. `CUDA_VISIBLE_DEVICES` is set automatically from the assigned GPU slot — no need to set it here. |
| `gpu_ids` | `[]` | Filled by the pool from the assigned slot. Set explicitly only when sending `MsgRun` directly to a worker without a pool. |
| `remote_dir` | `""` | Filled by the pool from `WorkerConfig.remote_dir`. |
| `scratch` | `""` | Filled by the pool from `WorkerConfig.scratch_dir`. |
| `run_from` | `None` | Subdirectory of `remote_dir` to use as cwd instead of `remote_dir` itself. Ignored when `files` is non-empty (workspace becomes cwd). |
| `set_dist_env` | `False` | Auto-populate `RANK`/`LOCAL_RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT` for distributed training. |
| `files` | `{}` | File payload — see below. |
| `return_files` | `[]` | Files to copy back — see below. |

## File payloads and workspaces

### Sending files to the worker

Set `MsgRun.files` to a `dict[str, str]` of workspace-relative paths and their
text content. When non-empty, the worker creates a per-run isolated workspace
before executing the command:

```
scratch_dir/{experiment}/{run_id}/workspace/
```

**Workspace setup:** the worker hard-links everything from `remote_dir` into
the workspace (fast even for multi-GB datasets — hard links cost no extra
disk), then writes the `files=` payload on top, overwriting just those paths.
The command's `cwd` becomes the workspace, and `MLSWEEP_WORKSPACE` is set in
the environment.

```python
MsgRun(
    ...,
    remote_dir="/home/user/project",   # large pre-staged files live here
    files={"train.py": new_train_py},  # only the changed file is sent
)
```

If `remote_dir` is empty or missing, the workspace is created from the
`files=` payload alone (no hard-link base).

### Getting files back

Set `MsgRun.return_files` to a list of paths relative to the command's cwd.
After the command exits, the worker copies those files into its `artifacts/`
directory before the normal rsync back to the controller. They appear in
`RunResult.files` as `{rel_path: text_content}`.

`return_files` is independent of `files`: it works whether or not a workspace
was created. When `files` is non-empty the cwd is the workspace; when `files`
is empty the cwd is `remote_dir` (or `remote_dir/run_from`).

```python
result = pool.run(MsgRun(
    ...,
    files={"train.py": original_train_py},
    return_files=["train.py"],
))
modified = result.files["train.py"]   # whatever the command left on disk
```

### `remote_dir` vs `scratch_dir`

- **`remote_dir`** — your project on the worker: source code, datasets,
  fixed assets. Large files live here and are hard-linked into each run's
  workspace for free. You populate this via `workers.toml` `setup` commands
  or manual staging.

- **`scratch_dir`** — mlsweep's working area on the worker. Each run gets
  `scratch_dir/{experiment}/{run_id}/` containing logs, metrics, `artifacts/`,
  and the `workspace/` directory. Managed entirely by mlsweep; you don't
  write here directly.

## `WorkerPool` methods

```python
pool.start()               # launch all workers, block until all connected
pool.submit(msg) -> run_id # dispatch a run; blocks until a slot is free
pool.wait(run_id)          # block until the run completes; return RunResult
pool.run(msg)              # submit + wait
pool.shutdown()            # send MsgShutdown to all workers
```

`WorkerPool` also works as a context manager (`with WorkerPool(...) as pool:`).

## Slot model

Slots are computed from `WorkerConfig.devices`, `gpus_per_run`, and `jobs`
using the same GPU topology-aware grouping as `mlsweep_run`. The pool keeps a
queue of available `(worker, gpu_ids)` slots; `submit()` blocks until one is
free, then dispatches immediately.

For CPU-only workers (`devices=[]`), `jobs` slots are created, each with
`gpu_ids=[]`.

## Output files

Artifacts are rsynced to `output_dir/{run_id}/artifacts/` on the controller
after each run. `return_files` contents are read from there and returned in
`RunResult.files`. `output_dir` defaults to `/tmp/mlsweep_pool` and can be
set in the `WorkerPool` constructor.
