# mlsweep

`mlsweep` is slightly opinionated but very general solution for managing tons of machine learning runs. It takes flexible combinations of hyperparameters and schedules them across your hardware.

The project contains a persistent manager that owns GPU scheduling and a web dashboard, and workers that execute training jobs. You aren't forced to use our web UI. You can export mlsweep to wandb or tensorboard, and use whatever you use for viewing. The mlsweep logger is also extensible, should you wish. Logs end up on the manager machine.

The main feature of mlsweep is not the logger, or the cluster managment features, but the sweep configuration file. The good stuff. The thing that I've been missing all my machine learning life, and the reason I wrote this library.

`mlsweep` does pretty much everything that wandb does. If you're missing anything, let me know on [Discord](https://discord.gg/w2K2JWJGUb) or [Twitter](https://twitter.com/apaz_cli).

But first, let's install it, and add the logging.

## Setup

Install mlsweep on the machine will run `mlsweep_manager`:

```sh
pip install 'mlsweep[all]'
```

That's it. Remote workers get mlsweep bootstrapped automatically over SSH, no install needed.

You con't need to run the manager (`mlsweep_manager`), controller (`mlsweep_run`), or worker (managed automatically by the manager) on separate machines. By default they run on the same machine.

## Add logging to your training script

```python
from mlsweep.logger import MLSweepLogger

# If you don't want to use it as a context manager, remember to call .close().
with MLSweepLogger() as logger:
    for step in range(1, num_steps + 1):
        loss = train_step()
        logger.log({"loss": loss}, step=step)

        # Write checkpoints to MLSWEEP_RUN_DIR, they get rsynced back automatically.
        # Call logger.sync() to trigger an immediate rsync mid-run (fire-and-forget).
        if step % 1000 == 0:
            save_checkpoint(os.environ["MLSWEEP_RUN_DIR"], step)
            logger.sync()
```

`MLSweepLogger` is only active when your script is launched by a worker. It checks for `MLSWEEP_WORKER_SOCKET`, which the worker sets. Run your script directly and it's a no-op. The worker always has mlsweep available (bootstrapped or from your venv), so the logger just works.

If your script doesn't use the logger at all that's fine too, mlsweep still dispatches the job and captures stdout/stderr to `training.log`. You just won't get metrics plots.

Metrics land in `outputs/sweeps/<experiment>/<run>/metrics.jsonl` on the manager. Anything written to `MLSWEEP_RUN_DIR` is rsynced to `outputs/sweeps/<experiment>/<run>/artifacts/` at the end of every run, and immediately on `logger.sync()`.

## Write a sweep configuration file

Add the following shebang, and use `chmod +x` so that your sweep file can be directly executable.

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]

OPTIONS = {
    ".lr": {
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

Running this produces 9 runs named `my_sweep_lr1e-4_bs32`, `my_sweep_lr1e-4_bs64`, etc.

Each run receives its flags appended to `COMMAND`: `python train.py --optimizer.lr 0.0001 --training.batch_size 32`.

See [sweep_configuration.md](docs/sweep_configuration.md) for the full format: subdimensions, monotonic/singular skipping, `EXCLUDE`, `NODES_PER_RUN` and `GPUS_PER_RUN` for training with `torchrun` (see `SET_DIST_ENV`), and more. For end-to-end examples with real frameworks (Prime-RL, TorchTitan), see [examples.md](docs/examples.md).

## Bayesian optimization

If your sweep is specifically for hyperparameter optimization, you can add an `OPTIMIZE` dict to save compute. It uses TPE (via [optuna](https://github.com/optuna/optuna)) to intelligently sample the space and find good configs faster than trying all combinations.

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]

OPTIMIZE = {
    "method": "bayes",
    "metric": "val_loss",
    "goal": "minimize",
    "budget": 40,
}

OPTIONS = {
    # Discrete dim
    ".optimizer": {
        "name": "opt",
        ".adam": {"flags": ["--optimizer", "adam"]},
        ".muon": {"flags": ["--optimizer", "muon"]},
    },
    # Continuous dims
    ".lr": {
        "distribution": "log_uniform",
        "min": 1e-5,
        "max": 1e-1,
        "flags": "--optimizer.lr",
        "name": "lr",
    },
    ".wd": {
        "distribution": "log_uniform",
        "min": 0.0,
        "max": 0.2,
        "flags": "--optimizer.weight_decay",
        "name": "wd",
    },
}
```

See [sweep_configuration.md](docs/sweep_configuration.md) for continuous ranges, singular dims, and all `OPTIMIZE` fields.

## Run

mlsweep uses a manager daemon that owns GPU scheduling and persists state. Start it once, then submit sweeps against it.

### 1. Start the manager

```sh
mlsweep_manager                                        # local GPUs, dashboard at http://localhost:7891
mlsweep_manager --workers workers.toml                 # remote workers
mlsweep_manager --port 7891 --host my.server.com       # custom port and externally-reachable hostname
mlsweep_manager --mlsweep-dir /data/mlsweep            # custom state dir (DB, token, experiment outputs)
```

Launching the manager also creates a worker process on localhost, unless --workers is passed.

The manager prints dashboard URLs on startup, including a token for authentication:

```
Dashboard: http://localhost:7891/?token=abc123...
```

The token is also saved to `~/.mlsweep/manager.token` so local workers find it automatically.

### 2. Submit a sweep

```sh
mlsweep_run sweeps/my_sweep.py --manager http://localhost:7891
mlsweep_run sweeps/my_sweep.py --manager http://localhost:7891 --stream   # live status
```

If the manager is on localhost, `mlsweep_run` auto-reads the token from `~/.mlsweep/manager.token`. For remote managers, pass `--token` or set `MLSWEEP_TOKEN`.

### Remote workers

The manager installs mlsweep on remote machines automatically over SSH, with no manual setup needed. It builds wheels from the local source at startup, SCPs them to the remote, and installs them into `/tmp/mlsweep_venv/`.

#### 1. Create a workers.toml

```toml
[[workers]]
host = "user@host1"
remote_dir = "/absolute/path/to/project"
ssh_key = "~/.ssh/id_ed25519"
devices = [0, 1, 2, 3]
jobs = 2
```

|    Field     | Required | Notes |
|--------------|----------|-------|
| `host`       | yes      | SSH target |
| `remote_dir` | yes      | Project root on the remote |
| `ssh_key`    | no       | Path to identity file (`-i`) |
| `pass`       | no       | SSH password (needs `sshpass`); or set `MLSWEEP_SSH_PASS` env var |
| `venv`       | no       | Existing venv to prefer over the auto-bootstrapped one. Accepts a project root, venv root, `bin/` dir, activate script, or python binary. |
| `devices`    | no       | Specific GPU IDs to use |
| `gpus`       | no       | Total GPU count (default: all visible) |
| `jobs`       | no       | Max concurrent jobs per GPU on this worker. Composes with `-j` on `mlsweep_run` via `min()`; 0 or omitted means no worker-level cap. |
| `port`       | no       | Worker TCP port (default: 7890; `0` = ephemeral). |

#### 2. Start the manager with the workers file

```sh
mlsweep_manager --workers workers.toml
```

## Dashboard

The manager serves a web dashboard at the URL printed at startup (default `http://localhost:7891`). It shows live metrics, per-run logs, file browser, and system status. Open it in a browser while your sweep runs.

## Useful CLI flags

All flags below assume `--manager http://localhost:7891`:

| Flag | Effect |
|------|--------|
| `--dry-run` | Print commands without running |
| `--validate` | Check config, list all combos, exit |
| `--stream` | Live status in terminal |
| `--experiment NAME` | Custom experiment name |
| `--priority N` | Higher values run sooner (default: 0) |
| `--wandb-project P` | Stream metrics to W&B |
| `--tensorboard-dir D` | Write TensorBoard logs |

Subcommands:

```sh
mlsweep_run fetch --manager http://localhost:7891 --experiment EXP_ID    # download results
mlsweep_run watch EXP_ID --manager http://localhost:7891                 # watch live status
```

### Using with W&B

mlsweep can log all runs to Weights & Biases with no changes to your training script.

```sh
pip install 'mlsweep[wandb]'
export WANDB_API_KEY=your_key_here
mlsweep_run sweeps/my_sweep.py --manager http://localhost:7891 --wandb-project my-project
mlsweep_run sweeps/my_sweep.py --manager http://localhost:7891 --wandb-project my-project --wandb-entity my-team
```

### Using with TensorBoard

```sh
pip install 'mlsweep[tensorboard]'
mlsweep_run sweeps/my_sweep.py --manager http://localhost:7891 --tensorboard-dir ./tb_logs
tensorboard --logdir ./tb_logs
```

## Troubleshooting

If the error messages are bad or the docs are confusing, hit me up on [Discord](https://discord.gg/w2K2JWJGUb) or [Twitter](https://twitter.com/apaz_cli).
