# Integration

## Local setup

```bash
git clone <your-project>
cd <your-project>
pip install mlsweep
```

Write a sweep file (see [sweep_configuration.md](sweep_configuration.md)):

```python
# sweeps/my_sweep.py
COMMAND = ["python", "train.py"]
OPTIONS = {
    ".lr": {"values": [1e-4, 3e-4, 1e-3], "flags": "--lr"},
}
```

Run locally:

```bash
mlsweep_run sweeps/my_sweep.py             # 1 GPU
mlsweep_run sweeps/my_sweep.py -g 4        # 4 GPUs in parallel
mlsweep_run sweeps/my_sweep.py -g          # all visible GPUs
mlsweep_run sweeps/my_sweep.py -g 4 -j 5   # 5 jobs per GPU (20 jobs)
```

## Remote workers

### 1. Install mlsweep on each remote machine

```bash
ssh user@host -i path/to/key
cd path/to/project/
pip install mlsweep
```

Verify:

```bash
ssh user@host mlsweep_worker --help
```

### 2. Create a workers.toml with your remote worker

```toml
[[workers]]
host = "user@host1"
remote_dir = "/absolute/path/to/project"
ssh_key = "~/.ssh/id_ed25519"
venv = "/absolute/path/to/venv/"  # Optional, resolves .venv/, venv/, calls bin/activate, defaults to remote_dir
devices = [0, 1, 2, 3]
jobs = 2
```

|    Field     | Required | Notes |
|--------------|----------|-------|
| `host`       | yes      | SSH target |
| `remote_dir` | yes      | Project root on the remote |
| `ssh_key`    | no       | Path to identity file (`-i`) |
| `pass`       | no       | SSH password (needs `sshpass`); or set `MLSWEEP_SSH_PASS` env var |
| `venv`       | no       | Venv locator (default: `remote_dir`). Accepts a project root, venv root, `bin/` dir, activate script, or python binary. |
| `devices`    | no       | Specific GPU IDs to use |
| `gpus`       | no       | Total GPU count -g (default: all visible) |
| `jobs`       | no       | Concurrent jobs per GPU slot -j (default: 1) |

**`venv` accepts any of:**
- Project root containing `.venv/` or `venv/`
- Venv root directory (contains `bin/mlsweep_worker`)
- `bin/` directory
- Path to `activate` script
- Path to a python binary

### 3. Add logging metrics to your training script

```python
from mlsweep.logger import MLSweepLogger

hparams = {"lr": 0.002, "batch_size": 32}

# Or if you don't want to use it as a context manager, call .close().
with MLSweepLogger(**hparams) as logger:
    for step in range(1, num_steps + 1):
        loss = train_step()
        logger.log({"loss": loss}, step=step)
```



No-op when run outside of mlsweep_run. Results land in `outputs/sweeps/<experiment>/<run>/metrics.jsonl`.

### 4. Write a sweep.

```python
# sweeps/my_sweep.py
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

This produces 9 runs named `my_sweep_lr1e-4_bs32`, `my_sweep_lr1e-4_bs64`, etc.

Each run receives its flags appended to `COMMAND`: `python train.py --optimizer.lr 0.0001 --training.batch_size 32`.

See [sweep_configuration.md](sweep_configuration.md) for the full format: subdimensions, monotonic/singular skipping, `EXCLUDE`, `GPUS_PER_RUN`, and more.

### 5. Run.

```bash
mlsweep_run sweeps/my_sweep.py --workers workers.toml
```

### 6. Visualize outputs.

Once you've launched the sweep, on the machine and in the dir you called `mlsweep_run` from, run:

```bash
mlsweep_viz
# or
mlsweep_viz experiment_name
```

This will prompt you to open up a browser (or pass --open-browser to do so automatically) to see the sweep visualizer.

## Troubleshooting

If the error messages are bad or the docs are bad or for some reason you feel confused feel free to hit me up on discord at @apaz or twitter/x at @apaz_cli.
