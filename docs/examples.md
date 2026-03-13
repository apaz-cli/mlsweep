# Sweep Examples

Practical patterns for common training scenarios.

---

## Basic Hyperparameter Sweep

A single-GPU sweep over learning rate and batch size:

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
        "values": [32, 64, 128],
        "flags": "--batch_size",
        "name": "bs",
    },
}
```

Run with 4 GPUs in parallel:

```sh
mlsweep_run sweeps/my_sweep.py -g 4
```

This produces 9 runs (3×3 grid) with up to 4 running concurrently.

---

## Multi-GPU DDP

Set `GPUS_PER_RUN` to the number of GPUs each training run needs. mlsweep spawns one process per GPU, each receiving `MLSWEEP_GPU_RANK` (0-based local rank) and `CUDA_VISIBLE_DEVICES` set to the full assigned group.

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]
GPUS_PER_RUN = 4

OPTIONS = {
    ".lr": {
        "values": [1e-4, 3e-4, 1e-3],
        "flags": "--lr",
        "name": "lr",
    },
}
```

In `train.py`, read `MLSWEEP_GPU_RANK` as the local rank:

```python
import os
import torch
import torch.distributed as dist

local_rank = int(os.environ["MLSWEEP_GPU_RANK"])
world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

# MASTER_ADDR / MASTER_PORT are not set automatically for single-node runs;
# provide them via EXTRA_FLAGS, a wrapper script, or set defaults here.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
device = torch.device(f"cuda:{local_rank}")
```

> **Port conflicts:** If multiple runs execute concurrently on the same machine, each needs a distinct `MASTER_PORT`. One approach is to derive a port from the run name: `int(hashlib.md5(os.environ["MLSWEEP_RUN_NAME"].encode()).hexdigest()[:4], 16) % 10000 + 20000`.

Run with 8 GPUs to get 2 parallel 4-GPU jobs:

```sh
mlsweep_run sweeps/ddp_sweep.py -g 8
```

GPU groups are chosen to maximise NVLink/interconnect quality.

---

## Multi-Node

Set `NODES_PER_RUN` to span a run across multiple machines. mlsweep spawns `GPUS_PER_RUN` processes on each node and injects `MLSWEEP_NODE_RANK`, `MLSWEEP_MASTER_ADDR`, and `MLSWEEP_MASTER_PORT` for the inter-node rendezvous.

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["python", "train.py"]
GPUS_PER_RUN = 4
NODES_PER_RUN = 2  # 8 GPUs total per run: 2 nodes × 4 GPUs

OPTIONS = {
    ".lr": {
        "values": [1e-4, 3e-4, 1e-3],
        "flags": "--lr",
        "name": "lr",
    },
}
```

In `train.py`:

```python
import os
import torch
import torch.distributed as dist

node_rank  = int(os.environ.get("MLSWEEP_NODE_RANK", "0"))
local_rank = int(os.environ["MLSWEEP_GPU_RANK"])
nnodes     = int(os.environ.get("MLSWEEP_NNODES", "1"))
gpus_per_node = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

global_rank = node_rank * gpus_per_node + local_rank
world_size  = nnodes * gpus_per_node

os.environ["MASTER_ADDR"] = os.environ["MLSWEEP_MASTER_ADDR"]
os.environ["MASTER_PORT"] = os.environ["MLSWEEP_MASTER_PORT"]

dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
device = torch.device(f"cuda:{local_rank}")
```

Run with a workers file containing at least 2 entries:

```sh
mlsweep_run sweeps/multinode_sweep.py --workers workers.toml
```

mlsweep launches the command on all nodes simultaneously and records the run as succeeded only when every node exits 0.

---

## TorchTitan: Multi-GPU Pretraining

[TorchTitan](https://github.com/pytorch/torchtitan) expects one process per GPU with rank information in the standard `RANK` / `LOCAL_RANK` / `WORLD_SIZE` / `MASTER_ADDR` / `MASTER_PORT` env vars. Set `SET_DIST_ENV = True` and mlsweep populates these automatically.

### Single-node

```python
#!/usr/bin/env mlsweep_run

GPUS_PER_RUN = 8
SET_DIST_ENV = True

COMMAND = ["python", "-m", "torchtitan.train", "--module", "llama3", "--config", "llama3_8b"]

OPTIONS = {
    ".lr": {
        "values": [3e-4, 8e-4, 2e-3],
        "flags": "--optimizer.lr",
        "name": "lr",
    },
    ".bs": {
        "values": [2, 4, 8],
        "flags": "--training.local_batch_size",
        "name": "bs",
        "singular": True,
    },
}
```

Run with 16 GPUs to get 2 concurrent 8-GPU runs:

```sh
mlsweep_run sweeps/torchtitan_sweep.py -g 16
```

### Multi-node

```python
#!/usr/bin/env mlsweep_run

GPUS_PER_RUN = 8
NODES_PER_RUN = 2
SET_DIST_ENV = True

COMMAND = ["python", "-m", "torchtitan.train", "--module", "llama3", "--config", "llama3_8b"]

OPTIONS = {
    ".lr": {
        "values": [3e-4, 8e-4],
        "flags": "--optimizer.lr",
        "name": "lr",
    },
    ".bs": {
        "values": [2, 4, 8],
        "flags": "--training.local_batch_size",
        "name": "bs",
        "singular": True,
    },
}
```

```sh
mlsweep_run sweeps/torchtitan_multinode.py --workers workers.toml
```

---

## Prime-RL: Multi-GPU RL Training

[Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl) is a single-entry-point launcher that spawns its own internal processes (trainer, orchestrator, inference server) and manages its own GPU assignment via `CUDA_VISIBLE_DEVICES`. It should be launched once per node, not once per GPU.

With `GPUS_PER_RUN=8`, mlsweep spawns 8 processes. Guard the actual launch on `MLSWEEP_GPU_RANK=0`; the other 7 processes exit immediately, leaving rank 0's Prime-RL instance with all 8 GPUs visible.

### Single-node

```python
#!/usr/bin/env mlsweep_run

# 8 GPUs total: 4 inference + 4 trainer (split configured inside base_rl.toml)
GPUS_PER_RUN = 8

COMMAND = [
    "bash", "-c",
    'if [ "$MLSWEEP_GPU_RANK" = "0" ]; then'
    '  exec uv run rl @ configs/base_rl.toml "$@"; fi',
    "--",
]

OPTIONS = {
    ".lr": {
        "values": [1e-6, 3e-6, 1e-5],
        "flags": "--trainer.optim.lr",
        "name": "lr",
    },
    ".kl": {
        "values": [1e-4, 1e-3, 1e-2],
        "flags": "--trainer.loss.kl_tau",
        "name": "kl",
    },
}
```

Run with 16 GPUs to get 2 parallel jobs:

```sh
mlsweep_run sweeps/rl_sweep.py -g 16
```

This works because Prime-RL is designed to be isolated by `CUDA_VISIBLE_DEVICES`, which mlsweep sets automatically. It reads the variable on startup and maps all its internal GPU allocations (inference server, trainer, teacher) from it. Each mlsweep slot gets a disjoint GPU group, so two concurrent Prime-RL instances never touch each other's devices. Port conflicts are also not an issue, Prime-RL calls `get_free_port()` for its internal torchrun rendezvous, so each instance binds a different port automatically.

### Multi-node

For multi-node Prime-RL, rank 0 on each node launches with the node-rank rendezvous vars:

```python
#!/usr/bin/env mlsweep_run

GPUS_PER_RUN = 8
NODES_PER_RUN = 2

COMMAND = [
    "bash", "-c",
    'if [ "$MLSWEEP_GPU_RANK" = "0" ]; then'
    '  exec uv run rl @ configs/base_rl.toml'
    '    --trainer.nnodes $MLSWEEP_NNODES'
    '    --trainer.node_rank $MLSWEEP_NODE_RANK'
    '    --trainer.master_addr $MLSWEEP_MASTER_ADDR'
    '    --trainer.master_port $MLSWEEP_MASTER_PORT'
    '    "$@"; fi',
    "--",
]

OPTIONS = {
    ".lr": {
        "values": [1e-6, 3e-6, 1e-5],
        "flags": "--trainer.optim.lr",
        "name": "lr",
    },
    ".kl": {
        "values": [1e-4, 1e-3, 1e-2],
        "flags": "--trainer.loss.kl_tau",
        "name": "kl",
    },
}
```

Run with a workers file that has at least 2 entries (more for parallel runs):

```sh
mlsweep_run sweeps/rl_multinode.py --workers workers.toml
```
