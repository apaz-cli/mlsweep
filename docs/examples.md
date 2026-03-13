# Sweep Examples

Practical sweep file patterns for common training frameworks.

## Prime-RL: Multi-GPU RL Training

[Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl) runs three components — a trainer, an orchestrator, and an inference server — all launched from a single `uv run rl` command. The command accepts TOML config files (`@ path/to/config.toml`) and CLI overrides (`--key.subkey value`), so mlsweep just appends flags to override the base config.

Each RL run uses a pool of GPUs split between training and inference. Set `GPUS_PER_RUN` to the total GPU count the run needs and configure the split inside the base config (or via flags).

### Base config

Start with a base TOML that sets the fixed parts of the run — model, environment, output dir, deployment topology:

```toml
# configs/base_rl.toml
output_dir = "outputs/rl-sweep"
max_steps = 500
seq_len = 2048

[model]
name = "Qwen/Qwen3-0.6B"

[wandb]
project = "my-rl-sweep"

[[orchestrator.env]]
id = "math-env"
name = "gsm8k"
args = { dataset_name = "openai/gsm8k", dataset_subset = "main" }

[inference.parallel]
tp = 1
dp = 4   # 4 inference GPUs

[trainer.model]
attn = "flash_attention_2"
```

### Sweep file

```python
#!/usr/bin/env mlsweep_run

COMMAND = ["uv", "run", "rl", "@", "configs/base_rl.toml"]

# 8 GPUs total: 4 inference (set in base config) + 4 trainer
GPUS_PER_RUN = 8

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
    ".bs": {
        "values": [128, 256, 512],
        "flags": "--orchestrator.batch_size",
        "name": "bs",
    },
    ".rollouts": {
        "values": [8, 16],
        "flags": "--orchestrator.rollouts_per_example",
        "name": "r",
    },
}
```

Run with 16 GPUs to get 2 parallel jobs:

```sh
mlsweep_run sweeps/rl_sweep.py -g 16
```

### Sweeping the model

To compare models, use a subdim — each branch sets both the trainer and inference model together:

```python
OPTIONS = {
    ".model": {
        "name": "model",
        ".qwen_0_6b": {
            "flags": ["--model.name", "Qwen/Qwen3-0.6B"],
        },
        ".qwen_1_7b": {
            "flags": ["--model.name", "Qwen/Qwen3-1.7B"],
        },
        ".qwen_4b": {
            "flags": ["--model.name", "Qwen/Qwen3-4B"],
        },
    },
    ".lr": {
        "values": [1e-6, 3e-6, 1e-5],
        "flags": "--trainer.optim.lr",
        "name": "lr",
    },
}
```

### Multi-node RL

For multi-node Prime-RL (e.g. trainer on 2 nodes, inference on 2 nodes), set `NODES_PER_RUN` to the number of machines and let mlsweep inject the distributed env vars:

```python
#!/usr/bin/env mlsweep_run

COMMAND = [
    "bash", "-c",
    "uv run rl @ configs/base_rl.toml "
    "--trainer.nnodes $MLSWEEP_NNODES "
    "--trainer.node_rank $MLSWEEP_NODE_RANK "
    "--trainer.master_addr $MLSWEEP_MASTER_ADDR "
    "--trainer.master_port $MLSWEEP_MASTER_PORT",
]

# 8 GPUs per node, 2 nodes = 16 GPUs total per run
GPUS_PER_RUN = 8
NODES_PER_RUN = 2

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

Run with a workers file that has at least 4 entries (2 nodes × 2 parallel runs):

```sh
mlsweep_run sweeps/rl_multinode.py --workers workers.toml
```

mlsweep sets `MLSWEEP_NODE_RANK=0` on the first worker and `MLSWEEP_NODE_RANK=1` on the second, along with `MLSWEEP_MASTER_ADDR` and `MLSWEEP_MASTER_PORT` so the two sides can rendezvous.

---

## TorchTitan: Multi-GPU Pretraining

[TorchTitan](https://github.com/pytorch/torchtitan) is launched with `torchrun -m torchtitan.train`, with a `--module` and `--config` identifying the model and a built-in config preset, plus `--key value` CLI overrides for any config field.

Set `GPUS_PER_RUN` to the number of GPUs for each training run and put `torchrun --nproc_per_node $GPUS_PER_RUN` in `COMMAND`. mlsweep sets `CUDA_VISIBLE_DEVICES` to the assigned devices, and `torchrun` discovers the process count from `--nproc_per_node`.

```python
#!/usr/bin/env mlsweep_run

GPUS_PER_RUN = 8

COMMAND = [
    "torchrun",
    "--nproc_per_node", str(GPUS_PER_RUN),
    "--rdzv_backend", "c10d",
    "--rdzv_endpoint", "localhost:0",
    "-m", "torchtitan.train",
    "--module", "llama3",
    "--config", "llama3_8b",
]

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
        "singular": True,   # find the largest that fits; skip the rest
    },
    ".warmup": {
        "values": [200, 500, 1000],
        "flags": "--lr_scheduler.warmup_steps",
        "name": "wu",
    },
    ".wd": {
        "values": [0.0, 0.1],
        "flags": "--optimizer.weight_decay",
        "name": "wd",
    },
}
```

With 16 GPUs you get 2 concurrent 8-GPU runs:

```sh
mlsweep_run sweeps/torchtitan_sweep.py -g 16
```

### Multi-node with torchrun

Set `NODES_PER_RUN` to the number of machines and use `bash -c` to build the `torchrun` invocation from the env vars mlsweep injects:

```python
#!/usr/bin/env mlsweep_run

GPUS_PER_RUN = 8
NODES_PER_RUN = 2

COMMAND = [
    "bash", "-c",
    "torchrun"
    " --nproc_per_node $GPUS_PER_NODE"
    " --nnodes $MLSWEEP_NNODES"
    " --node_rank $MLSWEEP_NODE_RANK"
    " --master_addr $MLSWEEP_MASTER_ADDR"
    " --master_port $MLSWEEP_MASTER_PORT"
    " -m torchtitan.train"
    " --module llama3 --config llama3_8b"
    ' "$@"',
    "--",
]

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
    ".warmup": {
        "values": [500, 1000],
        "flags": "--lr_scheduler.warmup_steps",
        "name": "wu",
    },
}
```

Run with a workers file that has at least 2 entries:

```sh
mlsweep_run sweeps/torchtitan_multinode.py --workers workers.toml
```

mlsweep launches the command on both workers simultaneously, with `MLSWEEP_NODE_RANK=0` on the first and `MLSWEEP_NODE_RANK=1` on the second. The shell expands the env vars, and `torchrun` handles the intra-node process launch. The run is recorded as succeeded only when both nodes exit 0.

> **Note:** For clusters with a job scheduler (Slurm, PBS), use the scheduler's multi-node launcher (`srun`, `mpirun`) instead, and submit mlsweep itself as a single-node job on the head node.
