#!/usr/bin/env python3
"""Training script to verify SET_DIST_ENV injection.

Reads RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from env
(set by mlsweep when SET_DIST_ENV=True) and initializes a gloo process
group to confirm all ranks are reachable. No GPU required.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.distributed as dist

from mlsweep.logger import MLSweepLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    assert local_rank == int(os.environ["MLSWEEP_GPU_RANK"]), (
        f"LOCAL_RANK={local_rank} != MLSWEEP_GPU_RANK={os.environ['MLSWEEP_GPU_RANK']}"
    )

    # MASTER_ADDR and MASTER_PORT are already in the environment; init_process_group
    # picks them up automatically via the default env:// init method.
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    src = torch.tensor([float(rank)])
    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, src)
    values = [t.item() for t in gathered]
    assert values == list(range(world_size)), f"allgather mismatch: {values}"

    print(f"rank={rank}/{world_size} local_rank={local_rank} seed={args.seed} gathered={values}", flush=True)

    with MLSweepLogger() as logger:
        logger.log({"world_size": float(world_size)}, step=1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
