#!/usr/bin/env python3
"""Minimal distributed training script for torchrun integration tests.

Uses the gloo backend so no GPU is required.  Sets MLSWEEP_GPU_RANK from
LOCAL_RANK before constructing the logger so that only rank 0 logs metrics.
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

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["MLSWEEP_GPU_RANK"] = str(local_rank)

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    src = torch.tensor([float(rank)])
    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, src)
    values = [t.item() for t in gathered]

    # Every rank 0..world_size-1 must have contributed exactly its index
    assert values == list(range(world_size)), f"allgather mismatch: {values}"

    print(f"rank={rank}/{world_size} seed={args.seed} gathered={values}", flush=True)

    with MLSweepLogger() as logger:
        logger.log({"world_size": float(world_size)}, step=1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
