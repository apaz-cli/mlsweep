#!/usr/bin/env python3
"""Minimal CPU training script for integration tests.

Accepts --lr and --bs, logs one metric step, exits immediately.
Prints CUDA_VISIBLE_DEVICES so tests can verify GPU assignment.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from mlsweep.logger import MLSweepLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=32)
    args = parser.parse_args()

    loss = round(math.log(1 / args.lr) / 10 + 32 / args.bs, 4)

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"loss={loss}")

    with MLSweepLogger() as logger:
        logger.log({"loss": loss}, step=1)


if __name__ == "__main__":
    main()
