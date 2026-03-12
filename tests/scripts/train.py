#!/usr/bin/env python3
"""Simulated training script for sweep tests.

Accepts --lr and --batch-size, runs 1000 fake training steps, and logs
metrics via MLSweepLogger. Reads run dir/name/experiment from the env
vars set by mlsweep_run; falls back to sensible defaults for standalone use.
"""

import os
import argparse
import math
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlsweep.logger import MLSweepLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    logger = MLSweepLogger(hparams=vars(args))

    print(f"[fake_train] lr={args.lr}  batch_size={args.batch_size}")

    for step in range(1, 1001):
        base_loss = math.exp(-args.lr * 100 * step) + 0.5 / args.batch_size
        loss = round(base_loss, 4)
        acc = round(1.0 - loss * 0.6, 4)
        print(f"step={step}  loss={loss:.4f}  acc={acc:.4f}", flush=True)
        logger.log({"loss": loss, "acc": acc}, step=step)
        time.sleep(0.05)

    logger.close()
    print("[fake_train] done")


if __name__ == "__main__":
    main()
