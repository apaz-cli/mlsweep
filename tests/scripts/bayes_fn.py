#!/usr/bin/env python3
"""Fast mathematical objective for bayes sweep tests.

Evaluates a known function of lr, wd, and activation type, with a clear
optimum at lr≈0.002, wd≈0.005, act=gelu. Used to verify that Bayesian
optimization converges toward the optimum and that singular/monotonic dims
behave correctly.

Batch-size OOM simulation:
  batch_size > 64  → exit 1 (simulates GPU OOM)
  batch_size <= 64 → succeeds (used to test singular skipping)
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlsweep.logger import MLSweepLogger


_LR_OPT  = 0.002    # optimum lr
_WD_OPT  = 0.005    # optimum wd
_ACT_BONUS = {"gelu": 0.0, "relu": 0.3, "silu": 0.6}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--wd",         type=float, default=1e-2)
    parser.add_argument("--act",        type=str,   default="gelu",
                        choices=["relu", "gelu", "silu"])
    parser.add_argument("--batch-size", type=int,   default=32)
    args = parser.parse_args()

    # Simulate OOM: fail loudly for oversized batches
    if args.batch_size > 64:
        print(f"[eval_fn] OOM: batch_size={args.batch_size} > 64", flush=True)
        sys.exit(1)

    # Compute val_loss: 2-D bowl in log-space + activation penalty
    lr_term = (math.log(args.lr / _LR_OPT)) ** 2
    wd_term = (math.log(args.wd / _WD_OPT)) ** 2
    act_bonus = _ACT_BONUS.get(args.act, 0.0)
    val_loss = round(lr_term + wd_term + act_bonus, 6)

    print(f"[eval_fn] lr={args.lr}  wd={args.wd}  act={args.act}"
          f"  batch_size={args.batch_size}  →  val_loss={val_loss}", flush=True)

    logger = MLSweepLogger()
    logger.log({"val_loss": val_loss, "lr_term": lr_term, "wd_term": wd_term}, step=1)
    logger.close()
    print("[eval_fn] done")


if __name__ == "__main__":
    main()
