#!/usr/bin/env python3
"""Fast mathematical objective for bayes sweep tests.

Evaluates a known function of lr, wd, and activation type, with a clear
optimum at lr≈0.002, wd≈0.005, act=gelu. Used to verify that Bayesian
optimization converges toward the optimum and that singular/monotonic dims
behave correctly.

Simulates a multi-step training run so the web UI shows loss curves.
Both train_loss and val_loss are logged at each step; they converge
exponentially to the analytical optimum (determined by lr, wd, act).

Batch-size OOM simulation:
  batch_size > 64  → exit 1 (simulates GPU OOM)
  batch_size <= 64 → succeeds (used to test singular skipping)
"""

import argparse
import math
import random
import sys

from mlsweep.logger import MLSweepLogger


_LR_OPT    = 0.002
_WD_OPT    = 0.005
_ACT_BONUS = {"gelu": 0.0, "relu": 0.3, "silu": 0.6}

N_STEPS = 40


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
        print(f"[bayes_fn] OOM: batch_size={args.batch_size} > 64", flush=True)
        sys.exit(1)

    # Analytical final loss: 2-D bowl in log-space + activation penalty
    lr_term   = (math.log(args.lr / _LR_OPT)) ** 2
    wd_term   = (math.log(args.wd / _WD_OPT)) ** 2
    act_bonus = _ACT_BONUS.get(args.act, 0.0)
    val_loss_final = round(lr_term + wd_term + act_bonus, 6)

    print(
        f"[bayes_fn] lr={args.lr}  wd={args.wd}  act={args.act}"
        f"  batch_size={args.batch_size}  →  val_loss_final={val_loss_final}",
        flush=True,
    )

    # Deterministic noise seeded from hyperparams so the same combo always
    # produces the same curve shape.
    rng = random.Random(int(args.lr * 1e7) ^ int(args.wd * 1e9) ^ hash(args.act))

    # Simulate convergence from a "random init" loss level.
    # initial must always exceed final; use multiplicative scaling for large
    # finals (bad params) so the curve always goes downward.
    initial_val   = max(val_loss_final * 2.0, val_loss_final + 3.0)
    initial_train = initial_val * 0.92
    tau = N_STEPS / 3.5   # exponential decay time constant

    with MLSweepLogger() as logger:
        for step in range(1, N_STEPS + 1):
            decay = math.exp(-step / tau)

            noise_scale = decay * 0.12 + 0.005   # noise shrinks as training stabilises
            val_noise   = rng.gauss(0, noise_scale)
            train_noise = rng.gauss(0, noise_scale * 0.6)

            val_loss   = val_loss_final + (initial_val   - val_loss_final) * decay + val_noise
            train_loss = val_loss_final + (initial_train - val_loss_final) * decay + train_noise

            logger.log(
                {
                    "train_loss": max(1e-6, train_loss),
                    "val_loss":   max(1e-6, val_loss),
                    "lr_term":    lr_term,
                    "wd_term":    wd_term,
                },
                step=step,
            )

    print("[bayes_fn] done")


if __name__ == "__main__":
    main()
