#!/usr/bin/env mlsweep_run
"""Bayesian sweep over eval_fn.py — tests Bayes mode, continuous dims, and singular dims.

Known optimum: lr≈0.002, wd≈0.005, act=gelu  →  val_loss≈0.0

Singular dim (batch_size):
  eval_fn.py exits 1 for batch_size > 64, simulating GPU OOM.
  With values [256, 128, 64, 32] and singular=True, the sweep will probe 256
  (fail), 128 (fail), 64 (succeed), then skip 32 for every subsequent
  lex combo — demonstrating that the optimizer only sees the lex dims.

Budget of 12 means 12 successful lex evaluations (each at batch_size=64).
"""

COMMAND = ["tests/scripts/bayes_fn.py"]

OPTIMIZE = {
    "method": "bayes",
    "metric": "val_loss",
    "goal": "minimize",
    "budget": 12,
    "n_initial": 4,
}

OPTIONS = {
    # Singular: finds the largest batch size that fits in memory.
    # eval_fn.py fails (exit 1) for batch_size > 64.
    ".batch_size": {
        "values": [256, 128, 64, 32],
        "flags": "--batch-size",
        "name": None,           # hardware detail — omit from run name
        "singular": True,
    },
    # Continuous dims: optimizer samples in log-space
    ".lr": {
        "distribution": "log_uniform",
        "min": 1e-5,
        "max": 1e-1,
        "flags": "--lr",
        "name": "lr",
    },
    ".wd": {
        "distribution": "log_uniform",
        "min": 1e-6,
        "max": 1e-1,
        "flags": "--wd",
        "name": "wd",
    },
    # Discrete dim: categorical sampling
    ".act": {
        "values": ["relu", "gelu", "silu"],
        "flags": "--act",
        "name": "act",
    },
}
