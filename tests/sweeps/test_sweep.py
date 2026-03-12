#!/usr/bin/env mlsweep_run

"""Sweep config for tests/scripts/train.py.

Launches runs for all combinations of learning rates and batch sizes.
"""

COMMAND = ["python", "tests/scripts/train.py"]

OPTIONS = {
    ".lr": {
        "values": [1e-5, 5e-5, 2.5e-5, 1e-4, 1e-3],
        "flags": "--lr",
        "name": "lr",
    },
    ".batch_size": {
        "values": [16, 32, 64, 128, 256, 512],
        "flags": "--batch-size",
        "name": "bs",
    },
}
