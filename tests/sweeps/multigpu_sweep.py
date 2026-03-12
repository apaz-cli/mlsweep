#!/usr/bin/env mlsweep_run

COMMAND = ["python", "tests/scripts/fast_train.py"]
GPUS_PER_RUN = 2

OPTIONS = {
    ".lr": {"values": [1e-3, 1e-4], "flags": "--lr", "name": "lr"},
}
