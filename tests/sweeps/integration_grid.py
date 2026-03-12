#!/usr/bin/env mlsweep_run

COMMAND = ["python", "tests/scripts/fast_train.py"]

OPTIONS = {
    ".lr": {"values": [1e-3, 1e-4], "flags": "--lr", "name": "lr"},
    ".bs": {"values": [32, 64], "flags": "--bs", "name": "bs"},
}
