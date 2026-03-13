#!/usr/bin/env mlsweep_run

import sys

COMMAND = [sys.executable, "tests/scripts/set_dist_env_train.py"]
GPUS_PER_RUN = 2
SET_DIST_ENV = True

OPTIONS = {
    ".seed": {"values": [42], "flags": "--seed", "name": "seed"},
}
