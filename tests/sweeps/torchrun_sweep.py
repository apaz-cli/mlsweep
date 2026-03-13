#!/usr/bin/env mlsweep_run

import sys

COMMAND = [
    sys.executable, "-m", "torch.distributed.run",
    "--standalone", "--nproc_per_node=2",
    "tests/scripts/dist_train.py",
]

OPTIONS = {
    ".seed": {"values": [42], "flags": "--seed", "name": "seed"},
}
