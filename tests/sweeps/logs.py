#!/usr/bin/env mlsweep_run
"""Log viewer test sweep — runs log_train.py with varied lr and bs.

Each run outputs rich ANSI-colored terminal output (gradients, progress bars,
colored metrics) to exercise the Logs page in the web UI.

With 2 × 2 = 4 runs at ~18 epochs each, the full sweep completes in roughly
2 minutes on a single worker (or ~30 s if two runs are dispatched at once).
"""

COMMAND = ["python", "tests/scripts/log_train.py"]

OPTIONS = {
    ".lr": {
        "values": [1e-3, 5e-3],
        "flags":  "--lr",
        "name":   "lr",
    },
    ".bs": {
        "values": [32, 128],
        "flags":  "--bs",
        "name":   "bs",
    },
}
