#!/usr/bin/env mlsweep_run

"""Artifact test sweep — generates PNG plots, JSON results, and CSV metrics per run.

COMMAND paths are relative to the git repo root, which is always packed into
the artifact tarball.  When using ``--project-dir``, point it at the repo root.
"""

import os as _os

_SCRIPT = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "scripts", "artifact_train.py"
)
# Resolve and strip to a relative path from the repo root so it still
# resolves inside the extracted artifact workspace on the worker.
_REPO_REL = "tests/scripts/artifact_train.py"

COMMAND = ["python", _REPO_REL]

OPTIONS = {
    ".lr": {"values": [1e-4, 1e-3, 1e-2], "flags": "--lr", "name": "lr"},
    ".bs": {"values": [32, 128, 512], "flags": "--bs", "name": "bs"},
}

# Validate on load: the training script must exist at the path computed
# from __file__ so we catch CWD issues early.
if not _os.path.exists(_SCRIPT):
    raise RuntimeError(
        f"Training script not found at {_SCRIPT!r}. "
        "Run from the git repo root, or set --project-dir to the repo root."
    )
