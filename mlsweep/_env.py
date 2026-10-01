"""Default Python environment for runs that bring none.

A run whose workspace and remote_dir have no ``.venv`` gets one built from the
project's standard dependency declaration, the same thing ``pip install .`` reads:

  1. ``pyproject.toml`` with a ``[project]`` table
  2. ``requirements.txt``
  3. ``setup.py`` / ``setup.cfg``

Environments are not portable between machines, so each worker builds its own.
When the dependencies are a self-contained list (static ``[project].dependencies``,
or a ``requirements.txt`` with no local paths or includes), the venv holds only
those dependencies and is cached per machine under a hash of the list, the
interpreter and the platform. The project's own code is not installed into it:
it changes with every submission, and the worker already puts the workspace on
``PYTHONPATH``. Anything else (``setup.py``, dynamic dependencies, ``-e .``) is
installed per run into the run's scratch directory, with pip's own cache making
repeat downloads cheap.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from typing import Callable

try:
    import tomllib  # type: ignore[import-not-found]  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python < 3.11

MARKER = ".mlsweep-env"

# Runs a command with output streamed to the run's log; raises on failure.
RunCmd = Callable[[list[str], str], None]
Log = Callable[[str], None]


@dataclasses.dataclass
class EnvSpec:
    source: str               # the file the dependencies came from, for the log
    install: list[str]        # arguments to `pip install`
    cache_key: str | None     # None: not cacheable, build per run


def cache_root() -> str:
    """``$MLSWEEP_ENV_CACHE``, else ``$XDG_CACHE_HOME/mlsweep/envs`` (``~/.cache``)."""
    explicit = os.environ.get("MLSWEEP_ENV_CACHE")
    if explicit:
        return explicit
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "mlsweep", "envs")


def _key(source: str, deps: object) -> str:
    """Hash of everything that makes a cached env valid on this machine."""
    blob = json.dumps({
        "source": source,
        "deps": deps,
        "python": sys.version,
        "implementation": sys.implementation.name,
        "platform": sys.platform,
        "machine": platform.machine(),
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


_LOCAL_REQ_PREFIXES = ("-r", "-c", "-e", "--requirement", "--constraint", "--editable",
                       ".", "/", "~", "file:")


def _requirements_self_contained(text: str) -> bool:
    """False when a line refers to other files or local paths, whose contents the hash
    of this file alone would not capture."""
    return not any(line.split(" #", 1)[0].strip().startswith(_LOCAL_REQ_PREFIXES)
                   for line in text.splitlines())


def detect(project_dir: str) -> EnvSpec | None:
    """Find the project's dependency declaration; None when it has none."""
    pyproject = os.path.join(project_dir, "pyproject.toml")
    if os.path.isfile(pyproject):
        with open(pyproject, "rb") as f:
            project = tomllib.load(f).get("project")
        if project is not None:
            if "dependencies" in project.get("dynamic", []):
                return EnvSpec("pyproject.toml", [project_dir], None)
            deps = sorted(project.get("dependencies", []))
            return EnvSpec("pyproject.toml", deps, _key("pyproject", deps))

    requirements = os.path.join(project_dir, "requirements.txt")
    if os.path.isfile(requirements):
        with open(requirements) as f:
            text = f.read()
        key = _key("requirements", text) if _requirements_self_contained(text) else None
        return EnvSpec("requirements.txt", ["-r", requirements], key)

    for name in ("setup.py", "setup.cfg"):
        if os.path.isfile(os.path.join(project_dir, name)):
            return EnvSpec(name, [project_dir], None)
    return None


def _build(venv_dir: str, spec: EnvSpec, project_dir: str, run: RunCmd) -> None:
    run([sys.executable, "-m", "venv", venv_dir], project_dir)
    if spec.install:
        run([os.path.join(venv_dir, "bin", "python"), "-m", "pip", "install",
             "--disable-pip-version-check", *spec.install], project_dir)


def ensure(spec: EnvSpec, project_dir: str, run_dir: str, run: RunCmd, log: Log) -> str:
    """Return a venv satisfying *spec*, building it if needed.

    Cacheable specs share one venv per machine, built once under a file lock so
    concurrent runs (and workers on the same host) wait for a single build. A venv
    is complete only once its marker is written; a build that died part-way is
    discarded and redone. Uncacheable specs get a fresh venv in *run_dir*.
    """
    if spec.cache_key is None:
        venv_dir = os.path.join(run_dir, "venv")
        log(f"[mlsweep] no .venv found; building a per-run env from {spec.source}\n")
        _build(venv_dir, spec, project_dir, run)
        return venv_dir

    root = cache_root()
    os.makedirs(root, exist_ok=True)
    venv_dir = os.path.join(root, spec.cache_key)
    marker = os.path.join(venv_dir, MARKER)
    with open(venv_dir + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.isfile(marker):
            os.utime(marker)  # last-used time, for pruning
            log(f"[mlsweep] no .venv found; using cached env {venv_dir} ({spec.source})\n")
            return venv_dir
        if os.path.exists(venv_dir):
            shutil.rmtree(venv_dir)
        log(f"[mlsweep] no .venv found; building cached env {venv_dir} from {spec.source}\n")
        _build(venv_dir, spec, project_dir, run)
        with open(marker, "w") as f:
            json.dump({"source": spec.source, "install": spec.install,
                       "python": sys.version, "built": time.time()}, f)
    return venv_dir
