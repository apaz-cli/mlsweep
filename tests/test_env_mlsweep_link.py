"""A venv built for a run can import mlsweep (for MLSweepLogger) even if its dependencies do not list it."""
import subprocess
import sys
from pathlib import Path

from mlsweep import _env


def test_built_venv_imports_mlsweep(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    _env._link_mlsweep(str(venv))
    _env._link_mlsweep(str(venv))  # idempotent
    out = subprocess.run([str(venv / "bin" / "python"), "-c", "import mlsweep.logger, mlsweep; print(mlsweep.__file__)"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out.endswith("mlsweep/__init__.py")
