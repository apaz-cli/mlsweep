"""UPLOAD_EXCLUDE / --exclude: name patterns match at any depth, path patterns match from the project root."""
from pathlib import Path

from mlsweep._sweep import load_sweep_file
from mlsweep._snapshot import project_files as _project_files


def _tree(root: Path) -> None:
    for f in ["a.py", "runs/x.pt", "yoco/runs/keep.txt", "data/big.bin", "data/small.txt", "sub/data/big.bin"]:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")


def _rel(root: Path, excludes: set[str]) -> set[str]:
    return {p.relative_to(root).as_posix() for p in _project_files(root, excludes)}


def test_name_pattern_matches_any_depth(tmp_path: Path) -> None:
    _tree(tmp_path)
    got = _rel(tmp_path, {"runs"})
    assert "runs/x.pt" not in got and "yoco/runs/keep.txt" not in got and "a.py" in got


def test_path_pattern_matches_from_root(tmp_path: Path) -> None:
    _tree(tmp_path)
    got = _rel(tmp_path, {"/runs", "data/*.bin"})
    assert "runs/x.pt" not in got and "data/big.bin" not in got
    assert {"yoco/runs/keep.txt", "data/small.txt", "sub/data/big.bin", "a.py"} <= got


def test_sweep_file_upload_exclude(tmp_path: Path) -> None:
    f = tmp_path / "s.py"
    f.write_text('COMMAND = ["python", "t.py"]\nOPTIONS = {}\nUPLOAD_EXCLUDE = ["/runs", "*.pt"]\n')
    assert load_sweep_file(f)["upload_exclude"] == ["/runs", "*.pt"]
    f.write_text('COMMAND = ["python", "t.py"]\nOPTIONS = {}\n')
    assert load_sweep_file(f)["upload_exclude"] == []
