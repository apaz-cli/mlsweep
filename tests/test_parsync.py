import getpass
import hashlib
import io
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

import mlsweep._parsync as _parsync
from mlsweep._parsync import fetch_parsync, parsync_bin


def _make_tarball(binary_content: bytes = b"fake-parsync-binary") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="parsync")
        info.size = len(binary_content)
        tf.addfile(info, io.BytesIO(binary_content))
    return buf.getvalue()


def test_verify_and_install_success(tmp_path: Path) -> None:
    binary_content = b"fake-parsync-binary"
    data = _make_tarball(binary_content)
    expected = hashlib.sha256(data).hexdigest()
    dest = tmp_path / "parsync"

    _parsync._verify_and_install(data, expected, dest)

    assert dest.read_bytes() == binary_content
    assert dest.stat().st_mode & stat.S_IXUSR


def test_verify_and_install_hash_mismatch(tmp_path: Path) -> None:
    data = _make_tarball()
    with pytest.raises(RuntimeError, match="integrity check failed"):
        _parsync._verify_and_install(data, "0" * 64, tmp_path / "parsync")


def test_fetch_parsync_installs_executable() -> None:
    fetch_parsync()
    binary = Path(parsync_bin())
    assert binary.exists()
    assert binary.stat().st_mode & stat.S_IXUSR
    result = subprocess.run([str(binary), "--help"], capture_output=True, timeout=10)
    assert b"parsync" in (result.stdout + result.stderr).lower()


@pytest.fixture(scope="module")
def ssh_localhost() -> None:
    """Skip if we can't SSH to localhost."""
    if not shutil.which("ssh"):
        pytest.skip("ssh not available")
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "127.0.0.1", "true"],
        capture_output=True,
    )
    if r.returncode != 0:
        pytest.skip("cannot SSH to localhost (no agent/key configured)")


def test_parsync_transfers_files(tmp_path: Path, ssh_localhost: None) -> None:
    fetch_parsync()

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    (src / "file1.txt").write_text("hello")
    (src / "subdir").mkdir()
    (src / "subdir" / "file2.txt").write_text("world")

    user = getpass.getuser()
    result = subprocess.run(
        [parsync_bin(), "-rlu", f"{user}@127.0.0.1:{src}/", f"{dst}/"],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert (dst / "file1.txt").read_text() == "hello"
    assert (dst / "subdir" / "file2.txt").read_text() == "world"


def test_parsync_skips_existing_files(tmp_path: Path, ssh_localhost: None) -> None:
    fetch_parsync()

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    (src / "a.txt").write_text("new")
    (dst / "a.txt").write_text("existing")
    # Touch dst file so it appears newer — parsync -u skips files newer on receiver
    import os, time
    future = time.time() + 3600
    os.utime(dst / "a.txt", (future, future))

    user = getpass.getuser()
    subprocess.run(
        [parsync_bin(), "-rlu", f"{user}@127.0.0.1:{src}/", f"{dst}/"],
        capture_output=True,
        timeout=30,
    )
    assert (dst / "a.txt").read_text() == "existing"
