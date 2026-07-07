import getpass
import hashlib
import io
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import time
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
def ssh_localhost(tmp_path_factory: pytest.TempPathFactory):
    """Start a temporary SSH server on an ephemeral port, yield, tear down.

    Generates a fresh key pair and host key inside a temporary directory.
    No persistent SSH configuration is touched.
    """
    if not shutil.which("ssh"):
        pytest.skip("ssh not available")

    if not shutil.which("ssh-keygen"):
        pytest.skip("ssh-keygen not available")

    # Check that paramiko is importable (needed by ssh_server.py)
    try:
        import paramiko  # noqa: F401
    except ImportError:
        pytest.skip("paramiko not installed")

    # Check that the SSH server script exists
    server_script = Path(__file__).parent / "ssh_server.py"
    if not server_script.is_file():
        pytest.skip(f"SSH server script not found: {server_script}")

    tmp = tmp_path_factory.mktemp("sshd")
    key_path = tmp / "id_ed25519"
    host_key_path = tmp / "ssh_host_rsa_key"
    auth_keys_path = tmp / "authorized_keys"

    # Generate client key
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", "", "-q"],
        check=True,
        capture_output=True,
    )
    pub_key = (tmp / "id_ed25519.pub").read_text().strip()

    # Generate host key
    subprocess.run(
        ["ssh-keygen", "-t", "rsa", "-f", str(host_key_path), "-N", "", "-q"],
        check=True,
        capture_output=True,
    )
    (host_key_path).chmod(0o600)

    auth_keys_path.write_text(pub_key + "\n")

    port = _find_free_port()

    # Launch SSH server
    proc = subprocess.Popen(
        [
            sys.executable, str(server_script),
            "--port", str(port),
            "--host-key", str(host_key_path),
            "--authorized-keys", str(auth_keys_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    ready = _wait_for_ssh(port, key_path, timeout=15)
    if not ready:
        proc.terminate()
        proc.wait()
        out, _ = proc.communicate()
        pytest.fail(f"SSH server did not become ready.\nstdout:\n{out}")

    # Create temp SSH config so parsync (libssh2) knows the port
    ssh_dir = tmp / ".ssh"
    ssh_dir.mkdir()
    ssh_config = ssh_dir / "config"
    ssh_config.write_text(
        f"Host 127.0.0.1\n"
        f"    Port {port}\n"
        f"    IdentityFile {key_path}\n"
        f"    StrictHostKeyChecking no\n"
        f"    UserKnownHostsFile /dev/null\n"
        f"    BatchMode yes\n"
    )
    (ssh_dir / "known_hosts").write_text("")  # avoid warnings

    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(tmp)

    yield

    # Teardown
    os.environ["HOME"] = old_home or ""
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_ssh(port: int, key_path: Path, timeout: int = 15) -> bool:
    """Poll until SSH to 127.0.0.1:<port> succeeds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = subprocess.run(
            [
                "ssh",
                "-p", str(port),
                "-i", str(key_path),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=2",
                "127.0.0.1", "true",
            ],
            capture_output=True,
        )
        if r.returncode == 0:
            return True
        time.sleep(0.3)
    return False


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
    future = time.time() + 3600
    os.utime(dst / "a.txt", (future, future))

    user = getpass.getuser()
    subprocess.run(
        [parsync_bin(), "-rlu", f"{user}@127.0.0.1:{src}/", f"{dst}/"],
        capture_output=True,
        timeout=30,
    )
    assert (dst / "a.txt").read_text() == "existing"
