"""Download and locate the bundled parsync binary."""

import hashlib
import io
import os
import platform
import stat
import tarfile
import urllib.request
from pathlib import Path

PARSYNC_VERSION = "0.2.0"

_BASE_URL = "https://github.com/AlpinDale/parsync/releases/download/v{ver}/{name}"

# (system, machine) -> (tarball filename, sha256)
_RELEASES: dict[tuple[str, str], tuple[str, str]] = {
    ("Linux",  "x86_64"):  ("parsync-v0.2.0-x86_64-linux.tar.gz",  "5716b5a5b0f4496f94d4190c8d14c5ae71c906f081e587f78c2bafde24db50aa"),
    ("Linux",  "aarch64"): ("parsync-v0.2.0-aarch64-linux.tar.gz", "7a8b1974f0e7e3218935f2f510ffc4b3e7761212dae77ca43556e5e56750023f"),
    ("Darwin", "x86_64"):  ("parsync-v0.2.0-x86_64-macos.tar.gz",  "a8968f61781dd05c717441f151558b93c80fcdcfefbe25f5405ff55322717e86"),
    ("Darwin", "arm64"):   ("parsync-v0.2.0-aarch64-macos.tar.gz", "132dcbce47f8f18eeace3c6cad9ae58fbc789e744fb88e33871eb544e0be8e42"),
}

_BIN_DIR = Path(__file__).parent / "_bin"


def fetch_parsync() -> None:
    """Download, verify, and install the parsync binary for the current platform.

    No-ops if the binary already exists. Raises RuntimeError on hash mismatch
    or if the platform is unsupported.
    """
    key = (platform.system(), platform.machine())
    if key not in _RELEASES:
        print(f"mlsweep: no parsync binary available for {key[0]}/{key[1]}, skipping")
        return
    filename, expected_sha256 = _RELEASES[key]
    dest = _BIN_DIR / "parsync"
    if dest.exists():
        return
    _BIN_DIR.mkdir(exist_ok=True)
    url = _BASE_URL.format(ver=PARSYNC_VERSION, name=filename)
    print(f"mlsweep: downloading parsync {PARSYNC_VERSION} for {key[0]}/{key[1]}...")
    with urllib.request.urlopen(url) as resp:
        data: bytes = resp.read()
    _verify_and_install(data, expected_sha256, dest)
    print("mlsweep: parsync installed")


def _verify_and_install(data: bytes, expected_sha256: str, dest: Path) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"parsync download integrity check failed\n"
            f"  expected: {expected_sha256}\n"
            f"  got:      {actual}"
        )
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        member = tf.getmember("parsync")
        f = tf.extractfile(member)
        assert f is not None
        binary = f.read()
    dest.write_bytes(binary)
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def parsync_bin() -> str:
    """Return the path to the bundled parsync binary.

    Raises RuntimeError if the binary has not been installed.
    """
    binary = _BIN_DIR / "parsync"
    if not binary.exists():
        raise RuntimeError("parsync binary not found; reinstall mlsweep")
    return str(binary)
