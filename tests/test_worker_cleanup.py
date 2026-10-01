"""Tests for the worker's MsgCleanup handler, against a real worker process."""

import os
import socket
import subprocess
import sys

import pytest

from mlsweep._shared import MsgCleaned, MsgCleanup, MsgHello, MsgWorkerHello, decode, encode, read_msg

_TOKEN = "cleanup-token"


@pytest.fixture
def worker(tmp_path):
    """Start a worker on an ephemeral port; yield ``(scratch_dir, socket)`` after the hello."""
    scratch = tmp_path / "scratch"
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlsweep.worker", "--port", "0",
         "--scratch-dir", str(scratch), "--token", _TOKEN, "--remote-dir", str(tmp_path)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        line = proc.stdout.readline()
        assert line.startswith("PORT="), line
        sock = socket.create_connection(("127.0.0.1", int(line[5:])), timeout=10)
        sock.sendall(encode(MsgHello(token=_TOKEN, controller_id="test")))
        _recv(sock, MsgWorkerHello)
        yield scratch, sock
        sock.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _recv(sock, cls):
    """Read messages until a *cls* arrives (skipping GPU stats, pings, ...)."""
    while True:
        payload = read_msg(sock)
        assert payload is not None, f"connection closed before {cls.__name__}"
        msg = decode(payload)
        if isinstance(msg, cls):
            return msg


def _cleanup(sock, experiment, run_id, final):
    sock.sendall(encode(MsgCleanup(run_id=run_id, experiment=experiment, final=final)))
    ack = _recv(sock, MsgCleaned)
    assert (ack.experiment, ack.run_id) == (experiment, run_id)


def _make_run_dir(scratch, experiment, run_id):
    d = os.path.join(scratch, experiment, run_id)
    os.makedirs(os.path.join(d, "artifacts"))
    open(os.path.join(d, "training.log"), "w").close()
    return d


def test_cleanup_final_deletes_scratch(worker):
    scratch, sock = worker
    run_dir = _make_run_dir(scratch, "exp_1", "run_1")
    _cleanup(sock, "exp_1", "run_1", final=True)
    assert not os.path.exists(run_dir)


def test_cleanup_midrun_keeps_scratch(worker):
    scratch, sock = worker
    run_dir = _make_run_dir(scratch, "exp_1", "run_1")
    _cleanup(sock, "exp_1", "run_1", final=False)
    assert os.path.isdir(run_dir)


def test_cleanup_rejects_traversal(worker, tmp_path):
    scratch, sock = worker
    outside = tmp_path / "outside"
    run_dir = _make_run_dir(tmp_path, "outside", "run_1")
    _cleanup(sock, "../outside", "run_1", final=True)
    assert os.path.isdir(outside)
    assert os.path.isdir(run_dir)
