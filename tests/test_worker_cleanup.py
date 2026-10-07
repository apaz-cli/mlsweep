"""Tests for the worker's MsgCleanup handler, and for runs whose scratch
directory disappears under them, against a real worker process."""

import os
import socket
import subprocess
import sys
import time

import pytest

from mlsweep._shared import (
    MsgCleaned,
    MsgCleanup,
    MsgHello,
    MsgLog,
    MsgMetric,
    MsgResult,
    MsgRun,
    MsgWorkerHello,
    decode,
    encode,
    read_msg,
)

_TOKEN = "cleanup-token"


@pytest.fixture
def worker(tmp_path):
    """Start a worker on an ephemeral port; yield ``(scratch_dir, socket)`` after the hello."""
    yield from _worker(tmp_path, tmp_path / "scratch")


def _worker(tmp_path, scratch):
    """Start a worker given *scratch*; yield the scratch dir its hello reports, and
    a connected socket."""
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
        hello = _recv(sock, MsgWorkerHello)
        yield hello.scratch_dir, sock
        sock.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _recv(sock, cls):
    """Read messages until a *cls* arrives (skipping GPU stats, pings, ...)."""
    while True:
        payload = read_msg(sock)
        assert payload is not None, f"connection closed before {cls}"
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


# ── Failures after a run is accepted ───────────────────────────────────────────
#
# A run may fail in setup or after its processes exit, but the worker must still
# report a result; otherwise the job stays dispatched or running forever.  This
# happens, for one, when another worker sharing the scratch root (or someone
# cleaning /tmp) deletes the run's directory while it runs.


def _result_and_log(sock, run_id, timeout=30):
    """The run's MsgResult and log text.  GPU stats keep arriving whether or not
    a result ever does, so the wait is bounded by a deadline."""
    log = []
    deadline = time.time() + timeout
    while True:
        assert time.time() < deadline, f"no result for {run_id} within {timeout}s"
        payload = read_msg(sock)
        assert payload is not None, "connection closed before the result"
        msg = decode(payload)
        if isinstance(msg, MsgResult) and msg.run_id == run_id:
            return msg, "".join(log)
        if isinstance(msg, MsgLog) and msg.run_id == run_id:
            log.append(msg.data)


def test_setup_failure_is_reported_when_the_run_dir_is_gone(worker):
    scratch, sock = worker
    run_dir = os.path.join(scratch, "e", "setup_gone")
    sock.sendall(encode(MsgRun(
        command=["true"], run_id="setup_gone", experiment="e",
        setup_command=["sh", "-c", f"rm -rf {run_dir}; exit 3"])))
    result, log = _result_and_log(sock, "setup_gone")
    assert not result.success
    assert "run setup failed" in log


def test_result_is_reported_when_returning_files_fails(worker):
    """A return_files entry that cannot be copied is noted in the log; the run
    keeps its own exit status and its result is still sent."""
    _, sock = worker
    sock.sendall(encode(MsgRun(
        command=["true"], run_id="bad_return", experiment="e",
        files={"main.py": ""}, return_files=["../../escape.txt"])))
    result, log = _result_and_log(sock, "bad_return")
    assert result.success
    assert "could not return '../../escape.txt'" in log


def test_metrics_arrive_when_the_scratch_path_is_too_long_for_a_socket(tmp_path):
    """The loggers' unix socket normally lives in the scratch dir; a path past the
    socket address limit (about 108 bytes) must not cost the runs their metrics."""
    scratch = tmp_path / ("x" * 120)
    gen = _worker(tmp_path, scratch)
    _, sock = next(gen)
    try:
        sock.sendall(encode(MsgRun(
            command=[sys.executable, "-c",
                     "from mlsweep.logger import MLSweepLogger\n"
                     "with MLSweepLogger() as lg: lg.log({'loss': 1.5}, step=3)"],
            run_id="m", experiment="e")))
        deadline = time.time() + 30
        metrics = []
        while True:
            assert time.time() < deadline, "no result within 30s"
            msg = decode(read_msg(sock))
            if isinstance(msg, MsgMetric) and msg.run_id == "m":
                metrics.append((msg.step, msg.data))
            if isinstance(msg, MsgResult) and msg.run_id == "m":
                break
        assert msg.success and metrics == [(3, {"loss": 1.5})]
    finally:
        gen.close()


def test_two_workers_given_one_scratch_dir_keep_their_runs_apart(tmp_path):
    """The second worker on a scratch dir claims a subdirectory and reports it in
    its hello, so the manager looks for that worker's runs there."""
    scratch = tmp_path / "shared"
    first, second = _worker(tmp_path, scratch), _worker(tmp_path, scratch)
    try:
        dirs = [next(first)[0], next(second)[0]]
        assert dirs[0] == str(scratch)
        assert os.path.dirname(dirs[1]) == str(scratch) and os.path.basename(dirs[1]).startswith(".worker-")
    finally:
        first.close()
        second.close()

