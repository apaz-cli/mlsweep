"""The worker's first message on a connection must be its MsgWorkerHello.

The manager drops a connection whose first message is anything else.  Building
the hello can take seconds (nvidia-smi topology, device probes), and the GPU
stats thread and run traffic used to reach a connection during that window, so
on slow hosts every connection was dropped and the worker relaunched forever.

This runs a real worker whose ``nvidia-smi`` (first on its PATH) answers the
topology query only after the stats thread has broadcast at least once.
"""
import os
import socket
import subprocess
import sys
import textwrap

import pytest

from conftest import _find_free_port
from mlsweep._shared import MsgGpuStats, MsgHello, MsgWorkerHello, decode, encode, read_msg

STATS_INTERVAL = 5.0  # the worker's _gpu_stats_thread period

FAKE_NVIDIA_SMI = textwrap.dedent("""\
    #!/bin/sh
    case "$*" in
      *topo*) sleep {topo_seconds}; exit 0 ;;
      *--query-gpu=index,utilization.gpu*) echo "0, 0, 0, 1, Fake GPU"; exit 0 ;;
    esac
    exit 1
""")


@pytest.mark.xdist_group("timing")
def test_stats_and_run_traffic_wait_for_hello(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text(FAKE_NVIDIA_SMI.format(topo_seconds=STATS_INTERVAL + 2))
    smi.chmod(0o755)
    port = _find_free_port()
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "CUDA_VISIBLE_DEVICES": "0"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlsweep.worker", "--token", "tok", "--port", str(port), "--no-device-probe",
         "--scratch-dir", str(tmp_path / "scratch"), "--remote-dir", str(tmp_path)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert proc.stdout is not None and proc.stdout.readline().strip() == f"PORT={port}"
        mgr = socket.create_connection(("127.0.0.1", port), timeout=30)
        mgr.sendall(encode(MsgHello(token="tok", controller_id="manager")))
        # The hello waits on the slow topology query, while the stats thread
        # broadcasts; the hello still comes first.
        assert isinstance(decode(read_msg(mgr)), MsgWorkerHello)
        # After the hello, stats flow as before.
        mgr.settimeout(STATS_INTERVAL * 3)
        assert isinstance(decode(read_msg(mgr)), MsgGpuStats)
        mgr.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
