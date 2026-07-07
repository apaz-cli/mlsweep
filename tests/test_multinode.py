"""Multi-node scheduling tests on a single machine.

Multi-node normally needs several physical hosts.  We make it testable on one
box with two tricks:

  1. Two local worker *processes* (the ``manager_with_two_workers`` fixture) act
     as two distinct nodes — the scheduler dispatches one node to each.
  2. A plain-socket rendezvous stands in for NCCL/torch: the per-node command
     reads the env the manager injects (NODE_RANK / NNODES / MASTER_ADDR /
     MASTER_PORT) and actually connects across "nodes".  This proves the wiring
     enables cross-node comms without needing GPUs.

Together these exercise the real multi-node dispatch + DB-backed result
aggregation path (the run finalises exactly once when every node reports).
"""

import os
import time

from conftest import _api_get, _wait_for_job
from mlsweep.run_sweep import (
    _http_request,
    _manager_url,
    manager_create_experiment,
    manager_submit_jobs_bulk,
)

_TOKEN = "test-token"

# Each node rendezvouses over MASTER_ADDR:MASTER_PORT (rank 0 listens, the rest
# connect), writes a per-rank marker, then optionally fails if MN_FAIL is set.
_RDV_SCRIPT = r"""
import os, socket, sys, time
rank = int(os.environ["MLSWEEP_NODE_RANK"])
nnodes = int(os.environ["MLSWEEP_NNODES"])
addr = os.environ["MLSWEEP_MASTER_ADDR"]
port = int(os.environ["MLSWEEP_MASTER_PORT"])
rdv = os.environ["RDV_DIR"]
if rank == 0:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(nnodes)
    s.settimeout(30)
    for _ in range(nnodes - 1):
        c, _a = s.accept(); c.recv(16); c.close()
    s.close()
else:
    deadline = time.time() + 30
    while True:
        try:
            c = socket.create_connection((addr, port), timeout=5)
            c.sendall(str(rank).encode()); c.close(); break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.2)
open(os.path.join(rdv, "rank_%d.done" % rank), "w").write("ok")
if os.environ.get("MN_FAIL") == str(rank):
    sys.exit(7)
"""


def _submit_multinode(url, exp, run_id, rdv_dir, extra_env=None):
    env = {"RDV_DIR": str(rdv_dir)}
    if extra_env:
        env.update(extra_env)
    jobs = [{
        "run_id": run_id,
        "experiment_id": exp,
        "command": ["python", "-c", _RDV_SCRIPT],
        "env": env,
        "nodes_per_run": 2,
        "gpus_per_run": 1,
        "files": {},
        "return_files": [],
    }]
    return manager_submit_jobs_bulk(url, _TOKEN, jobs)


def test_multinode_dispatch_and_rendezvous(manager_with_two_workers, tmp_path):
    """A 2-node job is placed on two workers, both nodes get the right env and
    rendezvous, and the single run finalises as 'done'."""
    server, url = manager_with_two_workers
    exp = "exp_mn_ok"
    manager_create_experiment(url, _TOKEN, exp, "mn_ok")

    rdv_dir = tmp_path / "rdv"
    rdv_dir.mkdir()

    assert _submit_multinode(url, exp, "mn", rdv_dir) is not None

    job = _wait_for_job(url, _TOKEN, "mn", exp, timeout=60)
    assert job is not None, "multi-node job did not finish"
    assert job["status"] == "done", f"unexpected status: {job['status']}"

    # Both nodes ran and completed the cross-node handshake.
    assert (rdv_dir / "rank_0.done").exists(), "node 0 did not run"
    assert (rdv_dir / "rank_1.done").exists(), "node 1 did not run"


def test_multinode_fails_if_any_node_fails(manager_with_two_workers, tmp_path):
    """If one node exits non-zero, the aggregated multi-node job is 'failed'
    (DB-backed aggregation: success is the AND across all node results)."""
    server, url = manager_with_two_workers
    exp = "exp_mn_fail"
    manager_create_experiment(url, _TOKEN, exp, "mn_fail")

    rdv_dir = tmp_path / "rdv"
    rdv_dir.mkdir()

    # Both nodes rendezvous; node 1 then exits non-zero.
    assert _submit_multinode(url, exp, "mn", rdv_dir, extra_env={"MN_FAIL": "1"}) is not None

    job = _wait_for_job(url, _TOKEN, "mn", exp, timeout=60)
    assert job is not None, "multi-node job did not finish"
    assert job["status"] == "failed", f"expected failed, got {job['status']}"
    # Both nodes still rendezvoused before node 1 failed.
    assert (rdv_dir / "rank_0.done").exists()
    assert (rdv_dir / "rank_1.done").exists()
