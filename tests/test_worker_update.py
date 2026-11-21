"""Which workers the manager flags as running older mlsweep code.

A source checkout changes code without changing its version or protocol, so
each worker reports a fingerprint of its worker modules in its hello, and the
manager compares it with those modules on disk, which is what an update
installs.  The hello tests drive ``_handle_worker_hello`` against a real
SQLite database.
"""

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import aiosqlite

from mlsweep import _manager_workers as mw
from mlsweep._manager_db import DbWriter, init_db, upsert_worker
from mlsweep._manager_state import ManagerState, WorkerConn
from mlsweep._shared import WORKER_MODULES, PROTOCOL_VERSION, MsgWorkerHello, code_fingerprint


def _outdated_after_hello(tmp_path, host: str, code: str, on_disk: str | None = None) -> bool:
    """Whether the worker is listed as outdated after a hello reporting *code*,
    with *on_disk* as the worker code on disk by then (default: as it is)."""
    async def body():
        db = await aiosqlite.connect(tmp_path / "manager.db")
        await init_db(db)
        writer = DbWriter(db)
        writer_task = asyncio.create_task(writer.run())
        try:
            await upsert_worker(db, worker_id="w", host=host, remote_dir="", status="offline")
            state = ManagerState()
            state.db_writer = writer
            wc = WorkerConn(worker_id="w", host=host, port=0)
            state.workers["w"] = wc
            hello = MsgWorkerHello(gpus=[0], topo={}, resuming=[], scratch_dir="/s",
                                   protocol=PROTOCOL_VERSION, code=code)
            await mw._handle_worker_hello(db, state, wc, wc.conn_gen, hello)
            assert wc.status == "connected"
            return mw.worker_outdated(state, "w", on_disk)
        finally:
            writer_task.cancel()
            await db.close()

    return asyncio.run(body())


def test_a_remote_worker_without_a_fingerprint_is_outdated(tmp_path):
    assert _outdated_after_hello(tmp_path, "user@gpu1", "")


def test_a_remote_worker_with_other_code_is_outdated(tmp_path):
    assert _outdated_after_hello(tmp_path, "user@gpu1", "0123456789abcdef")


def test_a_remote_worker_with_the_code_on_disk_is_current(tmp_path):
    assert not _outdated_after_hello(tmp_path, "user@gpu1", code_fingerprint())


def test_a_worker_code_edit_after_the_hello_marks_the_worker_outdated(tmp_path):
    """The comparison is with the code on disk when listed, not with what the manager
    loaded: an update installs the code on disk, so it must clear the flag."""
    assert _outdated_after_hello(tmp_path, "user@gpu1", code_fingerprint(), on_disk="edited-since")


def test_only_worker_modules_change_the_fingerprint(tmp_path):
    pkg = tmp_path / "mlsweep"
    shutil.copytree(Path(mw.__file__).parent, pkg, ignore=shutil.ignore_patterns("_wheels", "__pycache__"))
    before = code_fingerprint(pkg)
    (pkg / "_notes.py").write_text("# edited\n")
    (pkg / "_manager_http.py").write_text("# edited\n")
    assert code_fingerprint(pkg) == before
    with open(pkg / "worker.py", "a") as f:
        f.write("# edited\n")
    assert code_fingerprint(pkg) != before


def test_worker_modules_are_what_the_worker_imports():
    out = subprocess.run(
        [sys.executable, "-c", "import sys, mlsweep.worker; "
         "print(' '.join(m for m in sys.modules if m.startswith('mlsweep')))"],
        capture_output=True, text=True, check=True).stdout.split()
    loaded = {"__init__.py" if m == "mlsweep" else m.split(".", 1)[1] + ".py" for m in out}
    assert loaded <= set(WORKER_MODULES)
    assert set(WORKER_MODULES) - loaded == {"logger.py"}  # imported by runs, not the worker


def test_a_local_worker_is_never_flagged(tmp_path):
    """It runs the manager's own install; updating it could not change anything."""
    assert not _outdated_after_hello(tmp_path, "localhost", "")


def test_retiring_a_worker_closes_its_tunnel(tmp_path):
    """An update relaunches the worker with a new tunnel on the same remote port,
    which the old tunnel would keep holding."""
    class Tunnel:
        returncode = None
        terminated = False

        def terminate(self):
            self.terminated = True

    async def body():
        db = await aiosqlite.connect(tmp_path / "manager.db")
        await init_db(db)
        writer = DbWriter(db)
        writer_task = asyncio.create_task(writer.run())
        try:
            await upsert_worker(db, worker_id="w", host="user@gpu1", remote_dir="")
            state = ManagerState()
            state.db_writer = writer
            tunnel = Tunnel()
            state.workers["w"] = WorkerConn(worker_id="w", host="user@gpu1", port=0,
                                            status="connected", tunnel_proc=tunnel)
            await mw.declare_worker_dead(db, state, "w", "", shutdown=True)
            return tunnel.terminated, state.workers["w"].tunnel_proc
        finally:
            writer_task.cancel()
            await db.close()

    assert asyncio.run(body()) == (True, None)


def test_a_worker_that_fails_to_launch_is_listed_and_not_flagged(tmp_path):
    """A worker from the workers file has no row until something creates one.
    One that cannot start must still be listed, with the reason, and a failed
    ssh or install is not a sign of old code."""
    async def failing_launch(**kwargs):
        raise mw.BootstrapError("copying the wheel failed on user@gpu1: Connection refused")

    async def body():
        db = await aiosqlite.connect(tmp_path / "manager.db")
        await init_db(db)
        writer = DbWriter(db)
        writer_task = asyncio.create_task(writer.run())
        try:
            state = ManagerState()
            state.db_writer = writer
            state.reserve_worker_id("user@gpu1:7890")
            wc = await mw.connect_single_worker(
                db, state, "user@gpu1", "/proj", worker_id="user@gpu1:7890", port=7890,
                launch=failing_launch)
            record = await mw.get_worker(db, "user@gpu1:7890")
            return wc, record, mw.worker_outdated(state, "user@gpu1:7890")
        finally:
            writer_task.cancel()
            await db.close()

    wc, record, outdated = asyncio.run(body())
    assert wc is None
    assert record is not None and record.status == "dead"
    assert "Connection refused" in (record.last_error or "")
    assert not outdated
