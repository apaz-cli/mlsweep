"""Scalability of experiment logging: log and metric volume, many concurrent
runs, and large sweeps.

Each test pushes a realistic-to-heavy volume through a real manager and
worker and checks that
  * everything is stored, exactly once, and reads back quickly;
  * storage stays compact (logs are compressed, metrics packed per attempt);
  * the manager stays responsive while it ingests the stream.

Thresholds are generous (several times what a laptop needs) so the tests only
fail on a real regression, such as quadratic work or a blocked event loop.
"""

import json
import sqlite3
import statistics
import sys
import threading
import time
import urllib.error

import pytest

from cluster_harness import TOKEN, Cluster, wait_until
from conftest import _api_get, _api_post
from test_reconnect import _Proxy, proxied  # noqa: F401  (proxied is a fixture)

# The latency and duration bounds assume a machine that is not also running
# the other heavy tests here; under `pytest -n` the group shares one worker.
pytestmark = pytest.mark.xdist_group("timing")

PAD = "x" * 40


@pytest.fixture
def cluster(tmp_path):
    c = Cluster(tmp_path, [{"devices": [0], "jobs": 0}])
    yield c
    c.close()


def _printer(n: int, tag: str = "") -> list[str]:
    """A command printing *n* numbered lines as fast as it can."""
    return [sys.executable, "-c",
            f"import sys\nw = sys.stdout.write\n"
            f"for i in range({n}): w(f'{tag}line {{i}} {PAD}\\n')\nsys.stdout.flush()"]


def _metric_logger(n: int, every: float = 0.0) -> list[str]:
    """A command logging *n* steps of five metrics through MLSweepLogger."""
    return [sys.executable, "-c",
            "import time\nfrom mlsweep.logger import MLSweepLogger\n"
            "with MLSweepLogger() as log:\n"
            f"    for i in range({n}):\n"
            "        log.log({'loss': 1.0 / (i + 1), 'acc': i / 1e6, 'lr': 3e-4, 'gn': 0.5, 'tok': i * 512}, step=i)\n"
            f"        if {every}: time.sleep({every})\n"]


class _Latency:
    """Probe the manager's API from a background thread while a test runs."""

    def __init__(self, url: str, path: str = "/api/health"):
        self.url, self.path = url, path
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            _api_get(self.url, TOKEN, self.path)
            self.samples.append(time.perf_counter() - t0)
            time.sleep(0.05)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    @property
    def p95(self) -> float:
        return statistics.quantiles(self.samples, n=20)[-1]

    @property
    def worst(self) -> float:
        return max(self.samples)


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _db(c: Cluster) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{c.tmp / 'm.db'}?mode=ro", uri=True)


def _job_key(c: Cluster, eid: str, rid: str) -> int:
    with _db(c) as db:
        return db.execute("SELECT job_key FROM jobs WHERE experiment_id = ? AND run_id = ?",
                          (eid, rid)).fetchone()[0]


def _stored_bytes(c: Cluster, table: str, job_key: int) -> tuple[int, int]:
    """(rows, bytes of data) stored for a job in *table*."""
    with _db(c) as db:
        return db.execute(f"SELECT COUNT(*), COALESCE(SUM(LENGTH(data)), 0) FROM {table} "
                          "WHERE job_key = ?", (job_key,)).fetchone()


def _run_lines(text: str) -> list[str]:
    """A run log's lines from the job itself, without the worker's ``[mlsweep]`` notices."""
    return [ln for ln in text.splitlines() if not ln.startswith("[mlsweep] ")]


def _metrics(c: Cluster, eid: str, rid: str) -> list[dict]:
    """A run's metric rows (the endpoint serves JSONL, and 404 before the first one)."""
    try:
        text = c.get_text(f"/api/experiments/{eid}/jobs/{rid}/metrics")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise
    return [json.loads(line) for line in text.splitlines()]


def _submit(c: Cluster, eid: str, rid: str, command: list[str]):
    _api_post(c.url, TOKEN, "/api/jobs", {"run_id": rid, "experiment_id": eid,
                                           "gpus_per_run": 0, "command": command})


# ── Logs ──────────────────────────────────────────────────────────────────────


def test_a_very_chatty_run_is_stored_whole_compact_and_fast_to_read(cluster):
    c = cluster
    n = 200_000  # ~10 MB of output
    eid = c.experiment("chatty")
    with _Latency(c.url) as lat:
        _submit(c, eid, "r", _printer(n))
        wait_until(lambda: c.job(eid, "r")["status"] == "done", 60, "r to finish")
        time.sleep(1)  # keep probing while the tail is stored

    text, read_s = _timed(lambda: c.get_text(f"/api/experiments/{eid}/jobs/r/logs"))
    lines = _run_lines(text)
    assert len(lines) == n
    assert lines == [f"line {i} {PAD}" for i in range(n)]
    raw = len(text.encode())
    rows, stored = _stored_bytes(c, "logs", _job_key(c, eid, "r"))
    print(f"\nchatty: {raw/1e6:.1f} MB in {rows} chunks, {stored/1e6:.2f} MB stored; "
          f"read {read_s:.2f}s; api p95 {lat.p95*1e3:.0f} ms, worst {lat.worst*1e3:.0f} ms")
    assert read_s < 5
    assert stored < raw / 4  # compressed
    assert lat.p95 < 0.25 and lat.worst < 2


def test_many_chatty_runs_at_once_keep_their_logs_separate_and_whole(cluster):
    c = cluster
    runs, n = 8, 40_000
    eid = c.experiment("crowd")
    with _Latency(c.url) as lat:
        for i in range(runs):
            _submit(c, eid, f"r{i}", _printer(n, tag=f"r{i} "))
        c.wait_statuses(eid, {"done": runs}, timeout=90)
    for i in range(runs):
        lines = _run_lines(c.get_text(f"/api/experiments/{eid}/jobs/r{i}/logs"))
        assert lines == [f"r{i} line {k} {PAD}" for k in range(n)], f"r{i}"
    print(f"\ncrowd: api p95 {lat.p95*1e3:.0f} ms, worst {lat.worst*1e3:.0f} ms")
    assert lat.p95 < 0.25 and lat.worst < 2


def test_a_chatty_run_survives_a_dropped_connection_with_its_log_whole(proxied, tmp_path):
    url, proxy = proxied
    n = 150_000
    cmd = [sys.executable, "-c",
           f"import sys, time\nw = sys.stdout.write\n"
           f"for i in range({n}):\n"
           f"    w(f'line {{i}} {PAD}\\n')\n"
           f"    if i % 5000 == 0: sys.stdout.flush(); time.sleep(0.05)\n"]
    _api_post(url, TOKEN, "/api/jobs", {"run_id": "big", "experiment_id": "rc",
                                        "gpus_per_run": 0, "command": cmd})
    wait_until(lambda: _api_get(url, TOKEN, "/api/jobs/big?experiment_id=rc")["status"] == "running",
               20, "big to start")
    time.sleep(0.5)
    proxy.sever()
    wait_until(lambda: _api_get(url, TOKEN, "/api/jobs/big?experiment_id=rc")["status"] == "done",
               90, "big to finish")
    c = Cluster.__new__(Cluster)
    c.url = url
    lines = _run_lines(c.get_text("/api/experiments/rc/jobs/big/logs"))  # complete once done
    assert len(lines) == n
    assert lines == [f"line {i} {PAD}" for i in range(n)]


# ── Metrics ───────────────────────────────────────────────────────────────────


def test_many_metric_steps_are_stored_packed_and_read_back_fast(cluster):
    c = cluster
    n = 50_000
    eid = c.experiment("metrics")
    with _Latency(c.url) as lat:
        _submit(c, eid, "r", _metric_logger(n))
        wait_until(lambda: c.job(eid, "r")["status"] == "done", 90, "r to finish")
    rows, read_s = _timed(lambda: _metrics(c, eid, "r"))
    assert [r["step"] for r in rows] == list(range(n))
    assert rows[-1]["tok"] == (n - 1) * 512
    assert rows[9]["loss"] == pytest.approx(0.1)
    stored_rows, stored = _stored_bytes(c, "metrics", _job_key(c, eid, "r"))
    print(f"\nmetrics: {n} steps -> {stored_rows} row(s), {stored/1e6:.2f} MB; read {read_s:.2f}s; "
          f"api p95 {lat.p95*1e3:.0f} ms, worst {lat.worst*1e3:.0f} ms")
    assert stored_rows == 1  # packed when the attempt finished
    assert stored < n * 20
    assert read_s < 5
    assert lat.p95 < 0.25 and lat.worst < 2


def test_metrics_of_a_running_job_are_readable_while_it_logs(cluster):
    c = cluster
    eid = c.experiment("live")
    _submit(c, eid, "r", _metric_logger(2000, every=0.002))
    wait_until(lambda: len(_metrics(c, eid, "r")) > 200, 30,
               "metrics to stream in")
    assert c.job(eid, "r")["status"] == "running"
    wait_until(lambda: c.job(eid, "r")["status"] == "done", 60, "r to finish")
    assert [r["step"] for r in _metrics(c, eid, "r")] == list(range(2000))


def test_many_runs_logging_metrics_at_once_lose_nothing(cluster):
    c = cluster
    runs, n = 8, 10_000
    eid = c.experiment("mcrowd")
    with _Latency(c.url) as lat:
        for i in range(runs):
            _submit(c, eid, f"r{i}", _metric_logger(n))
        c.wait_statuses(eid, {"done": runs}, timeout=120)
    for i in range(runs):
        steps = [r["step"] for r in _metrics(c, eid, f"r{i}")]
        assert steps == list(range(n)), f"r{i}"
    print(f"\nmcrowd: api p95 {lat.p95*1e3:.0f} ms, worst {lat.worst*1e3:.0f} ms")
    assert lat.p95 < 0.25 and lat.worst < 2


# ── Large sweeps ──────────────────────────────────────────────────────────────


def test_a_large_sweep_is_cheap_to_submit_list_and_schedule_around(cluster):
    """Thousands of pending jobs that cannot run (no free GPU): submitting,
    listing and summarising stay fast, and the scheduler passes over them
    without making the manager sluggish."""
    c = cluster
    n = 5000
    c.patch(f"/api/workers/{c.worker_ids[0]}/devices", {"remove": [0]})
    eid = c.experiment("large")
    bodies = [c.job_body(eid, f"r{i}", 1, gpus=1, combo={"lr": i, "seed": i % 7})
              for i in range(n)]
    _, submit_s = _timed(lambda: [c.post("/api/jobs/bulk", bodies[i:i + 1000])
                                  for i in range(0, n, 1000)])
    jobs, list_s = _timed(lambda: c.get(f"/api/experiments/{eid}/jobs"))
    summary, summary_s = _timed(lambda: c.get(f"/api/experiments/{eid}/summary"))
    assert len(jobs) == n
    with _Latency(c.url) as lat:
        time.sleep(11)  # at least two periodic scheduler passes over every pending job
    print(f"\nlarge: submit {submit_s:.2f}s, list {list_s:.2f}s, summary {summary_s:.2f}s; "
          f"api p95 {lat.p95*1e3:.0f} ms, worst {lat.worst*1e3:.0f} ms")
    assert submit_s < 10 and list_s < 3 and summary_s < 1
    assert lat.p95 < 0.25 and lat.worst < 1
    assert c.statuses(eid) == {"pending": n}


def test_skip_rules_cost_stays_linear_in_the_sweep_size():
    """Each result re-applies the skip rules to the experiment's pending jobs,
    under the manager's lock, so it must not compare every pending job with
    every finished one."""
    import asyncio
    import itertools

    import aiosqlite

    from mlsweep._manager_db import apply_result_rules, create_experiment, init_db, insert_jobs_bulk

    async def per_result_seconds() -> float:
        db = await aiosqlite.connect(":memory:")
        try:
            await init_db(db)
            lrs, bss, seeds = range(40), range(20), range(15)  # 12,000 jobs
            rules = {"lr": {"monotonic": True, "singular": False, "_values": list(lrs)},
                     "bs": {"monotonic": False, "singular": True, "_values": list(bss)},
                     "seed": {"monotonic": False, "singular": False, "_values": list(seeds)}}
            await create_experiment(db, experiment_id="e", name="e", skip_rules=rules,
                                    singular_dims=["bs"])
            combos = [{"lr": a, "bs": b, "seed": c} for a, b, c in itertools.product(lrs, bss, seeds)]
            await insert_jobs_bulk(db, [{"run_id": f"r{i}", "experiment_id": "e", "command": ["x"],
                                         "combo": cb} for i, cb in enumerate(combos)])
            # A third of the sweep has finished, half of it failed.
            await db.execute("UPDATE jobs SET exit_code = 1, status = CASE WHEN job_key % 2 "
                             "THEN 'done' ELSE 'failed' END WHERE job_key % 3 = 0")
            await db.commit()
            t0 = time.perf_counter()
            await apply_result_rules(db, "e", "r2", False)
            await apply_result_rules(db, "e", "r4", True)
            return (time.perf_counter() - t0) / 2
        finally:
            await db.close()

    seconds = asyncio.run(per_result_seconds())
    print(f"\nskip rules: {seconds*1e3:.0f} ms per result over 12,000 jobs")
    assert seconds < 0.5
