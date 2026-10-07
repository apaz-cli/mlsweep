"""Unit tests for the scheduler's GPU placement policies ("pack" and "spread").

Drives ``_place`` directly with fake workers, applying each placement to the
occupancy the way ``_schedule_pass_locked`` does.  No manager process needed.
"""

from mlsweep._manager_db import SchedulableJob
from mlsweep._manager_state import WorkerConn
from mlsweep._manager_workers import _book, _place


def _worker(worker_id: str, n_gpus: int, cap: int) -> WorkerConn:
    return WorkerConn(worker_id=worker_id, host=worker_id, port=0,
                      gpus=list(range(n_gpus)), max_jobs_per_gpu=cap)


def _job(gpus: int = 1, nodes: int = 1) -> SchedulableJob:
    return SchedulableJob("exp", "run", gpus, nodes)


class _Cluster:
    def __init__(self, *workers: WorkerConn, topo: dict[tuple[int, int], int] | None = None):
        self.workers = list(workers)
        self.occ = {w.worker_id: {g: 0 for g in w.gpus} for w in workers}
        self.on_worker = {w.worker_id: 0 for w in workers}
        self.topos = {w.worker_id: topo or {} for w in workers}

    def place(self, job: SchedulableJob, placement: str):
        """Place *job* and book it.  Returns ``[(worker_id, gpus), ...]`` or None."""
        placed = _place(job, self.workers, self.occ, self.on_worker, self.topos, placement)  # type: ignore[arg-type]
        if placed is None:
            return None
        _book(placed, self.occ, self.on_worker)
        return [(wc.worker_id, gpus) for wc, gpus in placed]

    def fill(self, n: int, placement: str, job: SchedulableJob | None = None):
        return [self.place(job or _job(), placement) for _ in range(n)]


def test_spread_puts_one_job_on_each_gpu_before_doubling_up():
    c = _Cluster(_worker("a", 4, cap=2))
    gpus = [p[0][1][0] for p in c.fill(8, "spread")]
    assert gpus == [0, 1, 2, 3, 0, 1, 2, 3]
    assert c.place(_job(), "spread") is None


def test_pack_fills_each_gpu_before_the_next():
    c = _Cluster(_worker("a", 4, cap=2))
    gpus = [p[0][1][0] for p in c.fill(8, "pack")]
    assert gpus == [0, 0, 1, 1, 2, 2, 3, 3]
    assert c.place(_job(), "pack") is None


def test_spread_alternates_workers_and_pack_fills_one_first():
    spread = _Cluster(_worker("a", 2, cap=1), _worker("b", 2, cap=1))
    assert spread.fill(4, "spread") == [
        [("a", [0])], [("b", [0])], [("a", [1])], [("b", [1])],
    ]
    pack = _Cluster(_worker("a", 2, cap=1), _worker("b", 2, cap=1))
    assert pack.fill(4, "pack") == [
        [("a", [0])], [("a", [1])], [("b", [0])], [("b", [1])],
    ]


def test_spread_with_unlimited_jobs_per_gpu_still_uses_every_gpu():
    c = _Cluster(_worker("a", 4, cap=0))
    gpus = [p[0][1][0] for p in c.fill(8, "spread")]
    assert sorted(gpus) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_spread_sends_cpu_jobs_to_the_least_busy_worker():
    c = _Cluster(_worker("a", 1, cap=1), _worker("b", 1, cap=1))
    assert c.fill(4, "spread", _job(gpus=0)) == [
        [("a", [])], [("b", [])], [("a", [])], [("b", [])],
    ]


def test_pack_keeps_a_whole_worker_free_for_a_multi_gpu_job():
    """The trade-off between the policies: spreading leaves no worker idle."""
    for placement, fits in (("pack", True), ("spread", False)):
        c = _Cluster(_worker("a", 2, cap=1), _worker("b", 2, cap=1))
        c.fill(2, placement)
        assert (c.place(_job(gpus=2), placement) is not None) == fits, placement


def test_pack_prefers_a_partly_used_gpu_over_an_idle_one():
    c = _Cluster(_worker("a", 4, cap=2))
    c.occ["a"][2] = 1
    assert c.place(_job(), "pack") == [("a", [2])]


def test_spread_multi_gpu_job_takes_the_idle_gpus():
    c = _Cluster(_worker("a", 4, cap=2))
    c.occ["a"].update({0: 1, 1: 1})
    assert c.place(_job(gpus=2), "spread") == [("a", [2, 3])]


def test_topology_breaks_ties_between_equally_loaded_gpus():
    nvlink = {(1, 3): 10, (3, 1): 10}
    for placement in ("pack", "spread"):
        c = _Cluster(_worker("a", 4, cap=1), topo=nvlink)
        assert c.place(_job(gpus=2), placement) == [("a", [1, 3])], placement


def test_multinode_job_uses_distinct_workers():
    c = _Cluster(_worker("a", 2, cap=1), _worker("b", 2, cap=1), _worker("c", 2, cap=1))
    c.place(_job(), "spread")  # a is now the busiest
    placed = c.place(_job(gpus=1, nodes=2), "spread")
    assert [wid for wid, _ in placed] == ["b", "c"]
    assert c.place(_job(gpus=2, nodes=3), "spread") is None
