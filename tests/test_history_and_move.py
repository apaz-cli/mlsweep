"""Run history (``mlsweep history``) and moving runs that haven't finished.

The history records what the manager did to each run (dispatched where,
finished how, requeued why) and what people asked for, with where the request
came from: the CLI, the dashboard, or a bare API call.  ``move`` takes
pending runs at once and moves running ones when they finish.
"""

import json
import sys
import time
import urllib.request

import pytest

from conftest import _api_get, _api_post
from mlsweep import ctl
from mlsweep.run_sweep import compare_with_experiment

_TOKEN = "test-token"


def _base(server, url):
    return ["--manager", url, "--token", server.token, "--campaign", "test"]


def _exp(url, eid):
    _api_post(url, _TOKEN, "/api/experiments", {"campaign": "test", "experiment_id": eid, "name": eid})


def _run(url, eid, rid, seconds=0.0, code=0):
    _api_post(url, _TOKEN, "/api/jobs", {
        "run_id": rid, "experiment_id": eid, "gpus_per_run": 0,
        "command": [sys.executable, "-c", f"import time, sys; time.sleep({seconds}); sys.exit({code})"]})


def _until(what, check, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not (value := check()):
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.1)
    return value


def _status(url, eid, rid):
    try:
        return _api_get(url, _TOKEN, f"/api/jobs/{rid}?experiment_id={eid}")["status"]
    except Exception:
        return None


def _history(server, url, eid, *run, capsys):
    capsys.readouterr()
    ctl.history_cmd([eid, *run, "--json", "--manager", url, "--token", server.token])
    return json.loads(capsys.readouterr().out)


def test_history_says_what_happened_and_who_asked(manager_with_worker, capsys):
    server, url = manager_with_worker
    _exp(url, "h")
    _run(url, "h", "r1", code=1)
    _until("r1 to fail", lambda: _status(url, "h", "r1") == "failed")
    ctl.retry_cmd(["h", "r1", *_base(server, url)])
    _until("the retry to finish", lambda: _api_get(url, _TOKEN, "/api/jobs/r1?experiment_id=h")["attempt"] == 2
           and _status(url, "h", "r1") == "failed")
    events = _history(server, url, "h", "r1", capsys=capsys)["events"]
    seen = [(e["action"], e["source"], e["attempt"]) for e in events]
    assert seen == [("submitted", "api", None), ("dispatched", "manager", 1), ("finished", "manager", 1),
                    ("retried", "cli", None), ("dispatched", "manager", 2), ("finished", "manager", 2)], seen
    assert events[2]["detail"].startswith("failed, exit 1")
    assert events[1]["detail"].startswith("to localhost")

    # The text form, one line per entry.
    ctl.history_cmd(["h", "--manager", url, "--token", server.token])
    out = capsys.readouterr().out
    assert "# History of h" in out and "cli" in out and "retried" in out


def test_the_dashboard_names_itself(manager_server):
    server, url = manager_server
    _exp(url, "d")
    req = urllib.request.Request(
        f"{url}/api/experiments/d/status", method="PUT", data=json.dumps({"status": "paused"}).encode(),
        headers={"Authorization": f"Bearer {_TOKEN}", "Content-Type": "application/json",
                 "X-Mlsweep-Client": "dashboard"})
    urllib.request.urlopen(req).close()
    [event] = _api_get(url, _TOKEN, "/api/history?experiment_id=d")["events"]
    assert (event["run_id"], event["action"], event["source"]) == (None, "paused", "dashboard")


def test_a_deleted_experiment_keeps_its_history(manager_with_worker, capsys):
    server, url = manager_with_worker
    _exp(url, "gone")
    _run(url, "gone", "live", seconds=120)
    _until("the run to start", lambda: _status(url, "gone", "live") == "running")
    ctl.delete_cmd(["gone", "--yes", *_base(server, url)])
    resp = _history(server, url, "gone", capsys=capsys)
    assert resp["exists"] is False
    last = resp["events"][-1]
    assert (last["action"], last["source"]) == ("deleted", "cli")
    assert last["detail"] == "stopped 1 in-flight run(s): live"


def test_a_running_run_moves_when_it_finishes(manager_with_worker, capsys):
    server, url = manager_with_worker
    _exp(url, "wave")
    _run(url, "wave", "slow", seconds=2)
    _run(url, "wave", "later", seconds=0)
    _until("slow to start", lambda: _status(url, "wave", "slow") == "running")
    capsys.readouterr()
    ctl.move_cmd(["wave", "slow", "--to", "refs", "--create", *_base(server, url)])
    out = capsys.readouterr().out
    assert "created refs" in out and "1 running run(s) move when they finish: slow" in out
    assert _status(url, "wave", "slow") == "running"  # not restarted
    _until("slow to land in refs", lambda: _status(url, "refs", "slow") == "done")
    assert _status(url, "wave", "slow") is None
    exp = _api_get(url, _TOKEN, "/api/experiments/refs")
    assert exp["campaign"] == "test"
    src = [(e["action"], e["detail"]) for e in _history(server, url, "wave", "slow", capsys=capsys)["events"]]
    assert ("will move", "to refs when it finishes") in src and ("moved", "to refs") in src
    dst = [(e["action"], e["detail"]) for e in _history(server, url, "refs", "slow", capsys=capsys)["events"]]
    assert ("moved", "from wave") in dst


def test_a_running_run_is_refused_without_after_finish(manager_with_worker):
    server, url = manager_with_worker
    for eid in ("a", "b"):
        _exp(url, eid)
    _run(url, "a", "busy", seconds=120)
    _until("busy to start", lambda: _status(url, "a", "busy") == "running")
    with pytest.raises(SystemExit) as exc:
        ctl.move_cmd(["a", "busy", "--to", "b", "--now-only", *_base(server, url)])
    assert exc.value.code == 1


def test_a_move_that_cannot_happen_any_more_says_so(manager_with_worker, capsys):
    server, url = manager_with_worker
    for eid in ("src", "dst"):
        _exp(url, eid)
    _run(url, "src", "r", seconds=2)
    _until("r to start", lambda: _status(url, "src", "r") == "running")
    ctl.move_cmd(["src", "r", "--to", "dst", *_base(server, url)])
    _run(url, "dst", "r", seconds=120)  # the same run ID turns up in the target meanwhile
    _until("r to finish", lambda: _status(url, "src", "r") == "done")
    events = _history(server, url, "src", "r", capsys=capsys)["events"]
    assert events[-1]["action"] == "move failed" and "already has a run with this ID" in events[-1]["detail"]
    assert _api_get(url, _TOKEN, "/api/jobs/r?experiment_id=src")["move_to"] is None


def test_a_sweep_file_is_compared_with_an_experiment_by_command():
    def var(name, *flags):
        return {"name": name, "overrides": list(flags), "combo": {}}

    def job(rid, *flags, status="done"):
        return {"run_id": rid, "status": status, "command": json.dumps(["python", "t.py", *flags])}

    variations = [var("s_lr1", "--lr", "1"), var("s_lr2", "--lr", "2"), var("s_lr3", "--lr", "3"),
                  var("old", "--lr", "9")]
    jobs = [job("s.lr1", "--lr", "1"), job("s_lr2", "--lr", "2", status="running"),
            job("old", "--lr", "8"), job("extra", "--lr", "5", status="failed")]
    cmp = compare_with_experiment(variations, ["python", "t.py"], [], jobs)
    assert cmp["same"] == [{"name": "s_lr1", "run_id": "s.lr1", "status": "done"},
                           {"name": "s_lr2", "run_id": "s_lr2", "status": "running"}]
    assert [n["name"] for n in cmp["new"]] == ["s_lr3"]
    assert [c["name"] for c in cmp["clash"]] == ["old"]  # the name is taken by another command
    assert [o["run_id"] for o in cmp["only_in_experiment"]] == ["old", "extra"]
