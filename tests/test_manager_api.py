"""HTTP REST API tests for the mlsweep manager.

All tests run against a real manager subprocess (``manager_server`` fixture
from conftest.py).  They cover experiment / job / worker / artifact /
reachable / auth / health endpoints.
"""

import io
import json
import urllib.error
import urllib.request

import pytest

from conftest import (
    _api_get,
    _api_post,
    _wait_for_job,
)

_TOKEN = "test-token"


def _api_put(url, token, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{url}{path}", data=body, headers=headers, method="PUT",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _api_delete(url, token, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{url}{path}", data=body, headers=headers, method="DELETE",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _api_head(url, token, path):
    req = urllib.request.Request(
        f"{url}{path}", headers={"Authorization": f"Bearer {token}"}, method="HEAD",
    )
    return urllib.request.urlopen(req, timeout=10)


# ── Health / Auth ───────────────────────────────────────────────────────────────


def test_health_endpoint_requires_auth(manager_server):
    """Health endpoint should require authentication."""
    _, url = manager_server
    # Without auth token → 401
    req_no_auth = urllib.request.Request(f"{url}/api/health")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req_no_auth, timeout=10)
    assert exc.value.code == 401

    # With valid token → 200
    req = urllib.request.Request(
        f"{url}/api/health",
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    assert data["status"] == "ok"
    assert "workers_connected" in data
    assert "jobs_pending" in data
    assert "jobs_in_flight" in data


@pytest.mark.parametrize("headers,expected_code", [
    ({}, 401),
    ({"Authorization": "Bearer wrong-token"}, 401),
    ({"Authorization": f"Bearer {_TOKEN}"}, 200),
])
def test_auth_scenarios(manager_server, headers, expected_code):
    _, url = manager_server
    req = urllib.request.Request(f"{url}/api/experiments", headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        assert resp.status == expected_code
    except urllib.error.HTTPError as e:
        assert e.code == expected_code


# ── Experiments ─────────────────────────────────────────────────────────────────


def test_create_and_get_experiment(manager_server):
    _, url = manager_server
    data = _api_post(url, _TOKEN, "/api/experiments",
                     {"experiment_id": "e1", "name": "my_exp",
                      "note": "testing", "expected_jobs": 5})

    assert data["experiment_id"] == "e1"
    assert data["name"] == "my_exp"
    assert data["status"] == "running"
    assert data["note"] == "testing"

    # GET
    exp = _api_get(url, _TOKEN, "/api/experiments/e1")
    assert exp["experiment_id"] == "e1"


@pytest.mark.parametrize("body,expected_status", [
    (b"{bad json", 400),
    ({}, 400),
    ({"experiment_id": "invalid/id!"}, 400),
    ({"experiment_id": "a" * 129}, 400),
])
def test_create_experiment_errors(manager_server, body, expected_status):
    _, url = manager_server
    if isinstance(body, bytes):
        headers = {"Authorization": f"Bearer {_TOKEN}",
                   "Content-Type": "application/json"}
        req = urllib.request.Request(
            f"{url}/api/experiments", data=body, headers=headers, method="POST",
        )
    else:
        body_str = json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {_TOKEN}",
                   "Content-Type": "application/json"}
        req = urllib.request.Request(
            f"{url}/api/experiments", data=body_str, headers=headers,
            method="POST",
        )
    try:
        urllib.request.urlopen(req, timeout=10)
        pytest.fail("expected HTTP error")
    except urllib.error.HTTPError as e:
        assert e.code == expected_status


def test_update_experiment_status(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "e1"})

    updated = _api_put(url, _TOKEN, "/api/experiments/e1/status",
                       {"status": "completed"})
    assert updated["status"] == "completed"


def test_update_experiment_status_rejects_bad_value(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "e_bad"})
    try:
        _api_put(url, _TOKEN, "/api/experiments/e_bad/status", {"status": "bogus"})
        pytest.fail("expected 400 for invalid status")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_pause_and_resume_experiment_status(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "e_pause"})
    paused = _api_put(url, _TOKEN, "/api/experiments/e_pause/status", {"status": "paused"})
    assert paused["status"] == "paused"
    resumed = _api_put(url, _TOKEN, "/api/experiments/e_pause/status", {"status": "running"})
    assert resumed["status"] == "running"


def test_experiment_max_concurrent_create_and_update(manager_server):
    _, url = manager_server
    created = _api_post(url, _TOKEN, "/api/experiments",
                        {"experiment_id": "e_cap", "max_concurrent": 4})
    assert created["max_concurrent"] == 4

    updated = _api_put(url, _TOKEN, "/api/experiments/e_cap/max_concurrent",
                       {"max_concurrent": 2})
    assert updated["max_concurrent"] == 2

    # Negative is rejected.
    try:
        _api_put(url, _TOKEN, "/api/experiments/e_cap/max_concurrent",
                 {"max_concurrent": -1})
        pytest.fail("expected 400 for negative max_concurrent")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_delete_experiment(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "e_to_del"})

    resp = _api_delete(url, _TOKEN, "/api/experiments/e_to_del")
    assert resp["deleted"] == "e_to_del"

    # 404 on re-delete
    try:
        _api_delete(url, _TOKEN, "/api/experiments/e_to_del")
        pytest.fail("expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_list_experiments(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "ea"})
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "eb",
                                                  "status": "completed"})

    all_exp = _api_get(url, _TOKEN, "/api/experiments")
    assert len(all_exp) >= 2

    completed = _api_get(url, _TOKEN, "/api/experiments?status=completed")
    assert all(e["status"] == "completed" for e in completed)


def test_experiment_not_found(manager_server):
    _, url = manager_server
    try:
        _api_get(url, _TOKEN, "/api/experiments/nonexistent")
        pytest.fail("expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_experiment_summary(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments",
              {"experiment_id": "e_summ", "name": "summary_test"})

    summary = _api_get(url, _TOKEN, "/api/experiments/e_summ/summary")
    assert summary["name"] == "summary_test"
    assert "job_counts" in summary


# ── Jobs ────────────────────────────────────────────────────────────────────────


def test_insert_job(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "e_j"})

    data = _api_post(url, _TOKEN, "/api/jobs", {
        "run_id": "r1", "experiment_id": "e_j",
        "command": ["echo", "hello"],
    })
    assert data["run_id"] == "r1"
    assert data["status"] == "pending"


def test_insert_jobs_bulk(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "e_bulk"})

    jobs = [
        {"run_id": "ra", "experiment_id": "e_bulk", "command": ["echo", "a"]},
        {"run_id": "rb", "experiment_id": "e_bulk", "command": ["echo", "b"]},
    ]
    data = _api_post(url, _TOKEN, "/api/jobs/bulk", jobs)
    assert len(data) == 2
    assert {j["run_id"] for j in data} == {"ra", "rb"}


def test_get_job_not_found(manager_server):
    _, url = manager_server
    try:
        _api_get(url, _TOKEN, "/api/jobs/nonexist?experiment_id=e")
        pytest.fail("expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_update_job_status(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "e_status"})
    _api_post(url, _TOKEN, "/api/jobs", {
        "run_id": "r_status", "experiment_id": "e_status",
        "command": ["echo"],
    })
    job = _api_put(url, _TOKEN, "/api/jobs/r_status/status", {
        "experiment_id": "e_status", "status": "done", "exit_code": 0,
    })
    assert job["status"] == "done"
    assert job["exit_code"] == 0


def test_update_job_priority(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "e_pri"})
    _api_post(url, _TOKEN, "/api/jobs", {
        "run_id": "r_pri", "experiment_id": "e_pri",
        "command": ["echo"],
    })
    job = _api_put(url, _TOKEN, "/api/jobs/r_pri/priority", {
        "experiment_id": "e_pri", "priority": 99,
    })
    assert job["priority"] == 99


def test_cancel_job(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "e_cancel"})
    _api_post(url, _TOKEN, "/api/jobs", {
        "run_id": "r_cancel", "experiment_id": "e_cancel",
        "command": ["echo"],
    })
    job = _api_post(url, _TOKEN, "/api/jobs/r_cancel/cancel"
                    "?experiment_id=e_cancel")
    assert job["status"] == "cancelled"


def test_retry_job(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "e_retry"})
    # Create then fail a job
    _api_post(url, _TOKEN, "/api/jobs", {
        "run_id": "r_retry", "experiment_id": "e_retry",
        "command": ["echo"],
    })
    _api_put(url, _TOKEN, "/api/jobs/r_retry/status", {
        "experiment_id": "e_retry", "status": "failed", "exit_code": 1,
    })
    job = _api_post(url, _TOKEN, "/api/jobs/r_retry/retry"
                    "?experiment_id=e_retry")
    assert job["status"] == "pending"
    assert job["retry_count"] == 1


def test_retry_job_not_terminal_rejected(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "e_retry2"})
    _api_post(url, _TOKEN, "/api/jobs", {
        "run_id": "r_retry2", "experiment_id": "e_retry2",
        "command": ["echo"],
    })
    # Job is still pending → should 409
    try:
        _api_post(url, _TOKEN, "/api/jobs/r_retry2/retry?experiment_id=e_retry2")
        pytest.fail("expected 409")
    except urllib.error.HTTPError as e:
        assert e.code == 409


def test_list_pending_jobs(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/experiments", {"experiment_id": "e_pen"})
    _api_post(url, _TOKEN, "/api/jobs/bulk", [
        {"run_id": "rp", "experiment_id": "e_pen", "command": ["echo"]},
        {"run_id": "rd", "experiment_id": "e_pen", "command": ["echo"]},
    ])
    # The rd job gets dispatched; pending list should include it until worker
    # picks it up.  Just check the endpoint returns valid JSON.
    pending = _api_get(url, _TOKEN, "/api/jobs/pending")
    assert isinstance(pending, list)


# ── Workers ─────────────────────────────────────────────────────────────────────


def test_add_and_get_worker(manager_server):
    _, url = manager_server
    worker = _api_post(url, _TOKEN, "/api/workers", {
        "worker_id": "w1", "host": "127.0.0.1", "remote_dir": "/tmp",
    })
    assert worker["worker_id"] == "w1"
    assert worker["host"] == "127.0.0.1"

    fetched = _api_get(url, _TOKEN, "/api/workers/w1")
    assert fetched["worker_id"] == "w1"


# ── Artifacts ───────────────────────────────────────────────────────────────────


def test_artifact_register(manager_server):
    _, url = manager_server
    art = _api_post(url, _TOKEN, "/api/artifacts", {
        "artifact_id": "sha256:abc123", "size_bytes": 1024,
    })
    assert art["artifact_id"] == "sha256:abc123"
    assert art["ref_count"] == 1

    art2 = _api_post(url, _TOKEN, "/api/artifacts", {
        "artifact_id": "sha256:abc123", "size_bytes": 2048,
    })
    assert art2["ref_count"] == 2


def test_artifact_head(manager_server):
    _, url = manager_server
    # Not registered → 404
    try:
        _api_head(url, _TOKEN, "/api/artifacts/sha256:noexist")
        pytest.fail("expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404

    # Registered but not uploaded → still 404
    _api_post(url, _TOKEN, "/api/artifacts",
              {"artifact_id": "sha256:headtest"})
    try:
        _api_head(url, _TOKEN, "/api/artifacts/sha256:headtest")
        pytest.fail("expected 404 (registered but file missing)")
    except urllib.error.HTTPError as e:
        assert e.code == 404

    # Upload data → HEAD 200
    _api_put_raw(url, _TOKEN, "/api/artifacts/sha256:headtest/data",
                 b"fake tarball bytes")
    resp = _api_head(url, _TOKEN, "/api/artifacts/sha256:headtest")
    assert resp.status == 200


def test_artifact_upload_and_download(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/artifacts",
              {"artifact_id": "sha256:updown"})

    payload = b"binary artifact content here"
    resp = _api_put_raw(url, _TOKEN, "/api/artifacts/sha256:updown/data", payload)
    data = json.loads(resp.read())
    assert data["artifact_id"] == "sha256:updown"
    assert data["size_bytes"] == len(payload)

    # Download and verify
    req = urllib.request.Request(
        f"{url}/api/artifacts/sha256:updown",
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.read() == payload


def test_artifact_meta(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/artifacts",
              {"artifact_id": "sha256:metatest", "size_bytes": 42})

    meta = _api_get(url, _TOKEN, "/api/artifacts/sha256:metatest/meta")
    assert meta["artifact_id"] == "sha256:metatest"
    assert meta["size_bytes"] == 42

    # Non-existent → 404
    try:
        _api_get(url, _TOKEN, "/api/artifacts/sha256:nope/meta")
        pytest.fail("expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_artifact_ref(manager_server):
    _, url = manager_server
    _api_post(url, _TOKEN, "/api/artifacts", {"artifact_id": "sha256:ref"})

    updated = _api_put(url, _TOKEN, "/api/artifacts/sha256:ref/ref",
                       {"delta": 5})
    assert updated["ref_count"] == 6


# ── /api/reachable ──────────────────────────────────────────────────────────────


def _api_put_raw(url, token, path, body):
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/octet-stream"}
    req = urllib.request.Request(
        f"{url}{path}", data=body, headers=headers, method="PUT",
    )
    return urllib.request.urlopen(req, timeout=10)


@pytest.mark.parametrize("host,expected_code", [
    (None, 400),
    ("", 400),
    ("http://evil.com", 400),
    ("evil.com/path", 400),
    ("foo?bar=1", 400),
    ("user@host", 400),
])
def test_reachable_rejects_bad_hosts(manager_server, host, expected_code):
    _, url = manager_server
    # Build the URL manually to handle chars urllib would otherwise strip/encode
    path = f"/api/reachable?host={host}" if host else "/api/reachable"
    req = urllib.request.Request(
        f"{url}{path}",
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        pytest.fail(f"expected HTTP {expected_code} for host={host!r}")
    except urllib.error.HTTPError as e:
        assert e.code == expected_code


def test_reachable_loopback(manager_server):
    _, url = manager_server
    data = _api_get(url, _TOKEN, "/api/reachable?host=127.0.0.1")
    assert data["reachable"] is True


def test_reachable_unreachable(manager_server):
    _, url = manager_server
    data = _api_get(url, _TOKEN, "/api/reachable?host=192.0.2.1")
    assert data["reachable"] is False
