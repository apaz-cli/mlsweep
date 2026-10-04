"""HTTP and seeding helpers shared by the campaign tests."""

import json
import urllib.error
import urllib.request
import uuid

from conftest import _start_manager, _teardown_manager

TOKEN = "test-token"


def static_url(server, name, **params):
    """A web UI page URL on *server*, authenticated, with extra query *params*."""
    query = "&".join(f"{k}={v}" for k, v in {"token": TOKEN, **params}.items())
    return f"{server.url}/static/{name}?{query}"


def uid(prefix):
    """A fresh id, so tests sharing one manager never collide."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def shared_manager(tmp_path_factory, name):
    """Yield one manager (no workers) for a module; its tests use fresh ids."""
    proc, server, _ = _start_manager(tmp_path_factory.mktemp(name))
    yield server
    _teardown_manager(proc)


def call(server, method, path, body=None, raw=None):
    """Request *path*; return ``(status, parsed JSON or raw bytes)`` without raising."""
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(server.url + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status, payload, ctype = resp.status, resp.read(), resp.headers.get_content_type()
    except urllib.error.HTTPError as e:
        status, payload, ctype = e.code, e.read(), e.headers.get_content_type()
    if ctype == "application/json" and payload:
        return status, json.loads(payload)
    return status, payload


def q(path, campaign):
    """*path* with ``campaign=`` appended."""
    return f"{path}{'&' if '?' in path else '?'}campaign={campaign}"


def ok(server, method, path, body=None):
    status, data = call(server, method, path, body)
    assert 200 <= status < 300, (method, path, status, data)
    return data


def seed(server, campaign, prefix="e"):
    """An experiment in *campaign* with pending run r1, done run r2, and an artifact file."""
    eid = uid(prefix)
    ok(server, "POST", "/api/experiments", {"experiment_id": eid, "name": eid, "campaign": campaign})
    for rid in ("r1", "r2"):
        ok(server, "POST", "/api/jobs", {"experiment_id": eid, "run_id": rid, "command": ["echo"]})
    ok(server, "PUT", "/api/jobs/r2/status", {"experiment_id": eid, "status": "done", "exit_code": 0})
    art = server.mlsweep_dir / "experiments" / eid / "r1" / "artifacts"
    art.mkdir(parents=True)
    (art / "f.txt").write_text("hello")
    return eid


def snapshot(server, eid):
    """Everything a refused request must leave untouched."""
    status, exp = call(server, "GET", f"/api/experiments/{eid}")
    _, jobs = call(server, "GET", f"/api/experiments/{eid}/jobs")
    rows = sorted((j["run_id"], j["status"], j["priority"], j["label"], j["retry_count"])
                  for j in jobs) if isinstance(jobs, list) else jobs
    return status, exp, rows
