"""What keeps a stolen or limited foothold from going further; see INTERNET.md.

Every request here carries the real token unless it says otherwise.  None of
these routes are open, so the point is that the token alone does not reach
outside the data directory, run script in the dashboard, or leave a trace of
itself on disk.
"""

import asyncio
import json
import os
import re
import socket
import sqlite3
import struct
import subprocess
import sys
import types
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import aiosqlite
import pytest
from aiohttp import web

from browser import Browser, find_chromium
from campaign_helpers import ok, shared_manager, static_url, uid
from mlsweep._manager_db import init_db, insert_job, create_experiment, unfinished_jobs_with_unsafe_ids
from mlsweep._manager_http import _AuthThrottle, _MAX_AUTH_FAILURES, error_middleware
from mlsweep.run_sweep import _ws_refusal
from mlsweep._manager_workers import _parse_workers_file, validate_worker_host
from mlsweep._shared import MsgHello, encode, read_msg, validate_artifact_id, validate_run_id

WEBUI = Path(__file__).resolve().parent.parent / "mlsweep" / "webui"
CHROME = find_chromium()


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "security")


def _req(mgr, method, path, body=None, *, raw=None, auth=True, headers=None):
    """Send a request (quoting nothing, so a path can carry %2F); return (status, bytes, headers)."""
    h = dict(headers or {})
    if auth:
        h["Authorization"] = f"Bearer {mgr.token}"
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request(mgr.url + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return resp.status, resp.read(), resp.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def _experiment(mgr):
    eid = uid("sec")
    ok(mgr, "POST", "/api/experiments", {"campaign": "test", "experiment_id": eid})
    return eid


# ── IDs in paths ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["../x", "a/../../x", "/etc/passwd", "..", ".", "a//b", "a/", "x\ny", "", None, 7])
def test_run_ids_that_leave_their_directory_are_refused(bad):
    with pytest.raises(ValueError):
        validate_run_id(bad)


@pytest.mark.parametrize("good", ["run_lr0.1", "model=org/name_lr0.1", "a b", "x..y", "...", "ünï"])
def test_run_names_from_sweep_values_still_work(good):
    assert validate_run_id(good) == good


@pytest.mark.parametrize("bad", ["../x", "a/b", ".hidden", "", "x y", None])
def test_artifact_ids_must_be_a_file_name(bad):
    with pytest.raises(ValueError):
        validate_artifact_id(bad)


def test_artifact_ids_from_the_cli_still_work():
    aid = "sha256:" + "ab" * 32
    assert validate_artifact_id(aid) == aid


def test_artifact_routes_refuse_traversal(mgr):
    outside = Path(mgr.mlsweep_dir) / "secret.tar.gz"
    outside.write_bytes(b"READ-ME")
    assert _req(mgr, "POST", "/api/artifacts", {"artifact_id": "../../pwn"})[0] == 400
    assert _req(mgr, "PUT", "/api/artifacts/..%2F..%2Fpwn/data", raw=b"x")[0] == 400
    status, body, _ = _req(mgr, "GET", "/api/artifacts/..%2Fsecret")
    assert status == 400 and b"READ-ME" not in body
    assert _req(mgr, "HEAD", "/api/artifacts/..%2Fsecret")[0] == 400
    assert not list(Path(mgr.mlsweep_dir).parent.glob("pwn*"))


def test_artifact_upload_round_trip(mgr):
    aid = "sha256:" + uid("x")
    ok(mgr, "POST", "/api/artifacts", {"artifact_id": aid})
    quoted = urllib.parse.quote(aid, safe="")
    status, body, _ = _req(mgr, "PUT", f"/api/artifacts/{quoted}/data", raw=b"tarball bytes")
    assert status == 200 and json.loads(body)["size_bytes"] == 13
    assert _req(mgr, "GET", f"/api/artifacts/{quoted}")[1] == b"tarball bytes"
    assert not list((Path(mgr.mlsweep_dir) / "artifacts").glob("*.tmp"))


def test_job_file_route_refuses_traversal_in_either_id(mgr):
    leak = Path(mgr.mlsweep_dir) / "artifacts" / "leak.txt"
    leak.parent.mkdir(exist_ok=True)
    leak.write_text("LEAK")
    for path in ("/api/experiments/x%2F../jobs/x%2F..%2F../artifacts/leak.txt",
                 "/api/experiments/ok/jobs/..%2F..%2F../artifacts/leak.txt"):
        status, body, _ = _req(mgr, "GET", path)
        assert status == 400 and b"LEAK" not in body


def test_inserting_a_job_refuses_unsafe_ids(mgr):
    eid = _experiment(mgr)
    job = {"experiment_id": eid, "command": ["true"]}
    assert _req(mgr, "POST", "/api/jobs", {**job, "run_id": "../../evil"})[0] == 400
    assert _req(mgr, "POST", "/api/jobs/bulk", [{**job, "run_id": "../evil"}])[0] == 400
    assert _req(mgr, "POST", f"/api/experiments/{eid}/jobs/bulk",
                {"campaign": "test", "jobs": [{"run_id": "a/../../evil", "command": ["true"]}]})[0] == 400
    assert _req(mgr, "POST", "/api/jobs", {**job, "run_id": "r", "artifact_id": "../x"})[0] == 400
    assert _req(mgr, "POST", "/api/jobs", {**job, "run_id": "org/model_lr0.1"})[0] == 201


def test_old_unsafe_rows_are_found_for_the_manager_to_fail_at_startup(tmp_path):
    async def main():
        db = await aiosqlite.connect(tmp_path / "m.db")
        db.row_factory = aiosqlite.Row
        await init_db(db)
        await create_experiment(db, experiment_id="e", name="e", campaign="test")
        await insert_job(db, run_id="fine", experiment_id="e", command=["true"])
        await insert_job(db, run_id="old", experiment_id="e", command=["true"])
        await db.execute("UPDATE jobs SET run_id = '../other' WHERE run_id = 'old'")
        await db.commit()
        unsafe = await unfinished_jobs_with_unsafe_ids(db)
        await db.close()
        return unsafe
    assert asyncio.run(main()) == [("../other", "e")]


def test_purging_an_old_unsafe_row_leaves_other_experiments_alone(mgr):
    victim, eid = _experiment(mgr), _experiment(mgr)
    victim_dir = Path(mgr.mlsweep_dir) / "experiments" / victim
    victim_dir.mkdir(parents=True)
    (victim_dir / "keep.txt").write_text("keep")
    ok(mgr, "POST", "/api/jobs", {"experiment_id": eid, "run_id": "old", "command": ["true"]})
    ok(mgr, "PUT", "/api/jobs/old/status", {"experiment_id": eid, "status": "done", "exit_code": 0})
    bad = f"../{victim}"
    with sqlite3.connect(next(Path(mgr.mlsweep_dir).parent.rglob("*.db"))) as db:
        db.execute("UPDATE jobs SET run_id = ? WHERE experiment_id = ? AND run_id = 'old'", (bad, eid))
    status, body, _ = _req(mgr, "POST", f"/api/experiments/{eid}/jobs/purge", {"run_ids": [bad]})
    assert status == 200 and json.loads(body)["purged"] == [bad]
    assert (victim_dir / "keep.txt").exists()


# ── Files a job wrote ──────────────────────────────────────────────────────────


@pytest.fixture
def run_files(mgr):
    eid = uid("sec")
    rd = Path(mgr.mlsweep_dir) / "experiments" / eid / "r1" / "artifacts"
    rd.mkdir(parents=True)
    (rd / "x.html").write_text("<script>alert(1)</script>")
    (rd / "x.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>")
    (rd / "x.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (rd / "x.txt").write_text("hello")
    (rd / "x.bin").write_bytes(b"\0\1")
    return f"/api/experiments/{eid}/jobs/r1/artifacts"


def test_a_jobs_html_cannot_run_on_the_dashboard_origin(mgr, run_files):
    status, _, h = _req(mgr, "GET", f"{run_files}/x.html")
    assert status == 200
    assert h["Content-Type"] == "application/octet-stream"
    assert h["Content-Disposition"].startswith("attachment")
    assert "sandbox" in h["Content-Security-Policy"]


def test_a_jobs_svg_is_sandboxed(mgr, run_files):
    _, _, h = _req(mgr, "GET", f"{run_files}/x.svg")
    assert h["Content-Type"] == "image/svg+xml" and "sandbox" in h["Content-Security-Policy"]


def test_previews_still_get_their_bytes(mgr, run_files):
    status, body, h = _req(mgr, "GET", f"{run_files}/x.png")
    assert status == 200 and body.startswith(b"\x89PNG") and h["Content-Type"] == "image/png"
    assert h["Content-Disposition"].startswith("inline")
    status, body, h = _req(mgr, "GET", f"{run_files}/x.txt")
    assert status == 200 and body == b"hello" and h["Content-Type"].startswith("text/plain")
    assert _req(mgr, "GET", f"{run_files}/x.bin")[2]["Content-Disposition"].startswith("attachment")


# ── The token stays home ───────────────────────────────────────────────────────


def test_the_probe_that_sent_the_token_elsewhere_is_gone(mgr):
    assert _req(mgr, "GET", "/api/reachable?host=127.0.0.1")[0] == 404


def test_health_tells_the_dashboard_the_public_address(mgr):
    assert "public_url" in json.loads(_req(mgr, "GET", "/api/health")[1])


def test_wrong_tokens_of_any_shape_are_refused(mgr):
    status, body, _ = _req(mgr, "GET", "/api/health", auth=False)
    assert status == 401 and b"link the manager printed" in body
    # A link pasted twice: the second copy ends up inside the token.
    doubled = urllib.parse.quote(f"{mgr.token} {mgr.url}/?token={mgr.token}", safe="")
    assert _req(mgr, "GET", f"/?token={doubled}", auth=False)[0] == 401
    for wrong in ("nope", "ünï", mgr.token + "x", mgr.token[:-1]):
        quoted = urllib.parse.quote(wrong)
        assert _req(mgr, "GET", f"/api/health?token={quoted}", auth=False)[0] == 401


def test_the_audit_log_records_changes_without_the_token(mgr):
    eid = _experiment(mgr)
    assert _req(mgr, "GET", "/api/health?token=wrong-token", auth=False)[0] == 401
    assert _req(mgr, "POST", f"/api/experiments?token={mgr.token}", {"campaign": "test", "experiment_id": uid("sec")},
                auth=False)[0] == 201
    text = (Path(mgr.mlsweep_dir) / "audit.log").read_text()
    records = [json.loads(line) for line in text.splitlines()]
    assert any(r["method"] == "POST" and r["path"] == "/api/experiments" and r["status"] == 201 for r in records)
    assert any(r["path"] == "/api/health" and r["status"] == 401 for r in records)
    assert not any(r["method"] == "GET" and r["status"] == 200 for r in records)
    assert mgr.token not in text and "wrong-token" not in text
    assert eid


def test_repeated_bad_tokens_get_blocked_per_address():
    t = _AuthThrottle()
    for _ in range(_MAX_AUTH_FAILURES - 1):
        t.failed("203.0.113.9")
    assert t.retry_after("203.0.113.9") == 0
    t.failed("203.0.113.9")
    assert t.retry_after("203.0.113.9") > 0
    assert t.retry_after("203.0.113.10") == 0


@pytest.mark.parametrize("address", ["127.0.0.1", "::1"])
def test_loopback_is_never_blocked(address):
    t = _AuthThrottle()
    for _ in range(_MAX_AUTH_FAILURES * 2):
        t.failed(address)
    assert t.retry_after(address) == 0


# ── Worker hosts and the worker port ───────────────────────────────────────────


@pytest.mark.parametrize("bad", ["-oProxyCommand=touch x", "-F/tmp/c", "a b", "a\nb", "", None])
def test_a_worker_host_must_be_a_destination(bad):
    with pytest.raises(ValueError):
        validate_worker_host(bad)


@pytest.mark.parametrize("good", ["localhost", "gpu01", "me@gpu01.example.com", "10.0.0.5"])
def test_real_worker_hosts_pass(good):
    assert validate_worker_host(good) == good


def test_add_worker_refuses_an_option_as_a_host(mgr):
    status, body, _ = _req(mgr, "POST", "/api/workers", {"host": "-oProxyCommand=touch /tmp/x"})
    assert status == 400 and b"invalid worker host" in body


def test_a_workers_file_cannot_hold_an_option_as_a_host(tmp_path):
    f = tmp_path / "w.toml"
    f.write_text('[[workers]]\nhost = "-oProxyCommand=x"\nremote_dir = "/r"\n')
    with pytest.raises(ValueError):
        _parse_workers_file(str(f))


def test_an_oversized_frame_is_dropped_before_reading_it():
    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack(">I", 1 << 31))
        assert read_msg(b, limit=1024) is None
        a.sendall(struct.pack(">I", 2) + b"ok")
        assert read_msg(b, limit=1024) == b"ok"
    finally:
        a.close()
        b.close()


# ── Dashboard ──────────────────────────────────────────────────────────────────


def test_no_cross_origin_access(mgr):
    _, _, h = _req(mgr, "GET", "/api/health", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in h


def test_api_and_static_responses_carry_security_headers(mgr):
    _, _, h = _req(mgr, "GET", "/api/health")
    assert h["X-Content-Type-Options"] == "nosniff" and h["Referrer-Policy"] == "no-referrer"
    assert "default-src 'none'" in h["Content-Security-Policy"]
    _, _, h = _req(mgr, "GET", "/static/experiments.html", auth=False)
    csp = h["Content-Security-Policy"]
    assert "script-src 'self';" in csp and "frame-ancestors 'none'" in csp and "unsafe-inline" not in csp.split("style-src")[0]
    assert h["X-Content-Type-Options"] == "nosniff"


def test_static_directories_are_not_listed(mgr):
    assert _req(mgr, "GET", "/static/", auth=False)[0] != 200


@pytest.mark.parametrize("path", sorted(WEBUI.glob("*.html")), ids=lambda p: p.name)
def test_pages_have_no_inline_script_for_the_csp_to_block(path):
    html = path.read_text()
    assert not re.findall(r"<script(?![^>]*\ssrc=)[^>]*>", html)
    assert not re.findall(r"<[a-z][^>]*\son[a-z]+=", html)
    assert "https://" not in "".join(re.findall(r"<script[^>]*>", html))


@pytest.mark.parametrize("path", sorted(WEBUI.glob("page-*.js")), ids=lambda p: p.name)
def test_page_scripts_build_no_inline_handlers(path):
    assert not re.findall(r"""\son[a-z]+=["'`$]""", path.read_text())
    assert "ipify" not in path.read_text()


@pytest.fixture(scope="module")
def browser():
    b = Browser(CHROME)
    b.call("Page.addScriptToEvaluateOnNewDocument", source="""
        window.__csp = [];
        document.addEventListener('securitypolicyviolation', e =>
            window.__csp.push(e.violatedDirective + ' ' + e.blockedURI));
    """)
    yield b
    b.close()

@pytest.fixture(scope="module")
def seeded(mgr):
    """An experiment whose job data tries to run script wherever the pages show it."""
    eid = uid("xss")
    evil = "<img src=x onerror=window.__xss=1>"
    ok(mgr, "POST", "/api/experiments", {"campaign": "test", "experiment_id": eid, "name": evil})
    for i in range(2):
        rid = f"{evil}{i}"
        ok(mgr, "POST", "/api/jobs", {"experiment_id": eid, "run_id": rid, "command": ["echo", evil],
                                      "combo": {evil: f"{evil}{i}", "lr": i}})
        ok(mgr, "PUT", f"/api/jobs/{urllib.parse.quote(rid, safe='')}/status",
           {"experiment_id": eid, "status": "done", "exit_code": 0})
    return eid


@pytest.mark.skipif(CHROME is None, reason="no Chromium-family browser installed")
class TestInBrowser:
    @pytest.mark.parametrize("page", sorted(p.name for p in WEBUI.glob("*.html") if p.name != "index.html"))
    def test_every_page_loads_clean_under_its_csp(self, mgr, browser, seeded, page):
        browser.goto(static_url(mgr, page, experiment=seeded))
        browser.wait_for("document.readyState === 'complete'")
        browser.eval("new Promise(r => setTimeout(r, 800))")
        assert browser.eval("window.__csp") == []
        assert browser.eval("window.__xss") is None

    def test_results_draws_its_chart_with_the_vendored_plotly(self, mgr, browser, seeded):
        browser.goto(static_url(mgr, "results.html", experiment=seeded))
        browser.wait_for("typeof Plotly !== 'undefined'")
        assert browser.eval("new Promise(r => setTimeout(r, 1000)) .then(() => window.__csp)") == []
        assert browser.eval("window.__xss") is None
        assert browser.eval("!!document.querySelector('.axis-label')")
        assert browser.eval("[...document.querySelectorAll('.axis-label img')].length") == 0

    def test_inline_script_really_is_blocked(self, mgr, browser):
        browser.goto(static_url(mgr, "experiments.html"))
        browser.eval("""(() => { const s = document.createElement('script');
                         s.textContent = 'window.__ran = 1'; document.head.append(s); })()""")
        assert browser.eval("window.__ran") is None

    def test_action_buttons_work_through_delegation(self, mgr, browser, seeded):
        browser.goto(static_url(mgr, "experiments.html"))
        row = f"[...document.querySelectorAll('.exp-row')].find(r => r.dataset.expId === {json.dumps(seeded)})"
        browser.wait_for(f"!!{row}")
        browser.eval(f"{row}.querySelector('.exp-toggle').click()")
        browser.wait_for(f"runRowsOf({json.dumps(seeded)}).some(r => r.classList.contains('run-row'))")
        first = f"runRowsOf({json.dumps(seeded)}).find(r => r.classList.contains('run-row'))"
        # A click in the actions cell does not toggle the row around it.
        browser.eval(f"{first}.querySelector('.actions-cell').click()")
        assert browser.eval("[...document.querySelectorAll('.job-detail-row')].every(r => r.style.display === 'none')")
        browser.eval(f"{first}.cells[0].click()")
        browser.wait_for("[...document.querySelectorAll('.job-detail-row')].some(r => r.style.display !== 'none')")
        assert browser.eval("window.__xss") is None


def _middleware_request(output_size: int = 0):
    return types.SimpleNamespace(writer=types.SimpleNamespace(output_size=output_size))


def test_an_unexpected_error_is_a_json_500_without_its_details():
    async def boom(_request):
        raise RuntimeError("/secret/path in SQL")
    resp = asyncio.run(error_middleware(_middleware_request(), boom))
    assert resp.status == 500 and resp.content_type == "application/json"
    assert "secret" not in resp.text and "error" in json.loads(resp.text)


@pytest.mark.parametrize("exc", [web.HTTPNotFound(), ConnectionResetError()], ids=["http", "client gone"])
def test_error_middleware_leaves_expected_exceptions_to_aiohttp(exc):
    async def raises(_request):
        raise exc
    with pytest.raises(type(exc)):
        asyncio.run(error_middleware(_middleware_request(), raises))


def test_error_middleware_cannot_replace_a_response_already_started():
    async def boom(_request):
        raise RuntimeError("mid-stream")
    with pytest.raises(RuntimeError):
        asyncio.run(error_middleware(_middleware_request(output_size=100), boom))


# ── Mistakes get an answer that says what to fix ───────────────────────────────


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/experiments"), ("POST", "/api/jobs"), ("POST", "/api/workers"), ("POST", "/api/artifacts"),
    ("PATCH", "/api/scheduler"), ("PUT", "/api/experiments/nope/status"), ("PUT", "/api/jobs/r/priority"),
    ("PUT", "/api/artifacts/a/ref"), ("POST", "/api/experiments/nope/jobs/bulk"),
    ("PATCH", "/api/workers/w/concurrency"), ("POST", "/api/scheduler/pause"), ("POST", "/api/scheduler/unpause"),
])
def test_a_body_that_is_not_an_object_is_the_clients_mistake(mgr, method, path):
    status, body, _ = _req(mgr, method, path, [1, 2])
    assert status == 400 and json.loads(body)["error"] == "body must be a JSON object"


def test_bulk_insert_explains_bad_shapes(mgr):
    assert _req(mgr, "POST", "/api/jobs/bulk", 7)[0] == 400
    assert _req(mgr, "POST", "/api/jobs/bulk", [1])[0] == 400
    assert _req(mgr, "POST", "/api/jobs/bulk", [{"run_id": "r", "experiment_id": _experiment(mgr)}])[0] == 400
    assert _req(mgr, "POST", f"/api/experiments/{uid('sec')}/jobs/bulk", {"campaign": "test", "jobs": [1]})[0] == 400


def test_job_conflicts_name_the_problem(mgr):
    eid = _experiment(mgr)
    job = {"experiment_id": eid, "run_id": "r", "command": ["true"]}
    assert _req(mgr, "POST", "/api/jobs", job)[0] == 201
    status, body, _ = _req(mgr, "POST", "/api/jobs", job)
    assert status == 409 and b"duplicate run_id" in body
    assert _req(mgr, "POST", "/api/jobs/bulk", [job])[0] == 409
    status, body, _ = _req(mgr, "POST", "/api/jobs", {**job, "experiment_id": "no_such_experiment"})
    assert status == 404 and b"experiment not found" in body


@pytest.mark.parametrize("field", [
    {"status": "bogus"}, {"priority": "high"}, {"priority": True}, {"gpus_per_run": "x"}, {"gpus_per_run": -1},
    {"nodes_per_run": 1.5}, {"max_retries": "x"}, {"command": {"a": 1}}, {"command": [1]}, {"combo": [1]},
    {"env": "A=1"}, {"files": []}, {"return_files": "a"},
], ids=str)
def test_bad_job_fields_are_refused_rather_than_stored(mgr, field):
    eid = _experiment(mgr)
    status, body, _ = _req(mgr, "POST", "/api/jobs", {"experiment_id": eid, "run_id": "r", "command": ["true"], **field})
    assert status == 400 and next(iter(field)) in json.loads(body)["error"]
    assert json.loads(_req(mgr, "GET", f"/api/experiments/{eid}/jobs")[1]) == []


def test_good_job_fields_still_work(mgr):
    eid = _experiment(mgr)
    job = {"experiment_id": eid, "run_id": "r", "command": "echo hi", "priority": 1.5, "gpus_per_run": 0,
           "combo": {"lr": 0.1}, "env": {"A": "1"}, "files": {"a": "b"}, "return_files": ["x"], "max_retries": 0}
    assert _req(mgr, "POST", "/api/jobs", job)[0] == 201


@pytest.mark.parametrize("method,path,body", [
    ("POST", "/api/experiments", {"campaign": "test", "experiment_id": 123}),
    ("PUT", "/api/jobs/r/priority", {"priority": "high", "experiment_id": "e"}),
    ("PUT", "/api/jobs/r/status", {"status": "bogus", "experiment_id": "e"}),
    ("PUT", "/api/jobs/r/status", {"status": "done", "experiment_id": "e", "exit_code": "x"}),
    ("PUT", "/api/artifacts/a/ref", {"delta": "x"}),
    ("POST", "/api/artifacts", {"artifact_id": "a2", "size_bytes": "x"}),
    ("POST", "/api/workers", {"host": "h1", "port": "x"}),
    ("POST", "/api/workers", {"host": "h1", "port": 70000}),
    ("POST", "/api/workers", {"host": "h1", "devices": "0,1"}),
    ("POST", "/api/workers", {"host": "h1", "ssh_key": 5}),
], ids=str)
def test_bad_values_elsewhere_are_refused(mgr, method, path, body):
    assert _req(mgr, method, path, body)[0] == 400


@pytest.mark.parametrize("path", [
    "/api/jobs?limit=abc", "/api/jobs?limit=-1", "/api/jobs/pending?limit=abc",
    "/api/jobs?status=bogus", "/api/experiments/e/jobs?status=bogus",
])
def test_bad_query_parameters_are_refused(mgr, path):
    status, body, _ = _req(mgr, "GET", path)
    assert status == 400 and "error" in json.loads(body)


def test_notes_upload_explains_a_wrong_content_type(mgr):
    status, body, _ = _req(mgr, "POST", "/api/notes/upload", raw=b"x", headers={"Content-Type": "text/plain"})
    assert status == 400 and b"multipart" in body


def test_an_oversized_body_says_so(mgr):
    assert _req(mgr, "POST", "/api/experiments", raw=b"x" * (129 * 1024 * 1024),
                headers={"Content-Type": "application/json"})[0] == 413


def _cli(mgr, *args):
    env = {**os.environ, "MLSWEEP_MANAGER": mgr.url}
    return subprocess.run([sys.executable, "-m", "mlsweep", *args, "--token", "wrong", "--campaign", "test"],
                          capture_output=True, text=True, env=env, timeout=30)


@pytest.mark.parametrize("args", [("ls",), ("workers",), ("cancel", "x", "--all"), ("retry", "x", "--all")], ids=str)
def test_the_cli_says_why_and_fails_on_a_wrong_token(mgr, args):
    r = _cli(mgr, *args)
    assert r.returncode != 0, r.stdout
    assert "wrong or missing token" in r.stdout
    assert "No matching runs" not in r.stdout


def test_watch_reports_a_refused_websocket_in_one_line():
    refusal = _ws_refusal("HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json", b'{"error": "wrong token"}')
    assert refusal == "HTTP 401 Unauthorized: wrong token"
    assert _ws_refusal("HTTP/1.1 400 Bad Request", b"plain") == "HTTP 400 Bad Request: plain"
    assert _ws_refusal("HTTP/1.1 500 Oops", b"") == "HTTP 500 Oops"


def test_a_worker_says_why_it_refused_a_connection(tmp_path):
    w = subprocess.Popen([sys.executable, "-m", "mlsweep.worker", "--token", "right", "--port", "0",
                          "--scratch-dir", str(tmp_path / "s"), "--remote-dir", str(tmp_path)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        port = int(w.stdout.readline().strip().split("=")[1])
        with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
            c.settimeout(10)
            c.sendall(encode(MsgHello(token="wrong", controller_id="m")))
            assert c.recv(100) == b""
    finally:
        w.terminate()
    err = w.communicate(timeout=10)[1]
    assert "token does not match" in err and "wrong" not in err.replace("does not match", "")


@pytest.mark.skipif(CHROME is None, reason="no Chromium-family browser installed")
@pytest.mark.parametrize("page", sorted(p.name for p in WEBUI.glob("*.html") if p.name != "index.html"))
def test_every_page_explains_a_rejected_token(mgr, browser, page):
    browser.goto(f"{mgr.url}/static/{page}?token=wrong")
    text = browser.wait_for("document.getElementById('login-problem')?.textContent")
    assert "rejected this page's token" in text
    # Leave the remembered token right for the other tests in this browser.
    browser.goto(static_url(mgr, "experiments.html"))
