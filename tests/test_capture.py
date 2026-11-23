"""Session capture: the normalizers, attribution, and the manager's HTTP side.

The normalizer fixtures are real transcript lines (a few, trimmed) from pi and
Claude Code on this machine's formats.  The HTTP tests run against a real
manager process, like the other integration tests.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from mlsweep import _capture

# ── pi normalizer ─────────────────────────────────────────────────────────────

PI_SESSION_LINE = {
    "type": "session", "version": 3,
    "id": "01a2047e-e683-706c-8e4d-58c9e737378e",
    "timestamp": "2026-11-22T18:22:35.139Z",
    "cwd": "/home/apaz/git/mlsweep",
}

PI_USER_LINE = {
    "type": "message", "id": "a69a9d3c",
    "timestamp": "2026-11-22T18:22:36.000Z",
    "message": {"role": "user", "content": [{"type": "text", "text": "Run the sweep"}]},
}

PI_ASSISTANT_LINE = {
    "type": "message", "id": "68c798a0",
    "timestamp": "2026-11-22T18:22:37.000Z",
    "message": {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Let me look at the repo."},
            {"type": "text", "text": "On it."},
            {"type": "toolCall", "id": "call_00", "name": "bash",
             "arguments": {"command": "ls -la"}},
            {"type": "toolCall", "id": "call_01", "name": "edit",
             "arguments": {"path": "train.py", "oldText": "a\nb\nc", "newText": "x\ny"}},
        ],
        "usage": {"input": 100, "output": 10, "cost": {"total": 0.001}},
    },
}

PI_TOOL_RESULT_LINE = {
    "type": "message", "id": "a07555e2",
    "timestamp": "2026-11-22T18:22:38.000Z",
    "message": {"role": "toolResult", "toolCallId": "call_00", "toolName": "bash",
                "content": [{"type": "text", "text": "total 8\ndrwxr-xr-x lots of output"}],
                "isError": False, "durationMs": 24},
}

PI_SYSTEM_LINE = {
    "type": "message", "id": "a61a51a5",
    "timestamp": "2026-11-22T18:22:35.200Z",
    "message": {"role": "system", "content": [{"type": "text", "text": "You are an agent..."}]},
}


def _normalize(spec_name, lines):
    spec = next(s for s in _capture.agent_specs([spec_name]))
    meta: dict = {}
    events: list = []
    for line in lines:
        events.extend(spec.normalize(line, 0, meta))
    return meta, events


PI_MODEL_LINE = {
    "type": "model_change", "id": "e995c88a",
    "timestamp": "2026-11-22T18:22:35.300Z",
    "provider": "test", "modelId": "test-model",
}


def test_pi_normalizer_produces_expected_events():
    meta, events = _normalize("pi", [
        PI_SESSION_LINE, PI_MODEL_LINE, PI_SYSTEM_LINE, PI_USER_LINE, PI_ASSISTANT_LINE,
        PI_TOOL_RESULT_LINE,
    ])
    kinds = [e["kind"] for e in events]
    assert kinds == ["meta", "user", "thinking", "assistant", "tool", "tool", "tool_result"]
    assert meta["cwd"] == "/home/apaz/git/mlsweep"
    assert meta["native_id"] == "01a2047e-e683-706c-8e4d-58c9e737378e"
    # A model switch stays in the ledger, so later events make sense.
    model, user, thinking, assistant, tool, edit, result = events
    assert model["text"] == "model → test-model"
    assert user["text"] == "Run the sweep"
    assert thinking["text"] == "Let me look at the repo."
    assert assistant["text"] == "On it."
    # The harness reports a cost; the trace does not record it.
    assert assistant["model"] and "cost" not in assistant
    assert tool["name"] == "bash" and tool["text"] == "ls -la" and tool["tool_id"] == "call_00"
    # An edit's arguments never carry the file's contents.
    assert "a\nb\nc" not in edit["text"] and "<3 chars>" in edit["text"]
    assert "train.py" in edit["text"]
    # A tool result keeps its status, never its output.
    assert result["ok"] is True and result["duration_ms"] == 24
    assert "output" not in json.dumps(result)
    # Within one line the events number onward; across lines the caller
    # renumbers (see read_transcript).
    assert [e["seq"] for e in events] == [0, 0, 0, 1, 2, 3, 0]


def test_pi_session_id_is_the_native_id_without_prefix():
    spec = _capture.agent_specs(["pi"])[0]
    session_id, _ = _capture.read_transcript(spec, _write_transcript([
        PI_SESSION_LINE, PI_USER_LINE,
    ]))
    assert session_id == "01a2047e-e683-706c-8e4d-58c9e737378e"  # no agent prefix


def test_pi_prompt_is_kept_tool_output_is_not():
    spec = _capture.agent_specs(["pi"])[0]
    _, events = _capture.read_transcript(spec, _write_transcript([
        PI_SESSION_LINE, PI_USER_LINE, PI_ASSISTANT_LINE, PI_TOOL_RESULT_LINE,
    ]))
    text = json.dumps(events)
    assert "Run the sweep" in text and "On it." in text
    assert "lots of output" not in text and "total 8" not in text


# ── Claude Code normalizer ────────────────────────────────────────────────────

CLAUDE_USER_LINE = {
    "type": "user", "sessionId": "12f1729d-fc17-4245-bbdc-ff9d9231f36f",
    "cwd": "/home/apaz/git/mlsweep",
    "timestamp": "2026-11-11T04:02:13.356Z",
    "origin": {"kind": "human"},
    "message": {"role": "user", "content": "Add a way to auto-update."},
}

CLAUDE_ASSISTANT_LINE = {
    "type": "assistant", "sessionId": "12f1729d-fc17-4245-bbdc-ff9d9231f36f",
    "timestamp": "2026-11-11T04:02:20.000Z",
    "message": {
        "role": "assistant", "model": "claude-x",
        "content": [
            {"type": "thinking", "thinking": "Plan first."},
            {"type": "tool_use", "id": "tu_1", "name": "Read",
             "input": {"file_path": "/x/train.py", "limit": 40}},
        ],
        "usage": {"input_tokens": 1200, "output_tokens": 30},
    },
}

CLAUDE_TOOL_RESULT_LINE = {
    "type": "user", "sessionId": "12f1729d-fc17-4245-bbdc-ff9d9231f36f",
    "timestamp": "2026-11-11T04:02:21.000Z",
    "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tu_1", "is_error": False,
         "content": "the file's whole contents"}]},
}


def test_claude_normalizer():
    meta, events = _normalize("claude_code", [
        CLAUDE_USER_LINE, CLAUDE_ASSISTANT_LINE, CLAUDE_TOOL_RESULT_LINE,
    ])
    kinds = [e["kind"] for e in events]
    assert kinds == ["user", "thinking", "tool", "tool_result"]
    assert meta["native_id"] == "12f1729d-fc17-4245-bbdc-ff9d9231f36f"
    assert meta["cwd"] == "/home/apaz/git/mlsweep"
    tool = events[2]
    assert tool["name"] == "Read" and "train.py" in tool["text"]
    # Tokens are read from the Claude Code shape.
    assert events[2]["tokens"] == {"input": 1200, "output": 30}


# ── attribution ───────────────────────────────────────────────────────────────


def test_attribution_from_agent_environment():
    assert _capture.resolve_capture_session({
        "PI_SESSION_ID": "abc-123"}) == "abc-123"
    assert _capture.resolve_capture_session({
        "PI_SESSION_FILE": "/home/u/.pi/agent/sessions/x/2026-01-01T00-00-00-000Z_abc-456.jsonl",
    }) == "abc-456"
    assert _capture.resolve_capture_session({
        "CLAUDE_SESSION_ID": "cc-1"}) == "cc-1"


def test_mlsweep_session_id_overrides_everything():
    assert _capture.resolve_capture_session({
        "MLSWEEP_SESSION_ID": "manual-1", "PI_SESSION_ID": "agent-1"}) == "manual-1"


EMPTY_REGISTRY = {"sessions": [], "processes": []}


def test_no_agent_environment_means_no_trace():
    assert _capture.resolve_capture_session(
        {}, ancestors=[], registry=EMPTY_REGISTRY) is None


def test_attribution_never_refuses_a_submission():
    # Even a broken watcher is not the submission's business: the fallbacks
    # read whatever they can find, and answer None without raising.
    assert _capture.resolve_capture_session(
        {}, ancestors=[], registry=EMPTY_REGISTRY) is None


# ── the manager's HTTP side ───────────────────────────────────────────────────


def _post(url, token, path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{url}{path}", data=body, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _get(url, token, path):
    req = urllib.request.Request(f"{url}{path}", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _delete(url, token, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{url}{path}", data=body, method="DELETE", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


_SESSION = {
    "session_id": "01a2047e-e683-706c-8e4d-58c9e737378e",
    "agent": "pi", "host": "box", "user": "apaz", "cwd": "/home/apaz/git/mlsweep",
    "title": "Run the sweep", "model": "test-model",
    "started_at": 1000.0, "last_seen_at": 1100.0, "state": "active",
}

_EVENTS = [
    {"seq": 1, "ts": 1000.0, "kind": "user", "text": "Run the sweep"},
    {"seq": 2, "ts": 1001.0, "kind": "assistant", "text": "On it.",
     "model": "test-model"},
    {"seq": 3, "ts": 1002.0, "kind": "tool", "name": "bash", "text": "ls -la"},
]


def test_capture_upload_read_search_delete_roundtrip(manager_server):
    server, url = manager_server
    token = server.token

    status, body = _post(url, token, "/api/capture/sessions",
                         {"session": _SESSION, "events": _EVENTS})
    assert status == 201, body
    assert body["session_id"] == _SESSION["session_id"]
    assert body["events_stored"] == 3

    # A retried upload is a no-op, not a duplicate.
    status, body = _post(url, token, "/api/capture/sessions",
                         {"session": _SESSION, "events": _EVENTS})
    assert status == 201
    assert body["events_stored"] == 0

    status, body = _get(url, token, f"/api/capture/sessions/{_SESSION['session_id']}")
    assert status == 200 and body["event_count"] == 3 and body["experiments"] == []

    status, body = _get(url, token, f"/api/capture/sessions/{_SESSION['session_id']}/events")
    assert status == 200
    assert [e["kind"] for e in body["events"]] == ["user", "assistant", "tool"]
    assert body["next_from_seq"] == 4

    status, body = _get(url, token, "/api/capture/sessions")
    assert status == 200 and len(body) == 1 and body[0]["agent"] == "pi"

    status, body = _get(url, token, "/api/capture/search?q=sweep")
    assert status == 200 and len(body) == 1
    assert "sweep" in body[0]["snippet"].lower()

    # Not found is 404 for a well-formed id.
    status, _ = _get(url, token, "/api/capture/sessions/no-such-trace")
    assert status == 404


def test_capture_requires_a_session_before_events(manager_server):
    server, url = manager_server
    status, _ = _post(url, server.token,
                      "/api/capture/sessions/no-such-trace/close", {})
    assert status == 404


def test_capture_close_and_state_transitions(manager_server):
    server, url = manager_server
    token = server.token
    status, _ = _post(url, token, "/api/capture/sessions", {"session": _SESSION, "events": _EVENTS})
    assert status == 201
    status, body = _post(url, token, f"/api/capture/sessions/{_SESSION['session_id']}/close", {})
    assert status == 200 and body["state"] == "closed"

    # An active upload never revives a closed session (imported wins too).
    active = {**_SESSION, "state": "active"}
    status, body = _post(url, token, "/api/capture/sessions", {"session": active, "events": []})
    assert status == 201 and body["state"] == "closed"


def test_capture_auto_link_experiment(manager_server):
    server, url = manager_server
    token = server.token
    _post(url, token, "/api/capture/sessions", {"session": _SESSION, "events": _EVENTS})

    experiment = {
        "experiment_id": "sweep_test_1", "name": "test", "campaign": "main",
        "capture_session": _SESSION["session_id"],
    }
    status, body = _post(url, token, "/api/experiments", experiment)
    assert status == 201, body

    status, body = _get(url, token, "/api/experiments/sweep_test_1/capture")
    assert status == 200 and len(body) == 1
    assert body[0]["session_id"] == _SESSION["session_id"]

    # And back the other way.
    status, body = _get(url, token, f"/api/capture/sessions/{_SESSION['session_id']}")
    assert body["experiments"] == ["sweep_test_1"]


def test_capture_link_to_unknown_trace_is_refused(manager_server):
    server, url = manager_server
    _post(url, server.token, "/api/capture/sessions", {"session": _SESSION, "events": []})
    status, _ = _post(url, server.token, "/api/capture/link",
                      {"session_id": "ghost", "experiment_id": "sweep_test_2"})
    assert status == 404


def test_capture_manual_link_and_unlink(manager_server):
    server, url = manager_server
    token = server.token
    _post(url, token, "/api/capture/sessions", {"session": _SESSION, "events": []})
    _post(url, token, "/api/experiments",
          {"experiment_id": "sweep_link_1", "campaign": "main"})
    status, body = _post(url, token, "/api/capture/link",
                         {"session_id": _SESSION["session_id"], "experiment_id": "sweep_link_1",
                          "source": "manual"})
    assert status == 201 and body["linked"] is True
    status, body = _get(url, token, f"/api/capture/sessions/{_SESSION['session_id']}")
    assert body["experiments"] == ["sweep_link_1"]
    status, body = _delete(url, token, "/api/capture/link",
                           {"session_id": _SESSION["session_id"],
                            "experiment_id": "sweep_link_1"})
    assert status == 200 and body["unlinked"] is True
    status, body = _get(url, token, f"/api/capture/sessions/{_SESSION['session_id']}")
    assert body["experiments"] == []


def test_capture_search_titles_counts_and_link_errors(manager_server):
    server, url = manager_server
    token = server.token
    session = {**_SESSION, "session_id": "searchme", "title": "Warmup investigation"}
    _post(url, token, "/api/capture/sessions",
          {"session": session, "events": [{"seq": 1, "ts": 1.0, "kind": "user", "text": "hello"}]})

    # A title match finds the trace even though the FTS index holds only events.
    status, body = _get(url, token, "/api/capture/search?q=warmup")
    assert status == 200 and [s["session_id"] for s in body] == ["searchme"]
    assert body[0]["snippet"] == "Warmup investigation"

    # A quote is a search term, not an FTS5 syntax error.
    status, _ = _get(url, token, "/api/capture/search?q=%22warmup")
    assert status == 200

    # Counts drive the View menus, without listing every session.
    _post(url, token, "/api/experiments", {"experiment_id": "sweep_counts_1", "campaign": "main"})
    _post(url, token, "/api/capture/link",
          {"session_id": "searchme", "experiment_id": "sweep_counts_1", "source": "manual"})
    status, body = _get(url, token, "/api/capture/counts")
    assert status == 200 and body == {"sessions": 1, "experiments": {"sweep_counts_1": 1}}

    # Linking to an experiment the manager does not have is a 404, not a 500.
    status, _ = _post(url, token, "/api/capture/link",
                      {"session_id": "searchme", "experiment_id": "no_such_experiment"})
    assert status == 404


def test_capture_delete(manager_server):
    server, url = manager_server
    token = server.token
    _post(url, token, "/api/capture/sessions", {"session": _SESSION, "events": _EVENTS})
    status, body = _delete(url, token, f"/api/capture/sessions/{_SESSION['session_id']}")
    assert status == 200
    status, _ = _get(url, token, f"/api/capture/sessions/{_SESSION['session_id']}")
    assert status == 404
    status, body = _get(url, token, "/api/capture/sessions")
    assert body == []


def test_capture_wrong_token_is_401(manager_server):
    _, url = manager_server
    status, _ = _post(url, "wrong-token", "/api/capture/sessions", {"session": _SESSION})
    assert status == 401


# ── trace links from notes ────────────────────────────────────────────────────


def test_lint_checks_trace_refs():
    import tempfile
    from pathlib import Path

    from mlsweep import _notes_lint

    good = "01a2047e-e683-706c-8e4d-58c9e737378e"
    with tempfile.TemporaryDirectory() as root:
        note = Path(root) / "a.md"
        note.write_text(
            f"The sweep came out of [an idea](mlsweep:trace/{good}).\n\n"
            "And [a ghost](mlsweep:trace/ghost-0001).\n")
        text = note.read_text()
    evidence = {
        f"trace/{good}": {"ok": True, "ref": f"trace/{good}"},
        "trace/ghost-0001": {"ok": False, "ref": "trace/ghost-0001", "error": "no such trace"},
    }
    findings = []
    _notes_lint._check_blocks("a.md", text, None, None, evidence, findings)
    assert not any(f.kind == "evidence-ref" and good in f.message for f in findings)
    assert any(f.kind == "evidence-ref" and "ghost-0001" in f.message for f in findings)


def test_trace_ref_resolution_shape():
    # The resolver's output shape, without a manager: what the notes tab reads.
    assert _capture._from_env({"MLSWEEP_SESSION_ID": "x"}) == "x"


# ── helpers ───────────────────────────────────────────────────────────────────


def _write_transcript(lines):
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
        path = Path(f.name)
    return path


# ── the watcher, end to end ───────────────────────────────────────────────────


def test_capture_disabled_on_manager(tmp_path):
    import urllib.error
    import urllib.request

    from tests.conftest import _start_manager, _teardown_manager

    proc, server, url = _start_manager(tmp_path, with_worker=False, extra_args=("--no-capture",))
    try:
        payload = json.dumps({"session": _SESSION, "events": []}).encode()
        req = urllib.request.Request(
            f"{url}/api/capture/sessions", data=payload, method="POST",
            headers={"Authorization": f"Bearer {server.token}", "Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req, timeout=10)
        assert excinfo.value.code == 404  # the routes are simply not there
    finally:
        _teardown_manager(proc)


def test_watcher_uploads_session_end_to_end(manager_server, tmp_path):
    """A transcript on disk becomes a trace on the manager, tool output and all."""
    server, url = manager_server
    token = server.token
    sessions_root = tmp_path / ".pi" / "agent" / "sessions"  # what _sessions_root returns
    project_dir = sessions_root / "--home-apaz-git-mlsweep--"
    project_dir.mkdir(parents=True)
    transcript = project_dir / "2026-11-22T18-22-35-139Z_01a1cbca-f65f-71f6-888e-e3d1da5280ff.jsonl"
    transcript.write_text("\n".join(json.dumps(line) for line in [
        PI_SESSION_LINE, PI_USER_LINE, PI_ASSISTANT_LINE, PI_TOOL_RESULT_LINE,
    ]) + "\n")

    client = _capture.CaptureClient(directory=tmp_path / "capture")
    specs = _capture.agent_specs(["pi"], root=sessions_root)
    client.scan(specs)
    state = client.sessions[_capture.session_id_for("pi", "01a1cbca-f65f-71f6-888e-e3d1da5280ff")]
    assert state.cwd == "/home/apaz/git/mlsweep"
    assert state.offset == transcript.stat().st_size
    # The spool holds the events; the manager has never seen them.
    assert client.pending_count(state.session_id) == 6

    # An upload happens when the transcript goes quiet; drive it directly.
    _capture._upload(client, state, url, token)
    status, body = _get(url, token, f"/api/capture/sessions/{state.session_id}")
    assert status == 200
    assert body["event_count"] == 6
    assert body["title"] == "Run the sweep"
    assert client.pending_count(state.session_id) == 0  # accepted, so the spool is empty

    status, events = _get(url, token, f"/api/capture/sessions/{state.session_id}/events")
    kinds = [e["kind"] for e in events["events"]]
    assert kinds == ["user", "thinking", "assistant", "tool", "tool", "tool_result"]
    assert not any("lots of output" in json.dumps(e) for e in events["events"])


# ── the Traces tab in a real browser ──────────────────────────────────────────


def test_traces_page_renders_a_session(tmp_path_factory):
    """The uploaded session shows as ledger rows; the inspector opens on a click."""
    from browser import Browser, find_chromium
    from campaign_helpers import shared_manager, static_url, call

    if find_chromium() is None:
        import pytest
        pytest.skip("no Chromium")

    server = next(shared_manager(tmp_path_factory, "capture_webui"))
    call(server, "POST", "/api/capture/sessions", {
        "session": _SESSION, "events": _EVENTS,
    })
    call(server, "POST", "/api/experiments", {
        "experiment_id": "sweep_traced_1", "campaign": "main",
        "capture_session": _SESSION["session_id"],
    })
    b = Browser(find_chromium())
    try:
        b.call("Emulation.setDeviceMetricsOverride", width=1300, height=900,
               deviceScaleFactor=1, mobile=False)
        b.goto(static_url(server, "traces.html", session=_SESSION["session_id"]))
        b.wait_for("[...document.querySelectorAll('.session-item')].some(x => x.classList.contains('selected'))")
        b.wait_for("document.querySelectorAll('#ledger tbody tr').length > 0")
        assert b.eval("document.querySelectorAll('#ledger tbody tr').length") == len(_EVENTS)
        # The session header carries the trace's title and its linked experiment.
        assert b.eval("document.getElementById('trace-title').textContent") == "Run the sweep"
        assert b.eval("[...document.querySelectorAll('#trace-links .link-chip')].map(a => a.textContent)") == ["sweep_traced_1"]
        # The first event is the user's prompt; clicking opens the inspector.
        assert b.eval("document.querySelector('#ledger tbody tr .event-text').textContent") == "Run the sweep"
        b.eval("document.querySelector('#ledger tbody tr').click()")
        b.wait_for("document.getElementById('inspector').classList.contains('open')")
        assert b.eval("document.getElementById('inspector-text').textContent") == "Run the sweep"
        # A tool call is one line with its name; tool results show a status, not output.
        b.eval("document.querySelectorAll('#ledger tbody tr')[2].click()")
        b.wait_for("document.getElementById('inspector-kind').textContent === 'bash'")

        # The address bar holds the open trace, so a reload keeps it.
        assert "session=" + _SESSION["session_id"] in b.eval("location.search")
        # Time, turn counter and the filters are part of the ledger.
        assert b.eval("[...document.querySelectorAll('#ledger .time-col')]"
                      ".every(td => td.textContent.length > 0)")
        assert b.eval("document.getElementById('turn-pos').textContent") == "1 / 1"
        # The More menu carries the note reference; deleting lives there, not
        # next to Refresh.
        b.eval("document.getElementById('menu-btn').click()")
        b.wait_for("!!document.querySelector('.ml-ctx')")
        items = b.eval("[...document.querySelectorAll('.ml-ctx-item .l')].map(x => x.textContent)")
        assert "Copy note reference" in items and "Delete trace" in items, items
        b.eval("document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))")
        # Filtering by state narrows the list, and clearing brings it back.
        b.eval("[...document.querySelectorAll('#state-chips .chip')]"
               ".find(c => c.dataset.state === 'closed').click()")
        b.wait_for("document.querySelectorAll('.session-item').length === 0")
        assert "No traces match" in b.eval("document.getElementById('session-list').textContent")
        b.eval("[...document.querySelectorAll('#state-chips .chip')]"
               ".find(c => c.dataset.state === 'all').click()")
        b.wait_for("document.querySelectorAll('.session-item').length === 1")
        # View -> Traces on an experiment filters to its traces, not the whole list.
        b.goto(static_url(server, "traces.html", experiment="sweep_traced_1"))
        b.wait_for("document.querySelectorAll('.session-item').length === 1")
        info = b.eval("document.getElementById('list-info').textContent")
        assert info.startswith("1 trace") and "exp sweep_traced_1" in info, info
    finally:
        b.close()
