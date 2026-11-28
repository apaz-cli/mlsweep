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

from conftest import _api_status
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
        events.extend(spec.normalize(line, meta))
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
    # The caller numbers events across the session (see read_transcript).
    assert not any("seq" in e for e in events)


def test_pi_session_id_is_the_native_id_without_prefix():
    spec = _capture.agent_specs(["pi"])[0]
    session_id, _, _ = _capture.read_transcript(spec, _write_transcript([
        PI_SESSION_LINE, PI_USER_LINE,
    ]))
    assert session_id == "01a2047e-e683-706c-8e4d-58c9e737378e"  # no agent prefix


def test_pi_prompt_is_kept_tool_output_is_not():
    spec = _capture.agent_specs(["pi"])[0]
    _, events, _ = _capture.read_transcript(spec, _write_transcript([
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
        "CLAUDE_CODE_SESSION_ID": "cc-1"}) == "cc-1"


def test_mlsweep_session_id_overrides_everything():
    assert _capture.resolve_capture_session({
        "MLSWEEP_SESSION_ID": "manual-1", "PI_SESSION_ID": "agent-1"}) == "manual-1"


def test_no_agent_environment_means_no_trace():
    # Nothing is guessed from the working directory or recent transcripts.
    assert _capture.resolve_capture_session({}, ancestors=[]) is None


def test_attribution_reads_a_gone_ancestor_as_nothing():
    # A pid that no longer exists is skipped, not an error.
    assert _capture.resolve_capture_session({}, ancestors=[2**22 + 12345]) is None


# ── the manager's HTTP side ───────────────────────────────────────────────────


def _post(url, token, path, data):
    return _api_status(url, token, "POST", path, data)


def _get(url, token, path):
    return _api_status(url, token, "GET", path)


def _delete(url, token, path, data=None):
    return _api_status(url, token, "DELETE", path, data)


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


def test_capture_oversized_event_is_trimmed_not_broken(manager_server):
    # A thinking block over the per-event cap once made the whole upload fail.
    server, url = manager_server
    token = server.token
    session = {**_SESSION, "session_id": "oversized-1"}
    big = "zebra " + "x" * 100_000
    status, _ = _post(url, token, "/api/capture/sessions", {"session": session, "events": [
        {"seq": 1, "ts": 1.0, "kind": "user", "text": "short"},
        {"seq": 2, "ts": 2.0, "kind": "thinking", "text": big}]})
    assert status == 201
    status, body = _get(url, token, "/api/capture/sessions/oversized-1/events")
    assert status == 200
    long_event = body["events"][1]
    assert long_event["truncated"] is True
    assert big.startswith(long_event["text"]) and 30_000 < len(long_event["text"]) < 64 * 1024
    status, hits = _get(url, token, "/api/capture/search?q=zebra")
    assert status == 200 and "oversized-1" in json.dumps(hits)


def test_capture_close_and_state_transitions(manager_server):
    server, url = manager_server
    token = server.token
    status, _ = _post(url, token, "/api/capture/sessions", {"session": _SESSION, "events": _EVENTS})
    assert status == 201
    closed = {**_SESSION, "state": "closed"}
    status, body = _post(url, token, "/api/capture/sessions", {"session": closed, "events": []})
    assert status == 201 and body["state"] == "closed"

    # An import never replaces the live state.
    imported = {**_SESSION, "state": "imported"}
    status, body = _post(url, token, "/api/capture/sessions", {"session": imported, "events": []})
    assert status == 201 and body["state"] == "closed"

    # The watcher saw the session resume: it is active again.
    active = {**_SESSION, "state": "active"}
    status, body = _post(url, token, "/api/capture/sessions", {"session": active, "events": []})
    assert status == 201 and body["state"] == "active"


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


def test_capture_link_made_before_the_session_uploads(manager_server):
    # A sweep is submitted before the watcher has uploaded its session; the
    # link waits for the trace instead of being dropped.
    server, url = manager_server
    token = server.token
    status, _ = _post(url, token, "/api/experiments", {
        "experiment_id": "sweep_early_1", "campaign": "main",
        "capture_session": _SESSION["session_id"]})
    assert status == 201
    status, body = _get(url, token, "/api/capture/counts")
    assert body["experiments"] == {}  # no trace yet, so nothing to count
    _post(url, token, "/api/capture/sessions", {"session": _SESSION, "events": _EVENTS})
    status, body = _get(url, token, f"/api/capture/sessions/{_SESSION['session_id']}")
    assert body["experiments"] == ["sweep_early_1"]
    status, body = _get(url, token, "/api/capture/counts")
    assert body["experiments"] == {"sweep_early_1": 1}


def test_capture_search_spans_batches_and_tool_names(manager_server):
    server, url = manager_server
    token = server.token
    _post(url, token, "/api/capture/sessions", {"session": _SESSION, "events": _EVENTS[:1]})
    _post(url, token, "/api/capture/sessions", {"session": _SESSION, "events": [
        {"seq": 2, "ts": 1001.0, "kind": "tool", "name": "frobnicate", "text": "x=1"},
        {"seq": 3, "ts": 1002.0, "kind": "assistant", "text": "sweep again"}]})
    status, body = _get(url, token, "/api/capture/search?q=sweep")
    assert status == 200 and len(body) == 1  # one hit per session, not per batch
    status, body = _get(url, token, "/api/capture/search?q=frobnicate")
    assert status == 200 and [s["session_id"] for s in body] == [_SESSION["session_id"]]


def test_capture_transcript_has_every_event(manager_server):
    server, url = manager_server
    token = server.token
    events = [{"seq": i, "ts": float(i), "kind": "user", "text": f"p{i}"} for i in range(1, 2502)]
    status, _ = _post(url, token, "/api/capture/sessions", {"session": _SESSION, "events": events})
    assert status == 201
    req = urllib.request.Request(
        f"{url}/api/capture/sessions/{_SESSION['session_id']}/transcript",
        headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        lines = resp.read().decode().splitlines()
    assert len(lines) == 2501 and json.loads(lines[-1])["seq"] == 2501


def test_capture_has_no_manual_link(manager_server):
    # Links come only from the submission that names its session.
    server, url = manager_server
    _post(url, server.token, "/api/capture/sessions", {"session": _SESSION, "events": []})
    _post(url, server.token, "/api/experiments",
          {"experiment_id": "sweep_test_2", "campaign": "main"})
    req = urllib.request.Request(
        f"{url}/api/capture/link", method="POST",
        data=json.dumps({"session_id": _SESSION["session_id"], "experiment_id": "sweep_test_2"}).encode(),
        headers={"Authorization": f"Bearer {server.token}", "Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(req, timeout=10)
    assert refused.value.code in (404, 405)
    status, body = _get(url, server.token, f"/api/capture/sessions/{_SESSION['session_id']}")
    assert body["experiments"] == []


def test_capture_unlink_a_wrong_auto_link(manager_server):
    server, url = manager_server
    token = server.token
    _post(url, token, "/api/capture/sessions", {"session": _SESSION, "events": []})
    status, _ = _post(url, token, "/api/experiments",
                      {"experiment_id": "sweep_link_1", "campaign": "main",
                       "capture_session": _SESSION["session_id"]})
    assert status == 201
    status, body = _get(url, token, f"/api/capture/sessions/{_SESSION['session_id']}")
    assert body["experiments"] == ["sweep_link_1"]
    status, body = _delete(url, token, "/api/capture/link",
                           {"session_id": _SESSION["session_id"],
                            "experiment_id": "sweep_link_1"})
    assert status == 200 and body["unlinked"] is True
    status, body = _get(url, token, f"/api/capture/sessions/{_SESSION['session_id']}")
    assert body["experiments"] == []


def test_capture_search_titles_and_counts(manager_server):
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
    _post(url, token, "/api/experiments", {"experiment_id": "sweep_counts_1", "campaign": "main",
                                           "capture_session": "searchme"})
    status, body = _get(url, token, "/api/capture/counts")
    assert status == 200 and body == {"sessions": 1, "experiments": {"sweep_counts_1": 1}}


def test_trace_found_by_a_unique_prefix_of_its_id(manager_server):
    # A note may write mlsweep:trace/<first 8 characters>, like a short git hash.
    server, url = manager_server
    token = server.token
    for sid in ("abcdef01-1111-4000-8000-000000000001", "abcdef01-2222-4000-8000-000000000002",
                "fedcba98-3333-4000-8000-000000000003"):
        _post(url, token, "/api/capture/sessions", {"session": {**_SESSION, "session_id": sid},
                                                     "events": []})
    status, body = _get(url, token, "/api/capture/sessions/fedcba98")
    assert status == 200 and body["session_id"] == "fedcba98-3333-4000-8000-000000000003"
    status, body = _get(url, token, "/api/capture/sessions/abcdef01-2")
    assert status == 200 and body["session_id"] == "abcdef01-2222-4000-8000-000000000002"
    status, body = _get(url, token, "/api/capture/sessions/abcdef01")
    assert status == 400 and "more than one" in body["error"]
    status, _ = _get(url, token, "/api/capture/sessions/fedcb")  # too short to guess from
    assert status == 404


def test_note_trace_links_resolve_prefixes():
    import asyncio

    import aiosqlite

    from mlsweep._manager_capture import resolve_trace_refs
    from mlsweep._manager_db import init_db, store_capture_batch

    async def run():
        async with aiosqlite.connect(":memory:") as db:
            await init_db(db)
            for sid in ("9f6c0e22-aaaa-4000-8000-000000000001", "9f6c0e22-bbbb-4000-8000-000000000002",
                        "12345678-cccc-4000-8000-000000000003"):
                await store_capture_batch(db, {**_SESSION, "session_id": sid}, [])
            return await resolve_trace_refs(db, [
                "trace/12345678", "trace/9f6c0e22", "trace/9f6c0e22-bbbb-4000-8000-000000000002",
                "trace/00000000"])

    out = asyncio.run(run())
    assert out["trace/12345678"]["ok"] and out["trace/12345678"]["session_id"].startswith("12345678-")
    assert not out["trace/9f6c0e22"]["ok"] and "more than one" in out["trace/9f6c0e22"]["error"]
    assert out["trace/9f6c0e22-bbbb-4000-8000-000000000002"]["ok"]
    assert out["trace/00000000"] == {"ok": False, "ref": "trace/00000000", "error": "no such trace"}


def test_trace_ref_parses_kind_and_event():
    import pytest

    from mlsweep._manager_capture import parse_trace_ref

    assert parse_trace_ref("trace/abc12345") == ("abc12345", "", None)
    assert parse_trace_ref("trace/abc12345#E42") == ("abc12345", "", 42)
    assert parse_trace_ref("trace/abc12345?kind=mlsweep") == ("abc12345", "mlsweep", None)
    assert parse_trace_ref("trace/abc12345?kind=user#E7") == ("abc12345", "user", 7)
    with pytest.raises(ValueError, match="#E and its number"):
        parse_trace_ref("trace/abc12345#L10")
    with pytest.raises(ValueError, match="not a trace option"):
        parse_trace_ref("trace/abc12345?kind=bash")
    with pytest.raises(ValueError, match="not a trace option"):
        parse_trace_ref("trace/abc12345?seq=3")


def test_trace_links_to_an_event_resolve_it():
    import asyncio

    import aiosqlite

    from mlsweep._manager_capture import resolve_trace_refs
    from mlsweep._manager_db import init_db, store_capture_batch

    sid = _SESSION["session_id"]

    async def run():
        async with aiosqlite.connect(":memory:") as db:
            await init_db(db)
            await store_capture_batch(db, _SESSION, _EVENTS)
            return await resolve_trace_refs(db, [
                f"trace/{sid[:8]}#E3", f"trace/{sid}?kind=tools", f"trace/{sid}#E99",
                f"trace/{sid}#Efoo"])

    out = asyncio.run(run())
    hit = out[f"trace/{sid[:8]}#E3"]
    assert hit["ok"] and hit["session_id"] == sid, hit
    assert hit["event"]["kind"] == "tool" and hit["event"]["name"] == "bash" and hit["event"]["text"] == "ls -la"
    assert out[f"trace/{sid}?kind=tools"]["ok"] and out[f"trace/{sid}?kind=tools"]["kind"] == "tools"
    gone = out[f"trace/{sid}#E99"]
    assert not gone["ok"] and "no event 99" in gone["error"], gone
    assert not out[f"trace/{sid}#Efoo"]["ok"]


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

    from mlsweep import _notes_lint

    good = "01a2047e-e683-706c-8e4d-58c9e737378e"
    with tempfile.TemporaryDirectory() as root:
        note = Path(root) / "a.md"
        note.write_text(
            f"The sweep came out of [an idea](mlsweep:trace/{good}).\n\n"
            "And [a ghost](mlsweep:trace/ghost-0001).\n\n"
            f"And [an event it lacks](mlsweep:trace/{good}#E99).\n")
        text = note.read_text()
    evidence = {
        f"trace/{good}": {"ok": True, "ref": f"trace/{good}"},
        "trace/ghost-0001": {"ok": False, "ref": "trace/ghost-0001", "error": "no such trace"},
        f"trace/{good}#E99": {"ok": False, "ref": f"trace/{good}#E99", "error": "trace 01a2047e has no event 99"},
    }
    findings = []
    _notes_lint._check_blocks("a.md", text, None, None, evidence, findings)
    assert not any(f.kind == "evidence-ref" and f"mlsweep:trace/{good}:" in f.message for f in findings)
    assert any(f.kind == "evidence-ref" and "no event 99" in f.message for f in findings)
    assert any(f.kind == "evidence-ref" and "ghost-0001" in f.message for f in findings)


def test_trace_ref_resolution_shape():
    # The resolver's output shape, without a manager: what the notes tab reads.
    assert _capture._from_env({"MLSWEEP_SESSION_ID": "x"}) == "x"


# ── helpers ───────────────────────────────────────────────────────────────────


def _write_transcript(lines):
    import tempfile

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
    session_id = "01a1cbca-f65f-71f6-888e-e3d1da5280ff"
    _capture.request_follow(session_id, client.follow_dir, why="test")
    client.follow_requested(specs)
    client.scan(specs)
    state = client.sessions[session_id]
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


    # Quiet long enough, it closes; with nothing left to send, the close
    # still reaches the manager, once.
    state.closed = True
    assert client.needs_close_upload(state)
    _capture._upload(client, state, url, token)
    assert not client.needs_close_upload(state)
    status, body = _get(url, token, f"/api/capture/sessions/{state.session_id}")
    assert body["state"] == "closed" and body["event_count"] == 6


def test_post_session_sends_a_long_spool_in_batches(manager_server):
    server, url = manager_server
    events = [{"seq": i, "ts": float(i), "kind": "user", "text": f"p{i}"} for i in range(1, 6)]
    sent, error = _capture.post_session(url, server.token, _SESSION, events, batch_events=2)
    assert (sent, error) == (5, None)
    status, body = _get(url, server.token, f"/api/capture/sessions/{_SESSION['session_id']}")
    assert body["event_count"] == 5


# ── which sessions the watcher reads ──────────────────────────────────────────


CLAUDE_COMMAND_LINE = {
    "type": "user", "sessionId": "cc-follow-1", "cwd": "/home/apaz/git/mlsweep",
    "timestamp": "2026-11-11T04:02:00.000Z",
    "message": {"role": "user", "content": (
        "<command-name>/login</command-name>\n            "
        "<command-message>login</command-message>\n            "
        "<command-args></command-args>")},
}

CLAUDE_COMMAND_OUTPUT_LINE = {
    "type": "user", "sessionId": "cc-follow-1", "cwd": "/home/apaz/git/mlsweep",
    "timestamp": "2026-11-11T04:02:01.000Z",
    "message": {"role": "user", "content": "<local-command-stdout>Login successful</local-command-stdout>"},
}

CLAUDE_HUMAN_LINE = {
    "type": "user", "sessionId": "cc-follow-1", "cwd": "/home/apaz/git/mlsweep",
    "origin": {"kind": "human"}, "timestamp": "2026-11-11T04:02:05.000Z",
    "message": {"role": "user", "content": "Submit some test\nsweeps please"},
}


def _write_jsonl(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_watcher_reads_only_followed_sessions(tmp_path):
    root = tmp_path / "projects"
    followed = root / "-home-apaz-git-mlsweep" / "cc-follow-1.jsonl"
    other = root / "-home-apaz-elsewhere" / "cc-other-2.jsonl"
    _write_jsonl(followed, [CLAUDE_COMMAND_LINE, CLAUDE_HUMAN_LINE, CLAUDE_ASSISTANT_LINE])
    _write_jsonl(other, [CLAUDE_HUMAN_LINE, CLAUDE_ASSISTANT_LINE])
    client = _capture.CaptureClient(directory=tmp_path / "capture")
    specs = _capture.agent_specs(["claude_code"], root=root)

    # Turned on, following nothing: no transcript is read at all.
    client.follow_requested(specs)
    client.scan(specs)
    assert client.sessions == {}
    assert not client.pending_dir.exists()

    _capture.request_follow("cc-follow-1", client.follow_dir, why="capture this")
    client.follow_requested(specs)
    client.scan(specs)
    assert list(client.sessions) == ["cc-follow-1"]
    assert client.pending_count("cc-follow-1") > 0
    assert client.pending_count("cc-other-2") == 0
    # A follow request is idempotent, and survives a watcher restart.
    _capture.request_follow("cc-follow-1", client.follow_dir, why="again")
    assert len(_capture.followed_requests(client.follow_dir)) == 1


def test_followed_claude_session_meta(tmp_path):
    root = tmp_path / "projects"
    path = root / "-home-apaz-git-mlsweep" / "cc-follow-1.jsonl"
    _write_jsonl(path, [CLAUDE_COMMAND_LINE, CLAUDE_COMMAND_OUTPUT_LINE,
                        CLAUDE_HUMAN_LINE, CLAUDE_ASSISTANT_LINE])
    client = _capture.CaptureClient(directory=tmp_path / "capture")
    specs = _capture.agent_specs(["claude_code"], root=root)
    state = client.follow(specs, "cc-follow-1")
    assert state is not None
    client.scan(specs)
    assert state.cwd == "/home/apaz/git/mlsweep"
    assert state.model == "claude-x"
    # The title is the first real prompt, on one line; not the /login wrapper.
    assert state.title == "Submit some test sweeps please"
    with client.pending_path("cc-follow-1").open() as f:
        events = [json.loads(line) for line in f]
    users = [e["text"] for e in events if e["kind"] == "user"]
    # The slash command reads as typed; its local output is not a prompt.
    assert users == ["/login", "Submit some test\nsweeps please"]


def test_follow_waits_for_whole_lines_and_uploads_at_once(tmp_path):
    root = tmp_path / "projects"
    path = root / "-p" / "cc-follow-1.jsonl"
    _write_jsonl(path, [CLAUDE_HUMAN_LINE])
    with path.open("a") as f:
        f.write(json.dumps(CLAUDE_ASSISTANT_LINE)[:20])  # mid-write
    client = _capture.CaptureClient(directory=tmp_path / "capture")
    specs = _capture.agent_specs(["claude_code"], root=root)
    state = client.follow(specs, "cc-follow-1")
    client.scan(specs)
    assert client.pending_count("cc-follow-1") == 1
    # Never uploaded: due now, without waiting for a quiet minute.
    assert client.due_sessions(state.last_seen_at + 1) == [state]
    client.mark_uploaded(state, 1, now=state.last_seen_at + 1)
    path.write_text(path.read_text()[: -20] + json.dumps(CLAUDE_ASSISTANT_LINE) + "\n")
    client.scan(specs)
    assert client.pending_count("cc-follow-1") == 2  # thinking + tool, read whole
    # Uploaded a moment ago and still busy: it waits.
    assert client.due_sessions(state.last_seen_at + 5) == []


def test_only_the_capture_manager_pulls_a_session_in():
    same = _capture._same_manager
    assert same("http://localhost:7891", "http://127.0.0.1:7891/")
    assert same("http://localhost:7891", "http://localhost:7891/api/capture")
    # A test manager on another port, or another cluster, is not this one.
    assert not same("http://localhost:7891", "http://localhost:40123")
    assert not same("http://localhost:7891", "http://gpu-box:7891")


def test_unknown_session_is_not_followed_until_its_transcript_exists(tmp_path):
    root = tmp_path / "projects"
    client = _capture.CaptureClient(directory=tmp_path / "capture")
    specs = _capture.agent_specs(["claude_code"], root=root)
    assert client.follow(specs, "cc-later") is None
    assert client.follow(specs, "../escape") is None
    _write_jsonl(root / "-p" / "cc-later.jsonl", [CLAUDE_HUMAN_LINE])
    assert client.follow(specs, "cc-later") is not None


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
        # The kind buttons narrow the ledger to one kind, and say how many.
        b.eval("document.querySelector('#kind-chips [data-kind=user]').click()")
        b.wait_for("document.querySelectorAll('#ledger tbody tr').length === 1")
        assert "kind=user" in b.eval("location.search")
        assert b.eval("document.querySelector('#kind-chips [data-kind=tools] .n').textContent") == "1"
        b.eval("document.querySelector('#kind-chips [data-kind=\"\"]').click()")
        b.wait_for(f"document.querySelectorAll('#ledger tbody tr').length === {len(_EVENTS)}")
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

        # The palette finds the trace by its text; a source filter keeps to
        # the sources it names.
        b.wait_for("typeof MLSearch === 'object' && typeof MLChrome.pages === 'object'")
        kinds = ("(async q => (await MLSearch.search(q)).items.map(i => i.kind))")
        assert "trace" in b.eval(f"{kinds}('sweep')")
        assert set(b.eval(f"{kinds}('trace:sweep')")) == {"trace"}
        assert "trace" not in b.eval(f"{kinds}('docs:sweep')")
        assert b.eval("[...document.querySelectorAll('.session-item .session-title')]"
                      ".map(x => x.textContent)") == ["Run the sweep"]
    finally:
        b.close()


# ── continued sessions ────────────────────────────────────────────────────────


def test_claude_events_carry_their_transcript_uuid():
    line = {**CLAUDE_ASSISTANT_LINE, "uuid": "u-1"}
    _, events = _normalize("claude_code", [line])
    assert [e["src"] for e in events] == [f"u-1#{i}" for i in range(len(events))]
    _, untagged = _normalize("claude_code", [CLAUDE_ASSISTANT_LINE])
    assert all("src" not in e for e in untagged)


def test_a_continued_session_does_not_store_the_old_ones_events_again(manager_server):
    """Claude Code starts a continued session's transcript with a copy of the old one."""
    server, url = manager_server
    token = server.token
    old = {**_SESSION, "session_id": "aaaaaaaa-0000-4000-8000-000000000001", "agent": "claude_code"}
    new = {**old, "session_id": "bbbbbbbb-0000-4000-8000-000000000002", "started_at": 2000.0}
    first = [{"seq": i, "ts": 1000.0 + i, "kind": "user", "text": f"turn {i}", "src": f"u{i}#0"}
             for i in (1, 2, 3)]
    status, body = _post(url, token, "/api/capture/sessions", {"session": old, "events": first})
    assert status == 201 and body["events_stored"] == 3
    copied = [{**e} for e in first[1:]]  # the new file repeats turns 2 and 3, renumbered from 1
    for i, e in enumerate(copied, start=1):
        e["seq"] = i
    fresh = {"seq": 3, "ts": 2001.0, "kind": "user", "text": "the new turn", "src": "u9#0"}
    status, body = _post(url, token, "/api/capture/sessions", {"session": new, "events": copied + [fresh]})
    assert status == 201 and body["events_stored"] == 1
    assert body["continues"] == old["session_id"]
    status, events = _get(url, token, f"/api/capture/sessions/{new['session_id']}/events")
    assert [e["text"] for e in events["events"]] == ["the new turn"]
    status, got = _get(url, token, f"/api/capture/sessions/{old['session_id']}")
    assert got["continued_by"] == [new["session_id"]]
    # Uploading the old one again changes nothing; it still has all three.
    status, body = _post(url, token, "/api/capture/sessions", {"session": old, "events": first})
    assert body["events_stored"] == 0 and body["event_count"] == 3
    # Deleting the old trace leaves the new one standing alone.
    status, _ = _delete(url, token, f"/api/capture/sessions/{old['session_id']}")
    assert status == 200
    status, got = _get(url, token, f"/api/capture/sessions/{new['session_id']}")
    assert got["continues"] == ""
