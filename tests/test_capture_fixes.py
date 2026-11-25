"""Regressions for the capture review fixes: privacy, parsing, resume, caps."""

import asyncio
import json

import aiosqlite

from mlsweep._capture import (
    AgentSpec,
    CaptureClient,
    _iso_to_epoch,
    _summarize_args,
    find_transcript,
)
from mlsweep._manager_db import _capped_event_json, init_db, store_capture_batch


def test_short_file_contents_are_never_summarized():
    line = _summarize_args("Write", {"file_path": ".env", "content": "KEY=sk-live-123"})
    assert "sk-live" not in line and "file_path=.env" in line
    line = _summarize_args("Edit", {"file_path": "a.py", "old_string": "x=1", "new_string": "x=2"})
    assert "x=1" not in line and "x=2" not in line


def test_timestamps_keep_their_offset():
    assert _iso_to_epoch("2024-01-01T10:00:00+02:00") == 1704096000.0
    assert _iso_to_epoch("2024-01-01T08:00:00Z") == 1704096000.0
    assert _iso_to_epoch("2024-01-01T13:00:00.5+05:00") == 1704096000.5


def test_session_id_is_not_a_glob(tmp_path):
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "real.jsonl").write_text("{}\n")
    spec = AgentSpec(name="claude_code", directory=tmp_path, normalize=lambda line, meta: [])
    assert find_transcript([spec], "*") is None
    assert find_transcript([spec], "real") is not None


def _spec(tmp_path):
    def normalize(line, meta):
        if line.get("bad"):
            raise AttributeError("unexpected shape")
        return [{"ts": 1.0, "kind": "user", "text": line["text"]}]
    return AgentSpec(name="claude_code", directory=tmp_path / "agent", normalize=normalize)


def test_closed_session_reopens_and_bad_lines_are_skipped(tmp_path):
    spec = _spec(tmp_path)
    transcript = tmp_path / "agent" / "proj" / "s1.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({"text": "a"}) + "\n")
    client = CaptureClient(tmp_path / "capture")
    state = client.follow([spec], "s1")
    client.scan([spec])
    assert client.pending_count("s1") == 1
    assert state.started_at == 1.0

    state.closed = True
    with transcript.open("a") as f:
        f.write(json.dumps({"bad": True}) + "\n" + json.dumps({"text": "b"}) + "\n")
    client.scan([spec])
    assert not state.closed
    assert [e["text"] for e in client.pending_events("s1")] == ["a", "b"]
    # The offset was saved with the spool, so a restarted watcher does not re-read.
    again = CaptureClient(tmp_path / "capture")
    again.load()
    assert again.sessions["s1"].offset == state.offset


def test_long_non_ascii_text_keeps_a_prefix():
    data = json.loads(_capped_event_json({"seq": 1, "ts": 1.0, "kind": "thinking",
                                          "text": "漢" * 40_000}))
    assert data["truncated"] is True and len(data["text"]) > 5_000


def test_event_cap_keeps_the_header_moving():
    from mlsweep import _manager_db

    async def run():
        db = await aiosqlite.connect(":memory:")
        db.row_factory = aiosqlite.Row
        try:
            await init_db(db)
            meta = {"session_id": "cap1", "agent": "pi", "state": "active"}
            cap = _manager_db._MAX_EVENTS_PER_SESSION
            await db.execute("INSERT INTO capture_sessions (session_id, agent, event_count, state) "
                             "VALUES ('cap1', 'pi', ?, 'active')", (cap - 1,))
            await db.commit()
            events = [{"seq": i, "ts": 1.0, "kind": "user", "text": "x"} for i in (1, 2)]
            row, stored, error = await store_capture_batch(db, {**meta, "state": "closed"}, events)
            assert error is None and stored == 1
            assert row["state"] == "closed" and row["event_count"] == cap
        finally:
            await db.close()

    asyncio.run(run())
