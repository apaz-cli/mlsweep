"""Reading traces from the terminal: `mlsweep capture show / search / export`,
`capture ls --experiment`, and read-only `mlsweep sql`.

The pure parts (references, kinds, the ledger and its budget, ATIF) are tested
directly.  The commands run in-process against a real manager.
"""

import json
import re
from pathlib import Path

import pytest
from campaign_helpers import TOKEN, call, ok, shared_manager, uid

from mlsweep import _trace_view as tv
from mlsweep import cli


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "trace_read")


def _run(argv, capsys):
    code = 0
    try:
        cli.main(argv)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    out = capsys.readouterr()
    return code, out.out, out.err


def _base(mgr):
    return ["--manager", mgr.url, "--token", TOKEN]


# A session: a prompt, thinking, a reply, an mlsweep call and a plain call with
# their results, a second prompt, and redacted (empty) thinking.
EVENTS = [
    {"seq": 1, "ts": 1700000000.0, "kind": "user", "text": "Sweep the learning rate for the zebra model"},
    {"seq": 2, "ts": 1700000001.0, "kind": "thinking", "text": "Three rates should do.",
     "model": "claude-x", "tokens": {"input": 100, "output": 20}},
    {"seq": 3, "ts": 1700000001.0, "kind": "assistant", "text": "I'll write the sweep and validate it.",
     "model": "claude-x", "tokens": {"input": 100, "output": 20}},
    {"seq": 4, "ts": 1700000002.0, "kind": "tool", "name": "bash", "tool_id": "c1",
     "text": "cd proj && mlsweep run sweeps/lr.py --validate 2>&1 | tail -5", "model": "claude-x"},
    {"seq": 5, "ts": 1700000003.0, "kind": "tool_result", "name": "bash", "tool_id": "c1", "ok": True,
     "duration_ms": 1500},
    {"seq": 6, "ts": 1700000004.0, "kind": "tool", "name": "read", "tool_id": "c2", "text": "path=sweeps/lr.py"},
    {"seq": 7, "ts": 1700000004.5, "kind": "tool_result", "name": "read", "tool_id": "c2", "ok": False},
    {"seq": 8, "ts": 1700000005.0, "kind": "thinking", "text": ""},
    {"seq": 9, "ts": 1700000006.0, "kind": "user", "text": "Now submit it for real"},
    {"seq": 10, "ts": 1700000007.0, "kind": "tool", "name": "bash", "tool_id": "c3",
     "text": "MLSWEEP_CAMPAIGN=zebra mlsweep run sweeps/lr.py"},
    {"seq": 11, "ts": 1700000060.0, "kind": "tool_result", "name": "bash", "tool_id": "c3", "ok": True,
     "duration_ms": 52000},
]


def _upload(mgr, events=EVENTS, title="zebra sweep"):
    sid = uid("sess")
    ok(mgr, "POST", "/api/capture/sessions", {"session": {
        "session_id": sid, "agent": "claude_code", "title": title, "cwd": "/home/x/proj",
        "state": "active", "model": "claude-x"}, "events": events})
    return sid


# ── pure parts ────────────────────────────────────────────────────────────────


def test_the_mlsweep_pattern_is_the_pages():
    """The terminal and the Traces tab must agree on what an mlsweep command is."""
    js = (Path(__file__).parent.parent / "mlsweep" / "webui" / "page-traces.js").read_text()
    m = re.search(r"^const MLSWEEP_RE = /(.*)/;$", js, re.M)
    assert m, "MLSWEEP_RE not found in page-traces.js"
    assert m.group(1) == tv.MLSWEEP_PATTERN


@pytest.mark.parametrize("text,calls", [
    ("mlsweep ls", ["mlsweep ls"]),
    ("cd x && mlsweep run a.py --validate 2>&1 | tail", ["mlsweep run a.py --validate"]),
    ("FOO=1 uv run mlsweep watch e1; echo done", ["mlsweep watch e1"]),
    ("python3 -m mlsweep best --experiment e", ["mlsweep best --experiment e"]),
    ("echo mlsweep", []),
    ("grep mlsweep_dir file", []),
    ("~/.venv/bin/mlsweep status", ["mlsweep status"]),
])
def test_mlsweep_calls(text, calls):
    assert tv.mlsweep_calls(text) == calls
    assert tv.is_mlsweep({"kind": "tool", "text": text}) == bool(calls)


def test_trace_refs():
    assert tv.parse_trace_ref("mlsweep:trace/abc123?kind=mlsweep#E412") == ("abc123", "mlsweep", 412)
    assert tv.parse_trace_ref("trace/abc123") == ("abc123", "", None)
    assert tv.parse_trace_ref("abc123#E7") == ("abc123", "", 7)
    with pytest.raises(ValueError, match="kind="):
        tv.parse_trace_ref("abc123?kind=nope")
    with pytest.raises(ValueError, match="#E"):
        tv.parse_trace_ref("abc123#7")
    assert tv.trace_ref("abc", 4, "tools") == "mlsweep:trace/abc?kind=tools#E4"


def test_ledger_folds_results_and_filters_kinds():
    ledger = tv.Ledger()
    ledger.add(EVENTS)
    seqs = [r["seq"] for r in ledger.rows]
    assert seqs == [1, 2, 3, 4, 6, 9, 10]        # results folded, empty thinking left out
    line = tv.row_line(ledger.rows[3])
    assert line.startswith("E4 ") and "mlsweep run sweeps/lr.py --validate" in line and "✓ 1.5 s" in line
    assert "cd proj" not in line                 # only the mlsweep part of the command
    assert tv.row_line(ledger.rows[4]).endswith("path=sweeps/lr.py  ✗")

    ml = tv.Ledger("mlsweep")
    ml.add(EVENTS)
    assert [r["seq"] for r in ml.rows] == [4, 10]
    users = tv.Ledger("user")
    users.add(EVENTS)
    assert [r["seq"] for r in users.rows] == [1, 9]


def test_budget_keeps_at_least_one_line():
    lines = ["x" * 400] * 10
    assert tv.take_within_budget(lines, 100) == 1       # 400 chars = 100 tokens
    assert tv.take_within_budget(lines, 300) == 2
    assert tv.take_within_budget(["y" * 10000], 10) == 1


def test_atif_shape():
    doc = tv.to_atif({"session_id": "s1", "agent": "claude_code", "model": "claude-x",
                      "experiments": ["e1"]}, EVENTS)
    assert doc["schema_version"] == "ATIF-v1.8"
    assert doc["agent"] == {"name": "claude-code", "version": "unknown", "model_name": "claude-x"}
    steps = doc["steps"]
    assert [s["step_id"] for s in steps] == list(range(1, len(steps) + 1))
    # A call made after another's result came from a new response, so its own step.
    assert [s["source"] for s in steps] == ["user", "agent", "agent", "user", "agent"]
    first, second = steps[1], steps[2]
    assert first["reasoning_content"] == "Three rates should do."
    assert first["message"] == "I'll write the sweep and validate it."
    assert [c["tool_call_id"] for c in first["tool_calls"]] == ["c1"]
    assert first["metrics"] == {"prompt_tokens": 100, "completion_tokens": 20}
    assert [(r["source_call_id"], r["content"]) for r in first["observation"]["results"]] == [("c1", "ok (1.5 s)")]
    assert second["message"] == "" and [c["tool_call_id"] for c in second["tool_calls"]] == ["c2"]
    assert second["observation"]["results"][0]["content"] == "error"
    assert first["extra"]["mlsweep_seqs"] == [2, 3, 4, 5]
    # Every observation names a call in its own step, and every call has arguments.
    for s in steps:
        ids = {c["tool_call_id"] for c in s.get("tool_calls", [])}
        assert all(r.get("source_call_id") in ids for r in s.get("observation", {}).get("results", []))
        assert all(c["arguments"] == {} and "summary" in c["extra"] for c in s.get("tool_calls", []))
        assert s["source"] == "agent" or "tool_calls" not in s
    assert steps[4]["extra"]["mlsweep_seqs"] == [10, 11]
    assert doc["extra"]["mlsweep"]["ref"] == "mlsweep:trace/s1"
    json.dumps(doc)


# ── the commands, against a manager ──────────────────────────────────────────


def test_show_prints_a_ledger_and_pages_by_budget(mgr, capsys):
    sid = _upload(mgr)
    code, out, _ = _run(["capture", "show", sid, *_base(mgr)], capsys)
    assert code == 0, out
    assert f"cite as mlsweep:trace/{sid}" in out
    assert "E1 " in out and "E10 " in out and "(end of trace)" in out
    assert "E5 " not in out                       # the result is on its call's row

    # A small budget stops early and says how to go on.
    code, out, _ = _run(["capture", "show", sid, "--budget", "40", *_base(mgr)], capsys)
    more = re.search(r"^more: (mlsweep capture show \S+ --from (\d+))$", out, re.M)
    assert code == 0 and more, out
    code, out2, _ = _run([*more.group(1).split()[1:], *_base(mgr)], capsys)
    assert f"E{more.group(2)} " in out2

    # A note reference with a kind shows that kind only, and a prefix finds the trace.
    code, out, _ = _run(["capture", "show", f"mlsweep:trace/{sid[:12]}?kind=mlsweep", *_base(mgr)], capsys)
    rows = [ln for ln in out.splitlines() if ln.startswith("E")]
    assert [r.split()[0] for r in rows] == ["E4", "E10"], out
    assert "mlsweep run sweeps/lr.py" in rows[1] and "MLSWEEP_CAMPAIGN" not in rows[1]

    code, out, _ = _run(["capture", "show", sid, "--json", "--kind", "user", *_base(mgr)], capsys)
    data = json.loads(out)
    assert [r["seq"] for r in data["rows"]] == [1, 9] and data["next_from"] is None


def test_show_one_event_in_full(mgr, capsys):
    sid = _upload(mgr)
    code, out, _ = _run(["capture", "show", f"{sid}#E4", *_base(mgr)], capsys)
    assert code == 0
    assert out.startswith("E4  mlsweep  2023-11-14T22:13:22.000Z")
    assert "succeeded in 1.5 s" in out and f"cite as mlsweep:trace/{sid}#E4" in out
    assert "cd proj && mlsweep run sweeps/lr.py --validate 2>&1 | tail -5" in out   # the whole command
    code, out, err = _run(["capture", "show", sid, "--event", "99", *_base(mgr)], capsys)
    assert code == 1 and "has no event 99" in err
    code, _, err = _run(["capture", "show", "nosuchtrace", *_base(mgr)], capsys)
    assert code == 1 and "trace" in err


def test_search_names_the_events(mgr, capsys):
    sid = _upload(mgr)
    other = _upload(mgr, [{"seq": 1, "ts": 1.0, "kind": "user", "text": "nothing about it"}], title="other")
    word = sid.split("_")[-1]                      # unique to this test
    _upload(mgr, [{"seq": 1, "ts": 1.0, "kind": "user", "text": f"the {word} marker"},
                  {"seq": 2, "ts": 2.0, "kind": "assistant", "text": f"Found {word} again, Zebra too"}])
    code, out, _ = _run(["capture", "search", "zebra", "--trace", sid, *_base(mgr)], capsys)
    assert code == 0
    assert f"mlsweep:trace/{sid}#E1" in out and f"mlsweep:trace/{sid}#E10" in out and "2 events." in out

    code, out, _ = _run(["capture", "search", word, "zebra", "--json", *_base(mgr)], capsys)
    hits = json.loads(out)["hits"]
    assert [(h["seq"], h["kind"]) for h in hits] == [(2, "assistant")]   # every word, any case
    assert hits[0]["ref"].endswith("#E2") and other not in out

    code, out, _ = _run(["capture", "search", "mlsweep", "run", "--kind", "mlsweep", "--trace", sid,
                         "--json", *_base(mgr)], capsys)
    assert [h["seq"] for h in json.loads(out)["hits"]] == [4, 10]


def test_ls_by_experiment_and_export(mgr, capsys, tmp_path):
    sid = _upload(mgr)
    eid = uid("traced")
    ok(mgr, "POST", "/api/experiments", {"campaign": "test", "experiment_id": eid, "capture_session": sid})
    code, out, _ = _run(["capture", "ls", "--experiment", eid, "--json", *_base(mgr)], capsys)
    assert [s["session_id"] for s in json.loads(out)["sessions"]] == [sid]

    path = tmp_path / "t.json"
    code, _, err = _run(["capture", "export", sid, "-o", str(path), *_base(mgr)], capsys)
    assert code == 0 and "atif" in err
    doc = json.loads(path.read_text())
    assert doc["schema_version"] == "ATIF-v1.8" and doc["extra"]["mlsweep"]["experiments"] == [eid]
    code, out, _ = _run(["capture", "export", sid, "--format", "jsonl", *_base(mgr)], capsys)
    assert [json.loads(line)["seq"] for line in out.splitlines()] == [e["seq"] for e in EVENTS]


# ── sql ──────────────────────────────────────────────────────────────────────


def test_sql_reads(mgr, capsys):
    eid = uid("sqlexp")
    ok(mgr, "POST", "/api/experiments", {"campaign": "test", "experiment_id": eid})
    code, out, _ = _run(["sql", f"SELECT experiment_id, campaign FROM experiments WHERE experiment_id = '{eid}'",
                         *_base(mgr)], capsys)
    assert code == 0 and eid in out and "(1 row," in out
    code, out, _ = _run(["sql", "--json", *_base(mgr)], capsys)
    tables = {t["name"] for t in json.loads(out)["tables"]}
    assert {"experiments", "jobs", "capture_events"} <= tables
    status, data = call(mgr, "POST", "/api/sql",
                        {"sql": "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 50) "
                                "SELECT i FROM n", "limit": 5})
    assert status == 200 and data["row_count"] == 5 and data["truncated"] is True


@pytest.mark.parametrize("sql", [
    "DELETE FROM experiments",
    "UPDATE experiments SET name = 'x'",
    "INSERT INTO settings (key, value) VALUES ('a', 'b')",
    "DROP TABLE jobs",
    "CREATE TABLE t (x)",
    "ATTACH DATABASE '/tmp/x.db' AS x",
    "PRAGMA journal_mode = DELETE",
    "SELECT 1; DELETE FROM experiments",
    "SELECT load_extension('x')",
])
def test_sql_cannot_write(mgr, sql):
    before = call(mgr, "GET", "/api/experiments")[1]
    status, data = call(mgr, "POST", "/api/sql", {"sql": sql})
    assert status == 400, (sql, data)
    assert call(mgr, "GET", "/api/experiments")[1] == before


def test_sql_stops_a_long_query(mgr):
    status, data = call(mgr, "POST", "/api/sql", {
        "sql": "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n",
        "timeout": 0.5})
    assert status == 400 and "stopped after 0.5 s" in data["error"]


def test_show_names_the_trace_a_continued_session_continues():
    from mlsweep._capture_read import _header

    lines = _header({"session_id": "bbbb1111", "agent": "claude_code", "state": "active", "event_count": 4,
                     "continues": "aaaa0000", "continued_by": ["cccc2222"]})
    assert "  continues mlsweep:trace/aaaa0000: the events before this trace's first are there" in lines
    assert "  continued in mlsweep:trace/cccc2222" in lines
