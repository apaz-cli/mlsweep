"""`mlsweep complain`: reports land in the manager's database and read back."""

import json

import pytest
from campaign_helpers import TOKEN, call, shared_manager, uid

from mlsweep import _complain_cli


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "complain")


def _run(mgr, argv, capsys, session=lambda: None):
    code = 0
    try:
        _complain_cli.main([*argv, "--manager", mgr.url, "--token", TOKEN], session=session)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    out = capsys.readouterr()
    return code, out.out, out.err


def test_complain_records_and_lists(mgr, capsys):
    eid = uid("cexp")
    text = f"`mlsweep wait` hung on {eid}"
    code, out, _ = _run(mgr, [text, "--experiment", eid, "--command", "mlsweep wait", "--json"], capsys,
                        session=lambda: "sess-abc")
    assert code == 0
    cid = json.loads(out)["complaint_id"]

    _, out, _ = _run(mgr, ["--list", "--json"], capsys)
    row = next(c for c in json.loads(out)["complaints"] if c["complaint_id"] == cid)
    assert row["text"] == text and row["kind"] == "problem" and row["experiment_id"] == eid
    assert row["session_id"] == "sess-abc" and row["command"] == "mlsweep wait"
    assert row["source"] == "cli" and row["host"] and row["version"]

    code, out, _ = _run(mgr, ["--list"], capsys)
    assert code == 0 and f"#{cid}" in out and text in out


def test_improvement_kind(mgr, capsys):
    _, out, _ = _run(mgr, ["--improvement", "a --dry-run for purge", "--json"], capsys)
    cid = json.loads(out)["complaint_id"]
    status, data = call(mgr, "GET", "/api/complaints?limit=5")
    assert status == 200
    assert next(c for c in data["complaints"] if c["complaint_id"] == cid)["kind"] == "improvement"


def test_empty_text_is_refused(mgr, capsys):
    code, _, err = _run(mgr, ["   "], capsys)
    assert code == 2 and "say what went wrong" in err


def test_api_validates(mgr):
    assert call(mgr, "POST", "/api/complaints", {"text": ""})[0] == 400
    assert call(mgr, "POST", "/api/complaints", {"text": "x", "kind": "rant"})[0] == 400
    assert call(mgr, "POST", "/api/complaints", {"text": "x", "host": 3})[0] == 400
    assert call(mgr, "POST", "/api/complaints", {"text": "x" * (64 * 1024 + 1)})[0] == 400
    status, data = call(mgr, "POST", "/api/complaints", {"text": "from the api"})
    assert status == 200 and isinstance(data["complaint_id"], int)


def test_sql_sees_the_table(mgr):
    assert call(mgr, "POST", "/api/sql", {"sql": "SELECT count(*) FROM complaints"})[0] == 200
