"""Reading and searching experiments' code from the terminal.

`mlsweep code EXP PATH --lines A-B` / `-n` print numbered lines within a budget,
with the next-page command and the note reference.  A note's code link works
in place of EXP and PATH.  `mlsweep code search` finds lines in some
experiments or across a campaign, each with its reference.
"""

import json
import re

import pytest
from campaign_helpers import TOKEN, call, shared_manager, uid

from mlsweep import cli
from mlsweep._snapshot import scan_tree
from mlsweep.run_sweep import upload_snapshot

TRAIN = "".join(f"line_{i} = {i}\n" for i in range(1, 101))   # 100 lines
TRAIN = TRAIN.replace("line_42 = 42", "WARMUP_STEPS = 42  # the warmup")


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "code_read")


@pytest.fixture(scope="module")
def code(mgr, tmp_path_factory):
    """Two experiments in one campaign: train.py shared, warmup.py only in the second."""
    root = tmp_path_factory.mktemp("proj")
    (root / "train.py").write_text(TRAIN)
    (root / "blob.bin").write_bytes(b"\0WARMUP\0" * 10)
    campaign = uid("codecamp")
    out = {"campaign": campaign}
    for name in ("a", "b"):
        if name == "b":
            (root / "warmup.py").write_text("def warmup(step):\n    return min(1, step / 100)  # Warmup ramp\n")
        sid = upload_snapshot(mgr.url, TOKEN, scan_tree(root))[0]
        exp = uid(f"code_{name}")
        status, _ = call(mgr, "POST", f"/api/experiments/{exp}/jobs/bulk", {
            "campaign": campaign, "code": {"snapshot": sid},
            "jobs": [{"run_id": "r1", "command": ["true"], "artifact_id": sid}]})
        assert status == 201
        out[name], out[name + "_snap"] = exp, sid
    return out


def _run(argv, capsys, mgr, campaign):
    code = 0
    try:
        cli.main([*argv, "--manager", mgr.url, "--token", TOKEN, "--campaign", campaign])
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    out = capsys.readouterr()
    return code, out.out, out.err


def test_lines_are_numbered_and_cited(mgr, code, capsys):
    exp = code["a"]
    rc, out, _ = _run(["code", exp, "train.py", "--lines", "41-43"], capsys, mgr, code["campaign"])
    assert rc == 0, out
    assert out.splitlines()[1:4] == ["41  line_41 = 41", "42  WARMUP_STEPS = 42  # the warmup",
                                     "43  line_43 = 43"]
    assert f"cite as mlsweep:code/{exp}/train.py#L41-43" in out and "more:" not in out

    # Without -n or --lines a file still prints raw.
    rc, out, _ = _run(["code", exp, "train.py"], capsys, mgr, code["campaign"])
    assert out == TRAIN


def test_numbered_file_pages_within_the_budget(mgr, code, capsys):
    exp = code["a"]
    rc, out, _ = _run(["code", exp, "train.py", "-n", "--budget", "50"], capsys, mgr, code["campaign"])
    assert rc == 0
    more = re.search(r"^more: (mlsweep code .*)$", out, re.M)
    assert more, out
    assert more.group(1).endswith("--lines 13-")      # 50 tokens ≈ 200 chars ≈ 12 lines of "  n  line_n = n"
    rc, out, _ = _run(more.group(1).split()[1:], capsys, mgr, code["campaign"])
    assert out.splitlines()[1].startswith(" 13  line_13")


def test_a_notes_link_works_as_the_argument(mgr, code, capsys):
    exp = code["b"]
    rc, out, _ = _run(["code", f"mlsweep:code/{exp}/warmup.py#L2"], capsys, mgr, code["campaign"])
    assert rc == 0, out
    assert "2  " in out.splitlines()[1] and "Warmup ramp" in out
    assert f"cite as mlsweep:code/{exp}/warmup.py#L2" in out

    rc, out, _ = _run(["code", f"code/{exp}/train.py#L42", "--json"], capsys, mgr, code["campaign"])
    data = json.loads(out)
    assert data["text"].startswith("WARMUP_STEPS") and data["ref"].endswith("#L42") and data["next"] is None


def test_search_one_experiment(mgr, code, capsys):
    rc, out, err = _run(["code", "search", "WARMUP_STEPS", code["a"]], capsys, mgr, code["campaign"])
    assert rc == 0, (out, err)
    assert "42: WARMUP_STEPS = 42" in out
    assert f"mlsweep:code/{code['a']}/train.py#L42" in out and "blob.bin" not in out


def test_search_a_campaign_reads_shared_files_once(mgr, code, capsys):
    rc, out, _ = _run(["code", "search", "warmup", "-i", "--json"], capsys, mgr, code["campaign"])
    assert rc == 0
    res = json.loads(out)
    got = {(h["target"], h["path"], h["line"]) for h in res["hits"]}
    # Newest first: b has both files; a's train.py is the same file, so it is only counted.
    assert got == {(code["b"], "train.py", 42), (code["b"], "warmup.py", 1), (code["b"], "warmup.py", 2)}
    train = next(h for h in res["hits"] if h["path"] == "train.py")
    assert train["also_in"] == [code["a"]]
    assert res["files_skipped"] == 1                 # blob.bin is binary

    rc, out, _ = _run(["code", "search", r"return min\(", "-E", "-C", "1", "--json", "--path", "warmup.py"],
                      capsys, mgr, code["campaign"])
    hit, = json.loads(out)["hits"]
    assert hit["line"] == 2 and hit["before"] == ["def warmup(step):"] and hit["after"] == []


def test_search_reports_bad_patterns_and_limits(mgr, code, capsys):
    rc, _, err = _run(["code", "search", "(", "-E", code["a"]], capsys, mgr, code["campaign"])
    assert rc == 1 and "bad pattern" in err
    rc, out, _ = _run(["code", "search", "line_", code["a"], "--limit", "3", "--json"], capsys, mgr, code["campaign"])
    res = json.loads(out)
    assert len(res["hits"]) == 3 and res["truncated"] is True
