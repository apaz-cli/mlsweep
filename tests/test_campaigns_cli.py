"""Campaigns in the ``mlsweep`` command line.

Each command works in the campaign named by ``--campaign``, else
``$MLSWEEP_CAMPAIGN``, else ``default``; ``--all-campaigns`` covers every
campaign.  A command naming an experiment from another campaign exits 1 with a
hint and changes nothing.  Commands run in-process against a real manager,
except ``mlsweep run``, which runs as a subprocess from the repo root.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from campaign_helpers import TOKEN, call, ok, q, seed, shared_manager, snapshot, uid
from conftest import _wait_for_experiment_complete

from mlsweep import cli, ctl, run_sweep
from mlsweep.run_sweep import (
    _campaign_argv,
    _resolve_campaign,
    _with_campaign,
    require_campaign,
)

REPO_ROOT = Path(__file__).parent.parent
MLSWEEP_RUN = [sys.executable, "-m", "mlsweep.run_sweep"]


@pytest.fixture(autouse=True)
def _no_campaign_env(monkeypatch):
    monkeypatch.delenv("MLSWEEP_CAMPAIGN", raising=False)


@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "campaigns_cli")


def _base(server):
    return ["--manager", server.url, "--token", TOKEN]


def _run_cli(argv, capsys):
    """Run ``mlsweep *argv*`` in-process; return ``(exit code, stdout)``."""
    code = 0
    try:
        cli.main(argv)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    return code, capsys.readouterr().out


def _parse(argv):
    parser = run_sweep.argparse.ArgumentParser()
    run_sweep._add_manager_args(parser)
    return parser.parse_args(argv)


# ── Resolving the campaign ──────────────────────────────────────────────────────


def test_resolve_defaults_to_default():
    assert _resolve_campaign(_parse([])) == "default"


def test_resolve_reads_env(monkeypatch):
    monkeypatch.setenv("MLSWEEP_CAMPAIGN", "from_env")
    assert _resolve_campaign(_parse([])) == "from_env"


def test_resolve_flag_beats_env(monkeypatch):
    monkeypatch.setenv("MLSWEEP_CAMPAIGN", "from_env")
    assert _resolve_campaign(_parse(["--campaign", "from_flag"])) == "from_flag"


def test_resolve_empty_env_means_default(monkeypatch):
    monkeypatch.setenv("MLSWEEP_CAMPAIGN", "")
    assert _resolve_campaign(_parse([])) == "default"


@pytest.mark.parametrize("flag", ["--all-campaigns", "--all_campaigns"])
def test_resolve_all_campaigns(monkeypatch, flag):
    monkeypatch.setenv("MLSWEEP_CAMPAIGN", "from_env")
    assert _resolve_campaign(_parse([flag])) is None


def test_campaign_and_all_campaigns_are_exclusive(capsys):
    with pytest.raises(SystemExit) as exc:
        _parse(["--campaign", "x", "--all-campaigns"])
    assert exc.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["a b", "a/b", "*", "x" * 129])
def test_resolve_rejects_invalid_flag(capsys, bad):
    with pytest.raises(SystemExit) as exc:
        _resolve_campaign(_parse(["--campaign", bad]))
    assert exc.value.code == 1
    assert "campaign must be" in capsys.readouterr().out


def test_resolve_rejects_invalid_env(monkeypatch, capsys):
    monkeypatch.setenv("MLSWEEP_CAMPAIGN", "not valid")
    with pytest.raises(SystemExit):
        _resolve_campaign(_parse([]))
    assert "campaign must be" in capsys.readouterr().out


@pytest.mark.parametrize("path,campaign,expected", [
    ("/api/x", None, "/api/x"),
    ("/api/x", "c", "/api/x?campaign=c"),
    ("/api/x?a=1", "c", "/api/x?a=1&campaign=c"),
    ("/api/x?a=1", None, "/api/x?a=1"),
])
def test_with_campaign(path, campaign, expected):
    assert _with_campaign(path, campaign) == expected


def test_campaign_argv():
    assert _campaign_argv("c") == ["--campaign", "c"]
    assert _campaign_argv(None) == ["--all-campaigns"]


def test_require_campaign_none_makes_no_request(monkeypatch):
    monkeypatch.setattr(run_sweep, "_http_request", lambda *a, **k: pytest.fail("no request expected"))
    require_campaign("http://unused", "t", "e", None)


def test_require_campaign_passes_for_match_and_unknown(mgr):
    eid = seed(mgr, "alpha")
    require_campaign(mgr.url, TOKEN, eid, "alpha")
    require_campaign(mgr.url, TOKEN, "no_such_experiment", "alpha")


def test_require_campaign_exits_with_hint(mgr, capsys):
    eid = seed(mgr, "alpha")
    with pytest.raises(SystemExit) as exc:
        require_campaign(mgr.url, TOKEN, eid, "beta")
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert f"experiment {eid} is in campaign 'alpha', not 'beta'" in out
    assert "--campaign alpha or --all-campaigns" in out


def test_require_campaign_is_silent_when_manager_unreachable(capsys):
    require_campaign("http://127.0.0.1:9", "t", "e", "alpha")
    assert capsys.readouterr().out == ""


# ── Every experiment command honours the campaign ──────────────────────────────


# argv for each command, with {e} for the experiment id.
_COMMANDS = {
    "ls": ["ls", "{e}"],
    "logs": ["logs", "r1", "--experiment", "{e}"],
    "metrics": ["metrics", "--experiment", "{e}"],
    "cancel": ["cancel", "{e}", "--all", "--yes"],
    "retry": ["retry", "{e}", "r2"],
    "resume": ["resume", "{e}"],
    "stop": ["stop", "{e}", "--yes"],
    "pause": ["pause", "{e}"],
    "unpause": ["unpause", "{e}"],
    "wait": ["wait", "{e}", "--timeout", "0.5", "--interval", "0.1"],
    "best": ["best", "--experiment", "{e}", "--json"],
    "fetch": ["fetch", "--experiment", "{e}", "--json"],
    "watch": ["watch", "{e}", "--events"],
    "move": ["campaign", "move", "{e}", "zeta"],
}


def _argv(name, eid):
    return [a.replace("{e}", eid) for a in _COMMANDS[name]]


class _FakeWS:
    """Stands in for the WebSocket client; records the URL it was given."""

    urls: list = []

    def __init__(self, url, token, timeout=10.0):
        _FakeWS.urls.append(url)

    def connect(self):
        pass

    def iter_events(self, **_):
        yield {"type": "experiment_done", "experiment_id": "e"}

    def close(self):
        pass


@pytest.fixture
def fake_ws(monkeypatch):
    _FakeWS.urls = []
    monkeypatch.setattr(run_sweep, "_WebSocket", _FakeWS)
    return _FakeWS


@pytest.mark.parametrize("how", ["default", "flag", "env"])
@pytest.mark.parametrize("name", sorted(_COMMANDS))
def test_command_refuses_other_campaign(mgr, capsys, monkeypatch, fake_ws, name, how):
    eid = seed(mgr, "alpha")
    before = snapshot(mgr, eid)
    argv = _argv(name, eid) + _base(mgr)
    wanted = "default"
    if how == "flag":
        argv += ["--campaign", "beta"]
        wanted = "beta"
    elif how == "env":
        monkeypatch.setenv("MLSWEEP_CAMPAIGN", "beta")
        wanted = "beta"
    code, out = _run_cli(argv, capsys)
    assert code == 1, out
    assert f"is in campaign 'alpha', not '{wanted}'" in out
    assert "Pass --campaign alpha or --all-campaigns." in out
    assert snapshot(mgr, eid) == before
    assert fake_ws.urls == []


def _expect(name, mgr, eid, code, out, campaign):
    """What each command does once the campaign check lets it through."""
    exp = ok(mgr, "GET", f"/api/experiments/{eid}")
    jobs = {j["run_id"]: j for j in ok(mgr, "GET", f"/api/experiments/{eid}/jobs")}
    if name == "ls":
        assert code == 0 and "r1" in out and "r2" in out
    elif name == "logs":
        assert code == 1 and "No log found" in out
    elif name == "metrics":
        assert code == 1 and "No metrics" in out
    elif name == "cancel":
        assert code == 0 and jobs["r1"]["status"] == "cancelled" and jobs["r2"]["status"] == "done"
    elif name == "retry":
        assert code == 0 and jobs["r2"]["status"] == "pending"
    elif name == "resume":
        assert code == 0 and "Nothing to resume" in out
    elif name == "stop":
        assert code == 0 and exp["status"] == "aborted"
    elif name == "pause":
        assert code == 0 and exp["status"] == "paused"
    elif name == "unpause":
        assert code == 0 and exp["status"] == "running"
    elif name == "wait":
        assert code == ctl.WAIT_EXIT_TIMEOUT and "TIMEOUT" in out
    elif name == "best":
        assert code == 0 and {r["run_id"] for r in json.loads(out)} == {"r1", "r2"}
    elif name == "fetch":
        data = json.loads(out)
        assert code == 0 and data["campaign"] == "alpha" and len(data["runs"]) == 2
    elif name == "watch":
        assert code == 0
        url = _FakeWS.urls[-1]
        assert (f"campaign={campaign}" in url) if campaign else ("campaign=" not in url)
    elif name == "move":
        assert code == 0 and exp["campaign"] == "zeta"


@pytest.mark.parametrize("how", ["flag", "env", "all", "all_underscore"])
@pytest.mark.parametrize("name", sorted(_COMMANDS))
def test_command_works_in_its_campaign(mgr, capsys, monkeypatch, fake_ws, name, how):
    eid = seed(mgr, "alpha")
    argv = _argv(name, eid) + _base(mgr)
    campaign = "alpha"
    if how == "flag":
        argv += ["--campaign", "alpha"]
    elif how == "env":
        monkeypatch.setenv("MLSWEEP_CAMPAIGN", "alpha")
    else:
        monkeypatch.setenv("MLSWEEP_CAMPAIGN", "beta")  # --all-campaigns overrides it
        argv += ["--all-campaigns" if how == "all" else "--all_campaigns"]
        campaign = None
    code, out = _run_cli(argv, capsys)
    assert "is in campaign" not in out
    _expect(name, mgr, eid, code, out, campaign)


def test_unknown_experiment_is_reported_by_the_command(mgr, capsys):
    code, out = _run_cli(["logs", "r1", "--experiment", "no_such_exp"] + _base(mgr), capsys)
    assert code == 1 and "No log found" in out and "is in campaign" not in out


# ── ls ──────────────────────────────────────────────────────────────────────────


def test_ls_lists_only_the_current_campaign(mgr, capsys):
    cx = uid("cx")
    mine, other, dflt = seed(mgr, cx), seed(mgr, uid("cy")), seed(mgr, "default")
    code, out = _run_cli(["ls", "--campaign", cx] + _base(mgr), capsys)
    assert code == 0
    assert f"# Campaign {cx}: 1 experiment\n" in out
    assert mine in out and other not in out and dflt not in out
    code, out = _run_cli(["ls"] + _base(mgr), capsys)
    assert dflt in out and mine not in out and "# Campaign default:" in out


def test_ls_all_campaigns_heads_each_campaign(mgr, capsys):
    cx = uid("cx")
    mine, dflt = seed(mgr, cx), seed(mgr, "default")
    code, out = _run_cli(["ls", "--all-campaigns", "--all"] + _base(mgr), capsys)
    assert code == 0 and out.startswith("# All campaigns:")
    # Each experiment is listed under its own campaign's heading.
    heading = ""
    under = {}
    for line in out.splitlines():
        if line.startswith("## Campaign "):
            heading = line
        elif line.startswith("- "):
            under[line[2:].split(":")[0]] = heading
    assert under[mine].startswith(f"## Campaign {cx}:")
    assert under[dflt].startswith("## Campaign default:")


def test_ls_json_is_filtered(mgr, capsys):
    cx = uid("cx")
    a, b = seed(mgr, cx), seed(mgr, cx)
    seed(mgr, uid("cy"))
    code, out = _run_cli(["ls", "--json", "--campaign", cx] + _base(mgr), capsys)
    rows = json.loads(out)
    assert code == 0 and sorted(r["experiment_id"] for r in rows) == sorted([a, b])
    assert all(r["campaign"] == cx for r in rows)


def test_ls_runs_of_experiment_in_env_campaign(mgr, capsys, monkeypatch):
    cx = uid("cx")
    eid = seed(mgr, cx)
    monkeypatch.setenv("MLSWEEP_CAMPAIGN", cx)
    code, out = _run_cli(["ls", eid, "--json"] + _base(mgr), capsys)
    assert code == 0 and sorted(j["run_id"] for j in json.loads(out)) == ["r1", "r2"]


# ── status ──────────────────────────────────────────────────────────────────────


def test_status_json_reports_campaign_and_filters_recent(mgr, capsys):
    cx = uid("cx")
    mine = seed(mgr, cx)
    code, out = _run_cli(["status", "--json", "--campaign", cx] + _base(mgr), capsys)
    data = json.loads(out)
    assert code == 0 and data["campaign"] == cx
    assert data["recent_experiments"] == [mine] and data["n_experiments"] == 1


def test_status_json_all_campaigns(mgr, capsys):
    a, b = seed(mgr, uid("cx")), seed(mgr, uid("cy"))
    code, out = _run_cli(["status", "--json", "--all-campaigns"] + _base(mgr), capsys)
    data = json.loads(out)
    assert data["campaign"] is None
    assert data["n_experiments"] >= 2


def test_status_text_shows_campaign(mgr, capsys):
    code, out = _run_cli(["status", "--campaign", "alpha"] + _base(mgr), capsys)
    assert code == 0 and "campaign:   alpha" in out
    code, out = _run_cli(["status", "--all-campaigns"] + _base(mgr), capsys)
    assert "campaign:   all (--all-campaigns)" in out


# ── campaign ls / move ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("argv", [["campaign"], ["campaign", "ls"]])
def test_campaign_ls_text(mgr, capsys, argv):
    cx = uid("cx")
    seed(mgr, cx)
    seed(mgr, cx)
    code, out = _run_cli(argv + ["--campaign", cx] + _base(mgr), capsys)
    assert code == 0
    line = next(line for line in out.splitlines() if cx in line)
    assert line.lstrip().startswith("*")
    assert "2 experiments" in line and "2 done" in line and "2 pend" in line
    assert "default" in out
    assert "current campaign" in out


def test_campaign_ls_marks_nothing_for_all_campaigns(mgr, capsys):
    seed(mgr, uid("cx"))
    code, out = _run_cli(["campaign", "--all-campaigns"] + _base(mgr), capsys)
    assert code == 0
    assert not any(line.lstrip().startswith("*") for line in out.splitlines())
    assert "current campaign" not in out


def test_campaign_ls_singular_experiment(mgr, capsys):
    cx = uid("cx")
    seed(mgr, cx)
    _, out = _run_cli(["campaign"] + _base(mgr), capsys)
    line = next(line for line in out.splitlines() if cx in line)
    assert "1 experiment," in line


@pytest.mark.parametrize("argv", [["campaign", "--json"], ["campaign", "ls", "--json"]])
def test_campaign_ls_json(mgr, capsys, argv):
    cx = uid("cx")
    seed(mgr, cx)
    code, out = _run_cli(argv + _base(mgr), capsys)
    camps = json.loads(out)
    assert code == 0
    mine = next(c for c in camps if c["campaign"] == cx)
    assert mine["experiments"] == 1 and mine["job_counts"]["total"] == 2


def test_campaign_move(mgr, capsys):
    src, dst = uid("src"), uid("dst")
    eid = seed(mgr, src)
    code, out = _run_cli(["campaign", "move", eid, dst, "--campaign", src] + _base(mgr), capsys)
    assert code == 0 and f"moved {eid} to campaign {dst}" in out
    assert ok(mgr, "GET", f"/api/experiments/{eid}")["campaign"] == dst


@pytest.mark.parametrize("bad", ["a b", "*", "x" * 129])
def test_campaign_move_invalid_target(mgr, capsys, bad):
    eid = seed(mgr, "default")
    code, out = _run_cli(["campaign", "move", eid, bad] + _base(mgr), capsys)
    assert code == 1 and "campaign must be" in out
    assert ok(mgr, "GET", f"/api/experiments/{eid}")["campaign"] == "default"


def test_campaign_move_unknown_experiment(mgr, capsys):
    code, out = _run_cli(["campaign", "move", "no_such_exp", "x"] + _base(mgr), capsys)
    assert code == 1 and "FAIL" in out


def test_campaign_help(capsys):
    code, out = _run_cli(["campaign", "--help"], capsys)
    assert code == 0 and "move" in out and "ls" in out


def test_help_and_dispatch_know_campaign():
    assert "campaign" in cli._CTL_CMDS
    assert callable(ctl.campaign_cmd)
    h = cli._build_help()
    assert "campaign move EXP NAME" in h
    assert "--campaign" in h and "$MLSWEEP_CAMPAIGN" in h and "--all-campaigns" in h


# ── mlsweep run ─────────────────────────────────────────────────────────────────


def _submit(args, env_campaign=None, timeout=300):
    env = {k: v for k, v in os.environ.items() if k != "MLSWEEP_CAMPAIGN"}
    if env_campaign is not None:
        env["MLSWEEP_CAMPAIGN"] = env_campaign
    return subprocess.run(
        [*MLSWEEP_RUN, *args], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=timeout,
    )


def _grid(tmp_path, *extra):
    return ["tests/sweeps/integration_grid.py", "--output-dir", str(tmp_path), *extra]


@pytest.mark.parametrize("extra,env_campaign,expected", [
    (["--campaign", "c_flag"], None, "c_flag"),
    ([], "c_env", "c_env"),
    (["--campaign", "c_flag"], "c_env", "c_flag"),
    ([], None, "default"),
])
def test_dry_run_prints_campaign(tmp_path, extra, env_campaign, expected):
    proc = _submit(_grid(tmp_path, "--dry-run", *extra), env_campaign)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"Campaign: {expected}" in proc.stdout


@pytest.mark.parametrize("extra,env_campaign,expected", [
    (["--campaign", "c_flag"], None, "c_flag"),
    ([], "c_env", "c_env"),
    ([], None, "default"),
])
def test_run_submits_under_campaign(mgr, tmp_path, extra, env_campaign, expected):
    eid = uid("run")
    proc = _submit(_grid(tmp_path, "--manager", mgr.url, "--token", TOKEN,
                         "--experiment", eid, *extra), env_campaign)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"Experiment created: {eid} (campaign {expected})" in proc.stdout
    assert f"--campaign {expected}" in proc.stdout  # the printed watch/fetch hints
    assert ok(mgr, "GET", f"/api/experiments/{eid}")["campaign"] == expected
    assert len(ok(mgr, "GET", q(f"/api/experiments/{eid}/jobs", expected))) == 4


@pytest.mark.parametrize("flag", ["--all-campaigns", "--all_campaigns"])
def test_run_new_sweep_refuses_all_campaigns(mgr, tmp_path, flag):
    eid = uid("run")
    proc = _submit(_grid(tmp_path, "--manager", mgr.url, "--token", TOKEN,
                         "--experiment", eid, flag))
    assert proc.returncode == 1
    assert "a new sweep needs one campaign" in proc.stdout
    assert call(mgr, "GET", f"/api/experiments/{eid}")[0] == 404


def test_validate_allows_all_campaigns(tmp_path):
    proc = _submit(_grid(tmp_path, "--validate", "--all-campaigns"))
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_run_rejects_invalid_campaign(tmp_path):
    proc = _submit(_grid(tmp_path, "--dry-run", "--campaign", "bad name"))
    assert proc.returncode == 1 and "campaign must be" in proc.stdout


def test_run_rejects_campaign_with_all_campaigns(tmp_path):
    proc = _submit(_grid(tmp_path, "--dry-run", "--campaign", "x", "--all-campaigns"))
    assert proc.returncode == 2 and "not allowed with" in proc.stderr


def test_run_refuses_existing_experiment_in_other_campaign(mgr, tmp_path):
    eid = seed(mgr, "alpha")
    proc = _submit(_grid(tmp_path, "--manager", mgr.url, "--token", TOKEN,
                         "--experiment", eid, "--campaign", "beta"))
    assert proc.returncode == 1
    assert "Create experiment" in proc.stdout and "'alpha'" in proc.stdout
    assert sorted(j["run_id"] for j in ok(mgr, "GET", f"/api/experiments/{eid}/jobs")) == ["r1", "r2"]


def test_resume_refuses_experiment_in_other_campaign(mgr, tmp_path):
    eid = seed(mgr, "alpha")
    proc = _submit(["tests/sweeps/bayes_sweep.py", "--output-dir", str(tmp_path),
                    "--manager", mgr.url, "--token", TOKEN, "--resume", eid])
    assert proc.returncode == 1
    assert f"experiment {eid} is in campaign 'alpha', not 'default'" in proc.stdout


# ── End to end, with a worker ───────────────────────────────────────────────────


def test_sweep_runs_and_fetches_within_campaign(manager_with_worker, tmp_path, capsys):
    server, url = manager_with_worker
    base = ["--manager", url, "--token", server.token]
    proc = _submit(_grid(tmp_path, *base, "--experiment", "camp_e2e", "--campaign", "e2e"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _wait_for_experiment_complete(url, server.token, "camp_e2e", expected_jobs=4, timeout=180)

    code, out = _run_cli(["fetch", "--experiment", "camp_e2e", "--json", "--wait",
                          "--wait-interval", "1", "--campaign", "e2e"] + base, capsys)
    data = json.loads(out)
    assert code == 0 and data["campaign"] == "e2e"
    assert sorted(r["status"] for r in data["runs"]) == ["done"] * 4

    code, out = _run_cli(["fetch", "--experiment", "camp_e2e", "--campaign", "e2e",
                          "--output-dir", str(tmp_path / "dl")] + base, capsys)
    assert code == 0 and "Campaign:   e2e" in out
    assert any((tmp_path / "dl").iterdir())

    code, out = _run_cli(["best", "--experiment", "camp_e2e"] + base, capsys)
    assert code == 1 and "is in campaign 'e2e', not 'default'" in out

    code, out = _run_cli(["ls", "--all-campaigns"] + base, capsys)
    assert "## Campaign e2e: 1 experiment" in out and "- camp_e2e:" in out
