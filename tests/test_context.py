"""Client contexts (``mlsweep context``, ``--context``) and shell completion.

Each test runs the real CLI in a subprocess with an environment of its own, so
nothing in this process is patched.
"""

import json
import os
import subprocess
import sys

import pytest

from mlsweep._completion import SCRIPTS


@pytest.fixture
def env(tmp_path):
    """An environment with a private config file and no connection variables set."""
    out = {k: v for k, v in os.environ.items()
           if k not in ("MLSWEEP_MANAGER", "MLSWEEP_TOKEN", "MLSWEEP_CAMPAIGN", "MLSWEEP_CONTEXT")}
    out["MLSWEEP_CONFIG"] = str(tmp_path / "config.toml")
    out["MLSWEEP_DIR"] = str(tmp_path / "state")  # no manager.token to fall back to
    (tmp_path / "state").mkdir()
    return out


def _cli(env, *args, check=True):
    r = subprocess.run([sys.executable, "-m", "mlsweep", *args], capture_output=True, text=True, env=env,
                       timeout=60)
    if check:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


def _resolved(env, *args, **extra_env):
    """(manager, campaign, token) a command would use, from `mlsweep status --json`."""
    r = _cli({**env, **extra_env}, "status", "--json", *args)
    data = json.loads(r.stdout[r.stdout.index("{"):])
    return data["manager"], data["campaign"], data["token_present"]


def _add_two(env, tmp_path):
    (tmp_path / "lab.token").write_text("lab-secret\n")
    _cli(env, "context", "add", "lab", "--manager", "http://127.0.0.1:9/", "--token-file",
         str(tmp_path / "lab.token"), "--campaign", "lab-c", "--use")
    _cli(env, "context", "add", "other", "--manager", "http://127.0.0.1:8")


def test_add_list_show_use_rm(env, tmp_path):
    assert "No contexts" in _cli(env, "context").stdout
    _add_two(env, tmp_path)
    listing = json.loads(_cli(env, "context", "--json").stdout)
    assert listing["current"] == "lab"
    lab = next(c for c in listing["contexts"] if c["name"] == "lab")
    assert lab["manager"] == "http://127.0.0.1:9"  # trailing slash dropped
    assert "secret" not in (tmp_path / "config.toml").read_text()
    assert json.loads(_cli(env, "context", "show", "--json").stdout)["token"] == "present"

    _cli(env, "context", "use", "other")
    assert json.loads(_cli(env, "context", "--json").stdout)["current"] == "other"
    _cli(env, "context", "rm", "other")
    assert json.loads(_cli(env, "context", "--json").stdout)["current"] is None
    r = _cli(env, "context", "use", "nope", check=False)
    assert r.returncode == 1 and "no context named 'nope'" in r.stderr


def test_precedence(env, tmp_path):
    assert _resolved(env) == ("http://localhost:7891", "default", False)
    _add_two(env, tmp_path)
    # The current context applies by itself...
    assert _resolved(env) == ("http://127.0.0.1:9", "lab-c", True)
    # ...but not over $MLSWEEP_MANAGER, whose token and campaign it can't know.
    assert _resolved(env, MLSWEEP_MANAGER="http://127.0.0.1:7") == ("http://127.0.0.1:7", "default", False)
    # $MLSWEEP_CAMPAIGN beats the current context's campaign.
    assert _resolved(env, MLSWEEP_CAMPAIGN="env-c")[1] == "env-c"
    # A named context beats the environment, and takes nothing from the current one;
    # what it doesn't set (here, a campaign) falls back to the environment.
    assert _resolved(env, "--context", "other", MLSWEEP_MANAGER="http://127.0.0.1:7",
                     MLSWEEP_CAMPAIGN="env-c") == ("http://127.0.0.1:8", "env-c", False)
    assert _resolved(env, "--context", "lab", MLSWEEP_CAMPAIGN="env-c")[1] == "lab-c"
    assert _resolved(env, MLSWEEP_CONTEXT="other") == ("http://127.0.0.1:8", "default", False)
    # Flags beat everything, before or after --context.
    assert _resolved(env, "--manager", "http://127.0.0.1:6", "--context", "other")[0] == "http://127.0.0.1:6"
    assert _resolved(env, "--context", "other", "--campaign", "f")[1] == "f"


def test_unknown_or_broken_contexts(env, tmp_path):
    r = _cli(env, "status", "--context", "nope", check=False)
    assert r.returncode == 2 and "no context named 'nope'" in r.stderr
    r = _cli({**env, "MLSWEEP_CONTEXT": "nope"}, "status", check=False)
    assert r.returncode == 2
    # A broken config only warns, so commands that don't ask for a context still work.
    (tmp_path / "config.toml").write_text('current = "x"\n[contexts.x]\ntoken = "inline"\n')
    r = _cli(env, "status", "--json")
    assert "keep the token in a file" in r.stderr


def _complete(env, *words):
    return _cli(env, "__complete", "--", *words).stdout.split()


def test_completion(env, tmp_path):
    assert "context" in _complete(env, "con") and "completion" in _complete(env, "com")
    assert _complete(env, "context", "") == ["list", "show", "add", "use", "rm"]
    _add_two(env, tmp_path)
    assert _complete(env, "context", "use", "") == ["lab", "other"]
    assert _complete(env, "ls", "--context", "o") == ["other"]
    assert "--timeout" in _complete(env, "wait", "--ti")
    proj = tmp_path / "proj"
    (proj / "sub").mkdir(parents=True)
    (proj / "s.py").write_text("")
    (proj / "notes.txt").write_text("")
    assert _complete(env, "run", f"{proj}/") == [f"{proj}/s.py", f"{proj}/sub/"]
    # A manager that is down gives no experiment IDs, quickly.
    assert _complete(env, "wait", "") == []


@pytest.mark.parametrize("shell", sorted(SCRIPTS))
def test_completion_scripts(env, shell):
    script = _cli(env, "completion", shell).stdout
    assert "mlsweep __complete --" in script
    if shell == "bash":
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
