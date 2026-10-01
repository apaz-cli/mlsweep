"""Unit tests for the mlsweep CLI helpers, help text, and Makefile template."""

from collections.abc import Iterator
from importlib.resources import files

import pytest

from mlsweep import cli
from mlsweep import ctl
from mlsweep._colors import color_enabled, set_color, strip_color_flag
from mlsweep._shared import _GREEN, _RESET


@pytest.fixture(autouse=True)
def _reset_color() -> Iterator[None]:
    """Keep the global color toggle from leaking between tests."""
    set_color(False)
    yield
    set_color(False)


def test_visible_width_strips_ansi():
    set_color(True)
    assert cli._visible_width(f"{_GREEN}mlsweep{_RESET}") == 7
    set_color(False)
    assert cli._visible_width("plain") == 5
    assert cli._visible_width("") == 0


def test_pad():
    assert cli._pad("ab", 5) == "ab   "
    assert cli._pad("abc", 2) == "abc"  # never truncates


def test_align_two_columns():
    out = cli._align([("a", "x"), ("longer", "y")])
    lines = out.splitlines()
    assert len(lines) == 2
    # right column starts at the same visible column
    assert lines[0].index("x") == lines[1].index("y")
    assert lines[0].startswith("  a")
    assert lines[1].startswith("  longer")


def test_align3_three_columns():
    out = cli._align3([("run", "S=x", "DDD"), ("manager", "", "MMM")])
    lines = out.splitlines()
    # description column is aligned
    assert lines[0].index("DDD") == lines[1].index("MMM")
    assert "S=x" in lines[0]
    assert "manager" in lines[1]


def test_label_pads_to_shared_column():
    assert cli._label("token:") == "token:      "  # padded to width 10 + 2
    assert cli._label("manager:") == "manager:    "
    assert cli._label("dashboard:") == "dashboard:  "


def test_help_lists_all_subcommands():
    h = cli._build_help()
    for cmd in [
        "manager", "run", "worker", "watch", "fetch", "best", "wait", "status",
        "ls", "logs", "cancel", "retry", "resume", "stop", "pause",
        "unpause", "docs", "gen_makefile", "version",
    ]:
        assert cmd in h, cmd
    # skill is referenced as a docs topic
    assert "skill" in h


def test_wait_registered_as_ctl_command():
    assert "wait" in cli._CTL_CMDS
    assert callable(ctl.wait_cmd)


def test_doc_aliases():
    assert cli._DOC_ALIASES["skill"] == "SKILL.md"
    assert cli._DOC_ALIASES["readme"] == "README.md"
    assert cli._DOC_ALIASES["index"] == "README.md"
    assert cli._DOC_ALIASES["config"] == "sweep_configuration.md"
    assert cli._DOC_ALIASES["mlsweep"] == "mlsweep.md"
    assert cli._DOC_ALIASES["examples"] == "examples.md"


def test_makefile_template_has_all_targets():
    t = cli._build_makefile_template()
    for target in [
        "sweep-manager", "sweep-status", "sweep-docs", "sweep-validate",
        "sweep-dry-run", "sweep-run", "sweep-watch", "sweep-fetch", "sweep-gen",
        "sweep-best", "sweep-ls", "sweep-logs", "sweep-cancel", "sweep-retry",
        "sweep-resume", "sweep-stop",
    ]:
        assert f"{target}:" in t, target
    assert "mlsweep run $(S) --manager $(MLSWEEP_MANAGER) --stream" in t
    assert "mlsweep fetch --manager $(MLSWEEP_MANAGER) --experiment $(E)" in t
    assert "mlsweep best --manager $(MLSWEEP_MANAGER) --experiment $(E)" in t
    assert "mlsweep logs $(R) --experiment $(E) --manager $(MLSWEEP_MANAGER)" in t
    assert "R ?=" in t


def test_skill_doc_is_shipped():
    docs = files("mlsweep") / "docs"
    names = [p.name for p in docs.iterdir()]
    assert "SKILL.md" in names
    text = (docs / "SKILL.md").read_text()
    assert "name: mlsweep" in text
    assert "mlsweep run" in text


def test_main_help(capsys):
    cli.main(["--help"])
    out = capsys.readouterr().out
    # The banner is only for a bare `mlsweep`; --help must not print it.
    assert "experiment sweep engine" not in out
    assert "Control:" in out
    assert "\033[" not in out


def test_main_bare_prints_banner(capsys):
    cli.main([])
    out = capsys.readouterr().out
    assert "experiment sweep engine" in out
    assert "Control:" in out


def test_color_flag_enables_ansi(capsys):
    cli.main(["--color", "--help"])
    out = capsys.readouterr().out
    assert "\033[" in out
    assert "experiment sweep engine" not in out  # still no banner for --help


def test_strip_color_flag_preserves_run_passthrough():
    assert not color_enabled()
    rest = strip_color_flag(["--color", "run", "sweep.py", "--", "--color"])
    assert rest == ["run", "sweep.py", "--", "--color"]
    assert color_enabled()


def test_main_help_topic_skill(capsys):
    cli.main(["--help", "skill"])
    out = capsys.readouterr().out
    assert "mlsweep" in out


def test_main_version(capsys):
    from importlib.metadata import version
    cli.main(["version"])
    assert capsys.readouterr().out.strip() == version("mlsweep")


def test_main_unknown_subcommand(capsys):
    import pytest
    with pytest.raises(SystemExit):
        cli.main(["definitely-not-a-command"])
    out = capsys.readouterr().out
    assert "Unknown subcommand" in out
