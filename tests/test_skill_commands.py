"""Every command the skill names must exist, with the flags it shows.

Agents copy these lines as written, so a renamed flag or a missing required one
fails for them with no human to notice.  Checked against each command's --help.
"""

from __future__ import annotations

import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from mlsweep import cli

SKILL = (Path(__file__).resolve().parent.parent / "mlsweep/docs/SKILL.md").read_text()
GROUPS = {"notes", "workers", "campaign", "export", "capture"}
NO_HELP = {"docs", "version", "gen_makefile"}


def _commands() -> list[str]:
    """``mlsweep ...`` from the shell block's lines (comments kept) and from inline code."""
    found = []
    for block in re.findall(r"```sh\n(.*?)```", SKILL, re.S):
        found += [line.strip() for line in block.splitlines() if line.strip().startswith("mlsweep ")]
    prose = re.sub(r"```.*?```", "", SKILL, flags=re.S)
    found += [c for c in re.findall(r"`([^`\n]+)`", prose) if c.startswith("mlsweep ")]
    return list(dict.fromkeys(found))


def _subcommand(command: str) -> tuple[str, ...]:
    words = command.split()[1:]
    if words[0] in GROUPS and len(words) > 1 and re.fullmatch(r"[a-z_-]+", words[1]):
        return tuple(words[:2])
    return (words[0],)


@lru_cache(maxsize=None)
def _help(sub: tuple[str, ...]) -> str:
    done = subprocess.run(
        [sys.executable, "-c", "from mlsweep.cli import main; main()", *sub, "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, f"mlsweep {' '.join(sub)} --help: {done.stderr}"
    return done.stdout


def _checkable() -> list[str]:
    return [c for c in _commands() if not c.split()[1].startswith("-") and c.split()[1] not in NO_HELP]


def test_finds_the_commands() -> None:
    assert len(_checkable()) > 30


@pytest.mark.parametrize("command", _checkable())
def test_command_and_flags_exist(command: str) -> None:
    text = _help(_subcommand(command))
    for flag in re.findall(r"(?<![\w-])--[a-z][a-z-]*", command):
        assert re.search(rf"(?<![\w-]){flag}(?![\w-])", text), f"{flag} not in help for: {command}"


@pytest.mark.parametrize("topic", re.findall(r"^- `(\w+)`:", SKILL, re.M))
def test_doc_topics_exist(topic: str) -> None:
    assert topic in cli._DOC_ALIASES or topic == "readme"
