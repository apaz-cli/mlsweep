"""How a notes repo's config file is read, and which paths it marks as originals."""

from __future__ import annotations

import fnmatch
import json
from typing import Any

CONFIG_FILE = ".mlsweep-notes.json"
DEFAULT_CONFIG: dict[str, Any] = {
    "protected": ["sources/"],
    "max_file_mb": 50,
}


def parse_config(raw: str | None) -> dict[str, Any]:
    """The settings in the text of a config file; defaults for anything missing or malformed."""
    cfg = dict(DEFAULT_CONFIG)
    if not raw:
        return cfg
    try:
        data = json.loads(raw)
    except ValueError:
        return cfg
    if isinstance(data, dict):
        if isinstance(data.get("protected"), list):
            cfg["protected"] = [x for x in data["protected"] if isinstance(x, str)]
        if isinstance(data.get("max_file_mb"), (int, float)) and data["max_file_mb"] > 0:
            cfg["max_file_mb"] = data["max_file_mb"]
    return cfg


def is_protected(path: str, patterns: list[str]) -> bool:
    """True if repo-relative *path* falls under a protected pattern.

    A pattern ending in ``/`` protects that directory; anything else is a
    glob matched against the whole path.  This is a convention: lint skips these
    files and the dashboard marks them, but nothing refuses a change to them.
    """
    for pat in patterns:
        pat = pat.lstrip("/")
        if not pat:
            continue
        if pat.endswith("/"):
            if path.startswith(pat):
                return True
        elif fnmatch.fnmatchcase(path, pat):
            return True
    return False
