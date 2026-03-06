"""Shared utilities used across mlsweep modules."""

import subprocess
from typing import Any


def _git_root(path: str) -> str | None:
    """Return the root directory of the git repo containing path, or None."""
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path,
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _parse_tag_value(s: str) -> bool | int | float | str:
    """Convert a tag value string to a typed Python value."""
    if s == "True":
        return True
    if s == "False":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _val_sort_key(v: Any) -> tuple:
    """Sort key for axis values: bools first, then numbers, then strings."""
    if isinstance(v, bool):
        return (0, str(v))
    if isinstance(v, (int, float)):
        return (1, v)
    return (2, str(v))


def _detect_sub_axes(
    runs: list[dict[str, Any]],
    axes: dict[str, list[Any]],
) -> dict[str, dict[str, Any]]:
    """Detect axes that only appear when a parent axis has a specific value.

    runs:  list of dicts with "hash" and "combo" keys.
    axes:  {axis_name: [sorted values]}.
    Returns {child_axis: {"parentAxis": ..., "parentValue": ...}}.
    """
    all_names = {r["hash"] for r in runs}
    names_with = {ax: {r["hash"] for r in runs if ax in r["combo"]} for ax in axes}
    sub_axes: dict[str, dict[str, Any]] = {}
    for axis in axes:
        if names_with[axis] == all_names:
            continue  # universal axis — not a sub-axis
        for parent_axis in axes:
            if parent_axis == axis:
                continue
            for parent_val in axes[parent_axis]:
                names_with_parent = {
                    r["hash"] for r in runs if r["combo"].get(parent_axis) == parent_val
                }
                if names_with_parent == names_with[axis]:
                    sub_axes[axis] = {"parentAxis": parent_axis, "parentValue": parent_val}
                    break
            if axis in sub_axes:
                break
    return sub_axes
