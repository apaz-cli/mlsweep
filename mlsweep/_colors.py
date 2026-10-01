"""Global ANSI color toggle and color constants.

Colored human output is **off by default** and only turned on when ``--color``
is passed to an entry point.  The constants below are lightweight sentinel
objects, not plain strings: their string rendering resolves against the
current global setting at call time, so existing f-strings such as
``f"{_GREEN}ok{_RESET}"`` automatically produce no escapes unless color is
enabled.

Machine-readable formats (``--json``/``--csv``) and raw log/metric tables are
never routed through these constants, so they stay byte-pure regardless of the
setting.
"""

from __future__ import annotations

__all__ = [
    "set_color",
    "color_enabled",
    "strip_color_flag",
    "_GREEN",
    "_RED",
    "_YELLOW",
    "_CYAN",
    "_MAGENTA",
    "_BLUE",
    "_RESET",
    "_BOLD",
    "_DIM",
    "_BRIGHT_GREEN",
    "_BRIGHT_BLUE",
]

_enabled: bool = False


def set_color(enabled: bool) -> None:
    """Enable or disable ANSI color globally."""
    global _enabled
    _enabled = bool(enabled)


def color_enabled() -> bool:
    """Return whether ANSI color is currently enabled."""
    return _enabled


class _Color:
    """A color escape (or ``_RESET``) that resolves when converted to text."""

    __slots__ = ("_code",)

    def __init__(self, code: str) -> None:
        self._code = code

    def __str__(self) -> str:
        return self._code if _enabled else ""

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)

    def __add__(self, other: object) -> str:
        return str(self) + str(other)

    def __radd__(self, other: object) -> str:
        return str(other) + str(self)

    def __bool__(self) -> bool:
        return bool(str(self))

    def __repr__(self) -> str:
        return f"_Color({self._code!r})"


_GREEN = _Color("\033[32m")
_RED = _Color("\033[31m")
_YELLOW = _Color("\033[33m")
_CYAN = _Color("\033[36m")
_MAGENTA = _Color("\033[35m")
_BLUE = _Color("\033[34m")
_RESET = _Color("\033[0m")
_BOLD = _Color("\033[1m")
_DIM = _Color("\033[2m")
_BRIGHT_GREEN = _Color("\033[92m")
_BRIGHT_BLUE = _Color("\033[94m")


def strip_color_flag(argv: list[str]) -> list[str]:
    """Remove a leading ``--color`` flag, enabling color.

    Only arguments *before* a bare ``--`` are considered, so training-run
    passthrough args (``mlsweep run sweep.py -- --color``) are left intact.
    """
    out: list[str] = []
    seen_ddash = False
    for arg in argv:
        if arg == "--":
            seen_ddash = True
        elif not seen_ddash and arg == "--color":
            set_color(True)
            continue
        out.append(arg)
    return out
