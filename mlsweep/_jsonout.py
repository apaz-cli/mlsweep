"""The ``--json`` contract every mlsweep command keeps.

With ``--json`` (or ``watch --events``), stdout carries data only:

* one JSON value, the command's result (``watch --events`` streams one JSON
  object per line instead);
* or, when the command fails, ``{"error": MESSAGE, "exit_code": N}``, with that
  same nonzero exit code.  Bad arguments and an unreachable manager are failures
  like any other.

Everything meant for people (progress, OK/FAIL lines, warnings, tracebacks)
goes to stderr.  A command that ends with a nonzero exit code after printing
its result, such as ``wait`` reporting a failed run, prints no error object:
the result says what happened.

``contract(argv)`` sets this up around one command.  Commands print their
result with ``emit`` (or ``emit_line``), and keep printing everything else as
usual: while the contract is in force, plain ``print`` goes to stderr.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
from collections.abc import Iterator
from typing import Any, TextIO

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Prefixes the CLI puts on failure lines, dropped from the error message.
_FAIL_PREFIX = re.compile(r"^\s*(?:FAIL\s+|Error:\s*|error:\s*)", re.IGNORECASE)

_out: TextIO | None = None  # the real stdout, while the contract is in force
_emitted = False
_last_line = ""


def wants_json(argv: list[str]) -> bool:
    """Whether *argv* asks for machine-readable output (before any bare ``--``)."""
    for arg in argv:
        if arg == "--":
            return False
        if arg in ("--json", "--events"):
            return True
    return False


def active() -> bool:
    return _out is not None


def emit(value: Any) -> None:
    """Print a command's result: its one JSON value on stdout."""
    global _emitted
    out = _out or sys.stdout
    out.write(json.dumps(value, indent=2, default=str) + "\n")
    out.flush()
    _emitted = True


def emit_line(value: Any) -> None:
    """Print one JSON object of a stream (``watch --events``), on its own line."""
    global _emitted
    out = _out or sys.stdout
    out.write(json.dumps(value, default=str) + "\n")
    out.flush()
    _emitted = True


# A line that starts a failure message: FAIL/Error lines, and argparse's "prog: error: ...".
_FAIL_START = re.compile(r"^\s*(?:FAIL\b|Error:|error:)|: error: ", re.IGNORECASE)


class _Tee:
    """stderr, remembering the last failure message written, which becomes the error.

    That is the last line that starts a failure (FAIL, Error:, argparse's
    error) with the lines after it, such as a hint, or else the last line.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._block: list[str] = []

    def write(self, text: str) -> int:
        global _last_line
        for line in text.splitlines():
            plain = _ANSI.sub("", line)
            clean = _FAIL_PREFIX.sub("", plain).strip()
            if not clean:
                continue
            if _FAIL_START.search(plain):
                self._block = [clean]
            elif self._block:
                self._block.append(clean)
            _last_line = " ".join(self._block) if self._block else clean
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


@contextlib.contextmanager
def contract(argv: list[str]) -> Iterator[None]:
    """Keep the ``--json`` contract for one command run with *argv* (a no-op without ``--json``)."""
    global _out, _emitted, _last_line
    if not wants_json(argv) or active():
        yield
        return
    real_out, real_err = sys.stdout, sys.stderr
    _out, _emitted, _last_line = real_out, False, ""
    tee = _Tee(real_err)
    sys.stdout = tee
    sys.stderr = tee
    code: int | None = 0
    try:
        try:
            yield
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            if not isinstance(e.code, (int, type(None))):
                print(e.code)  # sys.exit("message")
            raise
        except KeyboardInterrupt:
            code = 130
            raise
        except Exception as e:
            code = 1
            import traceback
            traceback.print_exc()
            _last_line = f"{type(e).__name__}: {e}"
            raise SystemExit(1) from None
    finally:
        sys.stdout, sys.stderr = real_out, real_err
        if code and not _emitted:
            real_out.write(json.dumps({"error": _last_line or "the command failed", "exit_code": code}) + "\n")
            real_out.flush()
        elif not code and not _emitted:
            # Every command that takes --json prints a result; say so rather than print nothing.
            real_out.write(json.dumps({"error": "this command printed no JSON result", "exit_code": 1}) + "\n")
            real_out.flush()
            _out = None
            raise SystemExit(1)
        _out = None
