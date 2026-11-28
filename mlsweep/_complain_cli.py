"""`mlsweep complain`: report a problem with mlsweep, or an idea for it.

    mlsweep complain "TEXT"                     a problem (the default kind)
    mlsweep complain --improvement "TEXT"       an idea
    mlsweep complain - < report.md              the text from stdin
    mlsweep complain --list                     the latest reports

Reports are stored in the manager's database (the ``complaints`` table) and go
nowhere else.  Each one records the agent trace it came from, when there is
one, so a reader can see what led to it.
"""

from __future__ import annotations

import argparse
import getpass
import socket
import sys
from collections.abc import Callable

_NO_ROUTE = "this manager runs an older mlsweep without `mlsweep complain`"


def _version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("mlsweep")
    except PackageNotFoundError:
        return None


def _user() -> str | None:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return None


def _default_session() -> str | None:
    from mlsweep.run_sweep import capture_session_id
    return capture_session_id()


def main(argv: list[str], *, session: Callable[[], str | None] = _default_session) -> None:
    from mlsweep._jsonout import emit
    from mlsweep.run_sweep import (
        _add_connection_args,
        _http_request,
        _manager_token,
        check_response,
    )

    ap = argparse.ArgumentParser(
        prog="mlsweep complain",
        description="Report a problem with mlsweep, or an idea for it. Reports are stored on the manager "
                    "(the complaints table, see `mlsweep sql`) and are not sent anywhere else. Say what you "
                    "ran, what happened, and what you expected.")
    ap.add_argument("text", nargs="?", metavar="TEXT", help="the report (or - to read it from stdin)")
    ap.add_argument("--improvement", action="store_const", const="improvement", dest="kind", default="problem",
                    help="an idea for making mlsweep better, not a problem")
    ap.add_argument("--experiment", metavar="EXP", help="the experiment this is about, if any")
    ap.add_argument("--command", metavar="CMD", help="the mlsweep command that misbehaved, if any")
    ap.add_argument("--list", action="store_true", help="show the latest reports instead")
    ap.add_argument("--limit", type=int, default=20, help="with --list: at most this many (default 20)")
    ap.add_argument("--json", action="store_true", dest="as_json")
    _add_connection_args(ap)
    args = ap.parse_args(argv)
    manager, token = _manager_token(args)

    if args.list:
        status, resp = _http_request("GET", f"{manager}/api/complaints?limit={max(1, args.limit)}", token,
                                     quiet=True)
        check_response(status, resp, manager, missing=_NO_ROUTE)
        if args.as_json:
            emit(resp)
            return
        complaints = resp.get("complaints") or []
        if not complaints:
            print("No reports.")
        for c in complaints:
            about = f"  [{c['experiment_id']}]" if c.get("experiment_id") else ""
            print(f"#{c['complaint_id']}  {c['time'][:16].replace('T', ' ')}  {c['kind']}{about}")
            for line in c["text"].splitlines():
                print(f"    {line}")
        return

    text = sys.stdin.read() if args.text == "-" else args.text
    if not text or not text.strip():
        ap.error("say what went wrong: mlsweep complain \"TEXT\" (or - to read stdin)")
    body = {
        "text": text, "kind": args.kind, "experiment_id": args.experiment, "command": args.command,
        "session_id": session(), "version": _version(), "host": socket.gethostname(), "user": _user(),
    }
    status, resp = _http_request("POST", f"{manager}/api/complaints", token, quiet=True,
                                 json_data={k: v for k, v in body.items() if v})
    check_response(status, resp, manager, missing=_NO_ROUTE)
    if body["session_id"]:
        # Best effort: upload the trace behind the report if capture is on.
        from mlsweep._capture import follow_if_on
        follow_if_on(body["session_id"], why=f"complaint #{resp['complaint_id']}", manager=manager)
    if args.as_json:
        emit(resp)
        return
    print(f"Recorded {args.kind} #{resp['complaint_id']}. Thanks.")
