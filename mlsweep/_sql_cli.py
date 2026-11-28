"""`mlsweep sql`: read-only SQL over the manager's database.

    mlsweep sql                       the tables and their columns
    mlsweep sql "SELECT …"            a table of the rows
    mlsweep sql "SELECT …" --json     {"columns", "rows", "truncated", …}

For questions the other commands don't answer.  Queries can only read (see
``_manager_sql``).  Column names and meanings can change between versions, so
prefer the other commands where one fits.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

_NO_SQL_ROUTE = "this manager runs an older mlsweep without `mlsweep sql`"


def main(argv: list[str]) -> None:
    from mlsweep._jsonout import emit
    from mlsweep.run_sweep import (
        _add_connection_args,
        _http_request,
        _manager_token,
        check_response,
    )

    ap = argparse.ArgumentParser(
        prog="mlsweep sql",
        description="Read-only SQL over the manager's database. With no query, list the tables.")
    ap.add_argument("query", nargs="?", metavar="SQL", help="one SELECT statement (or - to read it from stdin)")
    ap.add_argument("--limit", type=int, default=200, help="at most this many rows (default 200, max 5000)")
    ap.add_argument("--timeout", type=float, default=5.0, help="stop the query after this many seconds (max 30)")
    ap.add_argument("--width", type=int, default=60, help="cut cells at this many characters (0: never)")
    ap.add_argument("--json", action="store_true", dest="as_json")
    _add_connection_args(ap)
    args = ap.parse_args(argv)
    manager, token = _manager_token(args)

    query = sys.stdin.read() if args.query == "-" else args.query
    if not query:
        status, resp = _http_request("GET", f"{manager}/api/sql", token, quiet=True)
        check_response(status, resp, manager, missing=_NO_SQL_ROUTE)
        if args.as_json:
            emit(resp)
            return
        for t in resp["tables"]:
            cols = ", ".join(c["name"] for c in t["columns"])
            print(f"{t['name']}{' (view)' if t['type'] == 'view' else ''}: {cols}")
        return

    status, resp = _http_request("POST", f"{manager}/api/sql", token, quiet=True,
                                 json_data={"sql": query, "limit": args.limit, "timeout": args.timeout},
                                 timeout=int(args.timeout) + 30)
    check_response(status, resp, manager, missing=_NO_SQL_ROUTE)
    if args.as_json:
        emit(resp)
        return
    print(_table(resp["columns"], resp["rows"], args.width))
    n = resp["row_count"]
    note = f"{n} row{'s' if n != 1 else ''}"
    if resp.get("truncated"):
        note += ", cut off (raise --limit, at most 5000, or narrow the query)"
    print(f"({note}, {resp.get('seconds', 0)} s)")


def _table(columns: list[str], rows: list[list[Any]], width: int) -> str:
    def cell(v: Any) -> str:
        s = "NULL" if v is None else " ".join(str(v).split())
        return s if width <= 0 or len(s) <= width else s[:width - 1] + "…"

    body = [[cell(v) for v in r] for r in rows]
    widths = [max([len(c)] + [len(r[i]) for r in body]) for i, c in enumerate(columns)]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    return "\n".join([line(columns), line(["-" * w for w in widths])] + [line(r) for r in body])
