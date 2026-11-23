"""Read-only SQL over the manager's database: ``POST /api/sql``.

For the questions the other routes don't answer.  ``mlsweep sql`` sends the
query, and ``GET /api/sql`` lists the tables and their columns.

A query runs on its own connection, opened read-only (``mode=ro``) with
``query_only`` set and an authorizer that allows reading and nothing else, so
no statement can write, attach another file or change a setting.  It runs in a
thread, one query at a time, and is stopped after its time limit.  Rows past
the limit, and output past a size cap, are cut off and the answer says so.

The database holds nothing the API doesn't already serve to the same token
(the workers' ``ssh_key`` is a path, as ``/api/workers`` shows).
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from aiohttp import web

from mlsweep._manager_http import _bad_request, _json_body, _json_response

sql_routes = web.RouteTableDef()

SQL_DB_KEY: web.AppKey[str] = web.AppKey("mlsweep_sql_db", str)
_LOCK_KEY: web.AppKey[asyncio.Lock] = web.AppKey("mlsweep_sql_lock", asyncio.Lock)

DEFAULT_ROWS = 200
MAX_ROWS = 5000
DEFAULT_TIMEOUT = 5.0
MAX_TIMEOUT = 30.0
MAX_OUTPUT_CHARS = 4_000_000
_MAX_CELL_CHARS = 100_000

# Pragmas that only read the schema.
_READ_PRAGMAS = frozenset({"table_info", "table_xinfo", "index_list", "index_info", "index_xinfo",
                           "foreign_key_list"})
_ALLOWED = frozenset({sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                      getattr(sqlite3, "SQLITE_RECURSIVE", 33)})


def _authorize(action: int, arg1: str | None, arg2: str | None, _db: str | None, _trigger: str | None) -> int:
    if action in _ALLOWED:
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in ("load_extension", "readfile",
                                                                            "writefile", "edit"):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() in _READ_PRAGMAS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{quote(str(Path(db_path).resolve()))}?mode=ro", uri=True,
                          check_same_thread=False)
    con.execute("PRAGMA query_only = ON")
    return con


def _cell(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, str) and len(value) > _MAX_CELL_CHARS:
        return value[:_MAX_CELL_CHARS] + "…"
    return value


def run_query(db_path: str, sql: str, *, limit: int = DEFAULT_ROWS,
              timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Run one read-only statement.  Raises sqlite3.Error with SQLite's message."""
    con = _connect(db_path)
    try:
        con.set_authorizer(_authorize)
        deadline = time.monotonic() + timeout
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        started = time.monotonic()
        try:
            cur = con.execute(sql)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc):
                raise sqlite3.OperationalError(f"stopped after {timeout:g} s") from None
            raise
        columns = [d[0] for d in cur.description or []]
        rows: list[list[Any]] = []
        size = 0
        truncated = False
        try:
            for raw in cur:
                if len(rows) >= limit:
                    truncated = True
                    break
                row = [_cell(v) for v in raw]
                size += sum(len(str(v)) for v in row) + 2 * len(row)
                if size > MAX_OUTPUT_CHARS:
                    truncated = True
                    break
                rows.append(row)
        except sqlite3.OperationalError as exc:
            if "interrupted" not in str(exc):
                raise
            raise sqlite3.OperationalError(f"stopped after {timeout:g} s") from None
        return {"columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated,
                "seconds": round(time.monotonic() - started, 3)}
    finally:
        con.close()


def list_tables(db_path: str) -> list[dict[str, Any]]:
    """Every table and view, with its columns."""
    con = _connect(db_path)
    try:
        out = []
        for name, kind in con.execute(
                "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') "
                "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'capture_fts_%' ORDER BY name"):
            cols = [{"name": c[1], "type": c[2]} for c in con.execute(f'PRAGMA table_info("{name}")')]
            out.append({"name": name, "type": kind, "columns": cols})
        return out
    finally:
        con.close()


@sql_routes.get("/api/sql")
async def handle_sql_tables(request: web.Request) -> web.Response:
    """The tables and columns a query can read."""
    return _json_response({"tables": await asyncio.to_thread(list_tables, request.app[SQL_DB_KEY])})


@sql_routes.post("/api/sql")
async def handle_sql(request: web.Request) -> web.Response:
    """``{"sql": "SELECT …", "limit": 200, "timeout": 5}``: one read-only statement."""
    body = await _json_body(request)
    sql = body.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        return _bad_request("'sql' must be a non-empty string")
    try:
        limit = min(max(int(body.get("limit", DEFAULT_ROWS)), 1), MAX_ROWS)
        timeout = min(max(float(body.get("timeout", DEFAULT_TIMEOUT)), 0.1), MAX_TIMEOUT)
    except (TypeError, ValueError):
        return _bad_request("'limit' and 'timeout' must be numbers")
    async with request.app[_LOCK_KEY]:
        try:
            result = await asyncio.to_thread(run_query, request.app[SQL_DB_KEY], sql,
                                             limit=limit, timeout=timeout)
        except sqlite3.Warning as exc:   # "You can only execute one statement at a time."
            return _bad_request(str(exc))
        except sqlite3.DatabaseError as exc:
            message = str(exc)
            if "not authorized" in message:
                message += " (queries here can only read)"
            return _bad_request(message)
    return _json_response(result)


def setup_sql(app: web.Application, db_path: str) -> None:
    """Register the SQL routes over the database at *db_path*."""
    app[SQL_DB_KEY] = db_path
    app[_LOCK_KEY] = asyncio.Lock()
    app.add_routes(sql_routes)
