"""`mlsweep capture show | search | export`: reading traces from the terminal.

``show`` prints a trace as a compact ledger, one line per event, and stops at a
size budget (``--budget``, in tokens), ending with the command for the next
page.  ``--kind`` keeps one kind of event, as the Traces tab's buttons do, and
``--event N`` (or a ``#EN`` reference) prints one event in full.  ``search``
finds events across traces and gives each hit's reference, ready for a note.
``export`` writes a trace as ATIF or as the JSONL capture uploaded.

Every command takes a trace id, a unique prefix of one, or a note reference
(``mlsweep:trace/<id>?kind=mlsweep#E412``).  ``--json`` gives the data instead.
"""

from __future__ import annotations

import json
import sys
from typing import Any, NoReturn
from urllib.parse import quote, urlencode

from mlsweep._trace_view import (
    DEFAULT_BUDGET,
    TRACE_LINK_KINDS,
    Ledger,
    fmt_ms,
    iso_time,
    kind_label,
    mlsweep_calls,
    parse_trace_ref,
    row_line,
    take_within_budget,
    trace_ref,
)

_PAGE = 2000   # events fetched per request


def add_parsers(sub: Any, add_connection_args: Any) -> None:
    """Add show, search and export to `mlsweep capture`'s subcommands."""
    show = sub.add_parser(
        "show", help="Print a trace as a compact ledger, or one event in full",
        description="Print a trace one line per event, within a size budget, then the "
                    "command for the next page. A note reference works as REF, "
                    "#E<seq> included.")
    show.add_argument("ref", metavar="REF", help="trace id, unique prefix, or mlsweep:trace/… reference")
    show.add_argument("--kind", choices=TRACE_LINK_KINDS, help="only this kind of event")
    show.add_argument("--from", dest="from_seq", type=int, default=0, metavar="SEQ",
                      help="start at this event (the next-page command sets it)")
    show.add_argument("--event", type=int, metavar="SEQ", help="print this one event in full")
    show.add_argument("--limit", type=int, default=0, help="at most this many rows (default: no limit)")
    show.add_argument("--budget", type=int, default=DEFAULT_BUDGET, metavar="TOKENS",
                      help=f"stop after about this many tokens of output (default: {DEFAULT_BUDGET})")
    show.add_argument("--width", type=int, default=160, help="cut rows at this many characters (0: never)")
    show.add_argument("--json", action="store_true", dest="as_json")
    add_connection_args(show)

    search = sub.add_parser(
        "search", help="Find events across traces, each with its reference",
        description="Events whose text has every word (or \"quoted phrase\"), best-matching "
                    "traces first. Each hit's reference can go straight into a note.")
    search.add_argument("words", nargs="+", metavar="WORD")
    search.add_argument("--trace", metavar="REF", help="search only this trace")
    search.add_argument("--kind", choices=TRACE_LINK_KINDS, help="only this kind of event")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--json", action="store_true", dest="as_json")
    add_connection_args(search)

    export = sub.add_parser(
        "export", help="Write a trace as ATIF (default) or capture's JSONL",
        description="ATIF is the Agent Trajectory Interchange Format other agent tools read.")
    export.add_argument("ref", metavar="REF")
    export.add_argument("--format", choices=("atif", "jsonl"), default="atif")
    export.add_argument("-o", "--output", metavar="FILE", help="write here instead of stdout")
    add_connection_args(export)


def _fail(message: str, code: int = 1) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _get(manager: str, token: str, path: str, *, raw: bool = False) -> Any:
    from mlsweep.run_sweep import _http_request

    status, resp = _http_request("GET", f"{manager}{path}", token, quiet=True, raw=raw)
    if status == 0:
        _fail(f"cannot reach the manager at {manager}")
    if status == 404 and not (isinstance(resp, dict) and resp.get("error")):
        _fail("the manager has no such route: it takes no traces (--no-capture) or runs an older mlsweep")
    if status >= 400:
        error = resp.get("error") if isinstance(resp, dict) else resp
        _fail(str(error or f"manager answered {status}"))
    return resp


def _session(manager: str, token: str, ref: str) -> tuple[dict[str, Any], str, int | None]:
    try:
        sid, kind, seq = parse_trace_ref(ref)
    except ValueError as exc:
        _fail(str(exc), 2)
    session = _get(manager, token, f"/api/capture/sessions/{quote(sid, safe='')}")
    return session, kind, seq


def _header(session: dict[str, Any]) -> list[str]:
    sid = session["session_id"]
    title = " ".join(str(session.get("title") or "").split())
    lines = [f"trace {sid}  {session.get('agent', '?')}  {session.get('state', '?')}  "
             f"{session.get('event_count', 0)} events" + (f"  {title[:80]}" if title else "")]
    bits = []
    if session.get("cwd"):
        bits.append(f"cwd {session['cwd']}")
    if session.get("model"):
        bits.append(f"model {session['model']}")
    if session.get("experiments"):
        bits.append("experiments " + ", ".join(session["experiments"]))
    if bits:
        lines.append("  " + "  ·  ".join(bits))
    lines.append(f"  cite as {trace_ref(sid)}")
    return lines


def _events(manager: str, token: str, sid: str, from_seq: int, limit: int = _PAGE) -> dict[str, Any]:
    page: dict[str, Any] = _get(manager, token, f"/api/capture/sessions/{quote(sid, safe='')}/events?"
                                + urlencode({"from_seq": from_seq, "limit": limit}))
    return page


def cmd_show(args: Any) -> None:
    from mlsweep._jsonout import emit
    from mlsweep.run_sweep import _manager_token

    manager, token = _manager_token(args)
    session, ref_kind, ref_seq = _session(manager, token, args.ref)
    sid = session["session_id"]
    seq = args.event if args.event is not None else ref_seq
    if seq is not None:
        _show_event(manager, token, session, seq, args.as_json)
        return
    kind = args.kind or ref_kind
    ledger = Ledger(kind)
    cursor = max(args.from_seq, 0)
    lines: list[str] = []
    next_from: int | None = None
    budget = args.budget if args.budget > 0 else 10**9
    while next_from is None:
        page = _events(manager, token, sid, cursor)
        events = page.get("events") or []
        before = len(ledger.rows)
        ledger.add(events)
        for row in ledger.rows[before:]:
            line = json.dumps(row, default=str) if args.as_json else row_line(row, args.width)
            full = (args.limit and len(lines) >= args.limit) or \
                take_within_budget(lines + [line], budget) <= len(lines)
            if full:
                next_from = int(row["seq"])
                break
            lines.append(line)
        if not events or len(events) < _PAGE:
            break
        cursor = int(page["next_from_seq"])
    rows = ledger.rows[:len(lines)]
    again = f"mlsweep capture show {sid} --from {next_from}" + (f" --kind {kind}" if kind else "") \
        if next_from is not None else None
    if args.as_json:
        emit({"session": session, "kind": kind, "rows": rows, "next_from": next_from, "next": again})
        return
    print("\n".join(_header(session)))
    if kind:
        print(f"  showing {kind} events only")
    print()
    print("\n".join(lines) if lines else "(no events" + (f" of kind {kind}" if kind else "")
          + (f" from E{args.from_seq}" if args.from_seq else "") + ")")
    print()
    print(f"more: {again}" if again else "(end of trace)")


def _show_event(manager: str, token: str, session: dict[str, Any], seq: int, as_json: bool) -> None:
    from mlsweep._jsonout import emit

    sid = session["session_id"]
    page = _events(manager, token, sid, seq, limit=200)
    events = page.get("events") or []
    event = events[0] if events and events[0].get("seq") == seq else None
    if event is None:
        _fail(f"trace {sid[:8]} has no event {seq}")
    result = None
    if event.get("kind") == "tool" and event.get("tool_id"):
        result = next((e for e in events[1:] if e.get("kind") == "tool_result"
                       and e.get("tool_id") == event["tool_id"]), None)
    ref = trace_ref(sid, seq)
    if as_json:
        emit({"session_id": sid, "ref": ref, "event": event, "result": result})
        return
    label = kind_label(event)
    print(f"E{seq}  {label}  {iso_time(event.get('ts')) or 'no timestamp'}  trace {sid}")
    meta = [f"{k} {event[k]}" for k in ("name", "model") if event.get(k)]
    tokens = event.get("tokens")
    if isinstance(tokens, dict) and any(v is not None for v in tokens.values()):
        meta.append("tokens " + ", ".join(f"{k} {v}" for k, v in tokens.items() if v is not None))
    if result is not None:
        took = fmt_ms(result.get("duration_ms"))
        meta.append(("failed" if result.get("ok") is False else "succeeded") + (f" in {took}" if took else ""))
    if meta:
        print("  " + "  ·  ".join(meta))
    print(f"  cite as {ref}")
    print()
    text = str(event.get("text") or "")
    calls = mlsweep_calls(text) if label == "mlsweep" else []
    if calls:
        print("mlsweep: " + "\n         ".join(calls))
        print()
    print(text if text else "(no text)")
    if event.get("kind") == "tool":
        print("\n(capture keeps a tool call's one-line summary and its status, not its output)")
    print(f"\ncontext: mlsweep capture show {sid} --from {max(seq - 5, 0)}")


def cmd_search(args: Any) -> None:
    from mlsweep._jsonout import emit
    from mlsweep.run_sweep import _manager_token

    manager, token = _manager_token(args)
    query: dict[str, Any] = {"q": " ".join(args.words), "limit": args.limit}
    if args.kind:
        query["kind"] = args.kind
    if args.trace:
        try:
            query["session"] = parse_trace_ref(args.trace)[0]
        except ValueError as exc:
            _fail(str(exc), 2)
    hits = _get(manager, token, "/api/capture/search/events?" + urlencode(query))
    if args.as_json:
        emit({"hits": hits})
        return
    if not hits:
        print("No matching events.")
        return
    last = None
    width = max(len(h["ref"]) for h in hits)
    for h in hits:
        if h["session_id"] != last:
            last = h["session_id"]
            title = " ".join(str(h.get("title") or h.get("cwd") or "").split())
            print(f"\ntrace {last}  {h.get('agent') or '?'}" + (f"  {title[:80]}" if title else ""))
        print(f"  {h['ref']:<{width}}  {h['label'][:12]:<12} {h['snippet']}")
    print(f"\n{len(hits)} event{'s' if len(hits) != 1 else ''}. "
          "`mlsweep capture show REF` prints one in full.")


def cmd_export(args: Any) -> None:
    from mlsweep.run_sweep import _manager_token

    manager, token = _manager_token(args)
    session, _, _ = _session(manager, token, args.ref)
    sid = quote(session["session_id"], safe="")
    if args.format == "atif":
        body = json.dumps(_get(manager, token, f"/api/capture/sessions/{sid}/atif"), indent=2) + "\n"
    else:
        raw = _get(manager, token, f"/api/capture/sessions/{sid}/transcript", raw=True)
        body = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw or "")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"Wrote {args.output} ({args.format}).", file=sys.stderr)
    else:
        sys.stdout.write(body)


COMMANDS = {"show": cmd_show, "search": cmd_search, "export": cmd_export}
