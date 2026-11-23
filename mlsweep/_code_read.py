"""Reading and searching the code experiments ran, the way traces are read.

``mlsweep code EXP PATH --lines 40-60`` (or ``-n`` for the whole file) prints
numbered lines within a size budget, then the command for the next page and
the note reference for what was shown.  A note's link works in place of EXP and
PATH: ``mlsweep code mlsweep:code/EXP/train.py#L40-60``.

``mlsweep code search PATTERN [EXP...]`` finds lines in one or more
experiments' code, or across the campaign's experiments, each hit with its
``mlsweep:code/EXP/path#L12`` reference.  Without ``-n``/``--lines``, a file
still prints raw, so ``mlsweep code EXP PATH > file`` keeps working.
"""

from __future__ import annotations

import re
import sys
from typing import Any
from urllib.parse import quote

from mlsweep._trace_view import DEFAULT_BUDGET, take_within_budget

_OPEN_RANGE_RE = re.compile(r"^L?(\d+)-$")


def parse_line_range(text: str) -> tuple[int, int | None]:
    """``40``, ``40-60``, ``40-`` (to the end), ``L40-L60`` as (start, end or None)."""
    from mlsweep._notes_code import parse_lines

    m = _OPEN_RANGE_RE.match(text.strip())
    if m:
        start = int(m.group(1))
        if start < 1:
            raise ValueError(f"bad line range `{text}`")
        return start, None
    first, end = parse_lines(text)
    if first is None:
        raise ValueError("an empty line range")
    return first, end


def print_lines(data: bytes, *, cite_target: str, path: str, start: int, end: int | None,
                budget: int, again: str, as_json: bool) -> None:
    """Numbered lines *start*..*end* of a file, cut at *budget* tokens, with the
    next-page command (``again`` plus ``--lines``) and the reference shown."""
    from mlsweep._jsonout import emit
    from mlsweep._snapshot import looks_binary

    if looks_binary(data):
        print(f"Error: {path} is binary; `mlsweep code ... {path} > FILE` saves it", file=sys.stderr)
        raise SystemExit(1)
    lines = data.decode("utf-8", errors="replace").splitlines()
    total = len(lines)
    if start > max(total, 1):
        print(f"Error: {path} has {total} lines", file=sys.stderr)
        raise SystemExit(1)
    stop = total if end is None else min(end, total)
    width = len(str(stop))
    rendered = [f"{n:>{width}}  {lines[n - 1]}" for n in range(start, stop + 1)]
    shown = take_within_budget(rendered, budget if budget > 0 else 10**9)
    last = start + shown - 1
    ref = f"mlsweep:code/{cite_target}/{path}#L{start}" + (f"-{last}" if last != start else "")
    more = None
    if last < stop:
        more = f"{again} --lines {last + 1}-{end if end is not None else ''}"
    if as_json:
        emit({"path": path, "target": cite_target, "lines": total, "start": start, "end": last,
              "text": "\n".join(lines[start - 1:last]), "ref": ref, "next": more})
        return
    print(f"{path}  lines {start}-{last} of {total}  ({cite_target})")
    print("\n".join(rendered[:shown]))
    print()
    print(f"cite as {ref}")
    if more:
        print(f"more: {more}")


def search(manager: str, token: str, args: Any, campaign: str | None) -> None:
    from mlsweep._jsonout import emit
    from mlsweep.run_sweep import _http_request

    body: dict[str, Any] = {"q": args.pattern, "regex": args.regex, "ignore_case": args.ignore_case,
                            "context": args.context_lines, "limit": args.limit}
    if args.targets:
        body["targets"] = args.targets
    else:
        body["campaign"] = campaign
    if args.path:
        body["path"] = args.path
    status, resp = _http_request("POST", f"{manager}/api/code/search", token, json_data=body,
                                 quiet=True, timeout=300)
    if status == 0:
        print(f"Error: cannot reach the manager at {manager}", file=sys.stderr)
        raise SystemExit(1)
    if status == 404 and not (isinstance(resp, dict) and resp.get("error")):
        print("Error: this manager runs an older mlsweep without code search", file=sys.stderr)
        raise SystemExit(1)
    if status >= 400:
        print(f"Error: {resp.get('error') if isinstance(resp, dict) else resp}", file=sys.stderr)
        raise SystemExit(1)
    if args.json:
        emit(resp)
        return
    for target, why in (resp.get("errors") or {}).items():
        print(f"{target}: {why}", file=sys.stderr)
    hits = resp["hits"]
    if not hits:
        print(f"No matches in {resp['files_searched']} files.")
        return
    last = None
    for h in hits:
        where = (h["target"], h["path"])
        if where != last:
            last = where
            also = h.get("also_in") or []
            extra = f"  (same file in {len(also)} more: {', '.join(also[:3])}{', …' if len(also) > 3 else ''})" \
                if also else ""
            print(f"\n{h['path']}  in {h['target']}{extra}")
        for s in h.get("before", []):
            print(f"  {'':>6}  {s}")
        print(f"  {h['line']:>6}: {h['text']}")
        for s in h.get("after", []):
            print(f"  {'':>6}  {s}")
    files = {(h["target"], h["path"]) for h in hits}
    print(f"\n{len(hits)} line{'s' if len(hits) != 1 else ''} in {len(files)} file"
          f"{'s' if len(files) != 1 else ''} ({resp['files_searched']} searched"
          + (f", {resp['files_skipped']} binary or too large" if resp.get("files_skipped") else "")
          + (", more not shown: raise --limit or narrow the search" if resp.get("truncated") else "")
          + ").")
    first = hits[0]
    print(f"Cite a line as {first['ref']}, or read around it with "
          f"`mlsweep code {first['target']} {quote(first['path'], safe='/')} --lines {max(first['line'] - 10, 1)}-{first['line'] + 10}`.")
