"""Regex matching for ``mlsweep code search -E``, in a process the manager can kill.

Python's ``re`` can backtrack for ever on some patterns (``(a+)+$``), and a
thread stuck in it cannot be stopped.  The manager runs this module as a
child, sends it the pattern and then one file at a time, and kills it when
the search runs past its deadline.

Protocol, one JSON object per line: first ``{"q", "flags", "max_line"}``, then
``{"lines": [...]}`` per file, each answered with ``{"found": [line indices]}``.
"""

import json
import re
import sys


def main() -> None:
    first = json.loads(sys.stdin.readline())
    pattern = re.compile(first["q"], first["flags"])
    max_line = int(first["max_line"])
    for raw in sys.stdin:
        lines = json.loads(raw)["lines"]
        found = [i for i, line in enumerate(lines) if pattern.search(line[:max_line])]
        sys.stdout.write(json.dumps({"found": found}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
