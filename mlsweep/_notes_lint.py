"""Mechanical checks over a notes folder.  Report-only, except the safe fixes.

Three tiers, after the karpathy-llm-wiki pattern:

1. **Safe fixes** (``fix``): a link whose path is wrong but whose file name
   matches exactly one file is pointed at that file.  Nothing else is ever
   changed automatically.
2. **Reports** (``report``): broken and ambiguous links, experiment IDs the
   manager does not know, leftover conflict markers and empty files.  These are
   suspects, for a person or a reviewing agent to judge.
3. **Judgment** is not done here, and that includes numbers.  Whether a number
   matches a run takes knowing what it means, so agents check them against the
   logged metrics.  ``review_prompt`` writes the brief for an independent
   reviewing agent, which records its findings as a new file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from mlsweep import _notes
from mlsweep._notes_git import read_config
from mlsweep._notes_rules import is_protected

@dataclass
class Finding:
    tier: str              # "fix" | "report"
    kind: str              # short slug
    path: str
    line: int
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"tier": self.tier, "kind": self.kind, "path": self.path, "line": self.line,
                "message": self.message, **({"detail": self.detail} if self.detail else {})}


KIND_TITLES = {
    "link-path": "Link path fixed by file name",
    "broken-link": "Broken links",
    "ambiguous-link": "Ambiguous links",
    "unknown-experiment": "Unknown experiment IDs",
    "conflict-marker": "Leftover conflict markers",
    "empty-file": "Empty files",
}

_CONFLICT_RE = re.compile(r"^(<{7}|={7}|>{7})( |$)")


def _check_conflict_markers(path: str, text: str, findings: list[Finding]) -> None:
    lines = text.split("\n")
    mask = _notes.code_mask(text)
    for i, line in enumerate(lines):
        if _CONFLICT_RE.match(line) and not mask[i]:
            findings.append(Finding("report", "conflict-marker", path, i + 1,
                                    "leftover merge conflict marker"))


def lint(root: str | Path, *, known_experiments: set[str] | None = None,
         index: _notes.Index | None = None) -> dict[str, Any]:
    """Run tiers 1 and 2 over *root*.  Returns ``{"findings": [...], "stats": {...}}``.

    *known_experiments* enables the unknown-ID check.
    """
    root = Path(root).resolve()
    known = known_experiments or set()
    if index is None:
        index = _notes.build_index(root, known)
    findings: list[Finding] = []
    # Originals under protected paths are kept as they are: not checked or fixed.
    protected = read_config(root)["protected"]

    def skip(path: str) -> bool:
        return is_protected(path, protected)

    by_name: dict[str, list[str]] = {}
    for f in index.files:
        by_name.setdefault(PurePosixPath(f.path).name.lower(), []).append(f.path)

    # ── links ──
    for src, links in index.links.items():
        if skip(src):
            continue
        for link in links:
            if link.resolved and not link.ambiguous:
                continue
            if link.resolved and link.ambiguous:
                findings.append(Finding(
                    "report", "ambiguous-link", src, link.line,
                    f"[[{link.target}]] matches {len(link.ambiguous)} files; linked to {link.resolved}",
                    {"target": link.target, "candidates": link.ambiguous}))
                continue
            name = PurePosixPath(link.target).name.lower()
            cands = by_name.get(name, []) or by_name.get(name + ".md", [])
            if len(cands) == 1 and ("/" in link.target or not link.wiki):
                findings.append(Finding(
                    "fix", "link-path", src, link.line,
                    f"{link.raw} points to a missing path; {cands[0]} has that name",
                    {"raw": link.raw, "target": link.target, "new": cands[0], "wiki": link.wiki}))
            elif not link.wiki or link.embed or "/" in link.target:
                findings.append(Finding(
                    "report", "broken-link", src, link.line,
                    f"{link.raw} does not resolve to any file",
                    {"target": link.target, "candidates": cands}))
            else:
                # [[Name]] with no file yet: an intentional "note to write" link in
                # Obsidian.  Report it, but as a link to a missing note.
                findings.append(Finding(
                    "report", "broken-link", src, link.line,
                    f"[[{link.target}]] has no note yet", {"target": link.target, "missing_note": True}))

    # ── experiments ──
    if known_experiments is not None:
        for path, text in index.texts.items():
            if skip(path):
                continue
            for m in _notes.GENERATED_EXP_ID_RE.finditer(text):
                if m.group(0) not in known:
                    line = text.count("\n", 0, m.start()) + 1
                    findings.append(Finding("report", "unknown-experiment", path, line,
                                            f"{m.group(0)} is not an experiment on this manager",
                                            {"id": m.group(0)}))

    # ── merge leftovers ──
    for path, text in index.texts.items():
        if not skip(path):
            _check_conflict_markers(path, text, findings)
    for f in index.files:
        if f.size == 0 and f.kind in ("md", "text"):
            findings.append(Finding("report", "empty-file", f.path, 0, "file is empty"))

    findings.sort(key=lambda x: (x.tier != "fix", x.kind, x.path, x.line))
    stats = {"files": len(index.files), "notes": len(index.texts),
             "links": sum(len(v) for v in index.links.values()),
             "experiments_cited": len(index.cited()),
             "experiments_checked": known_experiments is not None}
    return {"findings": [f.as_dict() for f in findings], "stats": stats,
            "kinds": KIND_TITLES}


def apply_fixes(root: str | Path, findings: list[dict[str, Any]]) -> list[str]:
    """Apply tier-1 ``link-path`` fixes.  Returns the files changed.

    A fix is applied only if the line still contains the link text the lint
    saw, so a fix computed before an edit never touches the edited text.
    """
    root = Path(root).resolve()
    changed: set[str] = set()
    by_file: dict[str, list[dict[str, Any]]] = {}
    for f in findings:
        if f.get("tier") == "fix" and f.get("kind") == "link-path":
            by_file.setdefault(f["path"], []).append(f)
    for path, fixes in by_file.items():
        p = _notes.resolve(root, path)
        text = _notes.read_text(p)
        if text is None:
            continue
        lines = text.split("\n")
        for fx in fixes:
            d = fx.get("detail", {})
            i = int(fx["line"]) - 1
            raw, target, new = d.get("raw"), d.get("target"), d.get("new")
            if not (raw and target and new) or not (0 <= i < len(lines)) or raw not in lines[i]:
                continue
            if d.get("wiki"):
                new_target = str(PurePosixPath(new).with_suffix("")) if new.endswith(".md") else new
                replacement = raw.replace(target, new_target, 1)
            else:
                m = _notes.MDLINK_RE.fullmatch(raw)
                if m is None:
                    continue
                replacement = _notes.with_href_path(m, _notes.relative_href(path, new))
            lines[i] = lines[i].replace(raw, replacement, 1)
            changed.add(path)
        if path in changed:
            _notes.atomic_write(p, "\n".join(lines).encode("utf-8"))
    return sorted(changed)


REVIEW_PROMPT = """\
You are reviewing a shared research notes folder at {root}. You did not write
these notes. Do not edit or delete any existing file.

Work from the evidence, not from the notes' own claims:
- Notes cite mlsweep experiment IDs. Check claims against the logged metrics:
  `mlsweep metrics --experiment ID --last --json`, `mlsweep best --experiment ID`.
- Files under sources/ are originals (papers, posts). Check quotes and numbers
  attributed to them against the file itself.

The mechanical lint below lists suspects. Judge each one; a suspect is not a
finding until you have checked it. Then look for what lint cannot see:
- numbers that do not match the logged metrics (lint does not check numbers);
- claims contradicted by another note or by newer experiments;
- conclusions stated more broadly than their evidence (one seed, short runs,
  one model size), or missing the conditions they hold under;
- dead ends recorded without a reason, and repeated experiments;
- notes that rewrote earlier conclusions without saying so.

Write your findings to a NEW file reviews/{date}-review.md: one bullet per
finding with the file and line, the evidence you checked, and what should
change. Say plainly when something checked out. Then stop.

Mechanical lint report:
{report}
"""


def format_report(result: dict[str, Any]) -> str:
    """The lint result as plain text, grouped by kind."""
    out: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for f in result["findings"]:
        groups.setdefault(f["kind"], []).append(f)
    if not groups:
        return "No findings."
    for kind, items in groups.items():
        title = KIND_TITLES.get(kind, kind)
        tier = "safe fix" if items[0]["tier"] == "fix" else "report"
        out.append(f"{title} ({len(items)}, {tier}):")
        for f in items:
            loc = f"{f['path']}:{f['line']}" if f["line"] else f["path"]
            out.append(f"  {loc}  {f['message']}")
        out.append("")
    s = result.get("stats", {})
    out.append(f"{s.get('notes', 0)} notes, {s.get('links', 0)} links, "
               f"{s.get('experiments_cited', 0)} experiments cited"
               + ("" if s.get("experiments_checked") else "; experiment IDs not checked (no manager)"))
    return "\n".join(out)


def review_prompt(root: str | Path, result: dict[str, Any], date: str) -> str:
    return REVIEW_PROMPT.format(root=root, date=date, report=format_report(result))
