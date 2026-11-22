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

import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from mlsweep import _notes, _notes_blocks, _notes_code
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
    "unknown-run": "Unknown run IDs",
    "block-spec": "Chart and table specs",
    "evidence-ref": "Evidence links",
    "code-ref": "Code links and blocks",
    "stale-evidence": "Evidence numbers that do not match",
    "conflict-marker": "Leftover conflict markers",
    "empty-file": "Empty files",
}

_CONFLICT_RE = re.compile(r"^(<{7}|={7}|>{7})( |$)")
_FENCE_OPEN_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})\s*([^\s`]*)")
_EVIDENCE_LINK_RE = re.compile(r"\[([^\]\n]*)\]\(\s*<?mlsweep:([^()\s<>]+)>?\s*\)")
_CLAIM_NUM_CLEAN = re.compile(r"[^0-9eE+.\-]")
# The last part of an evidence link that names a file, not a metric: a code link without `code/`.
_LOOKS_LIKE_FILE = re.compile(
    r"\.(py|pyi|sh|bash|toml|ya?ml|json|cfg|ini|txt|md|c|cc|cpp|h|hpp|cu|cuh|rs|go|js|ts|java|ipynb)(#L\d.*)?$",
    re.IGNORECASE)


def _check_conflict_markers(path: str, text: str, findings: list[Finding]) -> None:
    lines = text.split("\n")
    mask = _notes.code_mask(text)
    for i, line in enumerate(lines):
        if _CONFLICT_RE.match(line) and not mask[i]:
            findings.append(Finding("report", "conflict-marker", path, i + 1,
                                    "leftover merge conflict marker"))


def _fences(text: str) -> list[tuple[str, str, int]]:
    """``(language, body, 1-based line)`` for every fenced block in *text*."""
    out: list[tuple[str, str, int]] = []
    lines = text.split("\n")
    fence: str | None = None
    lang, start = "", 0
    body: list[str] = []
    for i, line in enumerate(lines):
        if fence is not None:
            stripped = line.strip()
            if stripped.startswith(fence) and stripped.replace(fence[0], "") == "":
                out.append((lang, "\n".join(body), start + 1))
                fence, body = None, []
            else:
                body.append(line)
            continue
        m = _FENCE_OPEN_RE.match(line)
        if m:
            fence, lang, start, body = m.group(1), m.group(2).lower(), i, []
    return out


def _claimed_number(text: str) -> float | None:
    """The number a link displays, read the same way the Notes tab reads it."""
    cleaned = _CLAIM_NUM_CLEAN.sub("", text.replace(",", ""))
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _check_blocks(
    path: str, text: str, known_experiments: set[str] | None,
    known_runs: dict[str, list[str]] | None,
    evidence: dict[str, dict[str, Any]] | None,
    findings: list[Finding],
) -> int:
    """Check data blocks and ``mlsweep:`` links (evidence and code).  Returns the block count.

    *evidence* holds what the manager resolved for each reference, code ones
    included; without it, code references are only checked for form.
    """
    count = 0
    for lang, body, line_no in _fences(text):
        if lang in ("mlsweep-code", "mlsweep-diff"):
            count += 1
            try:
                refs = _notes_code.block_refs(lang, body)
                spec_warnings = (_notes_code.code_block_ref(body)[1] if lang == "mlsweep-code"
                                 else _notes_code.diff_block_refs(body)[3])
            except _notes_blocks.SpecError as exc:
                findings.append(Finding("report", "block-spec", path, line_no, f"{lang}: {exc}"))
                continue
            for warning in spec_warnings:
                findings.append(Finding("report", "block-spec", path, line_no, f"{lang}: {warning}"))
            for ref in refs:
                _check_code_ref(path, line_no, ref, evidence, findings, f"{lang} block")
            if evidence is not None:
                _check_code_block_extent(path, line_no, lang, body, evidence, findings)
            continue
        kind = {"mlsweep-chart": "chart", "mlsweep-table": "table"}.get(lang)
        if kind is None:
            continue
        count += 1
        try:
            spec, warnings = _notes_blocks.normalize_spec(body, kind)
        except _notes_blocks.SpecError as exc:
            findings.append(Finding("report", "block-spec", path, line_no, f"{lang}: {exc}"))
            continue
        for warning in warnings:
            findings.append(Finding("report", "block-spec", path, line_no, f"{lang}: {warning}"))
        if known_experiments is None:
            continue
        for exp in spec.get("experiments") or []:
            if exp not in known_experiments and not _notes.GENERATED_EXP_ID_RE.fullmatch(exp):
                findings.append(Finding(
                    "report", "unknown-experiment", path, line_no,
                    f"`{exp}` in a {kind} block is not an experiment on this manager",
                    {"id": exp, "block": kind}))
        for ref in spec.get("runs") or []:
            exp, run = ref.split("/", 1) if "/" in ref else (None, ref)
            if (exp is not None and exp not in known_experiments
                    and not _notes.GENERATED_EXP_ID_RE.fullmatch(exp)):
                findings.append(Finding(
                    "report", "unknown-experiment", path, line_no,
                    f"`{exp}` in a {kind} block is not an experiment on this manager",
                    {"id": exp, "block": kind}))
            if known_runs is None:
                continue
            owners = known_runs.get(run)
            if not owners:
                findings.append(Finding("report", "unknown-run", path, line_no,
                                        f"`{ref}` names no run on this manager", {"id": ref}))
            elif exp is not None and exp not in owners:
                findings.append(Finding(
                    "report", "unknown-run", path, line_no,
                    f"run `{run}` belongs to {', '.join(owners)}, not `{exp}`",
                    {"id": run, "experiments": owners}))

    mask = _notes.code_mask(text)
    for i, raw_line in enumerate(text.split("\n")):
        if mask[i]:
            continue
        line = _notes.strip_inline_code(raw_line)
        for m in _EVIDENCE_LINK_RE.finditer(line):
            link_text, raw = m.group(1), m.group(2)
            if _notes_code.is_code_ref(raw):
                _check_code_ref(path, i + 1, raw, evidence, findings, "code link")
                continue
            if raw.startswith("trace/"):
                # A trace: the manager says whether the session exists.  There
                # is no number to check the note's text against.
                if evidence is not None:
                    hit = evidence.get(raw)
                    if hit is not None and not hit.get("ok"):
                        findings.append(Finding(
                            "report", "evidence-ref", path, i + 1,
                            f"mlsweep:{raw}: {hit.get('error', 'does not resolve')}", {"ref": raw}))
                continue
            if _LOOKS_LIKE_FILE.search(raw):
                findings.append(Finding(
                    "report", "code-ref", path, i + 1,
                    f"mlsweep:{raw} names a file, but a code link starts with `mlsweep:code/` "
                    f"(mlsweep:code/{raw})", {"ref": raw}))
                continue
            try:
                exp, run, _metric, _at = _notes_blocks.parse_evidence_ref(raw)
            except _notes_blocks.SpecError as exc:
                findings.append(Finding("report", "evidence-ref", path, i + 1,
                                        f"mlsweep:{raw}: {exc}", {"ref": raw}))
                continue
            if known_experiments is not None and exp not in known_experiments:
                findings.append(Finding(
                    "report", "unknown-experiment", path, i + 1,
                    f"evidence link names `{exp}`, which is not on this manager",
                    {"id": exp}))
                continue
            if evidence is not None:
                hit = evidence.get(raw)
                got = hit.get("value") if hit and hit.get("ok") else None
                claimed = _claimed_number(link_text)
                if (isinstance(got, (int, float)) and not isinstance(got, bool)
                        and claimed is not None):
                    tolerance = max(abs(float(got)) * 0.005, 0.005)
                    if abs(float(got) - claimed) > tolerance:
                        findings.append(Finding(
                            "report", "stale-evidence", path, i + 1,
                            f"note says {link_text.strip()} but this run logged {float(got):.6g}",
                            {"ref": raw, "claimed": claimed, "value": float(got),
                             "step": hit.get("step") if hit else None}))
            if known_runs is None:
                continue
            owners = known_runs.get(run)
            if not owners:
                findings.append(Finding("report", "evidence-ref", path, i + 1,
                                        f"evidence link names run `{run}`, which does not exist",
                                        {"ref": raw}))
            elif exp not in owners:
                findings.append(Finding(
                    "report", "evidence-ref", path, i + 1,
                    f"run `{run}` belongs to {', '.join(owners)}, not `{exp}`",
                    {"ref": raw, "experiments": owners}))
    return count


def _check_code_ref(path: str, line: int, raw: str, resolved: dict[str, dict[str, Any]] | None,
                    findings: list[Finding], what: str) -> None:
    try:
        _notes_code.parse_code_ref(raw)
    except _notes_code.CodeRefError as exc:
        findings.append(Finding("report", "code-ref", path, line, f"{what} mlsweep:{raw}: {exc}", {"ref": raw}))
        return
    hit = (resolved or {}).get(raw)
    if hit is not None and not hit.get("ok"):
        findings.append(Finding("report", "code-ref", path, line,
                                f"{what} mlsweep:{raw}: {hit.get('error', 'does not resolve')}", {"ref": raw}))
    elif hit is not None and hit.get("experiment") and (hit.get("versions") or 1) > 1:
        # An experiment appended to with other code: its ID means the latest version.
        findings.append(Finding(
            "report", "code-ref", path, line,
            f"{what} mlsweep:{raw}: the runs of {hit['experiment']} used {hit['versions']} code "
            f"versions and this shows the latest ({hit.get('short')}); to cite another, name its "
            "snapshot instead of the experiment (`mlsweep code EXP` lists them)",
            {"ref": raw, "versions": hit["versions"]}))


def _check_code_block_extent(path: str, line: int, lang: str, body: str,
                             resolved: dict[str, dict[str, Any]], findings: list[Finding]) -> None:
    """A code block too long to show whole, or a diff whose path is in neither side's code."""
    if lang == "mlsweep-code":
        ref, _ = _notes_code.code_block_ref(body)
        hit = resolved.get(ref.text())
        if not hit or not hit.get("ok") or not hit.get("lines"):
            return
        shown = (ref.end or hit["lines"]) - (ref.start or 1) + 1
        if shown > _notes_code.MAX_EXCERPT_LINES:
            findings.append(Finding(
                "report", "code-ref", path, line,
                f"mlsweep-code block shows {shown} lines of {ref.path}, but only the first "
                f"{_notes_code.MAX_EXCERPT_LINES} fit; narrow it with `lines:`", {"ref": ref.text()}))
        return
    sides = [resolved.get(r) for r in _notes_code.diff_path_refs(body)]
    if sides and all(s is not None and not s.get("ok") and "no file" in str(s.get("error")) for s in sides):
        _a, _b, prefix, _ = _notes_code.diff_block_refs(body)
        findings.append(Finding("report", "code-ref", path, line,
                                f"mlsweep-diff block: `{prefix}` is in neither experiment's code, "
                                "so the diff is empty", {"path": prefix}))


def evidence_refs(index: _notes.Index, root: str | Path) -> list[str]:
    """Every ``mlsweep:`` reference in the notes, in order, without duplicates.

    Callers resolve these (the manager's metrics, or ``/api/notes/evidence``)
    and pass the result back so lint can check the numbers themselves.
    """
    protected = read_config(Path(root).resolve())["protected"]
    refs: list[str] = []
    for path, text in index.texts.items():
        if is_protected(path, protected):
            continue
        mask = _notes.code_mask(text)
        for i, raw_line in enumerate(text.split("\n")):
            if mask[i]:
                continue
            for m in _EVIDENCE_LINK_RE.finditer(_notes.strip_inline_code(raw_line)):
                ref = m.group(2)
                if ref not in refs:
                    refs.append(ref)
        for lang, body, _line in _fences(text):
            if lang not in ("mlsweep-code", "mlsweep-diff"):
                continue
            try:
                block = _notes_code.block_refs(lang, body)
                if lang == "mlsweep-diff":
                    block += _notes_code.diff_path_refs(body)
            except _notes_blocks.SpecError:
                continue
            refs.extend(r for r in block if r not in refs)
    return refs


def lint(root: str | Path, *, known_experiments: set[str] | None = None,
         known_runs: dict[str, list[str]] | None = None,
         evidence: dict[str, dict[str, Any]] | None = None,
         index: _notes.Index | None = None) -> dict[str, Any]:
    """Run tiers 1 and 2 over *root*.  Returns ``{"findings": [...], "stats": {...}}``.

    *known_experiments* enables the unknown-ID check; *known_runs* (run ID ->
    experiments) also checks run IDs and chart/table blocks.  *evidence* maps
    resolved references to values, enabling the stale-number check.
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

    # ── chart/table blocks and evidence links ──
    blocks = 0
    for path, text in index.texts.items():
        if not skip(path):
            blocks += _check_blocks(path, text, known_experiments, known_runs,
                                    evidence, findings)

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
             "data_blocks": blocks,
             "experiments_checked": known_experiments is not None,
             "runs_checked": known_runs is not None,
             "evidence_checked": evidence is not None,
             "evidence_refs": len(evidence or {})}
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

Chart/table blocks and evidence links (`[0.42](mlsweep:EXP/RUN/metric@step)`)
were checked for unknown experiments and runs, and each evidence number was
compared with the logged metric. Code links (`mlsweep:code/EXP/path#L40-60`) and
code/diff blocks were checked to resolve to a file and lines in the code that
experiment ran; read the code itself with `mlsweep code EXP PATH`. A stale-evidence finding means they disagree;
still judge whether the number is the right one to cite.
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
               + (f", {s['data_blocks']} data blocks" if s.get("data_blocks") else "")
               + ("" if s.get("experiments_checked") else "; experiment and run IDs not checked (no manager)"))
    return "\n".join(out)


def review_prompt(root: str | Path, result: dict[str, Any], date: str) -> str:
    return REVIEW_PROMPT.format(root=root, date=date, report=format_report(result))
