"""Code in notes: links to a file as an experiment ran it, code excerpts, and diffs.

A link names a file at the version an experiment ran::

    [the warmup](mlsweep:code/EXP/yoco/train.py#L40-60)

``EXP`` is an experiment ID, or a snapshot ID (``snap:3fa9c1e2...``, a unique
prefix of 8 or more hex digits is enough).  The path and the ``#L`` lines are
optional: ``mlsweep:code/EXP`` links to the whole snapshot.

A ``mlsweep-code`` block shows the lines themselves, and a ``mlsweep-diff``
block what changed between two experiments' code::

    ```mlsweep-code
    experiment: EXP
    path: yoco/train.py
    lines: 40-60
    ```

    ```mlsweep-diff
    from: EXP_A
    to: EXP_B
    path: yoco/
    ```

The manager resolves all of these against its code snapshots (see
``mlsweep._snapshot``), so the Notes tab, the code page and lint agree.
"""

from __future__ import annotations

import asyncio
import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote
from typing import Any, Sequence

from aiosqlite import Connection

from mlsweep._manager_db import (
    experiment_code_versions,
    get_experiment,
    get_snapshot_manifest,
    list_code_submissions,
    resolve_snapshot_prefix,
)
from mlsweep._notes_blocks import SpecError, parse_spec
from mlsweep._snapshot import (
    Manifest,
    ObjectStore,
    _check_rel_path,
    diff_manifests,
    is_snapshot_id,
    looks_binary,
    parse_manifest,
    short_id,
    store_root,
)

CODE_PREFIX = "code/"
MAX_TEXT_BYTES = 2 << 20        # larger files are not read as text
MAX_EXCERPT_LINES = 400
MAX_DIFF_FILES = 60
MAX_DIFF_LINES = 3000

CODE_KEYS = {"experiment", "snapshot", "code", "path", "lines", "caption", "title"}
DIFF_KEYS = {"from", "to", "path", "context", "caption", "title"}

_EXPERIMENT_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")
_SNAPSHOT_RE = re.compile(r"^snap:[0-9a-f]{8,64}$")
_LINES_RE = re.compile(r"^L?(\d{1,7})(?:\s*-\s*L?(\d{1,7}))?$", re.IGNORECASE)


class CodeRefError(ValueError):
    """A code reference that cannot be understood."""


@dataclass(frozen=True)
class CodeRef:
    target: str              # experiment ID or snapshot ID / prefix
    path: str = ""           # "" = the whole snapshot; a directory or a file
    start: int | None = None
    end: int | None = None

    def text(self) -> str:
        """The canonical ``code/...`` form."""
        out = CODE_PREFIX + self.target + (f"/{self.path}" if self.path else "")
        if self.start is not None:
            out += f"#L{self.start}" + (f"-{self.end}" if self.end is not None and self.end != self.start else "")
        return out


def is_code_ref(raw: str) -> bool:
    text = raw.strip()
    if text.startswith("mlsweep:"):
        text = text[len("mlsweep:"):]
    return text.startswith(CODE_PREFIX)


def _check_target(target: str) -> str:
    if not (_EXPERIMENT_RE.match(target) or _SNAPSHOT_RE.match(target)):
        raise CodeRefError(f"`{target}` is not an experiment ID or a snapshot ID")
    return target


def parse_lines(text: str) -> tuple[int | None, int | None]:
    """``40``, ``40-60``, ``L40-L60`` as (start, end); empty means no lines."""
    text = str(text).strip()
    if not text:
        return None, None
    m = _LINES_RE.match(text)
    if not m:
        raise CodeRefError(f"bad line range `{text}` (write 40 or 40-60)")
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else start
    if start < 1 or end < start:
        raise CodeRefError(f"bad line range `{text}`")
    return start, end


def parse_code_ref(raw: str) -> CodeRef:
    """``[mlsweep:]code/TARGET[/PATH][#Lstart[-end]]`` as a :class:`CodeRef`."""
    text = raw.strip()
    if text.startswith("mlsweep:"):
        text = text[len("mlsweep:"):]
    if not text.startswith(CODE_PREFIX):
        raise CodeRefError("a code link starts with code/")
    text = text[len(CODE_PREFIX):]
    frag = ""
    if "#" in text:
        text, frag = text.split("#", 1)
    target, _, path = text.partition("/")
    _check_target(target)
    # A markdown link can't hold a space, so a path with one is written %20.
    path = unquote(path).strip("/")
    if path:
        try:
            _check_rel_path(path)
        except ValueError as exc:
            raise CodeRefError(str(exc)) from None
    start, end = parse_lines(frag) if frag else (None, None)
    if start is not None and not path:
        raise CodeRefError("line numbers need a file")
    return CodeRef(target, path, start, end)


# ── resolving ─────────────────────────────────────────────────────────────────


@dataclass
class Resolved:
    snapshot: str
    manifest: Manifest
    experiment: str | None
    versions: int            # how many code versions the experiment's runs used


class _Snapshots:
    """Resolves targets to snapshots and reads files, caching within one request."""

    def __init__(self, db: Connection, mlsweep_dir: str | Path) -> None:
        self.db = db
        self.store = ObjectStore(store_root(mlsweep_dir))
        self._targets: dict[str, Resolved | str] = {}
        self._texts: dict[str, tuple[str | None, int]] = {}

    async def _manifest(self, snapshot: str) -> Manifest | None:
        raw = await get_snapshot_manifest(self.db, snapshot)
        return None if raw is None else await asyncio.to_thread(parse_manifest, raw)

    async def _resolve(self, target: str) -> Resolved | str:
        if is_snapshot_id(target):
            manifest = await self._manifest(target)
            if manifest is None:
                matches = await resolve_snapshot_prefix(self.db, target)
                if len(matches) > 1:
                    return f"`{target}` matches more than one snapshot"
                if not matches:
                    return f"no snapshot `{target}` on this manager"
                target, raw = matches[0]
                manifest = await asyncio.to_thread(parse_manifest, raw)
            return Resolved(target, manifest, None, 1)
        if await get_experiment(self.db, target) is None:
            return f"no experiment `{target}` on this manager"
        versions = [v["artifact_id"] for v in await experiment_code_versions(self.db, target)]
        submitted = [s["snapshot"] for s in await list_code_submissions(self.db, target)
                     if s.get("snapshot") in versions]
        if not versions:
            return f"no run of `{target}` has code recorded"
        chosen = submitted[-1] if submitted else versions[-1]
        if not is_snapshot_id(chosen):
            return f"`{target}` ran from a code archive made before snapshots, which can't be shown"
        manifest = await self._manifest(chosen)
        if manifest is None:
            return f"the code of `{target}` is no longer stored"
        return Resolved(chosen, manifest, target, len(versions))

    async def resolve(self, target: str) -> Resolved | str:
        """The snapshot *target* names, or why it can't be found."""
        if target not in self._targets:
            self._targets[target] = await self._resolve(target)
        return self._targets[target]

    async def text(self, h: str, size: int) -> tuple[str | None, int]:
        """(text or None if binary/too large, line count)."""
        if h not in self._texts:
            if size > MAX_TEXT_BYTES:
                self._texts[h] = (None, 0)
            else:
                data = await asyncio.to_thread(self.store.read_bytes, h)
                if looks_binary(data):
                    self._texts[h] = (None, 0)
                else:
                    text = data.decode("utf-8", errors="replace")
                    self._texts[h] = (text, len(text.splitlines()))
        return self._texts[h]


def _dir_exists(manifest: Manifest, path: str) -> bool:
    pre = path + "/"
    return any(f.path.startswith(pre) for f in manifest.files) or path in manifest.dirs


async def _resolve_one(snaps: _Snapshots, raw: str) -> dict[str, Any]:
    try:
        ref = parse_code_ref(raw)
    except CodeRefError as exc:
        return {"ok": False, "ref": raw, "error": str(exc)}
    out: dict[str, Any] = {"ref": raw, "target": ref.target, "path": ref.path,
                           "start": ref.start, "end": ref.end}
    hit = await snaps.resolve(ref.target)
    if isinstance(hit, str):
        return {**out, "ok": False, "error": hit}
    out.update(snapshot=hit.snapshot, short=short_id(hit.snapshot), experiment=hit.experiment,
               versions=hit.versions, canonical=ref.text())
    if not ref.path:
        return {**out, "ok": True, "kind": "tree", "files": len(hit.manifest.files)}
    entry = hit.manifest.by_path().get(ref.path)
    if entry is None:
        if ref.start is None and _dir_exists(hit.manifest, ref.path):
            return {**out, "ok": True, "kind": "dir"}
        return {**out, "ok": False, "error": f"no file `{ref.path}` in {short_id(hit.snapshot)}"}
    out.update(kind="file", size=entry.size)
    text, n_lines = await snaps.text(entry.hash, entry.size)
    out.update(binary=text is None, lines=n_lines if text is not None else None)
    if ref.start is not None:
        if text is None:
            return {**out, "ok": False, "error": f"`{ref.path}` is binary or too large to cite by line"}
        if ref.end is not None and ref.end > n_lines:
            return {**out, "ok": False, "error": f"`{ref.path}` has {n_lines} lines, not {ref.end}"}
    return {**out, "ok": True}


async def resolve_code_refs(db: Connection, mlsweep_dir: str | Path, refs: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Resolve ``code/...`` references in one batch."""
    snaps = _Snapshots(db, mlsweep_dir)
    return {raw: await _resolve_one(snaps, raw) for raw in refs}


# ── blocks ────────────────────────────────────────────────────────────────────


def _block_spec(raw: str | dict[str, Any], keys: set[str], kind: str) -> tuple[dict[str, Any], list[str]]:
    spec = parse_spec(raw) if isinstance(raw, str) else dict(raw)
    unknown = sorted(set(spec) - keys)
    warnings = [f"unknown key `{k}` (a {kind} block takes {', '.join(sorted(keys))})" for k in unknown]
    return spec, warnings


def _target_of(spec: dict[str, Any], *names: str) -> str:
    given = [str(spec[n]).strip() for n in names if spec.get(n) not in (None, "")]
    if len(given) != 1:
        raise SpecError(f"name exactly one of {', '.join(f'`{n}`' for n in names)}")
    try:
        return _check_target(given[0])
    except CodeRefError as exc:
        raise SpecError(str(exc)) from None


def code_block_ref(raw: str | dict[str, Any]) -> tuple[CodeRef, list[str]]:
    """The reference a ``mlsweep-code`` block makes, and spec warnings.  Pure, for lint."""
    spec, warnings = _block_spec(raw, CODE_KEYS, "code")
    target = _target_of(spec, "experiment", "snapshot", "code")
    path = str(spec.get("path") or "").strip().strip("/")
    if not path:
        raise SpecError("`path` is required: the file to show")
    try:
        _check_rel_path(path)
        start, end = parse_lines(str(spec.get("lines") or ""))
    except (ValueError, CodeRefError) as exc:
        raise SpecError(str(exc)) from None
    return CodeRef(target, path, start, end), warnings


def diff_block_refs(raw: str | dict[str, Any]) -> tuple[CodeRef, CodeRef, str, list[str]]:
    """(from, to, path prefix, warnings) of a ``mlsweep-diff`` block.  Pure, for lint."""
    spec, warnings = _block_spec(raw, DIFF_KEYS, "diff")
    a = _target_of(spec, "from")
    b = _target_of(spec, "to")
    prefix = str(spec.get("path") or "").strip().strip("/")
    if prefix:
        try:
            _check_rel_path(prefix)
        except ValueError as exc:
            raise SpecError(str(exc)) from None
    return CodeRef(a), CodeRef(b), prefix, warnings


def _describe(hit: Resolved) -> dict[str, Any]:
    return {"snapshot": hit.snapshot, "short": short_id(hit.snapshot),
            "experiment": hit.experiment, "versions": hit.versions}


async def build_code_block(db: Connection, mlsweep_dir: str | Path, raw: str | dict[str, Any]) -> dict[str, Any]:
    """The lines a ``mlsweep-code`` block shows."""
    ref, warnings = code_block_ref(raw)
    spec = parse_spec(raw) if isinstance(raw, str) else dict(raw)
    snaps = _Snapshots(db, mlsweep_dir)
    hit = await snaps.resolve(ref.target)
    if isinstance(hit, str):
        raise SpecError(hit)
    entry = hit.manifest.by_path().get(ref.path)
    if entry is None:
        raise SpecError(f"no file `{ref.path}` in {short_id(hit.snapshot)}")
    text, n_lines = await snaps.text(entry.hash, entry.size)
    if text is None:
        raise SpecError(f"`{ref.path}` is binary or larger than {MAX_TEXT_BYTES >> 20} MB")
    lines = text.splitlines()
    start = ref.start or 1
    end = min(ref.end or n_lines, n_lines)
    if start > n_lines:
        raise SpecError(f"`{ref.path}` has {n_lines} lines")
    truncated = end - start + 1 > MAX_EXCERPT_LINES
    if truncated:
        end = start + MAX_EXCERPT_LINES - 1
        warnings.append(f"showing the first {MAX_EXCERPT_LINES} lines of the range")
    lines_part = f" L{start}–{end}" if ref.start is not None else ""
    title = str(spec.get("title") or f"{ref.path}{lines_part} · {hit.experiment or short_id(hit.snapshot)}")
    link = CodeRef(ref.target, ref.path, ref.start, ref.end if ref.end != ref.start else None)
    return {
        "title": title, **_describe(hit), "path": ref.path, "start": start, "end": end,
        "total_lines": n_lines, "lines": lines[start - 1:end], "truncated": truncated,
        "ref": link.text(), "warnings": warnings,
    }


async def build_diff_block(db: Connection, mlsweep_dir: str | Path, raw: str | dict[str, Any]) -> dict[str, Any]:
    """What changed between two experiments' code, as unified diffs per file."""
    a_ref, b_ref, prefix, warnings = diff_block_refs(raw)
    spec = parse_spec(raw) if isinstance(raw, str) else dict(raw)
    try:
        context = int(spec.get("context", 3))
    except (TypeError, ValueError):
        raise SpecError("`context` must be a whole number") from None
    if not 0 <= context <= 50:
        raise SpecError("`context` must be 0 to 50")
    snaps = _Snapshots(db, mlsweep_dir)
    a, b = await snaps.resolve(a_ref.target), await snaps.resolve(b_ref.target)
    for target, hit in ((a_ref.target, a), (b_ref.target, b)):
        if isinstance(hit, str):
            raise SpecError(hit)
    assert not isinstance(a, str) and not isinstance(b, str)
    d = diff_manifests(a.manifest, b.manifest, prefix or None)
    pairs = ([(n.path, o, n) for o, n in d.changed] + [(f.path, None, f) for f in d.added]
             + [(f.path, f, None) for f in d.removed])
    pairs.sort(key=lambda t: t[0])
    files: list[dict[str, Any]] = []
    budget = MAX_DIFF_LINES
    truncated = False
    for path, old, new in pairs:
        status = "A" if old is None else "D" if new is None else "M"
        item: dict[str, Any] = {"path": path, "status": status,
                                "old_size": old.size if old else None, "new_size": new.size if new else None}
        if len(files) >= MAX_DIFF_FILES or budget <= 0:
            truncated = True
            break
        if old is not None and new is not None and old.hash == new.hash:
            item.update(status="mode", old_mode=oct(old.mode), new_mode=oct(new.mode), diff=[])
            files.append(item)
            continue
        before, _ = await snaps.text(old.hash, old.size) if old else ("", 0)
        after, _ = await snaps.text(new.hash, new.size) if new else ("", 0)
        if before is None or after is None:
            item.update(binary=True, diff=[])
            files.append(item)
            continue
        lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=context))[2:]
        item["added"] = sum(1 for ln in lines if ln.startswith("+"))
        item["removed"] = sum(1 for ln in lines if ln.startswith("-"))
        if len(lines) > budget:
            lines = lines[:budget]
            item["truncated"] = True
            truncated = True
        budget -= len(lines)
        item["diff"] = lines
        files.append(item)
    if truncated:
        warnings.append(f"showing {len(files)} of {len(pairs)} files, at most {MAX_DIFF_LINES} diff lines")
    name_a = a.experiment or short_id(a.snapshot)
    name_b = b.experiment or short_id(b.snapshot)
    title = str(spec.get("title") or f"{name_a} → {name_b}" + (f" · {prefix}/" if prefix else ""))
    return {
        "title": title, "from": _describe(a), "to": _describe(b), "path": prefix,
        "same": a.snapshot == b.snapshot,
        "counts": {"changed": len(d.changed), "added": len(d.added), "removed": len(d.removed)},
        "changes": [{"path": p, "status": "A" if o is None else "D" if n is None else "M"} for p, o, n in pairs],
        "files": files, "truncated": truncated, "warnings": warnings,
    }


def diff_path_refs(body: str) -> list[str]:
    """For a diff block with a ``path``, that path on each side (for lint to check it exists somewhere)."""
    a, b, prefix, _ = diff_block_refs(body)
    return [CodeRef(a.target, prefix).text(), CodeRef(b.target, prefix).text()] if prefix else []


def block_refs(lang: str, body: str) -> list[str]:
    """The ``code/...`` references a code or diff block makes (for lint).  Raises SpecError."""
    if lang == "mlsweep-code":
        ref, _ = code_block_ref(body)
        return [ref.text()]
    # Only the two sides: a diff's path may exist in just one of them.
    a, b, _prefix, _ = diff_block_refs(body)
    return [a.text(), b.text()]
