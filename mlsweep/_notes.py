"""Reading a notes folder: safe paths, the file list, links, mentions, search.

The notes folder is a plain directory (a git clone, see ``_notes_git``).  This
module never runs git; it only reads files.  Everything here is synchronous
and meant to run in a worker thread.

Links follow Obsidian's conventions:

* ``[[Name]]``, ``[[Name|shown text]]``, ``[[Name#Heading]]`` link to a note by
  file name (``.md`` optional), ``[[dir/Name]]`` by path;
* ``![[plot.png]]`` embeds a file;
* ``[text](relative/path.md)`` is an ordinary relative link.

Experiment IDs need no syntax: any word that is a known experiment ID is a
mention of it.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote

from mlsweep._shared import _resolve_safe_subpath

MAX_TEXT_BYTES = 4 * 1024 * 1024        # larger files are not parsed or searched
MAX_WRITE_BYTES = 8 * 1024 * 1024        # largest text file the UI may save
MAX_UPLOAD_BYTES = 100 * 1024 * 1024     # largest upload

MD_EXT = {".md", ".markdown"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".bmp", ".ico", ".svg"}
PDF_EXT = {".pdf"}
TEXT_EXT = {
    ".txt", ".log", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".csv", ".tsv", ".py", ".sh", ".bash", ".zsh", ".js", ".ts", ".css", ".html", ".htm",
    ".xml", ".tex", ".bib", ".rst", ".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".rs", ".go",
    ".java", ".lua", ".sql", ".r", ".jl", ".ipynb", ".diff", ".patch", ".env", ".mk", ".cmake",
}
TEXT_NAMES = {"Makefile", "Dockerfile", "LICENSE", "README", "TODO", ".gitignore", ".gitattributes"}

HIDDEN_DIRS = {".git", ".obsidian", ".trash", "__pycache__", ".ipynb_checkpoints", "node_modules"}


def kind_of(path: str) -> str:
    """``md``, ``image``, ``pdf``, ``text`` or ``binary`` by file name."""
    p = PurePosixPath(path)
    ext = p.suffix.lower()
    if ext in MD_EXT:
        return "md"
    if ext in IMAGE_EXT:
        return "image"
    if ext in PDF_EXT:
        return "pdf"
    if ext in TEXT_EXT or p.name in TEXT_NAMES:
        return "text"
    return "binary"


# ── paths ──────────────────────────────────────────────────────────────────────


class PathError(ValueError):
    """A path from a request is not allowed."""


_BAD_CHARS = re.compile(r"[\x00-\x1f\x7f\\]")


def clean_rel(path: str, *, allow_root: bool = False) -> str:
    """Validate a repo-relative POSIX path from a client and normalize it.

    Refuses absolute paths, ``..``, empty components, control characters,
    backslashes, and anything inside ``.git``.
    """
    if not isinstance(path, str):
        raise PathError("path must be a string")
    path = unicodedata.normalize("NFC", path.strip())
    if path in ("", ".", "/"):
        if allow_root:
            return ""
        raise PathError("path is empty")
    if _BAD_CHARS.search(path):
        raise PathError("path contains control characters or backslashes")
    if path.startswith("/"):
        raise PathError("path must be relative")
    parts = path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise PathError("path has empty, '.' or '..' components")
    if any(p.lower() == ".git" for p in parts):
        raise PathError("paths inside .git are not allowed")
    if len(path) > 1024 or any(len(p.encode()) > 255 for p in parts):
        raise PathError("path is too long")
    return "/".join(parts)


def resolve(root: Path, rel: str, *, must_exist: bool = False) -> Path:
    """Absolute path for *rel* inside *root*, refusing symlinks that leave it.

    The parent chain is resolved; the final component may be a symlink only
    if it points inside *root*.
    """
    rel = clean_rel(rel, allow_root=True)
    root = root.resolve()
    try:
        _resolve_safe_subpath(root, rel)
    except ValueError:
        raise PathError("path leaves the notes folder") from None
    target = root / rel if rel else root
    if must_exist and not target.exists():
        raise FileNotFoundError(rel)
    return target


# ── listing ────────────────────────────────────────────────────────────────────


@dataclass
class Entry:
    path: str
    size: int
    mtime: float
    kind: str

    @classmethod
    def of(cls, rel: str, st: os.stat_result) -> Entry:
        return cls(rel, st.st_size, st.st_mtime, kind_of(rel))

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "mtime": self.mtime, "kind": self.kind}


def walk(root: Path) -> tuple[list[Entry], list[str]]:
    """All files and directories under *root* (not following symlinked dirs).

    Hidden tool directories (``.git``, ``.obsidian``, ...) are skipped; other
    dotfiles are listed.
    """
    files: list[Entry] = []
    dirs: list[str] = []
    root = root.resolve()
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = list(os.scandir(d))
        except OSError:
            continue
        for e in it:
            if e.name in HIDDEN_DIRS:
                continue
            try:
                st = e.stat(follow_symlinks=False)
            except OSError:
                continue
            rel = os.path.relpath(e.path, root).replace(os.sep, "/")
            if stat.S_ISDIR(st.st_mode):
                dirs.append(rel)
                stack.append(Path(e.path))
            elif stat.S_ISREG(st.st_mode):
                files.append(Entry.of(rel, st))
            elif stat.S_ISLNK(st.st_mode):
                try:
                    real = Path(os.path.realpath(e.path))
                    if real.is_file() and real.is_relative_to(root):
                        files.append(Entry.of(rel, real.stat()))
                except OSError:
                    pass
    files.sort(key=lambda f: f.path.lower())
    dirs.sort(key=str.lower)
    return files, dirs


def read_text(path: Path, limit: int = MAX_TEXT_BYTES) -> str | None:
    """File contents as text, or None if too large or not valid UTF-8 text."""
    try:
        if path.stat().st_size > limit:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def atomic_write(path: Path, data: bytes) -> None:
    """Replace *path* in one step, so a sync never commits a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".mlsweep-tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        if path.exists():
            shutil.copymode(path, tmp)
        else:
            os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── markdown scanning ──────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^(\s{0,3})(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"(`+)(.+?)\1")
WIKILINK_RE = re.compile(r"(!?)\[\[([^\[\]\n|#^]*)(?:#([^\[\]\n|]*))?(?:\|([^\[\]\n]*))?\]\]")
MDLINK_RE = re.compile(r"(!?)\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\(\s*<?([^()\s<>]+(?:\([^()\s]*\))?[^()\s<>]*)>?(?:\s+\"[^\"]*\")?\s*\)")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")
ID_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{2,127}")
# Experiment IDs mlsweep generates: NAME_YYYYMMDD_HHMM_xxxx
GENERATED_EXP_ID_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_\-]*_20\d{6}_\d{4}_[0-9a-f]{4}\b")


def code_mask(text: str) -> list[bool]:
    """Per line: True if the line is inside (or delimits) a fenced code block."""
    mask = []
    fence: str | None = None
    for line in text.split("\n"):
        m = _FENCE_RE.match(line)
        if fence is None:
            if m:
                fence = m.group(2)
                mask.append(True)
            else:
                mask.append(False)
        else:
            mask.append(True)
            if m and m.group(2).startswith(fence[0]) and len(m.group(2)) >= len(fence) \
                    and line.strip() == m.group(2):
                fence = None
    return mask


def strip_inline_code(line: str) -> str:
    """Blank out inline code spans, keeping column positions."""
    return _INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), line)


@dataclass
class Link:
    target: str          # as written (wikilink name or URL-decoded relative path)
    line: int            # 1-based
    embed: bool
    wiki: bool
    heading: str = ""
    resolved: str | None = None
    ambiguous: list[str] = field(default_factory=list)
    raw: str = ""        # the matched text


def parse_href(href: str) -> tuple[str, str] | None:
    """A markdown link's file path (decoded) and its ``?query#fragment``.

    None if it does not name a file in the folder: an external URL, a
    protocol-relative ``//host/x``, or only a query or fragment.
    """
    href = href.strip()
    if not href or href.startswith(("#", "//")) or _SCHEME_RE.match(href):
        return None
    raw = href.split("#", 1)[0].split("?", 1)[0]
    path = unquote(raw)
    return (path, href[len(raw):]) if path else None


def with_href_path(m: re.Match[str], path: str) -> str:
    """The ``MDLINK_RE`` match *m* pointing at *path*.  Only the href changes: the
    link text may spell the same name, and the query and fragment stay."""
    parsed = parse_href(m.group(3))
    rest = parsed[1] if parsed else ""
    start, end = m.start(3) - m.start(), m.end(3) - m.start()
    return m.group(0)[:start] + path + rest + m.group(0)[end:]


def extract_links(text: str) -> list[Link]:
    """Wikilinks and relative markdown links, outside code."""
    links: list[Link] = []
    mask = code_mask(text)
    for i, line in enumerate(text.split("\n")):
        if mask[i]:
            continue
        line = strip_inline_code(line)
        for m in WIKILINK_RE.finditer(line):
            target = m.group(2).strip()
            if not target:
                continue
            links.append(Link(target=target, line=i + 1, embed=bool(m.group(1)), wiki=True,
                              heading=(m.group(3) or "").strip(), raw=m.group(0)))
        for m in MDLINK_RE.finditer(line):
            parsed = parse_href(m.group(3))
            if parsed is None:
                continue
            path, rest = parsed
            links.append(Link(target=path, line=i + 1, embed=bool(m.group(1)), wiki=False,
                              heading=rest.split("#", 1)[1] if "#" in rest else "", raw=m.group(0)))
    return links


class Resolver:
    """Resolve link targets to files the way Obsidian does."""

    def __init__(self, paths: Iterable[str]):
        ordered = sorted(paths)
        self.lower = {p.lower(): p for p in ordered}
        self.by_name: dict[str, list[str]] = {}
        for p in ordered:
            name = PurePosixPath(p).name.lower()
            self.by_name.setdefault(name, []).append(p)
            if kind_of(p) == "md":
                self.by_name.setdefault(PurePosixPath(p).stem.lower(), []).append(p)

    @staticmethod
    def _closest(src: str, candidates: list[str]) -> str:
        src_dir = str(PurePosixPath(src).parent)
        same = [c for c in candidates if str(PurePosixPath(c).parent) == src_dir]
        if same:
            return same[0]
        return sorted(candidates, key=lambda c: (c.count("/"), len(c), c))[0]

    def resolve(self, src: str, link: Link) -> None:
        """Fill ``link.resolved`` (and ``link.ambiguous`` for a name with several files)."""
        target = link.target.strip().lstrip("/") if link.wiki else link.target
        if link.wiki:
            candidates = [target, target + ".md"] if PurePosixPath(target).suffix.lower() not in MD_EXT else [target]
            if "/" in target:
                for c in candidates:
                    hit = self.lower.get(c.lower())
                    if hit:
                        link.resolved = hit
                        return
                # A path that does not exist: fall back to its file name.
                name = PurePosixPath(target).name
            else:
                name = target
            hits = self.by_name.get(name.lower())
            if hits:
                link.resolved = self._closest(src, hits)
                if len(hits) > 1:
                    link.ambiguous = hits
            return
        # Relative markdown link.
        base = PurePosixPath(src).parent
        joined = os.path.normpath(str(base / target)).replace(os.sep, "/")
        if joined.startswith("../") or joined == "..":
            return
        hit = self.lower.get(joined.lower())
        if hit is None and PurePosixPath(joined).suffix == "":
            hit = self.lower.get((joined + ".md").lower())
        if hit:
            link.resolved = hit
            return
        # Root-relative (Obsidian "absolute path in vault") links.
        hit = self.lower.get(target.lstrip("/").lower())
        if hit:
            link.resolved = hit


def find_mentions(text: str, known: set[str]) -> dict[str, list[int]]:
    """Known experiment IDs that appear in *text* -> 1-based line numbers."""
    found: dict[str, list[int]] = {}
    if not known:
        return found
    for i, line in enumerate(text.split("\n")):
        for m in ID_TOKEN_RE.finditer(line):
            tok = m.group(0)
            for cand in (tok, tok.rstrip(".-")):
                if cand in known:
                    lines = found.setdefault(cand, [])
                    if not lines or lines[-1] != i + 1:
                        lines.append(i + 1)
                    break
    return found


def run_mentions(texts: dict[str, str], known_runs: set[str]) -> dict[str, dict[str, list[int]]]:
    """Run ID -> {note path: 1-based lines} for the known run IDs cited in *texts*.

    Run IDs have no syntax of their own either; like experiment IDs they are
    recognized as words the manager knows from *texts* alone.
    """
    found: dict[str, dict[str, list[int]]] = {}
    if not known_runs:
        return found
    for src, text in texts.items():
        for run_id, lines in find_mentions(text, known_runs).items():
            found.setdefault(run_id, {})[src] = lines
    return found


def _line_context(lines: list[str], line: int) -> str:
    """The 1-based *line* of *lines*, trimmed for showing beside a link."""
    return lines[line - 1].strip()[:240] if 0 < line <= len(lines) else ""


@dataclass
class Index:
    """Links between files, and experiment mentions, for the whole folder."""

    files: list[Entry]
    links: dict[str, list[Link]]                 # src -> links
    mentions: dict[str, dict[str, list[int]]]    # src -> exp id -> lines
    texts: dict[str, str]                        # md path -> contents (for search/lint)

    def as_dict(self) -> dict[str, Any]:
        out_links: dict[str, list[dict[str, Any]]] = {}
        for src, ls in self.links.items():
            lines = self.texts.get(src, "").split("\n")
            out_links[src] = [{
                "target": link.target, "line": link.line, "embed": link.embed,
                "resolved": link.resolved, "heading": link.heading,
                "context": _line_context(lines, link.line),
                **({"ambiguous": link.ambiguous} if link.ambiguous else {}),
            } for link in ls]
        return {"links": out_links, "mentions": self.exp_mentions()}

    def exp_mentions(self) -> dict[str, list[str]]:
        """Experiment id -> sorted notes that mention it."""
        out: dict[str, list[str]] = {}
        for src, m in self.mentions.items():
            for exp in m:
                out.setdefault(exp, []).append(src)
        return {k: sorted(v) for k, v in out.items()}

    def cited(self) -> list[str]:
        """Every experiment ID mentioned anywhere in the folder."""
        return sorted({exp for m in self.mentions.values() for exp in m})


@dataclass
class _Parsed:
    """One file's parse, reusable until its size or mtime changes."""

    stamp: tuple[int, float]
    text: str | None                       # markdown only
    links: list[Link]                      # not yet resolved
    mentions: dict[str, list[int]]
    resolved: list[Link] | None = None     # copies of links, for the cache's paths


@dataclass
class ParseCache:
    """Parses kept between ``build_index`` calls, so one edit re-reads one file."""

    known: set[str] = field(default_factory=set)
    files: dict[str, _Parsed] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)
    resolver: Resolver | None = None
    reads: list[str] = field(default_factory=list)  # files the last build_index read from disk


def _update_mentions(p: _Parsed, path: Path, known: set[str], added: set[str], reads: list[str]) -> _Parsed:
    """*p* for a file that has not changed, now that the known IDs have.

    IDs that are gone are dropped.  Only a file containing a new ID is tokenized
    again, because rescanning the whole folder for each new experiment is slow.
    """
    mentions = {exp: lines for exp, lines in p.mentions.items() if exp in known}
    text = p.text
    if text is None and added:
        text = read_text(path)
        reads.append(path.name)
    new_here = {exp for exp in added if text and exp in text}
    if new_here and text:
        mentions.update(find_mentions(text, new_here))
    return replace(p, mentions=mentions)


def build_index(root: Path, known_experiments: set[str] | None = None,
                cache: ParseCache | None = None) -> Index:
    """Walk *root*, parse every markdown file, resolve links and mentions.

    Links are resolved afresh when the set of files changes, since any new or
    moved file can change what a name means.  Mentions depend on the known IDs,
    so they are brought up to date when those change.
    """
    known = known_experiments or set()
    files, _ = walk(root)
    paths = [f.path for f in files]
    resolver = cache.resolver if cache is not None and cache.paths == paths else None
    same_paths = resolver is not None
    if resolver is None:
        resolver = Resolver(paths)
    previous = cache.files if cache is not None else {}
    known_changed = cache is not None and cache.known != known
    added = known - cache.known if cache is not None else set()
    # "exp." and "exp" compete for the token "exp.", so with such an ID only a
    # full scan says which one each line mentions.
    if known_changed and cache is not None and any(e.endswith((".", "-")) for e in known | cache.known):
        previous = {}
    parsed: dict[str, _Parsed] = {}
    links: dict[str, list[Link]] = {}
    mentions: dict[str, dict[str, list[int]]] = {}
    texts: dict[str, str] = {}
    reads: list[str] = []
    for f in files:
        is_md = f.kind == "md"
        if not is_md and not (f.kind == "text" and known and f.size <= 512 * 1024):
            continue
        stamp = (f.size, f.mtime)
        p = previous.get(f.path)
        if p is None or p.stamp != stamp:
            text = read_text(root / f.path)
            reads.append(f.path)
            p = _Parsed(stamp, text if is_md else None,
                        extract_links(text) if is_md and text is not None else [],
                        find_mentions(text, known) if text else {})
        elif known_changed:
            p = _update_mentions(p, root / f.path, known, added, reads)
        parsed[f.path] = p
        if p.text is not None:
            texts[f.path] = p.text
            if p.resolved is None or not same_paths:
                # Copies, because an Index already handed out may still be in use.
                p.resolved = [replace(link) for link in p.links]
                for link in p.resolved:
                    resolver.resolve(f.path, link)
            links[f.path] = p.resolved
        if p.mentions:
            mentions[f.path] = p.mentions
    if cache is not None:
        cache.known, cache.files, cache.paths, cache.resolver = set(known), parsed, paths, resolver
        cache.reads = reads
    return Index(files=files, links=links, mentions=mentions, texts=texts)


# ── search ─────────────────────────────────────────────────────────────────────


def search(root: Path, files: list[Entry], query: str, *, limit: int = 100,
           per_file: int = 8) -> list[dict[str, Any]]:
    """Case-insensitive search for all whitespace-separated terms.

    A file matches when every term appears in its path or its text.  Lines
    with any term are returned as context.  ``"quoted phrases"`` stay whole.
    """
    terms = [t.lower() for t in re.findall(r'"([^"]+)"|(\S+)', query) for t in t if t]
    if not terms:
        return []
    results: list[dict[str, Any]] = []
    for f in files:
        path_l = f.path.lower()
        text = read_text(root / f.path) if f.kind in ("md", "text") else None
        hay = path_l + "\n" + (text.lower() if text else "")
        if not all(t in hay for t in terms):
            continue
        matches = []
        if text:
            for i, line in enumerate(text.split("\n")):
                ll = line.lower()
                if any(t in ll for t in terms):
                    matches.append({"line": i + 1, "text": line[:400]})
                    if len(matches) >= per_file:
                        break
        results.append({"path": f.path, "kind": f.kind, "name_match": all(t in path_l for t in terms),
                        "matches": matches})
    results.sort(key=lambda r: (not r["name_match"], -len(r["matches"]), r["path"].count("/"), r["path"]))
    return results[:limit]


# ── link rewriting (rename) ────────────────────────────────────────────────────


def relative_href(src: str, target: str) -> str:
    """A markdown link from the file *src* to the file *target*."""
    rel = os.path.relpath(target, str(PurePosixPath(src).parent) or ".").replace(os.sep, "/")
    return rel.replace(" ", "%20")


def rewrite_links(text: str, src: str, moved: dict[str, str], resolver: Resolver) -> tuple[str, int]:
    """Point links in *text* (the file *src*) that resolve to a key of *moved*
    at its value.

    Wikilinks keep their style: a bare name stays a bare name when the new
    name is unique, otherwise the new path is written.  Returns the new text
    and the number of links changed.
    """
    mask = code_mask(text)
    lines = text.split("\n")
    count = 0

    def wiki_target(link: Link, new: str) -> str:
        new_p = PurePosixPath(new)
        is_md = kind_of(new) == "md"
        bare = new_p.stem if is_md else new_p.name
        if "/" not in link.target and len(resolver.by_name.get(bare.lower(), [])) <= 1:
            return bare
        return str(new_p.with_suffix("")) if is_md else new

    for i, line in enumerate(lines):
        if mask[i] or ("[[" not in line and "](" not in line):
            continue
        code_spans = [(m.start(), m.end()) for m in _INLINE_CODE_RE.finditer(line)]

        def in_code(m: re.Match[str]) -> bool:
            return any(a <= m.start() < b for a, b in code_spans)

        def wiki_sub(m: re.Match[str]) -> str:
            nonlocal count
            if in_code(m):
                return m.group(0)
            link = Link(target=m.group(2).strip(), line=i + 1, embed=bool(m.group(1)), wiki=True)
            if not link.target:
                return m.group(0)
            resolver.resolve(src, link)
            new = moved.get(link.resolved or "")
            if new is None:
                return m.group(0)
            count += 1
            heading = f"#{m.group(3)}" if m.group(3) is not None else ""
            alias = f"|{m.group(4)}" if m.group(4) is not None else ""
            return f"{m.group(1)}[[{wiki_target(link, new)}{heading}{alias}]]"

        def md_sub(m: re.Match[str]) -> str:
            nonlocal count
            if in_code(m):
                return m.group(0)
            parsed = parse_href(m.group(3))
            if parsed is None:
                return m.group(0)
            link = Link(target=parsed[0], line=i + 1, embed=bool(m.group(1)), wiki=False)
            resolver.resolve(src, link)
            new = moved.get(link.resolved or "")
            if new is None:
                return m.group(0)
            count += 1
            return with_href_path(m, relative_href(src, new))

        line2 = WIKILINK_RE.sub(wiki_sub, line)
        code_spans = [(m.start(), m.end()) for m in _INLINE_CODE_RE.finditer(line2)]
        line2 = MDLINK_RE.sub(md_sub, line2)
        lines[i] = line2
    return "\n".join(lines), count


# ── moving files (shared by the API and ``mlsweep notes mv``) ──────────────────


def move_path(root: Path, src: str, dst: str, *, update_links: bool = True) -> dict[str, Any]:
    """Rename or move file or folder *src* to *dst* inside *root*, pointing
    links to the moved notes at their new names.

    Raises ``PathError`` / ``FileExistsError`` / ``FileNotFoundError``.
    """
    src, dst = clean_rel(src), clean_rel(dst)
    sp, dp = resolve(root, src, must_exist=True), resolve(root, dst)
    if dp.exists():
        raise FileExistsError(dst)
    if sp.is_dir() and (dst + "/").startswith(src + "/"):
        raise PathError("cannot move a folder into itself")
    files, _ = walk(root)
    resolver = Resolver(f.path for f in files)
    if sp.is_dir():
        moved = {f.path: dst + f.path[len(src):] for f in files if f.path.startswith(src + "/")}
    else:
        moved = {src: dst}
    dp.parent.mkdir(parents=True, exist_ok=True)
    os.rename(sp, dp)
    changed: list[str] = []
    if update_links:
        for f in files:
            if f.kind != "md":
                continue
            cur = moved.get(f.path, f.path)
            path = root / cur
            text = read_text(path)
            if text is None:
                continue
            new_text, n = rewrite_links(text, f.path, moved, resolver)
            if n:
                atomic_write(path, new_text.encode("utf-8"))
                changed.append(cur)
    return {"from": src, "to": dst, "moved": len(moved), "links_updated_in": changed}


def backlinks(index: Index, path: str) -> list[dict[str, Any]]:
    """Links into *path* from other notes: [{src, line, context}]."""
    out = []
    for src, links in index.links.items():
        if src == path:
            continue
        lines: list[str] | None = None
        for link in links:
            if link.resolved == path:
                if lines is None:
                    lines = index.texts.get(src, "").split("\n")
                out.append({"src": src, "line": link.line, "context": _line_context(lines, link.line)})
    return out
