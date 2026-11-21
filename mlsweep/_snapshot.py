"""Code snapshots: a content-addressed store of the files a sweep ships.

A *snapshot* is a manifest of the project tree: every file's path, size, mode
and the sha256 of its bytes, plus the empty directories.  Its ID is
``snap:<sha256 of the canonical manifest>``, so the same files always give the
same ID, whatever else was going on at submit time.

The bytes live once per distinct content in an :class:`ObjectStore`, keyed by
their sha256.  Submitting a tree that differs from an earlier one by a few
files uploads and stores only those files; the rest are shared.  The manager
keeps one store under ``<mlsweep-dir>/store/objects``, and every worker keeps a
cache of the objects its runs have used under ``<scratch>/.objects``.

Nothing here depends on a version control system.  Git metadata, when the
project has it, is recorded beside the snapshot by the client as provenance.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import re
import sqlite3
import stat
import tempfile
import time
import zlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO, Any

SNAP_PREFIX = "snap:"
MANIFEST_VERSION = 1

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1 << 20

# A file is stored compressed when compressing its first MiB saves this much.
_COMPRESS_RATIO = 0.9
_COMPRESSED_SUFFIX = ".z"

# Paths a manifest may not contain: they would reach outside the tree, or
# into a directory that is never shipped.
_FORBIDDEN_PARTS = frozenset({"", ".", ".."})

VCS_DIRS = frozenset({".git", ".svn", ".hg", ".jj"})

# Leave out of every snapshot: VCS metadata, caches, environments.
DEFAULT_EXCLUDES: frozenset[str] = VCS_DIRS | frozenset({
    "__pycache__", "*.pyc", "*.pyo",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules",
    "outputs",
    ".venv", "venv", ".env",
    "*.egg-info", "*.dist-info",
    ".DS_Store",
    ".mlsweep",
})


# ===============================================================================
# IDs
# ===============================================================================


def store_root(mlsweep_dir: str | Path) -> Path:
    """Where the manager keeps its object store."""
    return Path(mlsweep_dir) / "store" / "objects"


def is_snapshot_id(artifact_id: str | None) -> bool:
    """True for a ``snap:`` ID (as opposed to a legacy tarball's ``sha256:``)."""
    return bool(artifact_id) and artifact_id.startswith(SNAP_PREFIX)  # type: ignore[union-attr]


def validate_object_hash(h: Any) -> str:
    """Return *h* if it is a lowercase hex sha256; raise ValueError otherwise."""
    if not isinstance(h, str) or not _HASH_RE.match(h):
        raise ValueError("object hash must be 64 lowercase hex characters")
    return h


def snapshot_id_of(canonical: bytes) -> str:
    """The ID of the snapshot whose canonical manifest is *canonical*."""
    return SNAP_PREFIX + hashlib.sha256(canonical).hexdigest()


def short_id(snapshot_id: str, n: int = 12) -> str:
    """``snap:3fa9c1e2b0d4`` for display."""
    return snapshot_id[: len(SNAP_PREFIX) + n] if is_snapshot_id(snapshot_id) else snapshot_id[:n + 7]


# ===============================================================================
# Manifests
# ===============================================================================


@dataclass(frozen=True)
class FileEntry:
    path: str     # POSIX, relative to the tree root
    hash: str     # sha256 hex of the raw bytes
    size: int
    mode: int     # permission bits (0o644, 0o755, ...)


@dataclass
class Manifest:
    files: list[FileEntry] = field(default_factory=list)
    dirs: list[str] = field(default_factory=list)   # directories with no files under them

    def canonical(self) -> bytes:
        """The bytes the snapshot ID hashes: sorted, compact JSON."""
        doc = {
            "v": MANIFEST_VERSION,
            "files": [[f.path, f.hash, f.size, f.mode] for f in sorted(self.files, key=lambda f: f.path)],
            "dirs": sorted(set(self.dirs)),
        }
        return json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def snapshot_id(self) -> str:
        return snapshot_id_of(self.canonical())

    def to_json(self) -> dict[str, Any]:
        doc: dict[str, Any] = json.loads(self.canonical())
        return doc

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def by_path(self) -> dict[str, FileEntry]:
        return {f.path: f for f in self.files}

    def hashes(self) -> set[str]:
        return {f.hash for f in self.files}


def _check_rel_path(p: Any) -> str:
    if not isinstance(p, str) or not p or len(p) > 4096 or "\\" in p or "\0" in p:
        raise ValueError(f"bad path in manifest: {p!r}")
    pure = PurePosixPath(p)
    if pure.is_absolute() or any(part in _FORBIDDEN_PARTS for part in p.split("/")):
        raise ValueError(f"path leaves the tree: {p!r}")
    if pure.parts[0] in VCS_DIRS:
        raise ValueError(f"path inside VCS metadata: {p!r}")
    return p


def parse_manifest(doc: Any) -> Manifest:
    """Validate a manifest from JSON (as :meth:`Manifest.to_json` writes it)."""
    if isinstance(doc, (bytes, str)):
        doc = json.loads(doc)
    if not isinstance(doc, dict) or doc.get("v") != MANIFEST_VERSION:
        raise ValueError(f"manifest must be an object with v={MANIFEST_VERSION}")
    files_raw, dirs_raw = doc.get("files", []), doc.get("dirs", [])
    if not isinstance(files_raw, list) or not isinstance(dirs_raw, list):
        raise ValueError("manifest 'files' and 'dirs' must be lists")
    files: list[FileEntry] = []
    seen: set[str] = set()
    for item in files_raw:
        if not isinstance(item, list) or len(item) != 4:
            raise ValueError("each manifest file is [path, hash, size, mode]")
        path, h, size, mode = item
        _check_rel_path(path)
        validate_object_hash(h)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"bad size for {path!r}")
        if not isinstance(mode, int) or isinstance(mode, bool) or not 0 <= mode <= 0o7777:
            raise ValueError(f"bad mode for {path!r}")
        if path in seen:
            raise ValueError(f"duplicate path in manifest: {path!r}")
        seen.add(path)
        files.append(FileEntry(path, h, size, mode & 0o777))
    dirs = [_check_rel_path(d) for d in dirs_raw]
    # A file may not sit where another file needs a directory.
    for f in files:
        parent = PurePosixPath(f.path).parent
        while str(parent) != ".":
            if str(parent) in seen:
                raise ValueError(f"{f.path!r} is under a file")
            parent = parent.parent
    return Manifest(files, dirs)


# ===============================================================================
# Object store
# ===============================================================================


class HashMismatch(ValueError):
    """The bytes written do not hash to the expected object ID."""


class ObjectStore:
    """Files keyed by the sha256 of their content, under *root*.

    An object is stored at ``root/ab/cdef...`` as its raw bytes, or at
    ``root/ab/cdef....z`` zlib-compressed when that is worthwhile.  Writes go
    to a temporary file and are renamed into place once their hash checks out,
    so a reader never sees a partial object and concurrent writers of the same
    object are harmless.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _base(self, h: str) -> Path:
        validate_object_hash(h)
        return self.root / h[:2] / h[2:]

    def locate(self, h: str) -> Path | None:
        """The object's file, or None if it is not stored."""
        base = self._base(h)
        if base.is_file():
            return base
        z = base.with_name(base.name + _COMPRESSED_SUFFIX)
        return z if z.is_file() else None

    def has(self, h: str) -> bool:
        return self.locate(h) is not None

    def touch(self, h: str) -> bool:
        """Mark the object as just used (for GC age and cache eviction).  False if absent."""
        p = self.locate(h)
        if p is None:
            return False
        try:
            os.utime(p)
        except OSError:
            pass
        return True

    def writer(self, expected: str | None = None) -> "ObjectWriter":
        return ObjectWriter(self, expected)

    def put_stream(self, src: IO[bytes], expected: str | None = None) -> str:
        w = self.writer(expected)
        try:
            while chunk := src.read(_CHUNK):
                w.write(chunk)
            return w.commit()
        finally:
            w.abort()

    def put_bytes(self, data: bytes) -> str:
        return self.put_stream(io.BytesIO(data))

    def iter_chunks(self, h: str) -> Iterator[bytes]:
        """The object's raw bytes.  Raises FileNotFoundError if it is not stored."""
        p = self.locate(h)
        if p is None:
            raise FileNotFoundError(f"object {h} not in store")
        compressed = p.name.endswith(_COMPRESSED_SUFFIX)
        d = zlib.decompressobj() if compressed else None
        with open(p, "rb") as f:
            while chunk := f.read(_CHUNK):
                if d is not None:
                    chunk = d.decompress(chunk)
                if chunk:
                    yield chunk
        if d is not None:
            tail = d.flush()
            if tail:
                yield tail

    def read_bytes(self, h: str) -> bytes:
        return b"".join(self.iter_chunks(h))

    def copy_to(self, h: str, dest: str | Path, mode: int = 0o644) -> None:
        """Write the object's bytes to *dest* (replacing it) with permission *mode*."""
        dest = Path(dest)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".mlsweep-", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                for chunk in self.iter_chunks(h):
                    f.write(chunk)
            os.chmod(tmp, mode)
            os.replace(tmp, dest)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def remove(self, h: str) -> int:
        """Delete the object; return the bytes freed on disk."""
        freed = 0
        base = self._base(h)
        for p in (base, base.with_name(base.name + _COMPRESSED_SUFFIX)):
            try:
                freed += p.stat().st_size
                p.unlink()
            except FileNotFoundError:
                pass
        return freed

    def iter_objects(self) -> Iterator[tuple[str, Path]]:
        """Every stored object as (hash, file)."""
        if not self.root.is_dir():
            return
        for sub in sorted(self.root.iterdir()):
            if not sub.is_dir() or len(sub.name) != 2:
                continue
            for p in sub.iterdir():
                name = p.name
                if name.endswith(_COMPRESSED_SUFFIX):
                    name = name[: -len(_COMPRESSED_SUFFIX)]
                h = sub.name + name
                if _HASH_RE.match(h) and p.is_file():
                    yield h, p

    def disk_bytes(self) -> int:
        total = 0
        for _, p in self.iter_objects():
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    def trim(self, limit_bytes: int, keep: Iterable[str] = ()) -> int:
        """Evict least recently used objects until the store fits in *limit_bytes*.

        Objects in *keep* stay.  Returns the bytes freed.  Use is the file's
        mtime, which :meth:`touch` and writes update.
        """
        keep_set = set(keep)
        entries: list[tuple[float, int, str]] = []
        total = 0
        for h, p in self.iter_objects():
            try:
                st = p.stat()
            except OSError:
                continue
            total += st.st_size
            if h not in keep_set:
                entries.append((st.st_mtime, st.st_size, h))
        freed = 0
        for _, _, h in sorted(entries):
            if total - freed <= limit_bytes:
                break
            freed += self.remove(h)
        return freed


class ObjectWriter:
    """Streams one object into an :class:`ObjectStore`, hashing as it goes.

    The bytes are spooled raw to a temporary file.  Once the hash is known, an
    object the store already has is dropped without further work; a new one is
    renamed into place, or compressed into place if its first MiB compresses
    well.  So re-storing known content costs one write, never a compression.
    """

    def __init__(self, store: ObjectStore, expected: str | None) -> None:
        if expected is not None:
            validate_object_hash(expected)
        self.store = store
        self.expected = expected
        self.size = 0
        self._sha = hashlib.sha256()
        self._head = bytearray()
        store.root.mkdir(parents=True, exist_ok=True)
        fd, self._tmp = tempfile.mkstemp(dir=store.root, prefix=".incoming-", suffix=".tmp")
        self._file: IO[bytes] = os.fdopen(fd, "wb")
        self._tmp_live = True

    def write(self, data: bytes) -> None:
        self._sha.update(data)
        self.size += len(data)
        if len(self._head) < _CHUNK:
            self._head += data[: _CHUNK - len(self._head)]
        self._file.write(data)

    def _compressible(self) -> bool:
        head = bytes(self._head)
        return bool(head) and len(zlib.compress(head, 1)) < len(head) * _COMPRESS_RATIO

    def commit(self) -> str:
        """Verify the hash and move the object into place.  Returns its hash."""
        self._file.close()
        h = self._sha.hexdigest()
        if self.expected is not None and h != self.expected:
            raise HashMismatch(f"content hashes to {h}, not {self.expected}")
        if self.store.touch(h):
            return h
        base = self.store._base(h)
        base.parent.mkdir(parents=True, exist_ok=True)
        zpath = base.with_name(base.name + _COMPRESSED_SUFFIX)
        if self.size <= _CHUNK:
            # The head is the whole object: compress it once, in memory.
            data = bytes(self._head)
            packed = zlib.compress(data, 6)
            if data and len(packed) < len(data) * _COMPRESS_RATIO:
                self._write_into_place(zpath, lambda dst: dst.write(packed))
            else:
                os.replace(self._tmp, base)
                self._tmp_live = False
            return h
        if not self._compressible():
            os.replace(self._tmp, base)
            self._tmp_live = False
            return h
        # Large files get the fast level: most of the saving at a fraction of the time.
        z = zlib.compressobj(1 if self.size > 16 * _CHUNK else 6)

        def pack(dst: IO[bytes]) -> None:
            with open(self._tmp, "rb") as src:
                while chunk := src.read(_CHUNK):
                    dst.write(z.compress(chunk))
            dst.write(z.flush())

        self._write_into_place(zpath, pack)
        return h

    def _write_into_place(self, dest: Path, fill: Callable[[IO[bytes]], Any]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.store.root, prefix=".incoming-", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as dst:
                fill(dst)
            os.replace(tmp, dest)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def abort(self) -> None:
        """Discard the spooled bytes (safe to call after commit)."""
        if not self._file.closed:
            self._file.close()
        if self._tmp_live:
            try:
                os.unlink(self._tmp)
            except OSError:
                pass
            self._tmp_live = False


# ===============================================================================
# Reading a project tree
# ===============================================================================


def build_output_dirs(root: Path) -> frozenset[str]:
    """Top-level directories of *root* that are build output rather than source.

    A project with a virtualenv in its root installs into it, so a top-level
    ``build/`` there is setuptools' leftover, not code the runs need.  Without
    a venv, ``build/`` may be the project's own source and is shipped.
    """
    try:
        has_venv = any((d / "pyvenv.cfg").is_file() for d in root.iterdir() if d.is_dir())
    except OSError:
        has_venv = False
    return frozenset({"build"}) if has_venv else frozenset()


def project_files(root: Path, excludes: Iterable[str], top_level_excludes: frozenset[str] = frozenset()) -> list[Path]:
    """Directories and regular files under *root*, sorted, without
    anything that matches an exclude pattern or is a symlink, socket or fifo.
    A pattern without ``/`` matches any file or directory name at any depth; a pattern
    with ``/`` matches the path relative to *root* (``runs/old``, ``data/*.bin``; a
    leading ``/`` is ignored).
    *top_level_excludes* are directory names skipped directly under *root* only."""
    excludes = list(excludes)
    name_pats = [p for p in excludes if "/" not in p]
    path_pats = [p.strip("/") for p in excludes if "/" in p]

    def skipped(path: Path) -> bool:
        if any(fnmatch.fnmatchcase(path.name, pat) for pat in name_pats):
            return True
        if not path_pats:
            return False
        rel = path.relative_to(root).as_posix()
        return any(fnmatch.fnmatchcase(rel, pat) for pat in path_pats)

    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not skipped(here / d) and not (here / d).is_symlink()
                       and not (here == root and d in top_level_excludes)]
        out.extend(here / d for d in dirnames)
        for name in filenames:
            path = here / name
            if not skipped(path) and path.is_file() and not path.is_symlink():
                out.append(path)
    return sorted(out)


class HashCache:
    """Remembers file hashes by (path, size, mtime, ctime, inode), as git's index does.

    A file whose stat matches is not read again.  An entry is not trusted if
    the file changed within :data:`RACY_SECS` of being hashed, since a second
    write in the same timestamp tick would not change its mtime.
    """

    RACY_SECS = 2.0

    def __init__(self, path: str | Path | None) -> None:
        self._db: sqlite3.Connection | None = None
        if path is None:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path), timeout=5)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS hashes (path TEXT PRIMARY KEY, size INTEGER, "
                "mtime_ns INTEGER, ctime_ns INTEGER, ino INTEGER, hash TEXT, seen REAL)")
        except sqlite3.Error:
            self._db = None

    @staticmethod
    def _key(st: os.stat_result) -> tuple[int, int, int, int]:
        return (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)

    def get(self, path: str, st: os.stat_result) -> str | None:
        if self._db is None:
            return None
        try:
            row = self._db.execute(
                "SELECT size, mtime_ns, ctime_ns, ino, hash FROM hashes WHERE path = ?", (path,)).fetchone()
        except sqlite3.Error:
            return None
        if row is None or tuple(row[:4]) != self._key(st):
            return None
        return row[4]  # type: ignore[no-any-return]

    def put(self, path: str, st: os.stat_result, h: str) -> None:
        if self._db is None:
            return
        if time.time() - st.st_mtime < self.RACY_SECS or time.time() - st.st_ctime < self.RACY_SECS:
            return
        try:
            self._db.execute("INSERT OR REPLACE INTO hashes VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (path, *self._key(st), h, time.time()))
        except sqlite3.Error:
            pass

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.commit()
                self._db.close()
            except sqlite3.Error:
                pass
            self._db = None


def hash_file(path: str | Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            sha.update(chunk)
    return sha.hexdigest()


@dataclass
class ScannedTree:
    root: Path
    manifest: Manifest
    sources: dict[str, Path]   # object hash -> one local file with that content
    hashed: list[str] = field(default_factory=list)  # paths read and hashed, not found in the hash cache


def scan_tree(
    root: str | Path,
    *,
    exclude_patterns: Iterable[str] = (),
    cache_path: str | Path | None = None,
) -> ScannedTree:
    """Hash the project tree at *root* into a manifest (nothing is uploaded)."""
    root = Path(root).resolve()
    excludes = set(DEFAULT_EXCLUDES) | set(exclude_patterns)
    cache = HashCache(cache_path)
    files: list[FileEntry] = []
    dir_paths: list[str] = []
    sources: dict[str, Path] = {}
    hashed: list[str] = []
    try:
        for entry in project_files(root, excludes, build_output_dirs(root)):
            rel = entry.relative_to(root).as_posix()
            try:
                st = entry.stat()
            except OSError:
                continue
            if stat.S_ISDIR(st.st_mode):
                dir_paths.append(rel)
                continue
            key = str(entry)
            h = cache.get(key, st)
            if h is None:
                try:
                    h = hash_file(entry)
                except OSError:
                    continue
                hashed.append(rel)
                cache.put(key, st, h)
            files.append(FileEntry(rel, h, st.st_size, stat.S_IMODE(st.st_mode) & 0o777))
            sources.setdefault(h, entry)
    finally:
        cache.close()
    return ScannedTree(root, Manifest(files, leaf_dirs(files, dir_paths)), sources, hashed)


def leaf_dirs(files: Iterable[FileEntry], dirs: Iterable[str]) -> list[str]:
    """The directories in *dirs* that writing *files* would not create anyway."""
    covered: set[str] = set()

    def add_parents(path: str) -> None:
        p = PurePosixPath(path).parent
        while str(p) != "." and str(p) not in covered:
            covered.add(str(p))
            p = p.parent

    for f in files:
        add_parents(f.path)
    dirs = sorted(set(dirs))
    for d in dirs:
        add_parents(d)
    return [d for d in dirs if d not in covered]


# ===============================================================================
# Writing a snapshot out
# ===============================================================================


def fetch_object(store: ObjectStore, url: str, h: str, *,
                 headers: dict[str, str] | None = None, timeout: float = 300) -> None:
    """Download object *h* from *url* into *store*, checked against its hash."""
    from urllib.request import Request, urlopen
    with urlopen(Request(url, headers=headers or {}), timeout=timeout) as r:
        store.put_stream(r, expected=h)


def materialize(
    manifest: Manifest,
    dest: str | Path,
    store: ObjectStore,
    fetch: Callable[[str], None] | None = None,
) -> None:
    """Write the snapshot's files into *dest* by copying them out of *store*.

    *fetch(hash)* is called for an object the store lacks (or loses to
    eviction meanwhile); it should put the object into *store*.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for d in manifest.dirs:
        (dest / d).mkdir(parents=True, exist_ok=True)
    for f in manifest.files:
        target = dest / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            if not store.has(f.hash):
                if fetch is None:
                    raise FileNotFoundError(f"object {f.hash} for {f.path} is not available")
                fetch(f.hash)
            try:
                store.copy_to(f.hash, target, f.mode or 0o644)
                store.touch(f.hash)
                break
            except FileNotFoundError:
                if attempt or fetch is None:
                    raise


# ===============================================================================
# Comparing snapshots
# ===============================================================================


@dataclass
class TreeDiff:
    added: list[FileEntry]
    removed: list[FileEntry]
    changed: list[tuple[FileEntry, FileEntry]]   # (old, new), content or mode differs

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed)


def under_prefix(path: str, prefix: str | None) -> bool:
    """True if *path* is *prefix* or inside it (always, for an empty prefix)."""
    pre = (prefix or "").strip("/")
    return not pre or path == pre or path.startswith(pre + "/")


def diff_manifests(old: Manifest, new: Manifest, prefix: str | None = None) -> TreeDiff:
    """Which files differ between two snapshots, optionally under *prefix* only."""
    a, b = old.by_path(), new.by_path()

    def wanted(p: str) -> bool:
        return under_prefix(p, prefix)

    added = [b[p] for p in sorted(b) if p not in a and wanted(p)]
    removed = [a[p] for p in sorted(a) if p not in b and wanted(p)]
    changed = [(a[p], b[p]) for p in sorted(a) if p in b and wanted(p)
               and (a[p].hash != b[p].hash or a[p].mode != b[p].mode)]
    return TreeDiff(added, removed, changed)


def looks_binary(data: bytes) -> bool:
    """A NUL byte in the first 8 KB, as git decides."""
    return b"\0" in data[:8192]
