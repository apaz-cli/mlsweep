"""Code snapshots: the object store, manifests, upload, provenance and GC."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import _api_get, _api_post, _api_request
from mlsweep._snapshot import (
    FileEntry,
    HashCache,
    HashMismatch,
    Manifest,
    ObjectStore,
    diff_manifests,
    materialize,
    parse_manifest,
    scan_tree,
)

TOKEN = "test-token"


def _raw(url: str, path: str, *, method: str = "GET", data: bytes | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(f"{url}{path}", data=data, method=method,
                                 headers={"Authorization": f"Bearer {TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _tree(root: Path, files: dict[str, str | bytes]) -> Path:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode() if isinstance(content, str) else content)
    return root


# ── Store and manifests ───────────────────────────────────────────────────────


def test_store_round_trips_compressible_and_incompressible(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    text = b"hello world\n" * 200_000          # compresses well, crosses the 1 MiB decision point
    noise = os.urandom(3 << 20)               # does not compress
    for data in (text, noise, b""):
        h = store.put_bytes(data)
        assert store.read_bytes(h) == data
        assert store.has(h)
    # Compressible content is stored compressed; random bytes raw.
    assert store.locate(store.put_bytes(text)).name.endswith(".z")  # type: ignore[union-attr]
    assert not store.locate(store.put_bytes(noise)).name.endswith(".z")  # type: ignore[union-attr]
    assert store.disk_bytes() < len(text) + len(noise)


def test_store_refuses_content_that_does_not_match_its_hash(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path)
    with pytest.raises(HashMismatch):
        store.put_stream(io.BytesIO(b"actual"), expected="0" * 64)
    assert list(store.iter_objects()) == []
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())


def test_trim_evicts_least_recently_used_and_keeps_what_is_asked(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path)
    hashes = [store.put_bytes(os.urandom(1000)) for _ in range(4)]
    for i, h in enumerate(hashes):
        os.utime(store.locate(h), (1000 + i, 1000 + i))  # type: ignore[arg-type]
    store.trim(2000, keep=[hashes[0]])
    left = {h for h, _ in store.iter_objects()}
    assert left == {hashes[0], hashes[3]}


def test_manifest_rejects_paths_that_leave_the_tree() -> None:
    h = "a" * 64
    for bad in ("../x", "/etc/passwd", "a//b", "a/./b", ".git/config", "x\\y"):
        with pytest.raises(ValueError):
            parse_manifest({"v": 1, "files": [[bad, h, 1, 0o644]], "dirs": []})
    with pytest.raises(ValueError):
        parse_manifest({"v": 1, "files": [["a", h, 1, 0o644], ["a", h, 1, 0o644]], "dirs": []})
    with pytest.raises(ValueError):
        parse_manifest({"v": 1, "files": [["a", h, 1, 0o644], ["a/b", h, 1, 0o644]], "dirs": []})
    m = parse_manifest({"v": 1, "files": [["a/b.py", h, 1, 0o755]], "dirs": ["empty"]})
    assert m.files[0].mode == 0o755 and m.dirs == ["empty"]


def test_snapshot_id_covers_content_paths_and_modes_only(tmp_path: Path) -> None:
    root = _tree(tmp_path / "p", {"train.py": "print(1)\n", "data/x.bin": b"\0\1"})
    a = scan_tree(root).manifest
    assert a.snapshot_id().startswith("snap:")
    assert parse_manifest(a.to_json()).snapshot_id() == a.snapshot_id()
    os.chmod(root / "train.py", 0o755)
    b = scan_tree(root).manifest
    assert b.snapshot_id() != a.snapshot_id()
    d = diff_manifests(a, b)
    assert [n.path for _, n in d.changed] == ["train.py"] and not d.added and not d.removed


def test_hash_cache_skips_unchanged_files(tmp_path: Path) -> None:
    root = _tree(tmp_path / "p", {"a.py": "a", "b.py": "b"})
    # The cache only trusts a file that has not changed for RACY_SECS, ctime included.
    time.sleep(HashCache.RACY_SECS + 0.1)
    cache = tmp_path / "cache.db"
    first = scan_tree(root, cache_path=cache)
    assert sorted(first.hashed) == ["a.py", "b.py"]
    again = scan_tree(root, cache_path=cache)
    assert again.manifest.snapshot_id() == first.manifest.snapshot_id()
    assert again.hashed == []
    (root / "a.py").write_text("changed")
    assert scan_tree(root, cache_path=cache).hashed == ["a.py"]


def test_materialize_fetches_only_missing_objects(tmp_path: Path) -> None:
    src = ObjectStore(tmp_path / "src")
    h1, h2 = src.put_bytes(b"one"), src.put_bytes(b"two")
    m = Manifest([FileEntry("a.txt", h1, 3, 0o644), FileEntry("sub/run.sh", h2, 3, 0o755)], ["empty"])
    cache = ObjectStore(tmp_path / "cache")
    cache.put_bytes(b"one")
    fetched: list[str] = []

    def fetch(h: str) -> None:
        fetched.append(h)
        cache.put_bytes(src.read_bytes(h))

    out = tmp_path / "out"
    materialize(m, out, cache, fetch=fetch)
    assert fetched == [h2]
    assert (out / "a.txt").read_text() == "one" and (out / "sub/run.sh").read_text() == "two"
    assert os.stat(out / "sub/run.sh").st_mode & 0o777 == 0o755
    assert (out / "empty").is_dir()


# ── Manager API ───────────────────────────────────────────────────────────────


def _upload(url: str, root: Path):  # type: ignore[no-untyped-def]
    from mlsweep.run_sweep import upload_snapshot
    tree = scan_tree(root)
    result = upload_snapshot(url, TOKEN, tree)
    assert result is not None
    return tree, result


def test_upload_sends_only_new_files(manager_server, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _, url = manager_server
    big = os.urandom(2 << 20)
    root = _tree(tmp_path / "p", {"train.py": "print('v1')\n", "data/train.bin": big, "copy.bin": big})
    tree, (sid, stats) = _upload(url, root)
    assert stats == {"files": 3, "bytes": tree.manifest.total_bytes, "new_files": 3,
                     "new_bytes": len("print('v1')\n") + len(big)}

    _, (sid_again, stats) = _upload(url, root)
    assert sid_again == sid and stats["new_files"] == 0 and stats["new_bytes"] == 0

    (root / "train.py").write_text("print('v2')\n")
    _, (sid2, stats) = _upload(url, root)
    assert sid2 != sid and stats["new_files"] == 1 and stats["new_bytes"] == len("print('v2')\n")

    snap = _api_get(url, TOKEN, f"/api/snapshots/{sid2}")
    assert snap["artifact_id"] == sid2 and snap["files"] == 3
    status, body = _raw(url, f"/api/snapshots/{sid2}/files/train.py")
    assert status == 200 and body == b"print('v2')\n"
    status, body = _raw(url, f"/api/snapshots/{sid2}/files/data/train.bin")
    assert status == 200 and body == big
    assert _raw(url, f"/api/snapshots/{sid2}/files/nope.py")[0] == 404
    # A unique prefix resolves.
    assert _api_get(url, TOKEN, f"/api/snapshots/{sid2[:17]}")["artifact_id"] == sid2


def _safetensors(shape: list[int], data: bytes) -> bytes:
    header = json.dumps({"w": {"dtype": "F32", "shape": shape, "data_offsets": [0, len(data)]}}).encode()
    return len(header).to_bytes(8, "little") + header + data


def test_snapshot_model_file_is_summarized(manager_server, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    # As artifact-info does for a run's artifacts.  Zeros are stored compressed
    # and random bytes raw, so both ways of reading an object are covered.
    _, url = manager_server
    root = _tree(tmp_path / "p", {
        "zeros.safetensors": _safetensors([4, 8], bytes(128)),
        "weights/noise.safetensors": _safetensors([64, 64], os.urandom(64 * 64 * 4)),
        "train.py": "print('hi')\n",
    })
    _, (sid, _) = _upload(url, root)
    for path, shape in (("zeros.safetensors", [4, 8]), ("weights/noise.safetensors", [64, 64])):
        status, body = _raw(url, f"/api/snapshots/{sid}/file-info/{path}")
        info = json.loads(body)
        assert status == 200 and info["format"] == "safetensors", body
        assert [t["shape"] for t in info["tensors"]] == [shape]
    assert _raw(url, f"/api/snapshots/{sid}/file-info/train.py")[0] == 422
    assert _raw(url, f"/api/snapshots/{sid}/file-info/nope.pt")[0] == 404


def test_snapshot_refused_until_its_objects_are_uploaded(manager_server) -> None:  # type: ignore[no-untyped-def]
    _, url = manager_server
    h = "b" * 64
    status, body = _raw(url, "/api/snapshots", method="POST",
                        data=json.dumps({"manifest": {"v": 1, "files": [["x", h, 1, 0o644]], "dirs": []}}).encode())
    assert status == 409 and json.loads(body)["missing"] == [h]
    status, _ = _raw(url, f"/api/objects/{'c' * 64}", method="PUT", data=b"not c")
    assert status == 400
    assert _raw(url, "/api/objects/not-a-hash")[0] == 400


def test_experiment_records_code_and_provenance(manager_server, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _, url = manager_server
    root = _tree(tmp_path / "p", {"train.py": "x"})
    _, (sid, _) = _upload(url, root)
    code = {"snapshot": sid, "sweep_file": "sweeps/a.py", "host": "h",
            "vcs": {"type": "git", "commit": "abc", "dirty": False}}
    _api_post(url, TOKEN, "/api/experiments/e1/jobs/bulk", {
        "campaign": "test",
        "name": "e1", "code": code,
        "jobs": [{"run_id": "r1", "command": ["true"], "artifact_id": sid}],
    })
    info = _api_get(url, TOKEN, "/api/experiments/e1/code")
    assert info["versions"] == [{"artifact_id": sid, "runs": 1, "kind": "snapshot", "files": 1, "size_bytes": 1}]
    assert len(info["submissions"]) == 1
    assert info["submissions"][0]["snapshot"] == sid and info["submissions"][0]["meta"]["vcs"]["commit"] == "abc"
    status, body = _raw(url, "/api/experiments/e2/jobs/bulk", method="POST",
                        data=json.dumps({"campaign": "test", "code": {"no": "snapshot"},
                                         "jobs": [{"run_id": "r", "command": ["true"]}]}).encode())
    assert status == 400


def _db(server) -> sqlite3.Connection:  # type: ignore[no-untyped-def]
    return sqlite3.connect(str(Path(server.mlsweep_dir).parent / "manager.db"))


def test_gc_removes_objects_no_snapshot_lists(manager_server, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    server, url = manager_server
    root = _tree(tmp_path / "p", {"shared.py": "shared", "only_a.py": "a" * 5000})
    _, (sid_a, _) = _upload(url, root)
    (root / "only_a.py").unlink()
    (root / "only_b.py").write_text("b")
    _, (sid_b, _) = _upload(url, root)
    for exp, sid in (("ea", sid_a), ("eb", sid_b)):
        _api_post(url, TOKEN, f"/api/experiments/{exp}/jobs/bulk",
                  {"campaign": "test", "code": {"snapshot": sid}, "jobs": [{"run_id": "r", "command": ["true"], "artifact_id": sid}]})
    _api_request(url, TOKEN, "DELETE", "/api/experiments/ea")

    # Age everything past GC's grace period.
    store = ObjectStore(Path(server.mlsweep_dir) / "store" / "objects")
    for _, p in store.iter_objects():
        os.utime(p, (1, 1))
    with _db(server) as db:
        db.execute("UPDATE artifacts SET stored_at = 1")

    plan = _api_post(url, TOKEN, "/api/gc", {"dry_run": True})
    assert plan["objects"]["count"] == 1 and plan["artifact_rows"] == 1
    _api_post(url, TOKEN, "/api/gc", {"dry_run": False})
    left = {h for h, _ in store.iter_objects()}
    assert len(left) == 2  # shared.py and only_b.py
    assert _raw(url, f"/api/snapshots/{sid_b}/files/shared.py")[1] == b"shared"
    assert _raw(url, f"/api/snapshots/{sid_a}")[0] == 404
