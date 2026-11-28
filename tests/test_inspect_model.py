"""Model and array previews for the Artifacts page (``mlsweep._inspect_model``)."""

import base64
import json
import os
import pickle
import shlex
import struct
from pathlib import Path

import pytest

from campaign_helpers import shared_manager, uid
from conftest import _api_status
from mlsweep._inspect_model import InspectError, inspect_file

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3),
                               torch.nn.Linear(3, 2).to(torch.bfloat16))


def _by_name(info):
    return {t["name"]: t for t in info["tensors"]}


def test_state_dict(tmp_path):
    m = _model()
    torch.save(m.state_dict(), tmp_path / "sd.pt")
    info = inspect_file(tmp_path / "sd.pt")
    assert info["format"] == "pytorch"
    assert info["tensor_count"] == 6 and info["total_params"] == 29
    t = _by_name(info)
    assert t["0.weight"]["shape"] == [3, 4] and t["0.weight"]["dtype"] == "float32"
    assert t["2.weight"]["dtype"] == "bfloat16"
    assert t["1.weight"]["values"] == [1.0, 1.0, 1.0]          # LayerNorm init
    assert t["2.bias"]["values"] == pytest.approx(m[2].bias.float().tolist())


def test_training_checkpoint_keeps_its_scalars(tmp_path):
    m = _model()
    opt = torch.optim.Adam(m.parameters())
    m[0](torch.randn(2, 4)).sum().backward()
    opt.step()
    torch.save({"epoch": 3, "model": m.state_dict(), "optimizer": opt.state_dict(),
                "cfg": {"lr": 1e-3}, "device": torch.device("cuda", 0),
                "rng": np.random.get_state()}, tmp_path / "last.ckpt")
    info = inspect_file(tmp_path / "last.ckpt")
    assert info["tree"]["epoch"] == 3 and info["tree"]["cfg"] == {"lr": 1e-3}
    t = _by_name(info)
    assert t["model.0.weight"]["shape"] == [3, 4]
    assert t["optimizer.state.0.step"]["values"] == [1.0]
    assert t["rng.1"]["kind"] == "ndarray" and t["rng.1"]["dtype"] == "uint32"


def test_whole_module_legacy_and_torchscript(tmp_path):
    m = _model()
    torch.save(m, tmp_path / "full.pth")
    torch.save(m.state_dict(), tmp_path / "old.pt", _use_new_zipfile_serialization=False)
    torch.jit.script(m).save(tmp_path / "ts.pt")
    full = inspect_file(tmp_path / "full.pth")
    assert full["tree"]["__object__"].endswith("Sequential")
    assert _by_name(full)["0.weight"]["kind"] == "parameter"     # _modules/_parameters left out
    legacy = inspect_file(tmp_path / "old.pt")
    assert legacy["format"] == "pytorch-legacy" and set(_by_name(legacy)) == set(_by_name(full))
    ts = inspect_file(tmp_path / "ts.pt")
    assert ts["format"] == "torchscript" and ts["total_params"] == 29


def test_safetensors(tmp_path):
    header = {"__metadata__": {"format": "pt"},
              "w": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}}
    raw = json.dumps(header).encode()
    (tmp_path / "m.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(12))
    info = inspect_file(tmp_path / "m.safetensors")
    assert info["format"] == "safetensors" and info["metadata"] == {"format": "pt"}
    assert info["tensors"] == [{"name": "w", "dtype": "bfloat16", "shape": [2, 3], "numel": 6, "bytes": 12}]


def test_numpy(tmp_path):
    np.save(tmp_path / "a.npy", np.array([1.5, 2.5], dtype=np.float32))
    np.savez(tmp_path / "b.npz", x=np.zeros((3, 4)), y=np.arange(3))
    assert inspect_file(tmp_path / "a.npy")["tensors"][0]["values"] == [1.5, 2.5]
    t = _by_name(inspect_file(tmp_path / "b.npz"))
    assert t["x"]["shape"] == [3, 4] and t["x"]["dtype"] == "float64" and t["y"]["values"] == [0, 1, 2]
    with open(tmp_path / "p.pkl", "wb") as f:
        pickle.dump({"preds": np.ones((5, 2), dtype=np.float16), "n": 7}, f)
    info = inspect_file(tmp_path / "p.pkl")
    assert info["format"] == "pickle" and info["tree"]["n"] == 7
    assert _by_name(info)["preds"]["shape"] == [5, 2]


def _gguf_string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def test_gguf(tmp_path):
    kv = (_gguf_string("general.architecture") + struct.pack("<I", 8) + _gguf_string("llama")
          + _gguf_string("tokens") + struct.pack("<IIQ", 9, 8, 3) + b"".join(_gguf_string(c) for c in "abc"))
    tensor = _gguf_string("tok.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 64, 10) + struct.pack("<IQ", 8, 0)
    (tmp_path / "m.gguf").write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 2) + kv + tensor)
    info = inspect_file(tmp_path / "m.gguf")
    assert info["metadata"]["general.architecture"] == "llama"
    assert info["metadata"]["tokens"] == {"array": 3, "head": ["a", "b", "c"]}
    t = info["tensors"][0]
    assert t["shape"] == [10, 64] and t["dtype"] == "q8_0" and t["bytes"] == 20 * 34


# Written by onnx.helper: MatMul → Add → Relu with weights W[4,3] and B[3].
_ONNX = base64.b64decode(
    "CA4SBHRlc3Q6vAEKEQoBeAoBVxIBaCIGTWF0TXVsCg4KAWgKAUISAXkiA0FkZAoMCgF5EgF6IgRSZWx1EgFnKjsIBAgDEAFCAVdK"
    "MAAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPyoVCAMQAUIBQkoMAAAAAAAAAAAAAAAAWhgK"
    "AXgSEwoRCAESDQoHEgViYXRjaAoCCARiGAoBehITChEIARINCgcSBWJhdGNoCgIIA0ICEBxyDAoGYXV0aG9yEgJtZQ==")


def test_onnx(tmp_path):
    (tmp_path / "m.onnx").write_bytes(_ONNX)
    info = inspect_file(tmp_path / "m.onnx")
    assert info["ops"] == {"MatMul": 1, "Add": 1, "Relu": 1}
    assert info["inputs"] == [{"name": "x", "dtype": "float32", "shape": ["batch", 4]}]
    assert _by_name(info)["W"]["shape"] == [4, 3]
    assert info["metadata"]["producer_name"] == "test" and info["metadata"]["author"] == "me"


class _Evil:
    """Unpickling this runs ``touch TARGET``."""

    def __init__(self, target):
        self.target = target

    def __reduce__(self):
        return (os.system, (f"touch {shlex.quote(str(self.target))}",))


def test_a_hostile_pickle_describes_itself_without_running(tmp_path):
    with open(tmp_path / "evil.pkl", "wb") as f:
        pickle.dump({"x": _Evil(tmp_path / "PWNED")}, f)
    torch.save({"x": _Evil(tmp_path / "PWNED")}, tmp_path / "evil.pt")
    for name in ("evil.pkl", "evil.pt"):
        info = inspect_file(tmp_path / name)
        assert info["tree"]["x"]["__object__"] in ("posix.system", "os.system", "nt.system")
    assert not (tmp_path / "PWNED").exists()


def test_unknown_bytes_are_refused(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"\0\1\2\3" * 10)
    (tmp_path / "x.pt").write_bytes(b"\x80garbage")
    for name in ("x.bin", "x.pt"):
        with pytest.raises(InspectError):
            inspect_file(tmp_path / name)


# ── The route ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mgr(tmp_path_factory):
    yield from shared_manager(tmp_path_factory, "inspect_model")


def _get(mgr, path):
    return _api_status(mgr.url, mgr.token, "GET", path)


def test_route(mgr):
    eid = uid("insp")
    rd = Path(mgr.mlsweep_dir) / "experiments" / eid / "r1" / "artifacts" / "ckpt"
    rd.mkdir(parents=True)
    torch.save(_model().state_dict(), rd / "model.pt")
    (rd / "junk.bin").write_bytes(b"\0" * 16)
    base = f"/api/experiments/{eid}/jobs/r1/artifact-info"
    status, info = _get(mgr, f"{base}/ckpt/model.pt")
    assert status == 200 and info["tensor_count"] == 6
    assert _get(mgr, f"{base}/ckpt/junk.bin")[0] == 422
    assert _get(mgr, f"{base}/ckpt/missing.pt")[0] == 404
    assert _get(mgr, f"{base}/..%2F..%2F..%2Fx.pt")[0] in (400, 403)
