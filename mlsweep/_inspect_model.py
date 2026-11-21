"""Summaries of model and array files, read without the libraries that wrote them.

``inspect_file`` reads a checkpoint's structure (tensor names, dtypes, shapes,
metadata) from its header or pickle.  Nothing here imports torch or numpy, and
pickles are read by an unpickler that never imports or calls anything: every
global becomes an inert stub, so a hostile checkpoint can describe itself but
cannot run code on the manager.

Supported: safetensors, PyTorch (zip and legacy pickle, TorchScript included),
plain pickle, NumPy ``.npy``/``.npz``, GGUF, and ONNX.
"""

from __future__ import annotations

import ast
import io
import json
import math
import pickle
import struct
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import IO, Any, BinaryIO, Callable

MAX_TENSORS = 20_000        # tensors listed; the totals still count every one
MAX_TREE_NODES = 4_000      # nodes in the non-tensor structure summary
MAX_DEPTH = 24
MAX_PICKLE_BYTES = 256 << 20
MAX_HEADER_BYTES = 128 << 20
MAX_STR = 200
MAX_ITEMS = 64              # list/dict entries kept per container in the tree
VALUE_PREVIEW = 8           # tensors this small show their values

EXTENSIONS = frozenset({
    ".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle",
    ".npy", ".npz", ".gguf", ".onnx",
})


class InspectError(Exception):
    """The file is not in a format this module can read."""


# ── dtypes ─────────────────────────────────────────────────────────────────────

_ITEMSIZE = {
    "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2, "int64": 8, "int32": 4,
    "int16": 2, "int8": 1, "uint64": 8, "uint32": 4, "uint16": 2, "uint8": 1, "bool": 1,
    "complex128": 16, "complex64": 8, "complex32": 4, "float8_e4m3fn": 1, "float8_e5m2": 1,
    "float8_e4m3fnuz": 1, "float8_e5m2fnuz": 1, "qint8": 1, "quint8": 1, "qint32": 4,
    "quint4x2": 1,
}
_STRUCT = {"float64": "d", "float32": "f", "float16": "e", "int64": "q", "int32": "i",
           "int16": "h", "int8": "b", "uint64": "Q", "uint32": "I", "uint16": "H",
           "uint8": "B", "bool": "?"}

_TORCH_STORAGE = {
    "DoubleStorage": "float64", "FloatStorage": "float32", "HalfStorage": "float16",
    "BFloat16Storage": "bfloat16", "LongStorage": "int64", "IntStorage": "int32",
    "ShortStorage": "int16", "CharStorage": "int8", "ByteStorage": "uint8",
    "BoolStorage": "bool", "ComplexDoubleStorage": "complex128",
    "ComplexFloatStorage": "complex64", "QInt8Storage": "qint8", "QUInt8Storage": "quint8",
    "QInt32Storage": "qint32", "QUInt4x2Storage": "quint4x2",
    "Float8_e4m3fnStorage": "float8_e4m3fn", "Float8_e5m2Storage": "float8_e5m2",
}
_SAFETENSORS_DTYPE = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U64": "uint64",
    "U32": "uint32", "U16": "uint16", "U8": "uint8", "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2", "C64": "complex64",
}
_NUMPY_KIND = {"f": "float", "i": "int", "u": "uint", "c": "complex"}


def _numpy_dtype(descr: Any) -> str:
    """``'<f4'`` → ``float32``; anything structured is shown as given."""
    if not isinstance(descr, str) or len(descr) < 2:
        return str(descr)[:MAX_STR]
    code = descr.lstrip("<>|=")
    if code in ("b1", "?"):
        return "bool"
    kind, size = code[:1], code[1:]
    if kind in _NUMPY_KIND and size.isdigit():
        return f"{_NUMPY_KIND[kind]}{int(size) * 8}"
    return code


def _tensor(name: str, dtype: str, shape: list[int], **extra: Any) -> dict[str, Any]:
    numel = math.prod(shape) if shape else 1
    size = _ITEMSIZE.get(dtype)
    t: dict[str, Any] = {"name": name, "dtype": dtype, "shape": shape, "numel": numel,
                         "bytes": numel * size if size else None}
    t.update(extra)
    return t


def _clip(value: Any) -> Any:
    """*value* as small JSON: long strings cut, containers capped."""
    if isinstance(value, str):
        return value if len(value) <= MAX_STR else value[:MAX_STR] + "…"
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (list, tuple)):
        items = [_clip(v) for v in value[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"… {len(value) - MAX_ITEMS} more")
        return items
    if isinstance(value, dict):
        return {str(k)[:MAX_STR]: _clip(v) for k, v in list(value.items())[:MAX_ITEMS]}
    return value


def _result(fmt: str, tensors: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    dtypes: dict[str, int] = {}
    total_bytes = 0
    for t in tensors:
        dtypes[t["dtype"]] = dtypes.get(t["dtype"], 0) + t["numel"]
        total_bytes += t["bytes"] or 0
    out: dict[str, Any] = {
        "format": fmt,
        "tensor_count": len(tensors),
        "total_params": sum(t["numel"] for t in tensors),
        "total_tensor_bytes": total_bytes,
        "dtypes": dtypes,
        "tensors": tensors[:MAX_TENSORS],
        "tensors_truncated": len(tensors) > MAX_TENSORS,
    }
    out.update(extra)
    return out


# ── safetensors ────────────────────────────────────────────────────────────────

def _safetensors(f: BinaryIO) -> dict[str, Any]:
    raw = f.read(8)
    if len(raw) < 8:
        raise InspectError("too short for safetensors")
    (n,) = struct.unpack("<Q", raw)
    if n > MAX_HEADER_BYTES:
        raise InspectError("safetensors header too large")
    try:
        header = json.loads(f.read(n))
    except (ValueError, UnicodeDecodeError):
        raise InspectError("safetensors header is not JSON") from None
    if not isinstance(header, dict):
        raise InspectError("safetensors header is not an object")
    meta = header.pop("__metadata__", None)
    tensors = []
    for name, info in header.items():
        if not isinstance(info, dict):
            continue
        dtype = _SAFETENSORS_DTYPE.get(str(info.get("dtype")), str(info.get("dtype")).lower())
        shape = [int(d) for d in info.get("shape") or []]
        tensors.append(_tensor(name, dtype, shape))
    return _result("safetensors", tensors, metadata=_clip(meta) if isinstance(meta, dict) else None)


# ── pickle (PyTorch and plain) ─────────────────────────────────────────────────

class _Stub:
    """An object from a global the unpickler refused to import."""

    _qualname = "?"
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    state: Any
    items: dict[Any, Any]
    list: list[Any]

    # __new__ rather than __init__: pickle's NEWOBJ opcode never calls __init__.
    def __new__(cls, *args: Any, **kwargs: Any) -> _Stub:
        self = super().__new__(cls)
        self.args, self.kwargs, self.state, self.items, self.list = args, kwargs, None, {}, []
        return self

    def __setstate__(self, state: Any) -> None:
        self.state = state

    def __setitem__(self, key: Any, value: Any) -> None:
        self.items[key] = value

    def append(self, value: Any) -> None:
        self.list.append(value)

    def extend(self, values: Any) -> None:
        self.list.extend(values)


def _stub_class(qualname: str) -> type[_Stub]:
    return type("Stub", (_Stub,), {"_qualname": qualname})


class _Global:
    """A non-callable global such as ``torch.float32`` or a storage type."""

    def __init__(self, qualname: str) -> None:
        self.qualname = qualname

    def __call__(self, *args: Any, **kwargs: Any) -> _Stub:   # torch.device("cuda") and the like
        return _stub_class(self.qualname)(*args, **kwargs)


class _Storage:
    def __init__(self, dtype: str, key: str, numel: int | None) -> None:
        self.dtype, self.key, self.numel = dtype, key, numel


class _Tensor:
    def __init__(self, dtype: str, shape: list[int], storage: _Storage | None = None,
                 offset: int = 0, kind: str = "tensor", requires_grad: bool = False) -> None:
        self.dtype, self.shape, self.storage, self.offset = dtype, shape, storage, offset
        self.kind, self.requires_grad = kind, requires_grad


def _ints(value: Any) -> list[int]:
    try:
        return [int(v) for v in value]
    except (TypeError, ValueError):
        return []


def _storage_dtype(storage_type: Any) -> str:
    name = storage_type.qualname.rsplit(".", 1)[-1] if isinstance(storage_type, _Global) else ""
    if name in _TORCH_STORAGE:
        return _TORCH_STORAGE[name]
    return name.removesuffix("Storage").lower() or "unknown"


def _rebuild_tensor(storage: Any, offset: Any = 0, size: Any = (), *_rest: Any) -> Any:
    if not isinstance(storage, _Storage):
        return _stub_class("torch.Tensor")(storage, offset, size)
    return _Tensor(storage.dtype, _ints(size), storage, int(offset or 0))


def _rebuild_parameter(data: Any, requires_grad: Any = True, *_rest: Any) -> Any:
    if isinstance(data, _Tensor):
        data.kind, data.requires_grad = "parameter", bool(requires_grad)
    return data


def _rebuild_from_type(func: Any, _new_type: Any, args: Any, _state: Any = None) -> Any:
    return func(*args) if callable(func) and isinstance(args, tuple) else None


def _rebuild_qtensor(storage: Any, offset: Any, size: Any, *_rest: Any) -> Any:
    t = _rebuild_tensor(storage, offset, size)
    if isinstance(t, _Tensor):
        t.kind = "quantized"
    return t


def _rebuild_shape_only(kind: str) -> Callable[..., Any]:
    """Rebuilds whose dtype argument comes before the shape (meta, wrapper subclasses)."""
    def rebuild(*args: Any) -> Any:
        dtype = next((a.qualname.rsplit(".", 1)[-1] for a in args if isinstance(a, _Global)), "unknown")
        shape = next((_ints(a) for a in args if isinstance(a, (tuple, list))), [])
        return _Tensor(dtype, shape, kind=kind)
    return rebuild


_TORCH_FUNCS: dict[str, Callable[..., Any]] = {
    "torch._utils._rebuild_tensor": _rebuild_tensor,
    "torch._utils._rebuild_tensor_v2": _rebuild_tensor,
    "torch._utils._rebuild_tensor_v3": _rebuild_tensor,
    "torch._utils._rebuild_parameter": _rebuild_parameter,
    "torch._utils._rebuild_parameter_with_state": _rebuild_parameter,
    "torch._tensor._rebuild_from_type_v2": _rebuild_from_type,
    "torch._utils._rebuild_qtensor": _rebuild_qtensor,
    "torch._utils._rebuild_meta_tensor_no_storage": _rebuild_shape_only("meta"),
    "torch._utils._rebuild_wrapper_subclass": _rebuild_shape_only("subclass"),
}
_SAFE_GLOBALS: dict[str, Any] = {
    "collections.OrderedDict": OrderedDict,
    "builtins.set": set, "builtins.frozenset": frozenset, "builtins.complex": complex,
    "builtins.slice": slice,
    "__builtin__.set": set, "__builtin__.frozenset": frozenset,
}
# Globals pickled by reference, never called: dtypes, storage types, devices.
_NON_CALLABLE_PREFIXES = ("torch.",)


class _SafeUnpickler(pickle.Unpickler):
    def __init__(self, f: BinaryIO, torch_zip: bool) -> None:
        super().__init__(f, encoding="latin1")
        self.torch_zip = torch_zip

    def find_class(self, module: str, name: str) -> Any:
        qualname = f"{module}.{name}"
        if qualname in _TORCH_FUNCS:
            return _TORCH_FUNCS[qualname]
        if qualname in _SAFE_GLOBALS:
            return _SAFE_GLOBALS[qualname]
        if (qualname.startswith(_NON_CALLABLE_PREFIXES) and not name.startswith("_")
                and (name.endswith("Storage") or name[:1].islower())):
            return _Global(qualname)
        return _stub_class(qualname)

    def persistent_load(self, pid: Any) -> Any:
        # ('storage', storage_type, key, location, numel[, view_metadata])
        if isinstance(pid, tuple) and pid and pid[0] in ("storage", b"storage") and len(pid) >= 3:
            numel = pid[4] if len(pid) > 4 and isinstance(pid[4], int) else None
            return _Storage(_storage_dtype(pid[1]), str(pid[2]), numel)
        return _stub_class("persistent_id")(pid)


_MODULE_SLOTS = frozenset({"_parameters", "_buffers", "_modules"})


class _Walker:
    """Turns an unpickled object into tensors plus a JSON tree of everything else."""

    def __init__(self, read_storage: Callable[[_Storage, int, int], bytes | None] | None) -> None:
        self.tensors: list[dict[str, Any]] = []
        self.nodes = 0
        self.read_storage = read_storage
        self.seen: set[int] = set()

    def walk(self, obj: Any, path: str, depth: int = 0) -> Any:
        if isinstance(obj, _Tensor):
            extra: dict[str, Any] = {}
            if obj.kind != "tensor":
                extra["kind"] = obj.kind
            values = self._values(obj)
            if values is not None:
                extra["values"] = values
            self.tensors.append(_tensor(path or "<root>", obj.dtype, obj.shape, **extra))
            return {"__tensor__": len(self.tensors) - 1}
        self.nodes += 1
        if obj is None or isinstance(obj, (bool, int, float, str, bytes, bytearray)):
            return _clip(obj) if self.nodes <= MAX_TREE_NODES else "…"
        if isinstance(obj, (_Global, _Storage)):
            return f"<{obj.qualname if isinstance(obj, _Global) else 'storage ' + obj.dtype}>"
        if depth >= MAX_DEPTH or id(obj) in self.seen:
            return "…"
        self.seen.add(id(obj))
        try:
            return self._container(obj, path, depth)
        finally:
            self.seen.discard(id(obj))

    def _child(self, path: str, key: Any) -> str:
        if key in _MODULE_SLOTS:          # a pickled nn.Module: fc._parameters.weight → fc.weight
            return path
        return f"{path}.{key}" if path else str(key)

    def _container(self, obj: Any, path: str, depth: int) -> Any:
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for i, (k, v) in enumerate(obj.items()):
                key = k if isinstance(k, str) else repr(k) if not isinstance(k, _Stub) else k._qualname
                child = self.walk(v, self._child(path, key), depth + 1)
                if i < MAX_ITEMS or isinstance(child, dict) and "__tensor__" in child:
                    out[str(key)[:MAX_STR]] = child
            if len(obj) > len(out):
                out["…"] = f"{len(obj) - len(out)} more entries"
            return out
        if isinstance(obj, (list, tuple, set, frozenset)):
            seq = list(obj)
            items = [self.walk(v, self._child(path, i), depth + 1) for i, v in enumerate(seq)]
            kept = [c for i, c in enumerate(items)
                    if i < MAX_ITEMS or isinstance(c, dict) and "__tensor__" in c]
            if len(kept) < len(items):
                kept.append(f"… {len(items) - len(kept)} more")
            return kept
        if isinstance(obj, _Stub):
            return self._stub(obj, path, depth)
        return f"<{type(obj).__name__}>"

    def _stub(self, obj: _Stub, path: str, depth: int) -> Any:
        q = obj._qualname
        array = _numpy_array(obj)
        if array is not None:
            self.tensors.append(_tensor(path or "<root>", array[0], array[1], kind="ndarray"))
            return {"__tensor__": len(self.tensors) - 1}
        out: dict[str, Any] = {"__object__": q}
        if obj.args:
            out["args"] = self.walk(list(obj.args), path, depth + 1)
        if obj.kwargs:
            out["kwargs"] = self.walk(obj.kwargs, path, depth + 1)
        if obj.state is not None:
            state = obj.state
            # (dict_state, slot_state) from __reduce_ex__; nn.Module keeps its tensors
            # in _parameters/_buffers, which reads better without those names.
            if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict):
                state = state[0]
            walked = self.walk(state, path, depth + 1)
            if isinstance(walked, dict):
                # Drop a module's empty private slots (_forward_hooks: {} and so on).
                walked = {k: v for k, v in walked.items()
                          if not (k.startswith("_") and v in ({}, [], None))}
            out["state"] = walked
        if obj.items:
            out["items"] = self.walk(obj.items, path, depth + 1)
        if obj.list:
            out["list"] = self.walk(obj.list, path, depth + 1)
        return out

    def _values(self, t: _Tensor) -> list[Any] | None:
        numel = math.prod(t.shape) if t.shape else 1
        if (numel > VALUE_PREVIEW or t.storage is None or self.read_storage is None
                or t.dtype not in _STRUCT and t.dtype != "bfloat16"):
            return None
        size = _ITEMSIZE[t.dtype]
        raw = self.read_storage(t.storage, t.offset * size, numel * size)
        if raw is None or len(raw) != numel * size:
            return None
        if t.dtype == "bfloat16":
            vals: list[Any] = [struct.unpack("<f", b"\0\0" + raw[i:i + 2])[0] for i in range(0, len(raw), 2)]
        else:
            vals = list(struct.unpack(f"<{numel}{_STRUCT[t.dtype]}", raw))
        return [v if not isinstance(v, float) or math.isfinite(v) else str(v) for v in vals]


def _numpy_array(obj: _Stub) -> tuple[str, list[int]] | None:
    """(dtype, shape) of a pickled ndarray stub, else None."""
    if not obj._qualname.endswith("multiarray._reconstruct"):
        return None
    state = obj.state
    if not (isinstance(state, tuple) and len(state) >= 4):
        return None
    dtype = state[2]
    descr = dtype.args[0] if isinstance(dtype, _Stub) and dtype.args else "unknown"
    return _numpy_dtype(descr if not isinstance(descr, str) or descr[:1] in "<>|=" else "<" + descr), _ints(state[1])


def _unpickle(f: BinaryIO, torch_zip: bool = False) -> Any:
    try:
        return _SafeUnpickler(f, torch_zip).load()
    except RecursionError:
        raise InspectError("pickle nests too deeply") from None
    except Exception as exc:
        raise InspectError(f"could not read pickle: {type(exc).__name__}: {exc}"[:300]) from None


def _pickle_result(fmt: str, obj: Any, read_storage: Any = None, **extra: Any) -> dict[str, Any]:
    walker = _Walker(read_storage)
    tree = walker.walk(obj, "")
    return _result(fmt, walker.tensors, tree=tree, tree_truncated=walker.nodes > MAX_TREE_NODES, **extra)


def _torch_zip(zf: zipfile.ZipFile) -> dict[str, Any]:
    names = zf.namelist()
    pkl = next((n for n in names if n.endswith("/data.pkl") and n.count("/") == 1), None)
    if pkl is None:
        raise InspectError("zip archive has no data.pkl")
    prefix = pkl[: -len("data.pkl")]
    info = zf.getinfo(pkl)
    if info.file_size > MAX_PICKLE_BYTES:
        raise InspectError("data.pkl too large to inspect")
    obj = _unpickle(io.BytesIO(zf.read(pkl)), torch_zip=True)
    big_endian = False
    if f"{prefix}byteorder" in names:
        big_endian = zf.read(f"{prefix}byteorder").strip() == b"big"
    torchscript = f"{prefix}constants.pkl" in names or any(n.startswith(f"{prefix}code/") for n in names)

    def read_storage(storage: _Storage, start: int, length: int) -> bytes | None:
        if big_endian:
            return None
        try:
            with zf.open(f"{prefix}data/{storage.key}") as s:
                s.seek(start)
                return s.read(length)
        except (KeyError, OSError, ValueError):
            return None

    version = zf.read(f"{prefix}version").decode("ascii", "replace").strip() if f"{prefix}version" in names else None
    return _pickle_result("torchscript" if torchscript else "pytorch", obj, read_storage,
                          archive_version=version)


_TORCH_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C


def _pickle_file(f: BinaryIO, size: int) -> dict[str, Any]:
    if size > MAX_PICKLE_BYTES:
        # Only a legacy torch file is allowed past the cap: its pickles are small,
        # and the bulk after them is raw storage bytes this never reads.
        try:
            legacy = _unpickle(io.BytesIO(f.read(64))) == _TORCH_LEGACY_MAGIC
        except InspectError:
            legacy = False
        if not legacy:
            raise InspectError("pickle too large to inspect")
        f.seek(0)
    first = _unpickle(f)
    if first != _TORCH_LEGACY_MAGIC:
        return _pickle_result("pickle", first)
    _unpickle(f)                          # protocol version
    sys_info = _unpickle(f)
    obj = _unpickle(f)
    return _pickle_result("pytorch-legacy", obj, None,
                          metadata=_clip(sys_info) if isinstance(sys_info, dict) else None)


# ── NumPy ──────────────────────────────────────────────────────────────────────

def _npy_header(f: IO[bytes]) -> tuple[str, list[int], bool]:
    if f.read(6) != b"\x93NUMPY":
        raise InspectError("not an .npy file")
    major = f.read(2)[:1]
    if major == b"\x01":
        (n,) = struct.unpack("<H", f.read(2))
    else:
        (n,) = struct.unpack("<I", f.read(4))
    if n > MAX_HEADER_BYTES:
        raise InspectError(".npy header too large")
    try:
        header = ast.literal_eval(f.read(n).decode("latin1"))
    except (ValueError, SyntaxError):
        raise InspectError("bad .npy header") from None
    if not isinstance(header, dict):
        raise InspectError("bad .npy header")
    return _numpy_dtype(header.get("descr")), _ints(header.get("shape", ())), bool(header.get("fortran_order"))


def _npy_tensor(name: str, f: IO[bytes]) -> dict[str, Any]:
    dtype, shape, fortran = _npy_header(f)
    extra: dict[str, Any] = {"fortran_order": True} if fortran else {}
    numel = math.prod(shape) if shape else 1
    if numel <= VALUE_PREVIEW and dtype in _STRUCT and not fortran:
        raw = f.read(numel * _ITEMSIZE[dtype])
        if len(raw) == numel * _ITEMSIZE[dtype]:
            extra["values"] = [v if not isinstance(v, float) or math.isfinite(v) else str(v)
                               for v in struct.unpack(f"<{numel}{_STRUCT[dtype]}", raw)]
    return _tensor(name, dtype, shape, **extra)


def _npz(zf: zipfile.ZipFile) -> dict[str, Any]:
    tensors = []
    for info in zf.infolist()[:MAX_TENSORS * 2]:
        if not info.filename.endswith(".npy"):
            continue
        with zf.open(info) as f:
            try:
                tensors.append(_npy_tensor(info.filename[:-4], f))
            except InspectError:
                continue
    return _result("npz", tensors)


# ── GGUF ───────────────────────────────────────────────────────────────────────

_GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
}
_GGML_PLAIN = {"F32": "float32", "F16": "float16", "BF16": "bfloat16", "F64": "float64",
               "I8": "int8", "I16": "int16", "I32": "int32", "I64": "int64"}
_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                10: "<Q", 11: "<q", 12: "<d"}


class _Reader:
    def __init__(self, f: BinaryIO) -> None:
        self.f = f

    def read(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise InspectError("file ends early")
        return b

    def unpack(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]


def _gguf(f: BinaryIO) -> dict[str, Any]:
    r = _Reader(f)
    if r.read(4) != b"GGUF":
        raise InspectError("not a GGUF file")
    version = r.unpack("<I")
    if version < 2:
        raise InspectError(f"GGUF version {version} is not supported")
    n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
    if n_kv > 1_000_000 or n_tensors > 10_000_000:
        raise InspectError("GGUF counts out of range")

    def string() -> str:
        n = r.unpack("<Q")
        if n > MAX_HEADER_BYTES:
            raise InspectError("GGUF string too long")
        return r.read(n).decode("utf-8", "replace")

    def value(vtype: int, keep: bool = True) -> Any:
        if vtype in _GGUF_SCALAR:
            return r.unpack(_GGUF_SCALAR[vtype])
        if vtype == 8:
            s = string()
            return s if keep else None
        if vtype == 9:
            etype, n = r.unpack("<I"), r.unpack("<Q")
            if etype in _GGUF_SCALAR and etype != 9:
                size = struct.calcsize(_GGUF_SCALAR[etype])
                head = [value(etype) for _ in range(min(n, 16))]
                f.seek((n - len(head)) * size, io.SEEK_CUR)
            else:
                head = [value(etype, i < 16) for i in range(n)][:16]
            return {"array": n, "head": _clip(head)}
        raise InspectError(f"unknown GGUF value type {vtype}")

    meta: dict[str, Any] = {}
    for _ in range(n_kv):
        key = string()
        meta[key] = _clip(value(r.unpack("<I")))
    tensors = []
    for _ in range(n_tensors):
        name = string()
        ndim = r.unpack("<I")
        dims = [r.unpack("<Q") for _ in range(min(ndim, 8))]
        gtype = r.unpack("<I")
        r.unpack("<Q")                     # offset
        tname, block, block_bytes = _GGML_TYPES.get(gtype, (f"type{gtype}", 1, 0))
        shape = list(reversed(dims))       # GGUF lists the fastest axis first
        numel = math.prod(shape) if shape else 1
        t = _tensor(name, _GGML_PLAIN.get(tname, tname.lower()), shape)
        t["bytes"] = numel // block * block_bytes if block_bytes else None
        tensors.append(t)
    return _result("gguf", tensors, metadata=meta, gguf_version=version)


# ── ONNX (a minimal protobuf reader) ──────────────────────────────────────────

_ONNX_DTYPE = {1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32",
               7: "int64", 8: "string", 9: "bool", 10: "float16", 11: "float64", 12: "uint32",
               13: "uint64", 14: "complex64", 15: "complex128", 16: "bfloat16",
               17: "float8_e4m3fn", 18: "float8_e4m3fnuz", 19: "float8_e5m2", 20: "float8_e5m2fnuz"}


class _Source:
    """Random access to a file through a small window, so byte-at-a-time reads stay cheap."""

    WINDOW = 1 << 16

    def __init__(self, f: BinaryIO) -> None:
        self.f, self.buf, self.base = f, b"", 0

    def byte(self, pos: int) -> int:
        i = pos - self.base
        if not 0 <= i < len(self.buf):
            self.f.seek(pos)
            self.buf, self.base, i = self.f.read(self.WINDOW), pos, 0
            if not self.buf:
                raise InspectError("ONNX file ends early")
        return self.buf[i]

    def read(self, pos: int, n: int) -> bytes:
        if self.base <= pos and pos + n <= self.base + len(self.buf):
            return self.buf[pos - self.base: pos - self.base + n]
        self.f.seek(pos)
        return self.f.read(n)


class _Proto:
    """Reads protobuf fields from a byte range, skipping past big ones (weights) unread."""

    def __init__(self, src: _Source, start: int, end: int) -> None:
        self.src, self.pos, self.end = src, start, end

    def _varint(self) -> int:
        shift = result = 0
        while True:
            b = self.src.byte(self.pos)
            self.pos += 1
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result
            shift += 7
            if shift > 70:
                raise InspectError("bad varint")

    def fields(self) -> Any:
        """Yield (field, wire, value): ints for varints, (start, end) ranges otherwise."""
        while self.pos < self.end:
            key = self._varint()
            field, wire = key >> 3, key & 7
            if wire == 0:
                yield field, wire, self._varint()
            elif wire == 2:
                n = self._varint()
                start = self.pos
                self.pos += n
                if self.pos > self.end:
                    raise InspectError("ONNX field overruns its message")
                yield field, wire, (start, self.pos)
            elif wire == 1:
                self.pos += 8
            elif wire == 5:
                self.pos += 4
            else:
                raise InspectError(f"unsupported protobuf wire type {wire}")

    def sub(self, rng: tuple[int, int]) -> _Proto:
        return _Proto(self.src, rng[0], rng[1])

    def text(self, rng: tuple[int, int]) -> str:
        return self.src.read(rng[0], min(rng[1] - rng[0], MAX_STR * 4)).decode("utf-8", "replace")


def _packed_ints(p: _Proto, wire: int, value: Any) -> list[int]:
    if wire == 0:
        return [value]
    sub = p.sub(value)
    out = []
    while sub.pos < sub.end:
        out.append(sub._varint())
    return out


def _onnx_tensor(p: _Proto) -> dict[str, Any]:
    dims: list[int] = []
    dtype, name = 0, ""
    for field, wire, v in p.fields():
        if field == 1:
            dims += _packed_ints(p, wire, v)
        elif field == 2 and wire == 0:
            dtype = v
        elif field == 8 and wire == 2:
            name = p.text(v)
    return _tensor(name, _ONNX_DTYPE.get(dtype, f"type{dtype}"), dims)


def _onnx_value_info(p: _Proto) -> dict[str, Any]:
    name, dtype, dims = "", "", []
    for field, wire, v in p.fields():
        if field == 1 and wire == 2:
            name = p.text(v)
        elif field == 2 and wire == 2:
            for tf, tw, tv in p.sub(v).fields():
                if tf != 1 or tw != 2:
                    continue
                tensor_type = p.sub(tv)
                for ef, ew, ev in tensor_type.fields():
                    if ef == 1 and ew == 0:
                        dtype = _ONNX_DTYPE.get(ev, f"type{ev}")
                    elif ef == 2 and ew == 2:
                        for df, dw, dv in p.sub(ev).fields():
                            if df != 1 or dw != 2:
                                continue
                            dim: Any = "?"
                            for xf, xw, xv in p.sub(dv).fields():
                                dim = xv if xf == 1 and xw == 0 else p.text(xv) if xf == 2 and xw == 2 else dim
                            dims.append(dim)
    return {"name": name, "dtype": dtype, "shape": dims}


def _onnx(f: BinaryIO, size: int) -> dict[str, Any]:
    model = _Proto(_Source(f), 0, size)
    meta: dict[str, Any] = {}
    tensors: list[dict[str, Any]] = []
    ops: dict[str, int] = {}
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    opsets: dict[str, int] = {}
    saw_graph = False
    for field, wire, v in model.fields():
        if field == 1 and wire == 0:
            meta["ir_version"] = v
        elif field in (2, 3, 4) and wire == 2:
            meta[{2: "producer_name", 3: "producer_version", 4: "domain"}[field]] = model.text(v)
        elif field == 8 and wire == 2:
            domain, version = "", 0
            for of, ow, ov in model.sub(v).fields():
                domain = model.text(ov) if of == 1 and ow == 2 else domain
                version = ov if of == 2 and ow == 0 else version
            opsets[domain or "ai.onnx"] = version
        elif field == 14 and wire == 2:
            k = val = ""
            for mf, mw, mv in model.sub(v).fields():
                k = model.text(mv) if mf == 1 and mw == 2 else k
                val = model.text(mv) if mf == 2 and mw == 2 else val
            meta[k] = val
        elif field == 7 and wire == 2:
            saw_graph = True
            graph = model.sub(v)
            for gf, gw, gv in graph.fields():
                if gw != 2:
                    continue
                if gf == 1:
                    for nf, nw, nv in graph.sub(gv).fields():
                        if nf == 4 and nw == 2:
                            op = graph.text(nv)
                            ops[op] = ops.get(op, 0) + 1
                elif gf == 2:
                    meta["graph_name"] = graph.text(gv)
                elif gf == 5:
                    tensors.append(_onnx_tensor(graph.sub(gv)))
                elif gf == 11:
                    inputs.append(_onnx_value_info(graph.sub(gv)))
                elif gf == 12:
                    outputs.append(_onnx_value_info(graph.sub(gv)))
    if not saw_graph:
        raise InspectError("no ONNX graph found")
    weights = {t["name"] for t in tensors}
    if opsets:
        meta["opset_import"] = opsets
    return _result("onnx", tensors, metadata=meta,
                   ops=dict(sorted(ops.items(), key=lambda kv: -kv[1])),
                   inputs=[i for i in inputs if i["name"] not in weights][:MAX_ITEMS],
                   outputs=outputs[:MAX_ITEMS])


# ── Dispatch ───────────────────────────────────────────────────────────────────

def inspect_file(path: Path) -> dict[str, Any]:
    """A JSON-ready summary of the model or array file at *path*.

    Raises ``InspectError`` when the contents match no supported format.
    """
    ext = path.suffix.lower()
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(8)
        f.seek(0)
        if head.startswith(b"GGUF"):
            out = _gguf(f)
        elif head.startswith(b"\x93NUMPY"):
            out = _result("npy", [_npy_tensor(path.stem, f)])
        elif head.startswith(b"PK\x03\x04"):
            try:
                zf = zipfile.ZipFile(f)
            except zipfile.BadZipFile:
                raise InspectError("damaged zip archive") from None
            with zf:
                if ext == ".npz" or not any(n.endswith("data.pkl") for n in zf.namelist()[:4096]):
                    out = _npz(zf)
                    if not out["tensor_count"]:
                        raise InspectError("zip archive holds no checkpoint or arrays")
                else:
                    out = _torch_zip(zf)
        elif ext == ".safetensors":
            out = _safetensors(f)
        elif head[:1] == b"\x80" or ext in (".pkl", ".pickle"):
            out = _pickle_file(f, size)
        elif ext == ".onnx":
            out = _onnx(f, size)
        elif (len(head) == 8 and head[:1] != b"\0" and _looks_safetensors(f, size)):
            f.seek(0)
            out = _safetensors(f)
        else:
            raise InspectError("unrecognized file format")
    out["file_bytes"] = size
    return out


def _looks_safetensors(f: BinaryIO, size: int) -> bool:
    raw = f.read(9)
    if len(raw) < 9:
        return False
    (n,) = struct.unpack("<Q", raw[:8])
    return n + 8 <= size and raw[8:9] == b"{"
