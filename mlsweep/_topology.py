"""GPU topology discovery and group selection.

Extracted verbatim from run_sweep.py. No logic changes.
_discover_remote_gpus removed — workers now report their own GPUs via MsgWorkerHello.
"""

import functools
import itertools
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any


def visible_devices() -> list[int]:
    """Get list of visible GPU device indices."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "") or os.environ.get("HIP_VISIBLE_DEVICES", "")
    if cvd:
        devs: list[int] = []
        for p in cvd.split(","):
            p = p.strip()
            if "-" in p:
                a, b = p.split("-", 1)
                devs.extend(range(int(a), int(b) + 1))
            else:
                devs.append(int(p))
        return devs
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True)
        return [int(x) for x in r.stdout.strip().splitlines() if x.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    try:
        r = subprocess.run(["amd-smi", "topology", "--json"],
                           capture_output=True, text=True, check=True)
        data = json.loads(r.stdout)
        return [entry["gpu"] for entry in data if "gpu" in entry]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError):
        pass
    return []


# Create and release a CUDA primary context on the single device exposed via
# CUDA_VISIBLE_DEVICES.  This is the exact operation that fails when a GPU is in a
# bad state (e.g. "GPU Recovery Action: Reset" after a fault) even though
# `nvidia-smi` lists it as idle with no processes.  On failure it prints the failing
# call and CUDA error ("cuInit failed: CUDA_ERROR_SYSTEM_NOT_READY (802)") and exits 1.  If
# libcuda is absent (CPU-only / AMD host) it exits 0 so the device is treated as
# usable and nothing is excluded by accident.
_CUDA_PROBE = """\
import ctypes, ctypes.util, sys
name = ctypes.util.find_library("cuda") or "libcuda.so.1"
try:
    lib = ctypes.CDLL(name)
except OSError:
    sys.exit(0)
try:
    retain = lib.cuDevicePrimaryCtxRetain
    release = lib.cuDevicePrimaryCtxRelease
except AttributeError:
    sys.exit(0)
def fail(call, rc):
    s = ctypes.c_char_p()
    try:
        lib.cuGetErrorName(rc, ctypes.byref(s))
    except AttributeError:
        pass
    print("%s failed: %s (%d)" % (call, (s.value or b"CUDA error").decode(), rc), flush=True)
    sys.exit(1)
rc = lib.cuInit(0)
if rc != 0:
    fail("cuInit", rc)
ctx = ctypes.c_void_p()
rc = retain(ctypes.byref(ctx), 0)
if rc != 0:
    fail("context creation", rc)
release(0)
sys.exit(0)
"""

# Default deadline for one probe round.  The driver serialises context creation, so
# N concurrent probes take about N times one probe (~0.7s each on an idle B300,
# several times that on a loaded shared host).
PROBE_TIMEOUT = 60.0


def probe_devices(
    devices: list[int],
    timeout: float = PROBE_TIMEOUT,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    env: Mapping[str, str] | None = None,
) -> dict[int, str | None]:
    """Probe each of *devices* by creating a real CUDA context on it.

    Returns ``{device: None}`` for a usable device and ``{device: reason}`` for one
    that is not, where *reason* names the CUDA error ("cuInit failed:
    CUDA_ERROR_NO_DEVICE (100)") or says the probe timed out.

    Enumerating a device (``nvidia-smi --query-gpu=index``) is not enough: a device
    that has faulted still enumerates and shows 0% utilization, but every CUDA
    context creation on it fails.  A worker that advertises such a device gets jobs
    that die in their first ``.cuda()`` call, spending retries.  Each probe runs in an
    isolated subprocess so the worker process itself never holds a context.  They
    run concurrently against one deadline, so a device that hangs costs *timeout*
    once rather than once per device.

    *popen* starts each probe (like ``subprocess.Popen``), in *env* (default
    this process's environment) with only device *d* visible.
    """
    base = dict(os.environ if env is None else env)
    procs: dict[int, Any] = {}
    for d in devices:
        probe_env = dict(base)
        probe_env["CUDA_VISIBLE_DEVICES"] = str(d)
        probe_env.pop("HIP_VISIBLE_DEVICES", None)
        try:
            procs[d] = popen(
                [sys.executable, "-c", _CUDA_PROBE], env=probe_env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except OSError as e:
            procs[d] = f"probe could not start: {e}"
    deadline = time.monotonic() + timeout
    out: dict[int, str | None] = {}
    for d, proc in procs.items():
        if isinstance(proc, str):
            out[d] = proc
            continue
        try:
            stdout, stderr = proc.communicate(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            out[d] = f"CUDA probe timed out after {timeout:g}s"
            continue
        if proc.returncode == 0:
            out[d] = None
            continue
        detail = stdout.decode(errors="replace").strip()
        if detail:  # "cuInit failed: CUDA_ERROR_... (n)" from the probe itself
            out[d] = detail.splitlines()[-1]
        else:  # the interpreter itself failed; its last stderr line says why
            lines = stderr.decode(errors="replace").strip().splitlines()
            out[d] = f"CUDA probe exited with code {proc.returncode}" + (f": {lines[-1]}" if lines else "")
    return out


def _topo_score(conn_type: str) -> int:
    """Convert an nvidia-smi topology connection type to a numeric score (higher = better)."""
    if conn_type.startswith("NV"):
        try:
            return 100 + int(conn_type[2:])  # NV12/NV18 -> 12, 18, etc.
        except ValueError:
            return 100
    return {"PIX": 50, "PXB": 40, "PHB": 30, "NODE": 20, "SYS": 10}.get(conn_type, 0)


def _parse_topo_output(text: str) -> dict[tuple[int, int], int]:
    """Parse `nvidia-smi topo -m` stdout. Returns {(gpu_a, gpu_b): score}."""
    lines = [l for l in text.splitlines() if l.strip()]
    # Header line starts with whitespace (tab) before GPU0
    col_gpus = None
    for line in lines:
        if "GPU0" in line and not line[0].isalpha():
            col_gpus = [int(m.group(1)) for m in re.finditer(r"GPU(\d+)", line)]
            break
    if not col_gpus:
        return {}
    scores = {}
    for line in lines:
        if not line.startswith("GPU"):
            continue
        parts = line.split()
        try:
            row_gpu = int(parts[0][3:])
        except (ValueError, IndexError):
            continue
        for ci, col_gpu in enumerate(col_gpus):
            if col_gpu == row_gpu or ci + 1 >= len(parts):
                continue
            val = parts[ci + 1]
            if val != "X":
                scores[(row_gpu, col_gpu)] = _topo_score(val)
    return scores


def _amd_topo_score(link_type: str, num_hops: int) -> int:
    """Convert an amd-smi topology link type to a numeric score (higher = better)."""
    if link_type == "XGMI":
        # AMD Infinity Fabric / xGMI: high-speed GPU interconnect analogous to NVLink
        return max(100 - (num_hops - 1) * 10, 50)
    if link_type in ("PCIE", "PCIX"):
        return max(50 - (num_hops - 1) * 10, 10)
    return 10


def _parse_amd_topo_output(text: str) -> dict[tuple[int, int], int]:
    """Parse `amd-smi topology --json` stdout. Returns {(gpu_a, gpu_b): score}."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    scores = {}
    for gpu_entry in data:
        gpu_a = gpu_entry.get("gpu")
        if gpu_a is None:
            continue
        for link in gpu_entry.get("links", []):
            gpu_b = link.get("gpu")
            link_type = link.get("link_type", "")
            num_hops = link.get("num_hops", 1)
            if gpu_b is None or gpu_a == gpu_b or link_type == "SELF":
                continue
            scores[(gpu_a, gpu_b)] = _amd_topo_score(link_type, num_hops)
    return scores


@functools.lru_cache(maxsize=None)
def _gpu_topology() -> dict[tuple[int, int], int]:
    """Query this host's GPU interconnect topology via nvidia-smi or amd-smi.

    Returns {(gpu_a, gpu_b): score} where higher score means better connectivity
    (NVLink/XGMI >> PCIe switch >> PCIe host bridge >> NUMA >> cross-NUMA).
    Falls back to {} if all queries fail.
    """
    # Try nvidia-smi first
    cmd = ["nvidia-smi", "topo", "-m"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return _parse_topo_output(r.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Fall back to amd-smi (AMD GPUs)
    cmd = ["amd-smi", "topology", "--json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return _parse_amd_topo_output(r.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return {}


def _parse_topo_wire(wire_topo: dict[str, int]) -> dict[tuple[int, int], int]:
    """Convert wire-format topology (string keys ``"gpu_a,gpu_b"`` → score)
    to the internal format ``{(gpu_a, gpu_b): score}`` used by ``closest_gpu_group``.
    """
    result: dict[tuple[int, int], int] = {}
    for k, v in wire_topo.items():
        a_str, b_str = k.split(",")
        result[(int(a_str), int(b_str))] = v
    return result


def closest_gpu_group(devices: list[int], group_size: int,
                      topo: dict[tuple[int, int], int]) -> list[int]:
    """Pick *group_size* GPUs from *devices* with the best NVLink/PCIe interconnect.

    Seeds the group with the highest-scoring pair, then greedily adds the GPU
    that scores best against the group so far.  Ties keep *devices* order, so
    with no topology (all scores zero) this is ``devices[:group_size]``.
    The caller guarantees ``len(devices) >= group_size``.
    """
    if group_size <= 1:
        return devices[:group_size]

    def pair_score(a: int, b: int) -> int:
        return topo.get((a, b), 0) + topo.get((b, a), 0)

    # Highest-scoring pair (O(n^2), fine for <=64 GPUs).
    group = list(max(itertools.combinations(devices, 2), key=lambda p: pair_score(*p)))
    remaining = [d for d in devices if d not in group]
    while len(group) < group_size:
        best = max(remaining, key=lambda g: sum(pair_score(g, e) for e in group))
        group.append(best)
        remaining.remove(best)
    return group
