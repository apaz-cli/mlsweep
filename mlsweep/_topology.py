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


# Try to create and release a CUDA primary context on the single device exposed via
# CUDA_VISIBLE_DEVICES.  This is the exact operation that fails when a GPU is in a
# bad state (e.g. "GPU Recovery Action: Reset" after a fault) even though
# `nvidia-smi` lists it as idle with no processes.  Any non-zero exit (1 = context
# creation failed, 3 = cuInit failed) marks the device unhealthy.  If libcuda is absent (CPU-only / AMD host) it exits 0 so
# the device is treated as usable and nothing is excluded by accident.
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
if lib.cuInit(0) != 0:
    sys.exit(3)
ctx = ctypes.c_void_p()
if retain(ctypes.byref(ctx), 0) != 0:
    sys.exit(1)
release(0)
sys.exit(0)
"""


def probe_usable_devices(
    devices: list[int], timeout: float = 10.0,
) -> tuple[list[int], list[int]]:
    """Split *devices* into ``(usable, unhealthy)`` by creating a real CUDA context.

    Enumerating a device (``nvidia-smi --query-gpu=index``) is not enough: a device
    that has faulted still enumerates and shows 0% utilization, but every CUDA
    context creation on it fails.  A worker that advertises such a device gets jobs
    that die in their first ``.cuda()`` call, spending retries.  Each probe runs in an
    isolated subprocess so the worker process itself never holds a context, and
    all probes run concurrently so the hello waits for the slowest one only.
    """
    procs: dict[int, subprocess.Popen[bytes] | None] = {}
    for d in devices:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(d)
        env.pop("HIP_VISIBLE_DEVICES", None)
        try:
            procs[d] = subprocess.Popen(
                [sys.executable, "-c", _CUDA_PROBE], env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError:
            procs[d] = None
    # The probes are independent, so run them concurrently against one deadline.
    deadline = time.monotonic() + timeout
    usable: list[int] = []
    unhealthy: list[int] = []
    for d, proc in procs.items():
        ok = False
        if proc is not None:
            try:
                ok = proc.wait(max(0.0, deadline - time.monotonic())) == 0
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        (usable if ok else unhealthy).append(d)
    return usable, unhealthy


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
def _gpu_topology(worker: str | None = None) -> dict[tuple[int, int], int]:
    """Query GPU interconnect topology via nvidia-smi or amd-smi.

    Returns {(gpu_a, gpu_b): score} where higher score means better connectivity
    (NVLink/XGMI >> PCIe switch >> PCIe host bridge >> NUMA >> cross-NUMA).
    Falls back to {} if all queries fail.
    worker: None for local; SSH target string for remote.
    """
    ssh_prefix = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", worker] if worker else []

    # Try nvidia-smi first
    cmd = ssh_prefix + ["nvidia-smi", "topo", "-m"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return _parse_topo_output(r.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Fall back to amd-smi (AMD GPUs)
    cmd = ssh_prefix + ["amd-smi", "topology", "--json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return _parse_amd_topo_output(r.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return {}


def _parse_topo_wire(wire_topo: dict[str, int]) -> dict[tuple[int, int], int]:
    """Convert wire-format topology (string keys ``"gpu_a,gpu_b"`` → score)
    to the internal format ``{(gpu_a, gpu_b): score}`` used by ``_best_gpu_groups``.
    """
    result: dict[tuple[int, int], int] = {}
    for k, v in wire_topo.items():
        a_str, b_str = k.split(",")
        result[(int(a_str), int(b_str))] = v
    return result


def _best_gpu_groups(devices: list[int], group_size: int, n_groups: int,
                     worker: str | None = None,
                     topo: dict[tuple[int, int], int] | None = None) -> list[list[int]]:
    """Select n_groups non-overlapping groups of group_size GPUs from devices,
    preferring groups with the best NVLink/PCIe interconnect.

    Uses a greedy algorithm: seeds each group with the highest-scoring pair,
    then expands by adding the GPU that maximises total score to the existing group.
    Falls back to sequential grouping when topology is unavailable (all scores zero).
    worker=None means local; SSH target string for remote topology query.
    topo: pre-supplied topology dict (takes precedence over worker SSH query).
    """
    if group_size == 1:
        return [[d] for d in devices[:n_groups]]

    if topo is None:
        topo = _gpu_topology(worker)

    def pair_score(a: int, b: int) -> int:
        return topo.get((a, b), 0) + topo.get((b, a), 0)

    available = list(devices)
    groups = []

    for _ in range(n_groups):
        if len(available) < group_size:
            break
        # Seed with the highest-scoring pair (O(n^2), fine for ≤64 GPUs)
        best_pair = (available[0], available[1] if len(available) > 1 else available[0])
        best_pair_score = -1
        for a, b in itertools.combinations(available, 2):
            s = pair_score(a, b)
            if s > best_pair_score:
                best_pair_score, best_pair = s, (a, b)
        # Greedily expand to group_size
        group = list(best_pair)
        remaining = [d for d in available if d not in set(group)]
        while len(group) < group_size and remaining:
            best_g = max(remaining, key=lambda g: sum(pair_score(g, e) for e in group))
            group.append(best_g)
            remaining.remove(best_g)

        if len(group) < group_size:
            break
        groups.append(group)
        used = set(group)
        available = [g for g in available if g not in used]

    return groups
