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
from typing import Any


def _visible_devices() -> list[int]:
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
