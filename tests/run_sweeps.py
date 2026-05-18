#!/usr/bin/env python3
"""Run all sweeps in tests/sweeps/ against a running mlsweep manager.

Usage
-----
    python tests/run_sweeps.py --manager http://localhost:7891 --token TOKEN
    python tests/run_sweeps.py --manager http://localhost:7891 --token TOKEN \\
        --skip multigpu torchrun set_dist_env
    python tests/run_sweeps.py --manager http://localhost:7891 --token TOKEN \\
        --only logs integration_grid

Multi-GPU sweeps (multigpu_sweep, torchrun_sweep, set_dist_env_sweep) are
skipped by default because they require 2+ GPUs on the worker.  Pass --all
to include them.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
SWEEPS_DIR = REPO_ROOT / "tests" / "sweeps"

# Skipped by default — require 2+ GPUs.
MULTI_GPU = {"multigpu_sweep", "torchrun_sweep", "set_dist_env_sweep"}


def _print_rule(char="═", width=58):
    print(char * width)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all sweeps in tests/sweeps/ against a running mlsweep manager.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manager", required=True,
                        help="Manager URL, e.g. http://localhost:7891")
    parser.add_argument("--token", required=True,
                        help="Manager auth token")
    parser.add_argument("--skip", nargs="*", default=[], metavar="NAME",
                        help="Additional sweep stems to skip (no .py extension)")
    parser.add_argument("--only", nargs="*", default=None, metavar="NAME",
                        help="Run only these sweep stems; ignores --skip and --all")
    parser.add_argument("--all", action="store_true",
                        help="Include multi-GPU sweeps (skipped by default)")
    parser.add_argument("--stream", action="store_true",
                        help="Pass --stream to mlsweep_run for live job status")
    parser.add_argument("-j", "--jobs-per-gpu", type=int, default=None, metavar="N",
                        help="Pass -j N to every mlsweep_run call")
    args = parser.parse_args()

    # ── Resolve mlsweep_run ───────────────────────────────────────────────────
    runner = shutil.which("mlsweep_run")
    if runner is None:
        print("error: mlsweep_run not found on PATH", file=sys.stderr)
        sys.exit(1)

    # ── Collect sweep files ───────────────────────────────────────────────────
    all_files = sorted(f for f in SWEEPS_DIR.glob("*.py") if not f.stem.startswith("_"))

    if args.only is not None:
        only = set(args.only)
        selected = [f for f in all_files if f.stem in only]
        skipped  = []
    else:
        skip = set(args.skip)
        if not args.all:
            skip |= MULTI_GPU
        selected = [f for f in all_files if f.stem not in skip]
        skipped  = [f for f in all_files if f.stem in skip]

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    _print_rule()
    print(f"  mlsweep sweep runner")
    print(f"  manager : {args.manager}")
    print(f"  runner  : {runner}")
    print(f"  sweeps  : {len(selected)}")
    if skipped:
        print(f"  skipped : {', '.join(f.stem for f in skipped)}")
    _print_rule()

    if not selected:
        print("\nNothing to run.")
        sys.exit(0)

    # ── Run each sweep ────────────────────────────────────────────────────────
    results: list[tuple[str, bool, float]] = []

    for sweep_file in selected:
        name = sweep_file.stem
        print(f"\n{'─' * 58}")
        print(f"  {name}")
        print(f"{'─' * 58}")

        cmd = [runner, "--manager", args.manager, "--token", args.token]
        if args.stream:
            cmd.append("--stream")
        if args.jobs_per_gpu is not None:
            cmd += ["-j", str(args.jobs_per_gpu)]
        cmd.append(str(sweep_file))

        t0 = time.monotonic()
        try:
            proc = subprocess.run(cmd, cwd=REPO_ROOT)
            ok   = proc.returncode == 0
        except Exception as exc:
            print(f"  error: {exc}", file=sys.stderr)
            ok = False
        elapsed = time.monotonic() - t0

        results.append((name, ok, elapsed))
        label = "ok" if ok else "FAILED"
        print(f"\n  [{label}]  {name}  ({elapsed:.1f}s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    _print_rule()
    print(f"  Summary  —  {len(results)} sweep(s) run")
    _print_rule("─")
    passed = failed = 0
    for name, ok, elapsed in results:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {name:<36}  {elapsed:>6.1f}s")
        if ok:
            passed += 1
        else:
            failed += 1
    _print_rule("─")
    print(f"  {passed} passed  ·  {failed} failed")
    _print_rule()
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
