"""mlsweep_export: replay existing experiment data through external writers."""

import argparse
import json
import os
import sys
from typing import Any

from mlsweep._sweep import _load_sweep_status
from mlsweep._writers import MultiWriterFactory, WriterFactory


def main() -> None:
    try:
        parser = argparse.ArgumentParser(
            description="Export existing mlsweep experiment data to wandb/TensorBoard."
        )
        parser.add_argument("experiment_dir", help="Path to experiment output directory")
        parser.add_argument("--wandb-project", default=None, help="W&B project name")
        parser.add_argument("--wandb-entity", default=None, help="W&B entity/team")
        parser.add_argument("--tensorboard-dir", default=None,
                            help="TensorBoard output directory")
        parser.add_argument("--runs", nargs="*", default=None,
                            help="Run names to export (default: all)")
        args = parser.parse_args()

        exp_dir = os.path.abspath(args.experiment_dir)
        manifest_path = os.path.join(exp_dir, "sweep_manifest.json")

        try:
            with open(manifest_path) as f:
                manifest: dict[str, Any] = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error: cannot read sweep_manifest.json: {e}", file=sys.stderr)
            sys.exit(1)

        experiment: str = manifest.get("experiment", os.path.basename(exp_dir))
        runs_list: list[dict[str, Any]] = manifest.get("runs", [])
        dims: list[str] = list(manifest.get("dims", {}).keys())

        status_data = _load_sweep_status(exp_dir)

        # Build factories
        factories: list[WriterFactory] = []
        if args.wandb_project:
            from mlsweep._writer_wandb import WandbWriterFactory
            factories.append(WandbWriterFactory(
                project=args.wandb_project,
                entity=args.wandb_entity,
                resume="allow",
            ))
        if args.tensorboard_dir:
            from mlsweep._writer_tensorboard import TensorBoardWriterFactory
            factories.append(TensorBoardWriterFactory(
                tb_dir=args.tensorboard_dir,
            ))

        if not factories:
            print(
                "Error: specify at least one of --wandb-project or --tensorboard-dir",
                file=sys.stderr,
            )
            sys.exit(1)

        factory: WriterFactory = MultiWriterFactory(factories)

        runs_by_name: dict[str, dict[str, Any]] = {r["name"]: r for r in runs_list}
        run_names = list(runs_by_name)
        if args.runs:
            run_names = [n for n in run_names if n in args.runs]

        factory.on_sweep_start(experiment, dims, run_names)

        for run_name in run_names:
            run_entry = runs_by_name.get(run_name)
            combo: dict[str, Any] = run_entry.get("combo", {}) if run_entry is not None else {}

            run_dir = os.path.join(exp_dir, run_name)
            metrics_path = os.path.join(run_dir, "metrics.jsonl")

            run_status = status_data.get(run_name, {})
            status: str = run_status.get("status", "unknown")
            elapsed = float(run_status.get("elapsed", 0.0))

            writer = factory.make(run_name, combo, run_dir)

            if os.path.isfile(metrics_path):
                with open(metrics_path) as mf:
                    for line in mf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record: dict[str, Any] = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        step = int(record.pop("step", 0))
                        record.pop("t", None)
                        writer.on_metric(step, record)

            writer.on_finish(status, elapsed)
            print(f"  exported {run_name}")

        factory.on_sweep_end()
        print(f"Done. Exported {len(run_names)} runs from {experiment}.")


    except KeyboardInterrupt:
        sys.exit(130)

if __name__ == "__main__":
    main()
