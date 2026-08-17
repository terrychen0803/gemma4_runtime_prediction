from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from common import (
    DEFAULT_EXPERIMENTS,
    DEFAULT_RUNS_ROOT,
    PROJECT_ROOT,
    load_workloads,
    parse_workload_ids,
    print_command,
    run_with_log,
    sanitize_name,
)


TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_gemma4.py"
CHECK_ENV_SCRIPT = PROJECT_ROOT / "scripts" / "check_environment.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect profiler-free Gemma 4 iteration ground truth."
    )
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workloads", default="G01")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    experiments = args.experiments.resolve()
    runs_root = args.runs_root.resolve()
    workloads = load_workloads(experiments)
    workload_ids = parse_workload_ids(args.workloads, set(workloads))
    device_label = sanitize_name(args.device_id)

    environment_command = [
        sys.executable,
        str(CHECK_ENV_SCRIPT),
        "--gpu",
        "--json-output",
        str(runs_root / device_label / "environment.json"),
    ]
    print("[Environment check]")
    print_command(environment_command)

    if not args.dry_run:
        result = subprocess.run(environment_command, check=False)
        if result.returncode != 0:
            raise SystemExit("GPU environment check failed.")

    completed = 0
    skipped = 0

    for workload_id in workload_ids:
        for repeat in range(1, args.repeats + 1):
            run_dir = runs_root / device_label / workload_id / f"baseline_{repeat:02d}"
            summary = run_dir / "summary.json"

            if summary.exists() and not args.force:
                print(f"[SKIP] {workload_id} repeat={repeat}")
                skipped += 1
                continue

            if run_dir.exists() and args.force and not args.dry_run:
                shutil.rmtree(run_dir)

            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--workload-id",
                workload_id,
                "--device-id",
                device_label,
                "--device",
                args.device,
                "--run-type",
                "baseline",
                "--repeat",
                str(repeat),
                "--experiments",
                str(experiments),
                "--runs-root",
                str(runs_root),
            ]

            if args.allow_download:
                command.append("--allow-download")

            print(f"[BASELINE] {workload_id} repeat={repeat}")
            print_command(command)

            if args.dry_run:
                continue

            return_code = run_with_log(command, run_dir / "run.log", PROJECT_ROOT)

            if return_code != 0:
                raise SystemExit(return_code)
            if not summary.exists():
                raise RuntimeError(f"Missing summary: {summary}")

            completed += 1

    print(f"Baseline complete. completed={completed} skipped={skipped}")


if __name__ == "__main__":
    main()

