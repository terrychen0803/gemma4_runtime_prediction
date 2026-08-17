from __future__ import annotations

import argparse
import json
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


def resolve_binary(name: str, root: Path | None) -> str | None:
    if root is not None:
        candidate = root / "bin" / name
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect optional HPCToolkit Gemma 4 GPU profiles."
    )
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workloads", default="G01")
    parser.add_argument(
        "--profile-mode", choices=["cuda", "cuda-trace"], default="cuda"
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--hpct-root", type=Path, default=None)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    hpct_root = args.hpct_root.resolve() if args.hpct_root else None
    hpcrun = resolve_binary("hpcrun", hpct_root)

    if hpcrun is None:
        if args.dry_run:
            hpcrun = "hpcrun"
        else:
            raise SystemExit("hpcrun was not found.")

    experiments = args.experiments.resolve()
    runs_root = args.runs_root.resolve()
    workloads = load_workloads(experiments)
    workload_ids = parse_workload_ids(args.workloads, set(workloads))
    device_label = sanitize_name(args.device_id)
    mode_name = (
        "hpctoolkit_cuda"
        if args.profile_mode == "cuda"
        else "hpctoolkit_cuda_trace"
    )

    environment_command = [
        sys.executable,
        str(CHECK_ENV_SCRIPT),
        "--gpu",
        "--json-output",
        str(runs_root / device_label / "environment.json"),
    ]
    print("[Environment check]")
    print_command(environment_command)

    if not args.dry_run and subprocess.run(environment_command, check=False).returncode != 0:
        raise SystemExit("GPU environment check failed.")

    for workload_id in workload_ids:
        run_dir = runs_root / device_label / workload_id / f"{mode_name}_{args.repeat:02d}"
        measurements = run_dir / "hpctoolkit-measurements"
        marker = run_dir / "collection_complete.json"

        if marker.exists() and not args.force:
            print(f"[SKIP] {workload_id}: collection complete")
            continue

        if run_dir.exists() and args.force and not args.dry_run:
            shutil.rmtree(run_dir)

        command = [hpcrun, "-e", "gpu=cuda"]
        if args.profile_mode == "cuda-trace":
            command.append("-t")

        command.extend(
            [
                "-o",
                str(measurements),
                sys.executable,
                str(TRAIN_SCRIPT),
                "--workload-id",
                workload_id,
                "--device-id",
                device_label,
                "--device",
                args.device,
                "--run-type",
                mode_name,
                "--repeat",
                str(args.repeat),
                "--experiments",
                str(experiments),
                "--runs-root",
                str(runs_root),
            ]
        )

        if args.allow_download:
            command.append("--allow-download")

        print(f"[HPCTOOLKIT-{args.profile_mode.upper()}] {workload_id}")
        print_command(command)

        if args.dry_run:
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "device_id": device_label,
            "workload_id": workload_id,
            "profile_mode": args.profile_mode,
            "repeat": args.repeat,
            "hpct_root": str(hpct_root) if hpct_root else None,
            "event": "gpu=cuda",
            "trace": args.profile_mode == "cuda-trace",
            "command": command,
        }
        (run_dir / "profiling_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

        return_code = run_with_log(command, run_dir / "hpcrun.log", PROJECT_ROOT)
        if return_code != 0:
            raise SystemExit(return_code)
        if not measurements.exists():
            raise RuntimeError(f"Missing HPCToolkit measurements: {measurements}")

        marker.write_text(
            json.dumps(
                {
                    "device_id": device_label,
                    "workload_id": workload_id,
                    "profile_mode": args.profile_mode,
                    "repeat": args.repeat,
                    "hpcrun_pass": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("HPCToolkit collection complete.")


if __name__ == "__main__":
    main()

