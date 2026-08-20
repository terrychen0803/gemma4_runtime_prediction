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


def get_nsys_version() -> str | None:
    try:
        result = subprocess.run(
            ["nsys", "--version"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or result.stderr.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect oracle or production-like marker-free Nsight profiles."
    )
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workloads", default="G01")
    parser.add_argument(
        "--collection-mode",
        choices=["deployment", "oracle"],
        default="deployment",
        help=(
            "deployment is marker-free and externally bounded; oracle uses NVTX "
            "and cudaProfilerApi only to validate cycle detection"
        ),
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--gpu-metrics-devices", default="0")
    parser.add_argument("--gpu-metrics-frequency", type=int, default=1000)
    parser.add_argument("--gpu-metrics-set", default=None)
    parser.add_argument("--delay", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--deployment-iters",
        type=int,
        default=1000,
        help="Extend only the synthetic deployment run so delayed capture sees steady cycles.",
    )
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.gpu_metrics_frequency < 10:
        raise ValueError("--gpu-metrics-frequency must be at least 10 Hz")
    if args.delay < 0 or args.duration <= 0:
        raise ValueError("--delay must be >= 0 and --duration must be > 0")
    if args.deployment_iters < 1:
        raise ValueError("--deployment-iters must be positive")

    experiments = args.experiments.resolve()
    runs_root = args.runs_root.resolve()
    workloads = load_workloads(experiments)
    workload_ids = parse_workload_ids(args.workloads, set(workloads))
    device_label = sanitize_name(args.device_id)
    mode_name = f"nsys_{args.collection_mode}"

    environment_command = [
        sys.executable,
        str(CHECK_ENV_SCRIPT),
        "--gpu",
        "--json-output",
        str(runs_root / device_label / "environment.json"),
    ]
    print("[Environment check]")
    print_command(environment_command)

    nsys_version = None
    if not args.dry_run:
        if subprocess.run(environment_command, check=False).returncode != 0:
            raise SystemExit("GPU environment check failed.")
        if shutil.which("nsys") is None:
            raise SystemExit("The Nsight Systems CLI 'nsys' was not found.")
        nsys_version = get_nsys_version()

    for workload_id in workload_ids:
        run_dir = runs_root / device_label / workload_id / f"{mode_name}_{args.repeat:02d}"
        report_base = run_dir / "profile"
        report_file = Path(str(report_base) + ".nsys-rep")

        if report_file.exists() and not args.force:
            print(f"[SKIP] {workload_id}: {report_file.name} exists")
            continue
        if run_dir.exists() and args.force and not args.dry_run:
            shutil.rmtree(run_dir)

        command = [
            "nsys",
            "profile",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            f"--gpu-metrics-devices={args.gpu_metrics_devices}",
            f"--gpu-metrics-frequency={args.gpu_metrics_frequency}",
        ]
        if args.gpu_metrics_set:
            command.append(f"--gpu-metrics-set={args.gpu_metrics_set}")

        if args.collection_mode == "oracle":
            command.extend(
                [
                    "--trace=cuda,nvtx",
                    "--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop",
                ]
            )
        else:
            # CHANGE NOTE: deployment capture is externally controlled and does
            # not require any marker/profiler API in the workload.
            command.extend(
                [
                    "--trace=cuda",
                    f"--delay={args.delay}",
                    f"--duration={args.duration}",
                ]
            )

        command.extend(
            [
                "--output",
                str(report_base),
                "--",
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
        if args.collection_mode == "oracle":
            command.extend(["--nvtx", "--cuda-profiler-range"])
        else:
            command.extend(["--iteration-override", str(args.deployment_iters)])
        if args.allow_download:
            command.append("--allow-download")

        print(f"[NSYS-{args.collection_mode.upper()}] {workload_id}")
        print_command(command)
        if args.dry_run:
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "schema_version": 2,
            "device_id": device_label,
            "workload_id": workload_id,
            "collection_mode": args.collection_mode,
            "repeat": args.repeat,
            "nsys_version": nsys_version,
            "trace": ["cuda", "nvtx"] if args.collection_mode == "oracle" else ["cuda"],
            "uses_workload_markers": args.collection_mode == "oracle",
            "capture_control": "cudaProfilerApi" if args.collection_mode == "oracle" else "external_delay_duration",
            "delay_seconds": args.delay if args.collection_mode == "deployment" else None,
            "duration_seconds": args.duration if args.collection_mode == "deployment" else None,
            "deployment_iteration_override": args.deployment_iters if args.collection_mode == "deployment" else None,
            "gpu_metrics_devices": args.gpu_metrics_devices,
            "gpu_metrics_frequency_hz": args.gpu_metrics_frequency,
            "gpu_metrics_set": args.gpu_metrics_set,
            "command": command,
        }
        (run_dir / "profiling_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        return_code = run_with_log(command, run_dir / "nsys.log", PROJECT_ROOT)
        if return_code != 0:
            raise SystemExit(return_code)
        if not report_file.exists():
            raise RuntimeError(f"Nsight report was not generated: {report_file}")

    print("Nsight collection complete.")


if __name__ == "__main__":
    main()
