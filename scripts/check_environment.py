from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STACK_FILE = PROJECT_ROOT / "configs" / "software_stack.json"


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip() or result.stderr.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Gemma 4 experiment stack.")
    parser.add_argument("--gpu", action="store_true", help="Require a usable CUDA GPU.")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optionally save the detected environment as JSON.",
    )
    args = parser.parse_args()

    expected = json.loads(STACK_FILE.read_text(encoding="utf-8-sig"))
    actual_python = ".".join(platform.python_version().split(".")[:2])
    python_ok = actual_python == expected["python_major_minor"]

    packages: dict[str, str | None] = {}
    package_ok = True

    for package, expected_version in expected["expected_versions"].items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
        except Exception:
            version = None

        packages[package] = version

        if version is None or (
            expected_version is not None and version != expected_version
        ):
            package_ok = False

    cuda_available = False
    cuda_version = None
    gpu_name = None
    gpu_capability = None
    gpu_total_memory_bytes = None
    gpu_multiprocessor_count = None

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_capability = list(torch.cuda.get_device_capability(0))
            properties = torch.cuda.get_device_properties(0)
            gpu_total_memory_bytes = properties.total_memory
            gpu_multiprocessor_count = properties.multi_processor_count
    except Exception:
        pass

    record = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_target_match": python_ok,
        "packages": packages,
        "cuda_available": cuda_available,
        "torch_cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "gpu_compute_capability": gpu_capability,
        "gpu_total_memory_bytes": gpu_total_memory_bytes,
        "gpu_multiprocessor_count": gpu_multiprocessor_count,
        "nvidia_smi": command_output(["nvidia-smi"]),
        "nvidia_smi_query": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,power.limit,clocks.max.sm,clocks.max.memory,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nsys_version": command_output(["nsys", "--version"]),
    }

    print(json.dumps(record, indent=2))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(record, indent=2), encoding="utf-8")

    passed = python_ok and package_ok and (cuda_available if args.gpu else True)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

