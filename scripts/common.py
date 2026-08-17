from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS = PROJECT_ROOT / "configs" / "experiments.csv"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model_config.json"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "synthetic_sft"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_workloads(csv_path: Path) -> dict[str, dict[str, Any]]:
    workloads: dict[str, dict[str, Any]] = {}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            workload_id = row["workload_id"].strip()
            workloads[workload_id] = {
                "workload_id": workload_id,
                "model_id": row["model_id"].strip(),
                "micro_batch_size": int(row["micro_batch_size"]),
                "sequence_length": int(row["sequence_length"]),
                "dtype": row["dtype"].strip().lower(),
                "total_iters": int(row["total_iters"]),
                "warmup_iters": int(row["warmup_iters"]),
                "gradient_accumulation_steps": int(
                    row["gradient_accumulation_steps"]
                ),
            }

    return workloads


def load_workload(csv_path: Path, workload_id: str) -> dict[str, Any]:
    workloads = load_workloads(csv_path)

    if workload_id not in workloads:
        raise ValueError(f"Unknown workload ID: {workload_id}")

    return workloads[workload_id]


def parse_workload_ids(text: str, valid_ids: set[str]) -> list[str]:
    if text.strip().lower() == "all":
        return sorted(valid_ids)

    values = [item.strip() for item in text.split(",") if item.strip()]

    if not values:
        raise ValueError("No workload IDs were specified.")

    unknown = [value for value in values if value not in valid_ids]

    if unknown:
        raise ValueError(f"Unknown workload IDs: {unknown}")

    return values


def sanitize_name(text: str) -> str:
    allowed = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    )
    result = "".join(character if character in allowed else "_" for character in text)
    return result.strip("_") or "unknown"


def print_command(command: list[str]) -> None:
    print(" ".join(f'"{arg}"' if " " in arg else arg for arg in command))


def run_with_log(command: list[str], log_path: Path, cwd: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return process.wait()


def flatten_numeric(data: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numeric(value, child_prefix))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        flattened[prefix] = float(data)

    return flattened

