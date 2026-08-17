from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_EXPERIMENTS,
    DEFAULT_RUNS_ROOT,
    load_json,
    load_workloads,
    parse_workload_ids,
    print_command,
    sanitize_name,
)


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def table_columns(connection: sqlite3.Connection, table: str) -> dict[str, str]:
    rows = connection.execute(
        f"PRAGMA table_info({quote_identifier(table)})"
    ).fetchall()
    return {str(row[1]).lower(): str(row[1]) for row in rows}


def aggregate_duration_tables(
    connection: sqlite3.Connection, tables: list[str], token: str
) -> dict[str, float]:
    count = 0
    total_ns = 0.0
    maximum_ns = 0.0
    total_bytes = 0.0

    for table in tables:
        if token not in table.upper():
            continue

        columns = table_columns(connection, table)
        start = columns.get("start")
        end = columns.get("end")

        if start is None or end is None:
            continue

        byte_column = columns.get("bytes")
        byte_expression = (
            f", COALESCE(SUM({quote_identifier(byte_column)}), 0)"
            if byte_column is not None
            else ", 0"
        )
        query = (
            "SELECT COUNT(*), "
            f"COALESCE(SUM({quote_identifier(end)} - {quote_identifier(start)}), 0), "
            f"COALESCE(MAX({quote_identifier(end)} - {quote_identifier(start)}), 0)"
            f"{byte_expression} FROM {quote_identifier(table)}"
        )
        row = connection.execute(query).fetchone()

        if row is None:
            continue

        count += int(row[0])
        total_ns += float(row[1])
        maximum_ns = max(maximum_ns, float(row[2]))
        total_bytes += float(row[3])

    return {
        "count": float(count),
        "total_ms": total_ns / 1e6,
        "mean_ms": total_ns / count / 1e6 if count else 0.0,
        "max_ms": maximum_ns / 1e6,
        "total_bytes": total_bytes,
    }


def extract_features(database: Path) -> dict[str, Any]:
    connection = sqlite3.connect(database)

    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        table_counts = {
            table: int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {quote_identifier(table)}"
                ).fetchone()[0]
            )
            for table in tables
        }

        return {
            "cuda_kernel": aggregate_duration_tables(
                connection, tables, "CUPTI_ACTIVITY_KIND_KERNEL"
            ),
            "cuda_memcpy": aggregate_duration_tables(
                connection, tables, "CUPTI_ACTIVITY_KIND_MEMCPY"
            ),
            "cuda_runtime_api": aggregate_duration_tables(
                connection, tables, "CUPTI_ACTIVITY_KIND_RUNTIME"
            ),
            "nvtx": aggregate_duration_tables(connection, tables, "NVTX"),
            "metric_table_row_counts": {
                table: count
                for table, count in table_counts.items()
                if "METRIC" in table.upper()
            },
            "all_table_row_counts": table_counts,
        }
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Nsight reports and extract stable numeric features."
    )
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--workloads", default="G01")
    parser.add_argument(
        "--profile-mode", choices=["trace", "gpu-metrics"], default="trace"
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()

    if shutil.which("nsys") is None:
        raise SystemExit("The Nsight Systems CLI 'nsys' was not found.")

    workloads = load_workloads(args.experiments.resolve())
    workload_ids = parse_workload_ids(args.workloads, set(workloads))
    device_label = sanitize_name(args.device_id)
    mode_name = "nsys_trace" if args.profile_mode == "trace" else "nsys_gpu_metrics"

    for workload_id in workload_ids:
        run_dir = (
            args.runs_root.resolve()
            / device_label
            / workload_id
            / f"{mode_name}_{args.repeat:02d}"
        )
        report = run_dir / "profile.nsys-rep"
        database = run_dir / "profile.sqlite"

        if not report.exists():
            raise FileNotFoundError(f"Missing Nsight report: {report}")

        if args.force_export or not database.exists():
            command = [
                "nsys",
                "export",
                "--type",
                "sqlite",
                "--force-overwrite",
                "true",
                "--output",
                str(database),
                str(report),
            ]
            print_command(command)
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise SystemExit(result.returncode)

        features = extract_features(database)
        features["device_id"] = device_label
        features["workload_id"] = workload_id
        features["profile_mode"] = args.profile_mode
        features["repeat"] = args.repeat

        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
            features["profiled_training_summary"] = {
                key: value
                for key, value in summary.items()
                if key.startswith("gpu_") or key.startswith("peak_vram")
            }

        output = run_dir / "profile_features.json"
        output.write_text(json.dumps(features, indent=2), encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()

