from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_EXPERIMENTS,
    DEFAULT_RUNS_ROOT,
    load_workloads,
    parse_workload_ids,
    print_command,
    sanitize_name,
)
from marker_free_features import extract_marker_free_features, safe_name


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def table_columns(connection: sqlite3.Connection, table: str) -> dict[str, str]:
    rows = connection.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    return {str(row[1]).lower(): str(row[1]) for row in rows}


def all_tables(connection: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]


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
        start, end = columns.get("start"), columns.get("end")
        if start is None or end is None:
            continue
        byte_column = columns.get("bytes")
        byte_expression = (
            f", COALESCE(SUM({quote_identifier(byte_column)}), 0)"
            if byte_column
            else ", 0"
        )
        query = (
            "SELECT COUNT(*), "
            f"COALESCE(SUM({quote_identifier(end)}-{quote_identifier(start)}),0), "
            f"COALESCE(MAX({quote_identifier(end)}-{quote_identifier(start)}),0)"
            f"{byte_expression} FROM {quote_identifier(table)}"
        )
        row = connection.execute(query).fetchone()
        if row:
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


def interval_rows(
    connection: sqlite3.Connection, tables: list[str], token: str
) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    for table in tables:
        if token not in table.upper():
            continue
        columns = table_columns(connection, table)
        start, end = columns.get("start"), columns.get("end")
        if not start or not end:
            continue
        byte_column = columns.get("bytes")
        byte_expression = quote_identifier(byte_column) if byte_column else "0"
        query = (
            f"SELECT {quote_identifier(start)}, {quote_identifier(end)}, "
            f"{byte_expression} FROM {quote_identifier(table)} ORDER BY "
            f"{quote_identifier(start)}"
        )
        for start_ns, end_ns, byte_count in connection.execute(query):
            rows.append((float(start_ns), float(end_ns), float(byte_count or 0)))
    return rows


def add_intervals_to_grid(
    rows: list[tuple[float, float, float]],
    grid_start_ns: float,
    sample_count: int,
    step_ns: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.zeros(sample_count, dtype=float)
    starts = np.zeros(sample_count, dtype=float)
    byte_counts = np.zeros(sample_count, dtype=float)
    for start_ns, end_ns, bytes_value in rows:
        if end_ns <= start_ns:
            continue
        first = max(0, int((start_ns - grid_start_ns) // step_ns))
        last = min(sample_count - 1, int((end_ns - grid_start_ns) // step_ns))
        if first >= sample_count or last < 0:
            continue
        starts[first] += 1.0
        byte_counts[first] += bytes_value
        for index in range(first, last + 1):
            bin_start = grid_start_ns + index * step_ns
            overlap = max(0.0, min(end_ns, bin_start + step_ns) - max(start_ns, bin_start))
            active[index] += overlap / step_ns
    return active, starts, byte_counts


def metric_table_signals(
    connection: sqlite3.Connection,
    tables: list[str],
    grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Best-effort adapter for Nsight versions exporting wide metric tables."""
    result: dict[str, np.ndarray] = {}
    timestamp_candidates = ("timestamp", "time", "start")
    excluded = {"id", "typeid", "deviceid", "contextid", "streamid", "end"}
    for table in tables:
        if "METRIC" not in table.upper():
            continue
        columns = table_columns(connection, table)
        timestamp = next((columns.get(name) for name in timestamp_candidates if columns.get(name)), None)
        if timestamp is None:
            continue
        for lowered, original in columns.items():
            if original == timestamp or lowered in excluded or lowered.endswith("id"):
                continue
            query = (
                f"SELECT {quote_identifier(timestamp)}, {quote_identifier(original)} "
                f"FROM {quote_identifier(table)} WHERE {quote_identifier(original)} "
                "IS NOT NULL ORDER BY 1"
            )
            samples: list[tuple[float, float]] = []
            try:
                for timestamp_value, metric_value in connection.execute(query):
                    samples.append((float(timestamp_value), float(metric_value)))
            except (TypeError, ValueError, sqlite3.Error):
                continue
            if len(samples) < 4:
                continue
            times = np.asarray([item[0] for item in samples])
            values = np.asarray([item[1] for item in samples])
            if np.std(values) <= 1e-12:
                continue
            result[f"metric_{safe_name(table)}_{safe_name(original)}"] = np.interp(
                grid, times, values
            )
    return result


def extract_timeseries(
    connection: sqlite3.Connection, sample_interval_ms: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    tables = all_tables(connection)
    kernels = interval_rows(connection, tables, "CUPTI_ACTIVITY_KIND_KERNEL")
    memcopies = interval_rows(connection, tables, "CUPTI_ACTIVITY_KIND_MEMCPY")
    combined = kernels + memcopies
    if not combined:
        raise ValueError("No CUDA kernel or memcpy intervals were found in the report")
    start_ns = min(row[0] for row in combined)
    end_ns = max(row[1] for row in combined)
    step_ns = sample_interval_ms * 1e6
    sample_count = int(np.ceil((end_ns - start_ns) / step_ns)) + 1
    grid = start_ns + np.arange(sample_count, dtype=float) * step_ns

    kernel_active, kernel_starts, _ = add_intervals_to_grid(
        kernels, start_ns, sample_count, step_ns
    )
    memcpy_active, memcpy_starts, memcpy_bytes = add_intervals_to_grid(
        memcopies, start_ns, sample_count, step_ns
    )
    signals = {
        "cuda_kernel_active_fraction": kernel_active,
        "cuda_kernel_launches": kernel_starts,
        "cuda_memcpy_active_fraction": memcpy_active,
        "cuda_memcpy_starts": memcpy_starts,
        "cuda_memcpy_bytes": memcpy_bytes,
    }
    signals.update(metric_table_signals(connection, tables, grid))
    return grid, signals


def write_timeseries_csv(
    path: Path, timestamps: np.ndarray, signals: dict[str, np.ndarray]
) -> None:
    names = sorted(signals)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp_ns", *names])
        for index, timestamp in enumerate(timestamps):
            writer.writerow([int(timestamp), *(signals[name][index] for name in names)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract marker-free cycle features from Nsight profiles."
    )
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--workloads", default="G01")
    parser.add_argument(
        "--collection-mode", choices=["deployment", "oracle"], default="deployment"
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--sample-interval-ms", type=float, default=1.0)
    parser.add_argument("--min-period-ms", type=float, default=5.0)
    parser.add_argument("--max-period-ms", type=float, default=10_000.0)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()

    if shutil.which("nsys") is None:
        raise SystemExit("The Nsight Systems CLI 'nsys' was not found.")
    if args.sample_interval_ms <= 0:
        raise ValueError("--sample-interval-ms must be positive")

    workloads = load_workloads(args.experiments.resolve())
    workload_ids = parse_workload_ids(args.workloads, set(workloads))
    device_label = sanitize_name(args.device_id)
    mode_name = f"nsys_{args.collection_mode}"

    for workload_id in workload_ids:
        run_dir = args.runs_root.resolve() / device_label / workload_id / f"{mode_name}_{args.repeat:02d}"
        report = run_dir / "profile.nsys-rep"
        database = run_dir / "profile.sqlite"
        if not report.exists():
            raise FileNotFoundError(f"Missing Nsight report: {report}")
        if args.force_export or not database.exists():
            command = [
                "nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                "--output", str(database), str(report),
            ]
            print_command(command)
            if subprocess.run(command, check=False).returncode != 0:
                raise SystemExit("Nsight SQLite export failed")

        connection = sqlite3.connect(database)
        try:
            tables = all_tables(connection)
            diagnostics = {
                "cuda_kernel": aggregate_duration_tables(connection, tables, "CUPTI_ACTIVITY_KIND_KERNEL"),
                "cuda_memcpy": aggregate_duration_tables(connection, tables, "CUPTI_ACTIVITY_KIND_MEMCPY"),
                "cuda_runtime_api": aggregate_duration_tables(connection, tables, "CUPTI_ACTIVITY_KIND_RUNTIME"),
                "nvtx": aggregate_duration_tables(connection, tables, "NVTX"),
            }
            timestamps, signals = extract_timeseries(connection, args.sample_interval_ms)
        finally:
            connection.close()

        deployment_features = extract_marker_free_features(
            timestamps,
            signals,
            sample_interval_ms=args.sample_interval_ms,
            min_period_ms=args.min_period_ms,
            max_period_ms=args.max_period_ms,
        )
        features: dict[str, Any] = {
            "schema_version": 2,
            "device_id": device_label,
            "workload_id": workload_id,
            "collection_mode": args.collection_mode,
            "repeat": args.repeat,
            "deployment_features": deployment_features,
            # Diagnostics are saved for audit only. build_prediction_table.py
            # intentionally refuses to include them as model inputs.
            "audit_diagnostics": diagnostics,
        }
        write_timeseries_csv(run_dir / "marker_free_timeseries.csv", timestamps, signals)
        output = run_dir / "profile_features.json"
        output.write_text(json.dumps(features, indent=2), encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
