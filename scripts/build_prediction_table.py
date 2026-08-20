from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_EXPERIMENTS,
    DEFAULT_RUNS_ROOT,
    PROJECT_ROOT,
    flatten_numeric,
    load_json,
    load_workloads,
    parse_workload_ids,
    sanitize_name,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "prediction" / "rtx5090_to_rtx4090.csv"


def load_baselines(
    runs_root: Path,
    device_id: str,
    workload_id: str,
    repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []

    for repeat in range(1, repeats + 1):
        run_dir = runs_root / device_id / workload_id / f"baseline_{repeat:02d}"
        summary_path = run_dir / "summary.json"
        metadata_path = run_dir / "metadata.json"

        if not summary_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Incomplete baseline run: {run_dir}")

        summary = load_json(summary_path)

        if summary.get("run_type") != "baseline":
            raise ValueError(f"Ground truth must be profiler-free: {summary_path}")
        if summary.get("instrumentation") != "window_only":
            raise ValueError(
                f"Baseline predates the low-instrumentation v2 contract: {summary_path}. "
                "Recollect it with scripts/run_baseline.py."
            )

        summaries.append(summary)
        metadata.append(load_json(metadata_path))

    return summaries, metadata


def metadata_fingerprint(metadata: dict[str, Any]) -> str:
    controlled = {
        "model_config": metadata.get("model_config"),
        "resolved_model_revision": metadata.get("resolved_model_revision"),
        "dataset_manifest": metadata.get("dataset_manifest"),
        "torch_version": metadata.get("torch_version"),
        "torch_cuda_version": metadata.get("torch_cuda_version"),
        "transformers_version": metadata.get("transformers_version"),
        "peft_version": metadata.get("peft_version"),
    }
    payload = json.dumps(controlled, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def aggregate_runtime(summaries: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(item["steady_window_mean_iter_ms"]) for item in summaries]
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "std_ms": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def selected_profiler_features(features: dict[str, Any]) -> dict[str, float]:
    if features.get("collection_mode") != "deployment":
        raise ValueError("Prediction features must come from a deployment profile")
    selected = features.get("deployment_features")
    if not isinstance(selected, dict):
        raise ValueError("Missing marker-free deployment_features")
    # CHANGE NOTE: only x_* columns are legal model inputs. Oracle/NVTX,
    # profiler-run summaries, and workload metadata are never flattened here.
    return flatten_numeric(selected, prefix="x_profile")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pair RTX5090 profiler features with RTX5090 and RTX4090 "
            "profiler-free iteration ground truth."
        )
    )
    parser.add_argument("--source-device", default="RTX5090")
    parser.add_argument("--target-device", default="RTX4090")
    parser.add_argument("--workloads", default="all")
    parser.add_argument("--baseline-repeats", type=int, default=5)
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument(
        "--min-period-confidence",
        type=float,
        default=0.5,
        help="Reject uncertain marker-free period detections instead of forcing a label.",
    )
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-stack-mismatch",
        action="store_true",
        help="Allow source and target package/model/dataset fingerprints to differ.",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    if args.baseline_repeats < 1:
        raise ValueError("--baseline-repeats must be at least 1")
    if not 0.0 <= args.min_period_confidence <= 1.0:
        raise ValueError("--min-period-confidence must be between 0 and 1")

    source_device = sanitize_name(args.source_device)
    target_device = sanitize_name(args.target_device)
    runs_root = args.runs_root.resolve()
    workloads = load_workloads(args.experiments.resolve())
    workload_ids = parse_workload_ids(args.workloads, set(workloads))
    mode_name = "nsys_deployment"
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for workload_id in workload_ids:
        try:
            feature_path = (
                runs_root
                / source_device
                / workload_id
                / f"{mode_name}_{args.profile_repeat:02d}"
                / "profile_features.json"
            )
            if not feature_path.exists():
                raise FileNotFoundError(f"Missing source profiler features: {feature_path}")

            source_summaries, source_metadata = load_baselines(
                runs_root,
                source_device,
                workload_id,
                args.baseline_repeats,
            )
            target_summaries, target_metadata = load_baselines(
                runs_root,
                target_device,
                workload_id,
                args.baseline_repeats,
            )

            source_fingerprints = {
                metadata_fingerprint(item) for item in source_metadata
            }
            target_fingerprints = {
                metadata_fingerprint(item) for item in target_metadata
            }

            if len(source_fingerprints) != 1 or len(target_fingerprints) != 1:
                raise ValueError(f"Stack changed between repeats for {workload_id}")

            if (
                source_fingerprints != target_fingerprints
                and not args.allow_stack_mismatch
            ):
                raise ValueError(
                    f"Source and target controlled-stack fingerprints differ for {workload_id}"
                )

            source_runtime = aggregate_runtime(source_summaries)
            target_runtime = aggregate_runtime(target_summaries)
            workload = workloads[workload_id]
            row: dict[str, Any] = {
                "audit_workload_id": workload_id,
                "audit_source_device_id": source_device,
                "audit_target_device_id": target_device,
                "audit_model_id": workload["model_id"],
                "audit_micro_batch_size": workload["micro_batch_size"],
                "audit_sequence_length": workload["sequence_length"],
                "audit_tokens_per_iteration": (
                    workload["micro_batch_size"] * workload["sequence_length"]
                ),
                "audit_dtype": workload["dtype"],
                "audit_gradient_accumulation_steps": workload[
                    "gradient_accumulation_steps"
                ],
                "audit_source_baseline_mean_ms": source_runtime["mean_ms"],
                "audit_source_baseline_median_ms": source_runtime["median_ms"],
                "audit_source_baseline_std_ms": source_runtime["std_ms"],
                "y_target_iteration_mean_ms": target_runtime["mean_ms"],
                "y_target_iteration_median_ms": target_runtime["median_ms"],
                "audit_target_ground_truth_std_ms": target_runtime["std_ms"],
                "audit_target_to_source_runtime_ratio": (
                    target_runtime["mean_ms"] / source_runtime["mean_ms"]
                ),
                "audit_controlled_stack_fingerprint": next(
                    iter(source_fingerprints)
                ),
                "x_hardware_source_gpu_total_memory_bytes": source_metadata[0].get(
                    "gpu_total_memory_bytes"
                ),
                "x_hardware_source_gpu_multiprocessor_count": source_metadata[0].get(
                    "gpu_multiprocessor_count"
                ),
                "x_hardware_target_gpu_total_memory_bytes": target_metadata[0].get(
                    "gpu_total_memory_bytes"
                ),
                "x_hardware_target_gpu_multiprocessor_count": target_metadata[0].get(
                    "gpu_multiprocessor_count"
                ),
                "audit_source_gpu_name": source_metadata[0].get("gpu_name"),
                "audit_target_gpu_name": target_metadata[0].get("gpu_name"),
            }
            profile_features = load_json(feature_path)
            confidence = float(
                profile_features.get("deployment_features", {})
                .get("period_detection", {})
                .get("confidence", 0.0)
            )
            if confidence < args.min_period_confidence:
                raise ValueError(
                    f"Marker-free period confidence {confidence:.3f} is below "
                    f"{args.min_period_confidence:.3f} for {workload_id}"
                )
            row.update(selected_profiler_features(profile_features))
            rows.append(row)
        except (FileNotFoundError, ValueError) as exception:
            if not args.allow_incomplete:
                raise
            skipped.append({"workload_id": workload_id, "reason": str(exception)})

    if not rows:
        raise SystemExit("No complete workload rows were available.")

    fieldnames = sorted({key for row in rows for key in row})
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "source_device": source_device,
        "target_device": target_device,
        "profile_mode": "deployment_marker_free",
        "baseline_repeats": args.baseline_repeats,
        "min_period_confidence": args.min_period_confidence,
        "rows": len(rows),
        "skipped": skipped,
        "model_input_prefix": "x_",
        "audit_prefix": "audit_",
        "target_column": "y_target_iteration_median_ms",
        "output_csv": str(output),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
