from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_EXPERIMENTS,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_RUNS_ROOT,
    load_json,
    load_workload,
    sanitize_name,
)


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)

    if low == high:
        return ordered[low]

    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def select_rows(tensor, start: int, count: int):
    total = tensor.shape[0]

    if start + count <= total:
        return tensor.narrow(0, start, count)

    first = tensor.narrow(0, start, total - start)
    second = tensor.narrow(0, 0, count - first.shape[0])
    import torch

    return torch.cat((first, second), dim=0).pin_memory()


def load_dataset(dataset_root: Path, sequence_length: int) -> tuple[dict, dict]:
    import torch

    manifest_path = dataset_root / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest does not exist: {manifest_path}\n"
            "Run scripts/generate_dataset.py first."
        )

    manifest = load_json(manifest_path)
    entry = manifest.get("files", {}).get(str(sequence_length))

    if entry is None:
        raise ValueError(
            f"Dataset does not contain sequence length {sequence_length}."
        )

    tensor_path = dataset_root / entry["file"]
    dataset = torch.load(tensor_path, map_location="cpu", weights_only=True)

    expected_shape = (int(entry["samples"]), sequence_length)

    for key in ("input_ids", "attention_mask", "labels"):
        if key not in dataset or tuple(dataset[key].shape) != expected_shape:
            raise ValueError(f"Invalid {key} tensor shape in {tensor_path}")

    return dataset, manifest


def create_scaler(torch_module, enabled: bool):
    try:
        return torch_module.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch_module.cuda.amp.GradScaler(enabled=enabled)


class IterationRecorder:
    def __init__(
        self,
        torch_module,
        workload: dict[str, Any],
        output_dir: Path,
        device_id: str,
        run_type: str,
        repeat: int,
    ) -> None:
        self.torch = torch_module
        self.workload = workload
        self.output_dir = output_dir
        self.device_id = device_id
        self.run_type = run_type
        self.repeat = repeat
        self.records: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.window_start_ns: int | None = None
        self.window_end_ns: int | None = None

    def start_window(self) -> None:
        self.torch.cuda.synchronize()
        self.window_start_ns = time.perf_counter_ns()

    def end_window(self) -> None:
        self.torch.cuda.synchronize()
        self.window_end_ns = time.perf_counter_ns()

    def add_step(
        self,
        step: int,
        loss_tensor,
        host_launch_ms: float,
        events: dict[str, Any],
    ) -> None:
        self.records.append(
            {
                "workload_id": self.workload["workload_id"],
                "device_id": self.device_id,
                "run_type": self.run_type,
                "repeat": self.repeat,
                "iteration_id": step,
                "warmup": step <= self.workload["warmup_iters"],
                "micro_batch_size": self.workload["micro_batch_size"],
                "sequence_length": self.workload["sequence_length"],
                "tokens_per_iteration": (
                    self.workload["micro_batch_size"]
                    * self.workload["sequence_length"]
                ),
                "dtype": self.workload["dtype"],
                "loss": None,
                "host_launch_ms": host_launch_ms,
                "gpu_h2d_ms": None,
                "gpu_forward_ms": None,
                "gpu_backward_ms": None,
                "gpu_optimizer_ms": None,
                "gpu_total_ms": None,
            }
        )
        events["loss_tensor"] = loss_tensor.detach()
        self.events.append(events)

    def resolve_events(self) -> None:
        self.torch.cuda.synchronize()

        loss_values = (
            self.torch.stack([events["loss_tensor"] for events in self.events])
            .float()
            .cpu()
            .tolist()
        )

        if any(not math.isfinite(float(value)) for value in loss_values):
            raise FloatingPointError("One or more iterations produced a non-finite loss.")

        for record, events, loss_value in zip(
            self.records, self.events, loss_values, strict=True
        ):
            record["loss"] = float(loss_value)
            record["gpu_h2d_ms"] = events["start"].elapsed_time(events["h2d_end"])
            record["gpu_forward_ms"] = events["h2d_end"].elapsed_time(
                events["forward_end"]
            )
            record["gpu_backward_ms"] = events["forward_end"].elapsed_time(
                events["backward_end"]
            )
            record["gpu_optimizer_ms"] = events["backward_end"].elapsed_time(
                events["optimizer_end"]
            )
            record["gpu_total_ms"] = events["start"].elapsed_time(
                events["optimizer_end"]
            )

    def write(self, peak_vram_bytes: int) -> None:
        self.resolve_events()
        iteration_path = self.output_dir / "iterations.csv"

        with iteration_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.records[0].keys()))
            writer.writeheader()
            writer.writerows(self.records)

        valid = [record for record in self.records if not record["warmup"]]
        window_total_ms = None

        if self.window_start_ns is not None and self.window_end_ns is not None:
            window_total_ms = (self.window_end_ns - self.window_start_ns) / 1e6

        valid_count = len(valid)
        mean_iteration_ms = (
            window_total_ms / valid_count
            if window_total_ms is not None and valid_count > 0
            else None
        )
        tokens_per_iteration = (
            self.workload["micro_batch_size"] * self.workload["sequence_length"]
        )

        def values(name: str) -> list[float]:
            return [float(record[name]) for record in valid if record[name] is not None]

        gpu_total = values("gpu_total_ms")
        summary = {
            "workload_id": self.workload["workload_id"],
            "device_id": self.device_id,
            "run_type": self.run_type,
            "repeat": self.repeat,
            "model_id": self.workload["model_id"],
            "micro_batch_size": self.workload["micro_batch_size"],
            "sequence_length": self.workload["sequence_length"],
            "tokens_per_iteration": tokens_per_iteration,
            "dtype": self.workload["dtype"],
            "gradient_accumulation_steps": self.workload[
                "gradient_accumulation_steps"
            ],
            "total_iters": self.workload["total_iters"],
            "warmup_iters": self.workload["warmup_iters"],
            "measurement_iters": valid_count,
            "steady_window_total_ms": window_total_ms,
            "steady_window_mean_iter_ms": mean_iteration_ms,
            "steady_window_tokens_per_second": (
                tokens_per_iteration * 1000.0 / mean_iteration_ms
                if mean_iteration_ms
                else None
            ),
            "gpu_h2d_mean_ms": mean_or_none(values("gpu_h2d_ms")),
            "gpu_forward_mean_ms": mean_or_none(values("gpu_forward_ms")),
            "gpu_backward_mean_ms": mean_or_none(values("gpu_backward_ms")),
            "gpu_optimizer_mean_ms": mean_or_none(values("gpu_optimizer_ms")),
            "gpu_total_mean_ms": mean_or_none(gpu_total),
            "gpu_total_median_ms": statistics.median(gpu_total) if gpu_total else None,
            "gpu_total_p90_ms": percentile(gpu_total, 0.90),
            "gpu_total_p95_ms": percentile(gpu_total, 0.95),
            "peak_vram_bytes": peak_vram_bytes,
            "peak_vram_mb": peak_vram_bytes / (1024 * 1024),
        }

        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


def write_minimal_summary(
    output_dir: Path,
    workload: dict[str, Any],
    device_id: str,
    run_type: str,
    repeat: int,
    window_total_ms: float | None,
    peak_vram_bytes: int,
    final_loss: float,
) -> None:
    """Write baseline/deployment results without per-iteration instrumentation.

    Baseline timing deliberately uses one synchronization at each edge of the
    complete steady-state window. Deployment runs do not time the window at all;
    they exist only to be observed by an external, marker-free profiler.
    """
    measurement_iters = workload["total_iters"] - workload["warmup_iters"]
    mean_iteration_ms = (
        window_total_ms / measurement_iters
        if window_total_ms is not None and measurement_iters > 0
        else None
    )
    tokens_per_iteration = (
        workload["micro_batch_size"] * workload["sequence_length"]
    )
    summary = {
        "workload_id": workload["workload_id"],
        "device_id": device_id,
        "run_type": run_type,
        "repeat": repeat,
        "model_id": workload["model_id"],
        "micro_batch_size": workload["micro_batch_size"],
        "sequence_length": workload["sequence_length"],
        "tokens_per_iteration": tokens_per_iteration,
        "dtype": workload["dtype"],
        "gradient_accumulation_steps": workload["gradient_accumulation_steps"],
        "total_iters": workload["total_iters"],
        "warmup_iters": workload["warmup_iters"],
        "measurement_iters": measurement_iters,
        "steady_window_total_ms": window_total_ms,
        "steady_window_mean_iter_ms": mean_iteration_ms,
        "steady_window_tokens_per_second": (
            tokens_per_iteration * 1000.0 / mean_iteration_ms
            if mean_iteration_ms
            else None
        ),
        "final_loss": final_loss,
        "instrumentation": "window_only" if run_type == "baseline" else "none",
        "peak_vram_bytes": peak_vram_bytes,
        "peak_vram_mb": peak_vram_bytes / (1024 * 1024),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one fixed-shape Gemma 4 LoRA SFT workload."
    )
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--run-type",
        choices=[
            "baseline",
            "nsys_oracle",
            "nsys_deployment",
            "hpctoolkit_cuda",
            "hpctoolkit_cuda_trace",
        ],
        default="baseline",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--iteration-override",
        type=int,
        default=None,
        help=(
            "Extend a deployment run so an external delayed capture cannot miss "
            "a short synthetic workload; not allowed for baseline/oracle labels."
        ),
    )
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Call cudaProfilerStart/Stop around measured iterations for Nsight capture.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload = load_workload(args.experiments.resolve(), args.workload_id)

    if workload["gradient_accumulation_steps"] != 1:
        raise ValueError("Version 1 requires gradient_accumulation_steps=1")
    if args.iteration_override is not None:
        if args.run_type != "nsys_deployment":
            raise ValueError("--iteration-override is allowed only for nsys_deployment")
        if args.iteration_override <= workload["warmup_iters"]:
            raise ValueError("--iteration-override must exceed warmup_iters")
    execution_workload = dict(workload)
    if args.iteration_override is not None:
        execution_workload["total_iters"] = args.iteration_override

    device_label = sanitize_name(args.device_id)
    run_name = f"{args.run_type}_{args.repeat:02d}"
    output_dir = (
        args.runs_root.resolve() / device_label / workload["workload_id"] / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this workload on a GPU node.")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    if workload["dtype"] == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not provide native BF16 support.")
        torch_dtype = torch.bfloat16
    elif workload["dtype"] == "fp16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {workload['dtype']}")

    dataset, dataset_manifest = load_dataset(
        args.dataset_root.resolve(), workload["sequence_length"]
    )
    dataset = {key: value.pin_memory() for key, value in dataset.items()}
    model_config = load_json(args.model_config.resolve())

    if workload["model_id"] != model_config["model_id"]:
        raise ValueError("Workload model_id does not match model_config.json")

    import peft
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    load_kwargs: dict[str, Any] = {
        "revision": model_config.get("model_revision"),
        "local_files_only": (
            False if args.allow_download else model_config.get("local_files_only", True)
        ),
        "trust_remote_code": model_config.get("trust_remote_code", False),
        "attn_implementation": model_config["attention_implementation"],
        "dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    model = AutoModelForCausalLM.from_pretrained(workload["model_id"], **load_kwargs)
    model.config.use_cache = False
    resolved_model_revision = getattr(model.config, "_commit_hash", None)

    lora = model_config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias=lora["bias"],
            target_modules=list(lora["target_modules"]),
        ),
    )

    if model_config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    model.to(device)
    model.train()

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_config = model_config["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        fused=bool(optimizer_config["fused"]),
    )
    optimizer.zero_grad(set_to_none=True)
    scaler = create_scaler(torch, enabled=workload["dtype"] == "fp16")

    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    properties = torch.cuda.get_device_properties(device)
    metadata = {
        "workload": workload,
        "execution_total_iters": execution_workload["total_iters"],
        "device_id": device_label,
        "device_argument": args.device,
        "run_type": args.run_type,
        "repeat": args.repeat,
        "nvtx": args.nvtx,
        "cuda_profiler_range": args.cuda_profiler_range,
        "instrumentation_policy": (
            "oracle_per_iteration"
            if args.run_type == "nsys_oracle"
            else "window_only"
            if args.run_type == "baseline"
            else "none"
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_multiprocessor_count": properties.multi_processor_count,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
        "model_config": model_config,
        "resolved_model_revision": resolved_model_revision,
        "dataset_manifest": dataset_manifest,
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    oracle_mode = args.run_type == "nsys_oracle"
    if (args.nvtx or args.cuda_profiler_range) and not oracle_mode:
        raise ValueError("NVTX and cudaProfilerApi are allowed only in nsys_oracle runs")
    recorder = (
        IterationRecorder(
            torch_module=torch,
            workload=workload,
            output_dir=output_dir,
            device_id=device_label,
            run_type=args.run_type,
            repeat=args.repeat,
        )
        if oracle_mode
        else None
    )
    torch.cuda.reset_peak_memory_stats(device)
    profiler_range_active = False
    baseline_window_start_ns: int | None = None
    baseline_window_end_ns: int | None = None
    final_loss = None

    try:
        for step in range(1, execution_workload["total_iters"] + 1):
            if step == workload["warmup_iters"] + 1:
                if args.run_type == "baseline":
                    torch.cuda.synchronize()
                    baseline_window_start_ns = time.perf_counter_ns()
                elif recorder is not None:
                    recorder.start_window()
                if oracle_mode and args.cuda_profiler_range:
                    torch.cuda.cudart().cudaProfilerStart()
                    profiler_range_active = True

            batch_start = ((step - 1) * workload["micro_batch_size"]) % dataset[
                "input_ids"
            ].shape[0]
            host_start_ns = time.perf_counter_ns() if oracle_mode else None
            events = None
            if oracle_mode:
                events = {
                    name: torch.cuda.Event(enable_timing=True)
                    for name in (
                        "start",
                        "h2d_end",
                        "forward_end",
                        "backward_end",
                        "optimizer_end",
                    )
                }

            if args.nvtx:
                torch.cuda.nvtx.range_push(f"TRAIN_ITER_{step:04d}")

            if events is not None:
                events["start"].record()
            input_ids = select_rows(
                dataset["input_ids"], batch_start, workload["micro_batch_size"]
            ).to(device=device, dtype=torch.long, non_blocking=True)
            attention_mask = select_rows(
                dataset["attention_mask"], batch_start, workload["micro_batch_size"]
            ).to(device=device, non_blocking=True)
            labels = select_rows(
                dataset["labels"], batch_start, workload["micro_batch_size"]
            ).to(device=device, dtype=torch.long, non_blocking=True)
            if events is not None:
                events["h2d_end"].record()

            with torch.autocast(device_type="cuda", dtype=torch_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss
            if events is not None:
                events["forward_end"].record()

            scaler.scale(loss).backward()
            if events is not None:
                events["backward_end"].record()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(optimizer_config["max_grad_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if events is not None:
                events["optimizer_end"].record()

            if args.nvtx:
                torch.cuda.nvtx.range_pop()

            final_loss = loss.detach()
            if recorder is not None and events is not None and host_start_ns is not None:
                host_launch_ms = (time.perf_counter_ns() - host_start_ns) / 1e6
                recorder.add_step(step, loss, host_launch_ms, events)

            if step == execution_workload["total_iters"]:
                if args.run_type == "baseline":
                    torch.cuda.synchronize()
                    baseline_window_end_ns = time.perf_counter_ns()
                elif recorder is not None:
                    recorder.end_window()
                if profiler_range_active:
                    torch.cuda.cudart().cudaProfilerStop()
                    profiler_range_active = False

        if final_loss is None:
            raise RuntimeError("Training loop did not produce a loss")
        final_loss_value = float(final_loss.float().cpu())
        if not math.isfinite(final_loss_value):
            raise FloatingPointError("Training produced a non-finite final loss")
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        if recorder is not None:
            recorder.write(peak_vram_bytes)
        else:
            window_total_ms = None
            if baseline_window_start_ns is not None and baseline_window_end_ns is not None:
                window_total_ms = (
                    baseline_window_end_ns - baseline_window_start_ns
                ) / 1e6
            write_minimal_summary(
                output_dir,
                execution_workload,
                device_label,
                args.run_type,
                args.repeat,
                window_total_ms,
                peak_vram_bytes,
                final_loss_value,
            )
    except Exception as exception:
        if profiler_range_active:
            try:
                torch.cuda.cudart().cudaProfilerStop()
            except Exception:
                pass
        failure = {
            "workload_id": workload["workload_id"],
            "device_id": device_label,
            "run_type": args.run_type,
            "repeat": args.repeat,
            "exception_type": type(exception).__name__,
            "message": str(exception),
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise

    print(f"Experiment complete: {output_dir}")


if __name__ == "__main__":
    main()
