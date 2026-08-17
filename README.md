# Gemma 4 cross-hardware iteration runtime prediction

This project measures deterministic Gemma 4 LoRA SFT workloads for the
following transfer experiment:

1. Collect profiler features on `RTX5090`.
2. Collect profiler-free baseline iteration ground truth on `RTX5090` and
   `RTX4090`.
3. Pair the RTX5090 profiler features with both baseline summaries.
4. Train a predictor whose target is RTX4090 steady-state iteration runtime.

Profiler runs are never used as runtime labels because profiler overhead changes
iteration latency.

## Workload contract

- Model: `google/gemma-4-E2B-it`
- Task: text-only causal language-model SFT
- Fine-tuning: LoRA on attention and MLP projections
- Single GPU only; no CPU/disk offload
- Fixed sequence lengths: 256, 512, 1024
- Micro-batches: 1, 2, 4
- Dtypes: FP16 and BF16
- Gradient accumulation: 1
- Timing: 20 warm-up iterations followed by 100 measured iterations
- Baseline repeats: 3

If a workload does not fit on every device in the comparison set, record it as
unsupported. Do not automatically reduce its batch size.

## Intended collection order

```powershell
# Prepare exactly once, then copy the same model cache and generated dataset
# to both machines.
python scripts/generate_dataset.py --allow-download

# Ground truth without profiler overhead.
python scripts/run_baseline.py --device-id RTX5090 --workloads all --repeats 3
python scripts/run_baseline.py --device-id RTX4090 --workloads all --repeats 3

# Feature collection on RTX5090 only.
python scripts/run_nsys.py --device-id RTX5090 --workloads all --profile-mode trace
python scripts/extract_nsys_features.py --device-id RTX5090 --workloads all

# Build one row per workload for prediction.
python scripts/build_prediction_table.py --source-device RTX5090 --target-device RTX4090
```

BF16 workloads require native BF16 GPU support. Use the same Python, PyTorch,
Transformers, PEFT, CUDA, attention backend, model revision, and dataset hash on
both devices.

