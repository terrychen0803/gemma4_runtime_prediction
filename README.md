# Gemma 4 cross-hardware iteration runtime prediction

This project predicts RTX 4090 steady-state Gemma 4 fine-tuning iteration
runtime from a short RTX 5090 marker-free profiling window plus numeric hardware
descriptors.

> **Profiling v2 change:** profiling is now split into low-instrumentation
> baseline, marker-free deployment, and validation-only oracle modes. Old v1
> results must not be mixed with v2. See [PROFILING_V2.md](PROFILING_V2.md) for
> the full change record and data contract.

## Experiment contract

1. Profile the workload only on RTX 5090 in `deployment` mode.
2. Collect profiler-free ground truth on RTX 5090 and RTX 4090.
3. Detect iteration-like cycles from low-level source time series without
   markers or workload metadata.
4. Predict the RTX 4090 repeat-median iteration runtime.

Profiler runs are never runtime labels because profiling overhead changes
iteration latency. Source baseline runtime is retained as `audit_*` data and is
not a legal model input.

## Workloads

- Model: `google/gemma-4-E2B-it`
- Task: text-only causal language-model SFT
- Fine-tuning: LoRA on attention and MLP projections
- Single GPU; no CPU/disk offload
- Sequence lengths: 256, 512, 1024
- Micro-batches: 1, 2, 4
- Dtypes: FP16 and BF16
- Gradient accumulation: 1
- Timing: 20 warm-up iterations and 100 measured iterations
- Recommended baseline repeats: 5

If a workload does not fit every compared device, mark it unsupported. Do not
silently reduce its batch size.

### Gemma 4 text-only LoRA scope

> **Change note (2026-08-26):** `configs/model_config.json` explicitly excludes
> module paths under `vision_tower` and `audio_tower` from LoRA injection.

Gemma 4 uses projection names such as `q_proj`, `k_proj`, `v_proj`, and
`o_proj` in both the language model and its non-text towers. The latter use
`Gemma4ClippableLinear`, which is not a PEFT-supported LoRA target. Because this
project measures text-only causal-LM SFT, LoRA remains enabled for the language
model's attention and MLP projections while the vision/audio towers stay
frozen. This avoids monkey-patching model internals and preserves a clear,
repeatable workload contract.

If training still reports `Gemma4ClippableLinear is not supported`, verify that
the run uses the repository's updated `configs/model_config.json` and that the
installed PEFT exposes the `exclude_modules` argument on `LoraConfig`.

## Setup

Create an environment and install dependencies on both GPU nodes:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate the dataset once, then copy the identical model cache and generated
dataset to both machines:

```bash
python scripts/generate_dataset.py --allow-download
```

Use the same Python, PyTorch, Transformers, PEFT, CUDA, attention backend,
model revision, model cache, and dataset hash on both nodes.

## 1. Collect baseline ground truth

Run on RTX 5090 and RTX 4090 respectively:

```bash
python scripts/run_baseline.py --device-id RTX5090 --workloads all --repeats 5
python scripts/run_baseline.py --device-id RTX4090 --workloads all --repeats 5
```

Baseline v2 creates no per-iteration CUDA events. It synchronizes only around
the entire measured window. Delete or archive v1 baseline directories before
recollection; the table builder rejects their old summaries.

## 2. Collect marker-free deployment profiles

Run only on RTX 5090:

```bash
python scripts/run_nsys.py \
  --device-id RTX5090 \
  --workloads all \
  --collection-mode deployment \
  --gpu-metrics-frequency 1000 \
  --delay 60 \
  --duration 30 \
  --deployment-iters 1000
```

`--delay` and `--duration` are external Nsight controls. Adjust the delay so
loading/compilation have finished, and make the duration long enough to contain
at least three complete training cycles. Deployment mode passes no NVTX or CUDA
profiler API flag to the training process. `--deployment-iters` extends only
the number of repeated synthetic training steps so a delayed external capture
does not miss a fast workload; it does not change the computation inside an
iteration and is never used for baseline labels.

> **Change note (2026-08-26):** deployment profiling explicitly uses
> `--kill=sigterm`. When the duration expires, Nsight can return `143` or `-15`
> after successfully generating `profile.nsys-rep`. The runner treats those
> codes as an expected deployment stop only when the report exists and is
> non-empty. Oracle runs and every other non-zero return code still fail.

If a previous `--workloads all` run stopped on return code 143 after writing a
report, rerun the same command without `--force`. Completed workload reports are
skipped and collection resumes with the first missing workload.

Extract 1 ms time-series and marker-free cycle features:

```bash
python scripts/extract_nsys_features.py \
  --device-id RTX5090 \
  --workloads all \
  --collection-mode deployment \
  --sample-interval-ms 1
```

Each deployment directory contains:

- `profile.nsys-rep`: raw report.
- `profile.sqlite`: exported database.
- `marker_free_timeseries.csv`: resampled low-level signals.
- `profile_features.json`: period confidence and per-cycle features.
- `profiling_config.json`: exact collection settings and command.

## 3. Optional oracle validation

Oracle mode is not a deployment input. Use it only to compare marker-free
periods with known NVTX iteration boundaries:

```bash
python scripts/run_nsys.py \
  --device-id RTX5090 \
  --workloads all \
  --collection-mode oracle

python scripts/extract_nsys_features.py \
  --device-id RTX5090 \
  --workloads all \
  --collection-mode oracle
```

The table builder explicitly refuses oracle feature files.

## 4. Build the prediction table

After copying both nodes' run directories into one `runs/` tree:

```bash
python scripts/build_prediction_table.py \
  --source-device RTX5090 \
  --target-device RTX4090 \
  --baseline-repeats 5 \
  --min-period-confidence 0.5
```

The CSV column prefixes are enforced as follows:

- `x_*`: legal prediction inputs.
- `y_target_iteration_median_ms`: prediction target.
- `audit_*`: analysis and stratification only; never feed these to the model.

Rows below the configured marker-free period confidence are rejected rather
than silently receiving unreliable cycle features.

## HPCToolkit

Optional HPCToolkit collection remains available:

```bash
python scripts/run_hpctoolkit.py \
  --device-id RTX5090 \
  --workloads all \
  --profile-mode cuda-trace
```

HPCToolkit is currently an auxiliary/ablation source. Its output is not merged
into the v2 deployment prediction table.

## Dry-run checks

Commands can be inspected without a GPU run:

```bash
python scripts/run_baseline.py --device-id RTX5090 --workloads G01 --dry-run
python scripts/run_nsys.py --device-id RTX5090 --workloads G01 --collection-mode deployment --dry-run
python scripts/run_nsys.py --device-id RTX5090 --workloads G01 --collection-mode oracle --dry-run
```
