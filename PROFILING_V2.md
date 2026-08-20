# Profiling v2: marker-free deployment contract

This document records the profiling redesign introduced after the initial
Gemma workload scaffold. It is intentionally explicit so future users do not
accidentally train on oracle-only information.

## What changed

| Area | v1 | v2 |
|---|---|---|
| Baseline timing | Five CUDA events per iteration | One synchronization at each edge of the complete measured window |
| Deployment capture | `cudaProfilerStart/Stop` and NVTX | External Nsight `delay`/`duration`; no workload marker or profiler API |
| Oracle capture | Mixed with normal feature collection | Separate `nsys_oracle` directory and validation-only contract |
| Profiler features | Whole-trace counts/totals, NVTX, profiled summary | Resampled low-level time series, marker-free period detection, per-cycle features |
| Prediction columns | Metadata and features mixed | `x_*` model inputs, `y_*` target, `audit_*` analysis-only columns |
| Baseline repeats | 3 | 5 recommended; target is the repeat median |

## Three collection modes

### Baseline

Produces ground truth on both source and target GPUs. Baseline mode has no
profiler, NVTX, per-iteration CUDA event, callback, or per-phase timing. It
times all measured iterations as one synchronized wall-clock window.

### Deployment

Produces the only profiler features allowed into the prediction model. Nsight
starts and stops collection externally. The training process receives no NVTX
or CUDA profiler-range flags. The extractor uses GPU metric tables when the
installed Nsight version exports them in a numeric wide format, and always
constructs marker-free kernel/memcpy activity signals as a portable fallback.

### Oracle

Uses NVTX and `cudaProfilerStart/Stop` to expose true iteration boundaries.
Oracle reports are for evaluating period-detection error only. The prediction
table builder rejects oracle feature files.

## Data contract

The generated prediction CSV enforces prefixes:

- `x_profile_*`: marker-free source profiling features.
- `x_hardware_*`: allowed numeric source/target hardware descriptors.
- `y_target_iteration_median_ms`: target GPU ground truth.
- `audit_*`: workload metadata, source runtime, names, ratios, and diagnostics.

Training code must select model inputs with `column.startswith("x_")`. Never
use `audit_*` columns as features. Workload ID, micro-batch size, sequence
length, dtype, token count, source baseline runtime, NVTX, and oracle period
labels are deliberately excluded from `x_*`.

## Migration from v1

Old `baseline_*`, `nsys_trace_*`, and `nsys_gpu_metrics_*` results are not
compatible with v2. Keep them for historical comparison, but recollect the
experiment into `baseline_*`, `nsys_deployment_*`, and optionally
`nsys_oracle_*`. The table builder rejects old baseline summaries rather than
silently mixing contracts.

## Capture-window guidance

The default deployment capture waits 60 seconds and records 30 seconds. Tune
`--delay` so model loading and compilation have completed. The report needs at
least three complete training cycles; slower workloads may require a longer
duration. The synthetic deployment runner defaults to 1000 iterations so the
process remains alive through a delayed capture; this changes repetition count,
not computation inside an iteration. Baseline and oracle counts are never
overridden. The extractor removes five percent from each capture boundary
before period detection.

Run a sensitivity study at 250, 500, and 1000 Hz and at multiple capture
offsets before fixing the paper's final setting. The current default is 1000 Hz
with 1 ms feature resampling.

## HPCToolkit status

`run_hpctoolkit.py` remains available as an optional complementary profile.
HPCToolkit data is not currently part of the v2 deployment feature contract or
the prediction table. Add it only as a separate ablation until a marker-free,
time-aligned feature adapter has been validated.
