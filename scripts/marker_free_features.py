from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _percentile_features(values: np.ndarray) -> dict[str, float]:
    result = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    for quantile in (10, 25, 50, 75, 90):
        result[f"p{quantile}"] = float(np.percentile(values, quantile))
    threshold = result["p50"] + result["std"]
    result["high_duty_ratio"] = float(np.mean(values > threshold))
    return result


def _robust_standardize(matrix: np.ndarray) -> np.ndarray:
    median = np.median(matrix, axis=0)
    scale = np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0)
    fallback = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (matrix - median) / scale


def _dominant_component(matrix: np.ndarray) -> np.ndarray:
    normalized = _robust_standardize(matrix)
    _, _, right = np.linalg.svd(normalized, full_matrices=False)
    component = normalized @ right[0]
    component -= np.mean(component)
    return component


def _normalized_autocorrelation(signal: np.ndarray) -> np.ndarray:
    size = len(signal)
    fft_size = 1 << (2 * size - 1).bit_length()
    spectrum = np.fft.rfft(signal, fft_size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_size)[:size]
    overlap = np.arange(size, 0, -1, dtype=float)
    correlation = correlation / overlap
    if correlation[0] <= 1e-12:
        return np.zeros(size, dtype=float)
    return correlation / correlation[0]


def detect_period(
    matrix: np.ndarray,
    sample_interval_ms: float,
    min_period_ms: float,
    max_period_ms: float,
) -> dict[str, Any]:
    component = _dominant_component(matrix)
    correlation = _normalized_autocorrelation(component)
    min_lag = max(2, int(math.ceil(min_period_ms / sample_interval_ms)))
    max_lag = min(
        len(component) // 3,
        int(math.floor(max_period_ms / sample_interval_ms)),
    )
    if max_lag <= min_lag:
        raise ValueError("Capture is too short for the requested period range")

    candidates = [
        lag
        for lag in range(min_lag + 1, max_lag)
        if correlation[lag] >= correlation[lag - 1]
        and correlation[lag] > correlation[lag + 1]
    ]
    if not candidates:
        candidates = [int(np.argmax(correlation[min_lag : max_lag + 1])) + min_lag]

    scored: list[tuple[float, int, float]] = []
    for lag in candidates:
        cycle_count = len(component) // lag
        if cycle_count < 3:
            continue
        cycles = matrix[: cycle_count * lag].reshape(cycle_count, lag, matrix.shape[1])
        template = np.median(cycles, axis=0)
        denominator = float(np.linalg.norm(template)) + 1e-12
        similarities = []
        for cycle in cycles:
            similarities.append(
                1.0
                - min(1.0, float(np.linalg.norm(cycle - template)) / denominator)
            )
        template_score = float(np.mean(similarities))
        score = 0.65 * max(0.0, float(correlation[lag])) + 0.35 * template_score
        scored.append((score, lag, template_score))

    if not scored:
        raise ValueError("Fewer than three complete candidate cycles were captured")
    best_score = max(item[0] for item in scored)
    # Prefer the shortest near-optimal lag so repeated harmonics (2T, 3T, ...)
    # do not replace the fundamental training cycle T.
    score, lag, template_score = min(
        (item for item in scored if item[0] >= best_score - 0.02),
        key=lambda item: item[1],
    )
    return {
        "period_samples": int(lag),
        "period_ms": float(lag * sample_interval_ms),
        "confidence": float(max(0.0, min(1.0, score))),
        "autocorrelation": float(correlation[lag]),
        "template_similarity": template_score,
        "complete_cycles": int(len(component) // lag),
    }


def extract_marker_free_features(
    timestamps_ns: np.ndarray,
    signals: dict[str, np.ndarray],
    sample_interval_ms: float = 1.0,
    min_period_ms: float = 5.0,
    max_period_ms: float = 10_000.0,
) -> dict[str, Any]:
    """Resample low-level profiler signals and summarize detected cycles.

    No workload marker, iteration count, batch size, sequence length, dtype, or
    profiler-run timing summary is accepted by this function.
    """
    if len(timestamps_ns) < 4:
        raise ValueError("At least four timestamped samples are required")
    order = np.argsort(timestamps_ns)
    timestamps = np.asarray(timestamps_ns, dtype=np.float64)[order]
    unique_mask = np.concatenate(([True], np.diff(timestamps) > 0))
    timestamps = timestamps[unique_mask]
    start_ns, end_ns = float(timestamps[0]), float(timestamps[-1])
    step_ns = sample_interval_ms * 1e6
    grid = np.arange(start_ns, end_ns + step_ns * 0.5, step_ns)

    columns: list[np.ndarray] = []
    names: list[str] = []
    for name, raw_values in sorted(signals.items()):
        values = np.asarray(raw_values, dtype=np.float64)[order][unique_mask]
        finite = np.isfinite(values)
        if np.count_nonzero(finite) < 4:
            continue
        interpolated = np.interp(grid, timestamps[finite], values[finite])
        if float(np.std(interpolated)) <= 1e-12:
            continue
        columns.append(interpolated)
        names.append(safe_name(name))
    if not columns:
        raise ValueError("No varying numeric profiler signals were available")

    matrix = np.column_stack(columns)
    # Remove capture boundaries where profiler startup/shutdown often dominates.
    trim = max(1, int(len(grid) * 0.05))
    if len(grid) - 2 * trim >= 30:
        grid = grid[trim:-trim]
        matrix = matrix[trim:-trim]

    period = detect_period(
        matrix, sample_interval_ms, min_period_ms, max_period_ms
    )
    lag = period["period_samples"]
    cycle_count = len(matrix) // lag
    cycle_matrix = matrix[: cycle_count * lag].reshape(
        cycle_count, lag, matrix.shape[1]
    )

    per_signal: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        flattened = cycle_matrix[:, :, index].reshape(-1)
        features = _percentile_features(flattened)
        cycle_means = np.mean(cycle_matrix[:, :, index], axis=1)
        features["cycle_mean_cv"] = float(
            np.std(cycle_means) / (abs(np.mean(cycle_means)) + 1e-12)
        )
        per_signal[name] = features

    correlations: dict[str, float] = {}
    if len(names) >= 2:
        correlation_matrix = np.corrcoef(matrix, rowvar=False)
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                correlations[f"{names[left]}__{names[right]}"] = float(
                    correlation_matrix[left, right]
                )

    return {
        "schema_version": 2,
        "feature_contract": "marker_free_low_level_timeseries_only",
        "capture_duration_ms": float((grid[-1] - grid[0]) / 1e6),
        "sample_interval_ms": float(sample_interval_ms),
        "sample_count": int(len(grid)),
        "signal_count": int(len(names)),
        "signal_names": names,
        "period_detection": period,
        "per_signal": per_signal,
        "cross_signal_correlation": correlations,
    }
