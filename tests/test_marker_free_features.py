from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_prediction_table import selected_profiler_features  # noqa: E402
from marker_free_features import extract_marker_free_features  # noqa: E402


class MarkerFreeFeatureTests(unittest.TestCase):
    def test_detects_synthetic_period_without_markers(self) -> None:
        sample_count = 12_000
        period_samples = 200
        timestamps = np.arange(sample_count, dtype=float) * 1e6
        phase = 2 * np.pi * np.arange(sample_count) / period_samples
        signals = {
            "sm_active": 55 + 30 * np.sin(phase),
            "dram_active": 40 + 15 * np.sin(phase + 0.6),
            "kernel_launches": (np.sin(phase) > 0.75).astype(float),
        }

        result = extract_marker_free_features(
            timestamps,
            signals,
            sample_interval_ms=1.0,
            min_period_ms=50.0,
            max_period_ms=500.0,
        )

        self.assertAlmostEqual(
            result["period_detection"]["period_ms"], 200.0, delta=2.0
        )
        self.assertGreater(result["period_detection"]["confidence"], 0.7)
        self.assertNotIn("batch_size", result)
        self.assertNotIn("iteration_count", result)

    def test_prediction_selector_rejects_oracle(self) -> None:
        with self.assertRaises(ValueError):
            selected_profiler_features(
                {"collection_mode": "oracle", "deployment_features": {"value": 1}}
            )

    def test_prediction_selector_only_flattens_deployment_features(self) -> None:
        selected = selected_profiler_features(
            {
                "collection_mode": "deployment",
                "deployment_features": {"period": {"confidence": 0.9}},
                "audit_diagnostics": {"nvtx": {"count": 100}},
                "profiled_training_summary": {"gpu_total_ms": 10},
            }
        )
        self.assertEqual(selected, {"x_profile.period.confidence": 0.9})


if __name__ == "__main__":
    unittest.main()
