import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from spatial.station_contract import create_split_manifest, detailed_metrics, load_or_create_manifest
from spatial.station_features import ENGINEERED_FEATURES, FEATURES, LEGACY_FEATURES, add_causal_features
from spatial.station_model import StationSequenceModel


class StationFeatureTests(unittest.TestCase):
    def test_causal_features_are_ordered_and_expected(self):
        frame = pd.DataFrame({
            "valid_time": ["2026-01-01T00:00:00Z", "2026-07-02T12:00:00Z"],
            "hrrr_temp_c": [12.0, 30.0], "rtma_temp_c": [10.0, 25.0],
            "hrrr_rh": [40.0, 30.0], "rtma_rh": [50.0, 35.0],
            "hrrr_wind_ms": [5.0, 8.0], "rtma_wind_ms": [3.0, 6.0],
        })
        result = add_causal_features(frame)
        self.assertEqual(FEATURES, LEGACY_FEATURES + ENGINEERED_FEATURES)
        self.assertAlmostEqual(result.valid_hour_sin.iloc[0], 0.0, places=6)
        self.assertAlmostEqual(result.valid_hour_cos.iloc[0], 1.0, places=6)
        self.assertAlmostEqual(result.valid_hour_cos.iloc[1], -1.0, places=6)
        self.assertEqual(result.temp_change_c.tolist(), [2.0, 5.0])
        self.assertEqual(result.rh_change.tolist(), [-10.0, -5.0])
        self.assertEqual(result.wind_change_ms.tolist(), [2.0, 2.0])

    def test_model_rejects_single_layer_dropout(self):
        with self.assertRaises(ValueError):
            StationSequenceModel(len(FEATURES), hidden_size=16, num_layers=1, dropout=0.1)


class StationSplitTests(unittest.TestCase):
    def test_manifest_has_disjoint_locked_holdout_and_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "station.csv"
            dataset.write_text("fixture")
            dates = pd.date_range("2025-01-01", periods=20, tz="UTC")
            frame = pd.DataFrame({
                "run_id": [f"run-{index}" for index in range(20)],
                "forecast_init_time": dates, "station_id": ["TEST"] * 20,
                "lon": [-92.0] * 20,
            })
            manifest = create_split_manifest(frame, dataset)
            holdout = set(manifest["holdout_runs"])
            self.assertTrue(holdout)
            for fold in manifest["folds"]:
                train, validation = set(fold["train_runs"]), set(fold["validation_runs"])
                self.assertFalse(train & validation)
                self.assertFalse(holdout & (train | validation))

            manifest_path = Path(directory) / "split.json"
            manifest_path.write_text(json.dumps(manifest))
            dataset.write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "dataset_sha256"):
                load_or_create_manifest(manifest_path, frame, dataset)

    def test_detailed_metrics_include_threshold_and_groups(self):
        count = 120
        scored = pd.DataFrame({
            "target_fm": np.r_[np.full(60, 5.0), np.full(60, 10.0)],
            "prediction": np.r_[np.full(60, 5.5), np.full(60, 9.5)],
            "initial_fm": np.full(count, 8.0), "lead_hour": np.tile([4, 5], 60),
            "month": np.ones(count), "station_id": ["TEST"] * count,
        })
        quantiles = np.column_stack((scored.prediction - 1, scored.prediction, scored.prediction + 1))
        report = detailed_metrics(scored, quantiles)
        self.assertIn("rmse", report)
        self.assertIn("critical_low_fm", report["regimes"])
        self.assertEqual(report["critical_threshold"]["support"], 60)
        self.assertEqual(report["quantile_order_violation_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
