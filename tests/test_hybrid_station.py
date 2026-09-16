import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from spatial.evaluate_baselines import FEATURES as BASE_FEATURES
from spatial.hybrid_calibration import apply_calibration, fit_calibration
from spatial.hybrid_contract import create_manifest
from spatial.hybrid_features import FEATURES, add_hybrid_features
from spatial.hybrid_model import HybridStationModel
from spatial.hybrid_training import cross_fitted_base_predictions, train_model
from spatial.register_hybrid_station import validate_bundle
from spatial.search_hybrid_station import configs


class HybridContractTests(unittest.TestCase):
    def test_v2_split_is_disjoint_80_10_10(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "data.csv"; dataset.write_text("fixture")
            frame = pd.DataFrame({"run_id": [str(index) for index in range(100)],
                                  "forecast_init_time": pd.date_range("2025-01-01", periods=100, tz="UTC")})
            manifest = create_manifest(frame, dataset, FEATURES)
            development, calibration, locked = (set(manifest[key]) for key in
                                                 ("development_runs", "calibration_runs", "locked_test_runs"))
            self.assertEqual((len(development), len(calibration), len(locked)), (80, 10, 10))
            self.assertFalse(development & calibration); self.assertFalse(development & locked)
            self.assertFalse(calibration & locked)
            for fold in manifest["folds"]:
                self.assertFalse(set(fold["train_runs"]) & set(fold["validation_runs"]))
                self.assertFalse(locked & (set(fold["train_runs"]) | set(fold["validation_runs"])))

    def test_crossfit_excludes_scored_initialization_groups(self):
        rows = []
        for run in range(10):
            row = {name: 1.0 for name in BASE_FEATURES}
            row.update({"run_id": str(run), "forecast_init_time": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=run)})
            rows.append(row)
        frame = pd.DataFrame(rows)
        class Dummy:
            def predict(self, values): return np.zeros(len(values))
        with patch("spatial.hybrid_training.fit_base_model", return_value=Dummy()):
            prediction, provenance = cross_fitted_base_predictions(frame, frame.run_id, blocks=5)
        self.assertFalse(prediction.isna().any())
        for block in provenance:
            self.assertFalse(set(block["fit_runs"]) & set(block["scored_runs"]))

    def test_crossfit_allows_xgboost_native_missing_values(self):
        rows = []
        for run in range(10):
            row = {name: 1.0 for name in BASE_FEATURES}
            row.update({"run_id": str(run), "forecast_init_time": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=run)})
            rows.append(row)
        frame = pd.DataFrame(rows); frame.loc[0, BASE_FEATURES[0]] = np.nan
        class Dummy:
            def predict(self, values): return np.zeros(len(values))
        with patch("spatial.hybrid_training.fit_base_model", return_value=Dummy()):
            prediction, _ = cross_fitted_base_predictions(frame, frame.run_id, blocks=5)
        self.assertFalse(prediction.isna().any())


class HybridFeatureModelTests(unittest.TestCase):
    def test_features_and_sequence_deltas_are_causal(self):
        frame = pd.DataFrame({
            "run_id": ["1", "1"], "station_id": ["A", "A"], "lead_hour": [4, 5],
            "valid_time": pd.to_datetime(["2026-01-01T16:00Z", "2026-01-01T17:00Z"]),
            "hrrr_temp_c": [10.0, 12.0], "rtma_temp_c": [8.0, 8.0],
            "hrrr_rh": [50.0, 45.0], "rtma_rh": [55.0, 55.0],
            "hrrr_wind_ms": [3.0, 5.0], "rtma_wind_ms": [2.0, 2.0],
            "hrrr_precip_mm": [0.0, 2.0], "physics_fm": [9.0, 8.0],
        })
        result = add_hybrid_features(frame, [8.0, 7.0])
        self.assertTrue(set(FEATURES).issuperset({"incumbent_base_fm", "vpd_kpa", "forecast_emc"}))
        self.assertEqual(result.temp_lead_change_c.tolist(), [0.0, 2.0])
        self.assertEqual(result.rh_lead_change.tolist(), [0.0, -5.0])
        self.assertEqual(result.precip_lead_change_mm.tolist(), [0.0, 2.0])

    def test_ordered_quantiles_cannot_cross(self):
        model = HybridStationModel(len(FEATURES), hidden_size=32, num_layers=2, dropout=0.1)
        prediction = model(torch.randn(16, 12, len(FEATURES)), torch.randn(16, 12))
        self.assertTrue(torch.all(prediction[..., 0] < prediction[..., 1]))
        self.assertTrue(torch.all(prediction[..., 1] < prediction[..., 2]))

    def test_fixed_seed_training_is_deterministic(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(8, 12, len(FEATURES))).astype("float32")
        base = rng.normal(10, 1, size=(8, 12)).astype("float32")
        target = (base + 0.5).astype("float32"); mask = np.ones_like(target, dtype="float32")
        arguments = dict(hidden_size=32, num_layers=1, dropout=0, learning_rate=1e-3,
                         p50_loss="mae", batch_size=4, fixed_epochs=2, epochs=2, seed=417, device="cpu")
        _, first, _ = train_model((x, base, target, mask), None, **arguments)
        _, second, _ = train_model((x, base, target, mask), None, **arguments)
        for key in first["state_dict"]:
            self.assertTrue(torch.equal(first["state_dict"][key], second["state_dict"][key]))

    def test_search_space_has_exactly_24_unique_configs(self):
        values = configs()
        self.assertEqual(len(values), 24)
        self.assertEqual(len({str(sorted(item.items())) for item in values}), 24)


class HybridCalibrationTests(unittest.TestCase):
    def test_bias_is_clipped_and_intervals_are_calibrated(self):
        actual = np.linspace(0, 10, 101)
        quantiles = np.column_stack((actual - 0.1, actual - 3.0, actual + 0.1))
        calibration = fit_calibration(actual, quantiles)
        self.assertEqual(calibration["bias_offset"], 1.5)
        calibrated = apply_calibration(quantiles, calibration)
        self.assertTrue(np.all(calibrated[:, 0] <= calibrated[:, 1]))
        self.assertTrue(np.all(calibrated[:, 1] <= calibrated[:, 2]))
        coverage = np.mean((actual >= calibrated[:, 0]) & (actual <= calibrated[:, 2]))
        self.assertGreaterEqual(coverage, 0.8)


class HybridRegistryTests(unittest.TestCase):
    def test_failed_gate_cannot_validate_for_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "final gate"):
                validate_bundle({"pass": False, "checks": {"mae": False}}, directory)

    def test_partial_bundle_cannot_validate_for_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "asset mismatch"):
                validate_bundle({"pass": True, "checks": {"all": True}}, directory)


if __name__ == "__main__":
    unittest.main()
