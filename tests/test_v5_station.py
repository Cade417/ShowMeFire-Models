import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from spatial.register_v5 import validate_registration
from spatial.v5_contract import create_manifest, prospective_runs, reject_forbidden, validate_manifest
from spatial.v5_features import FEATURES, add_v5_features, regime_labels
from spatial.v5_guard import apply_guard, fit_guard, fit_uncertainty, intervals
from spatial.v5_training import base_configs, specialist_configs


def feature_frame():
    leads = [4, 5, 6, 7]
    return pd.DataFrame({
        "run_id": ["run"] * 4, "station_id": ["A"] * 4, "lead_hour": leads,
        "forecast_init_time": pd.to_datetime(["2026-07-01T12:00Z"] * 4),
        "valid_time": pd.to_datetime([f"2026-07-01T{hour}:00Z" for hour in (16, 17, 18, 19)]),
        "target_time": pd.to_datetime([f"2026-07-01T{hour}:00Z" for hour in (16, 17, 18, 19)]),
        "initial_fm": [10.] * 4, "initial_age_hours": [1.] * 4, "physics_fm": [9.] * 4,
        "rtma_temp_c": [25.] * 4, "rtma_rh": [40.] * 4, "rtma_wind_ms": [2.] * 4,
        "hrrr_temp_c": [30., 31., 32., 33.], "hrrr_rh": [45., 60., 50., 40.],
        "hrrr_wind_ms": [3.] * 4, "hrrr_precip_mm": [0., 2., 2., 2.],
        "hrrr_precip_accum_mm": [0., 2., 2., 2.], "hrrr_precip_increment_mm": [0., 2., 0., 0.],
        "precip_interval_hours": [4., 1., 1., 1.], "precip_available": [1.] * 4,
        "precip_reset_flag": [0.] * 4, "precip_partial_window_flag": [1., 0., 0., 0.],
        "lat": [38.] * 4, "lon": [-92.] * 4, "target_fm": [9.] * 4, "target_mask": [1.] * 4,
    })


class V5EvidenceTests(unittest.TestCase):
    def test_exposed_v4_calibration_and_locked_runs_are_forbidden(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "data.csv"; dataset.write_text("fixture")
            frame = pd.DataFrame({"run_id": list(map(str, range(12))),
                                  "forecast_init_time": pd.date_range("2026-01-01", periods=12, tz="UTC")})
            v4 = {"development_runs": list(map(str, range(8))), "calibration_runs": ["8", "9"],
                  "forbidden_v3_test_runs": ["10", "11"],
                  "folds": [{"name": "fold", "train_runs": ["0", "1"], "validation_runs": ["2"]}]}
            manifest = create_manifest(frame, dataset, v4, FEATURES)
            self.assertEqual(set(manifest["forbidden_runs"]), {"8", "9", "10", "11"})
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                reject_forbidden(manifest, ["8"], "search")
            validate_manifest(manifest, frame, dataset, FEATURES)
            appended = pd.concat([frame, pd.DataFrame({"run_id": ["new"],
                "forecast_init_time": [pd.Timestamp("2026-09-01T12:00Z")]})], ignore_index=True)
            self.assertEqual(prospective_runs(manifest, appended), ["new"])

    def test_registration_refuses_development_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "prospective"):
                validate_registration(Path(directory), {"status": "development", "pass": True,
                                                        "beta_registration_allowed": False})


class V5FeatureTests(unittest.TestCase):
    def test_rain_windows_and_regimes_are_causal(self):
        frame = feature_frame(); result = add_v5_features(frame, [8.] * 4, "rain25_scale2")
        self.assertEqual(result.precip_6h_mm.tolist(), [0., 2., 2., 2.])
        self.assertEqual(regime_labels(result).tolist(), ["summer_dry", "active_rain", "post_rain", "post_rain"])
        original = result[FEATURES].to_numpy().copy()
        changed = frame.assign(target_fm=[100.] * 4, target_rh=[1.] * 4, target_wind_ms=[99.] * 4)
        np.testing.assert_allclose(original, add_v5_features(changed, [8.] * 4, "rain25_scale2")[FEATURES], equal_nan=True)

    def test_actual_duration_and_quality_are_explicit(self):
        result = add_v5_features(feature_frame(), [8.] * 4, "rain25_scale2")
        self.assertEqual(result.precip_intensity_mmph.iloc[1], 2.)
        self.assertEqual(result.precip_quality_issue.iloc[0], 1.)
        self.assertEqual(result.precip_quality_issue.iloc[1], 0.)


class V5GuardTests(unittest.TestCase):
    def test_guard_corrects_supported_slice_and_falls_back_elsewhere(self):
        rows = []
        for fold in ("a", "b", "c"):
            for _ in range(150):
                rows.append({"fold": fold, "regime": "summer_dry", "lead_hour": 4.,
                             "actual": 10., "base": 11., "raw_correction": -1.})
                rows.append({"fold": fold, "regime": "other", "lead_hour": 5.,
                             "actual": 10., "base": 10., "raw_correction": 5.})
        scored = pd.DataFrame(rows); guard = fit_guard(scored, minimum_rows=300)
        self.assertGreater(guard["summer_dry|4.0"]["weight"], 0)
        self.assertEqual(guard["other|5.0"]["weight"], 0)
        prediction, weights, _, reasons = apply_guard([11., 10.], [-1., 5.], [4., 5.],
                                                       ["summer_dry", "other"], guard, [True, False])
        self.assertLess(prediction[0], 11.); self.assertEqual(prediction[1], 10.)
        self.assertEqual(weights[1], 0); self.assertEqual(reasons[1], "unavailable_features")

    def test_conformal_intervals_are_ordered(self):
        scored = pd.DataFrame({"actual": np.arange(500.), "base": np.arange(500.) + 1,
                               "raw_correction": [-1.] * 500, "lead_hour": [4.] * 500,
                               "regime": ["summer_dry"] * 500, "fold": ["a"] * 500})
        guard = fit_guard(scored, minimum_rows=100); uncertainty = fit_uncertainty(scored, guard, minimum_rows=100)
        prediction, *_ = apply_guard(scored.base, scored.raw_correction, scored.lead_hour, scored.regime, guard)
        quantiles = intervals(prediction, scored.regime, uncertainty)
        self.assertTrue(np.all(np.diff(quantiles, axis=1) >= 0))

    def test_search_spaces_are_controlled(self):
        self.assertEqual(len(base_configs()), 16); self.assertEqual(len(specialist_configs()), 8)


if __name__ == "__main__": unittest.main()
