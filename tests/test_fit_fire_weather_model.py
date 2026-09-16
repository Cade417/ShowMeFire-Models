import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fire_weather_ml import model_bundle
from fire_weather_ml.features import FEATURE_COLUMNS


def _synthetic_panel(n_rows: int = 200, seed: int = 0, with_null_labels: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {column: rng.uniform(0, 30, size=n_rows) for column in FEATURE_COLUMNS}
    frame = pd.DataFrame(data)
    # A simple, learnable synthetic relationship - not physically meaningful,
    # just something a GBM can genuinely fit better than a mean predictor.
    frame[model_bundle.DEFAULT_LABEL_COLUMN] = (
        frame["wind_ms"] * 1.5 + frame["kbdi"] * 0.2 - frame["fm10_pct"] * 0.8 + rng.normal(0, 0.5, size=n_rows)
    ).clip(lower=0.0)
    if with_null_labels:
        frame.loc[frame.index[:5], model_bundle.DEFAULT_LABEL_COLUMN] = np.nan
    return frame


class FitTests(unittest.TestCase):
    def test_fits_and_returns_expected_bundle_keys(self):
        bundle = model_bundle.fit(_synthetic_panel())
        for key in ("model", "feature_columns", "label_column", "xgb_params", "training_row_count", "feature_ranges"):
            self.assertIn(key, bundle)
        self.assertEqual(bundle["training_row_count"], 200)

    def test_feature_ranges_match_the_real_training_data(self):
        panel = _synthetic_panel()
        bundle = model_bundle.fit(panel)
        for column in model_bundle.MODEL_FEATURE_COLUMNS:
            self.assertAlmostEqual(bundle["feature_ranges"][column]["min"], panel[column].min(), places=6)
            self.assertAlmostEqual(bundle["feature_ranges"][column]["max"], panel[column].max(), places=6)

    def test_drops_null_label_rows_before_fitting(self):
        bundle = model_bundle.fit(_synthetic_panel(with_null_labels=True))
        self.assertEqual(bundle["training_row_count"], 195)
        self.assertEqual(bundle["dropped_null_label_rows"], 5)

    def test_raises_when_every_label_is_null(self):
        panel = _synthetic_panel()
        panel[model_bundle.DEFAULT_LABEL_COLUMN] = np.nan
        with self.assertRaises(ValueError):
            model_bundle.fit(panel)

    def test_raises_on_missing_feature_column(self):
        # kbdi/gdd_accum aren't in MODEL_FEATURE_COLUMNS (fit()'s default) -
        # drop one that actually is, so this test still exercises the
        # missing-column check it's meant to.
        panel = _synthetic_panel().drop(columns=["fm10_pct"])
        with self.assertRaises(ValueError):
            model_bundle.fit(panel)


class ScoreTests(unittest.TestCase):
    def test_returns_a_prediction_per_row(self):
        train = _synthetic_panel(seed=1)
        bundle = model_bundle.fit(train)
        test = _synthetic_panel(seed=2, n_rows=50)
        predictions = model_bundle.score(test, bundle)
        self.assertEqual(len(predictions), 50)
        self.assertTrue(np.isfinite(predictions).all())

    def test_a_well_fit_model_beats_predicting_the_training_mean(self):
        train = _synthetic_panel(seed=1, n_rows=400)
        bundle = model_bundle.fit(train)
        test = _synthetic_panel(seed=2, n_rows=100)
        predictions = model_bundle.score(test, bundle)
        truth = test[model_bundle.DEFAULT_LABEL_COLUMN]
        model_mae = float(np.mean(np.abs(truth - predictions)))
        naive_mae = float(np.mean(np.abs(truth - train[model_bundle.DEFAULT_LABEL_COLUMN].mean())))
        self.assertLess(model_mae, naive_mae)


class CalibrateRiskScoreTests(unittest.TestCase):
    def test_percentile_table_spans_0_to_100(self):
        rng = np.random.default_rng(0)
        predictions = rng.exponential(scale=0.5, size=1000)  # right-skewed, like the real distribution
        calibration = model_bundle.calibrate_risk_score(predictions)
        self.assertEqual(calibration["percentiles"][0], 0)
        self.assertEqual(calibration["percentiles"][-1], 100)
        self.assertEqual(len(calibration["values_ch_per_h"]), 101)
        self.assertEqual(calibration["sample_size"], 1000)

    def test_raises_when_nothing_is_finite(self):
        with self.assertRaises(ValueError):
            model_bundle.calibrate_risk_score(np.array([np.nan, np.nan]))

    def test_a_skewed_distribution_still_spreads_evenly_across_0_100(self):
        # The real motivation: even though raw ch/h values are heavily
        # right-skewed (most near zero, a long thin tail), the median raw
        # value must map close to score 50 - percentile rank is even by
        # construction, regardless of the raw distribution's shape.
        rng = np.random.default_rng(1)
        predictions = rng.exponential(scale=0.5, size=5000)
        calibration = model_bundle.calibrate_risk_score(predictions)
        median_score = model_bundle.risk_score_0_100(np.median(predictions), calibration)
        self.assertAlmostEqual(float(median_score), 50.0, delta=2.0)


class RiskScore0100Tests(unittest.TestCase):
    def test_endpoints_map_to_0_and_100(self):
        calibration = model_bundle.calibrate_risk_score(np.array([0.0, 1.0, 2.0, 3.0, 10.0]))
        self.assertAlmostEqual(float(model_bundle.risk_score_0_100(0.0, calibration)), 0.0, places=3)
        self.assertAlmostEqual(float(model_bundle.risk_score_0_100(10.0, calibration)), 100.0, places=3)

    def test_negative_input_is_clipped_to_zero_score(self):
        calibration = model_bundle.calibrate_risk_score(np.array([0.0, 1.0, 2.0]))
        self.assertAlmostEqual(float(model_bundle.risk_score_0_100(-5.0, calibration)), 0.0, places=3)

    def test_accepts_an_array(self):
        calibration = model_bundle.calibrate_risk_score(np.array([0.0, 1.0, 2.0, 3.0, 10.0]))
        scores = model_bundle.risk_score_0_100(np.array([0.0, 10.0]), calibration)
        np.testing.assert_allclose(scores, [0.0, 100.0], atol=1e-3)


class SaveLoadRoundTripTests(unittest.TestCase):
    def test_save_then_load_produces_matching_predictions(self):
        train = _synthetic_panel(seed=3)
        bundle = model_bundle.fit(train)
        bundle["risk_calibration"] = model_bundle.calibrate_risk_score(model_bundle.score(train, bundle))
        test = _synthetic_panel(seed=4, n_rows=30)
        original_predictions = model_bundle.score(test, bundle)

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            model_bundle.save(bundle, directory)
            for filename in model_bundle.BUNDLE_ASSET_FILENAMES.values():
                self.assertTrue((directory / filename).exists())
            reloaded = model_bundle.load(directory)

        reloaded_predictions = model_bundle.score(test, reloaded)
        np.testing.assert_allclose(original_predictions.to_numpy(), reloaded_predictions.to_numpy(), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
