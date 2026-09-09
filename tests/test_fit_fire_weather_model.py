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
        for key in ("model", "feature_columns", "label_column", "xgb_params", "training_row_count"):
            self.assertIn(key, bundle)
        self.assertEqual(bundle["training_row_count"], 200)

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
        panel = _synthetic_panel().drop(columns=["kbdi"])
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


class SaveLoadRoundTripTests(unittest.TestCase):
    def test_save_then_load_produces_matching_predictions(self):
        train = _synthetic_panel(seed=3)
        bundle = model_bundle.fit(train)
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
