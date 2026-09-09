import importlib.util
import unittest

import numpy as np
import pandas as pd

from fire_weather_ml import emulation_cost, model_bundle, rothermel_labels
from fire_weather_ml.features import FEATURE_COLUMNS

PYRETECHNICS_AVAILABLE = importlib.util.find_spec("pyretechnics") is not None


def _synthetic_labeled_panel(n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = {column: rng.uniform(0, 30, size=n) for column in FEATURE_COLUMNS}
    frame = pd.DataFrame(data)
    frame[model_bundle.DEFAULT_LABEL_COLUMN] = rng.uniform(0, 10, size=n)
    # LABEL_INPUT_COLUMNS not already in FEATURE_COLUMNS, with a real
    # burnable fuel model code so compute_labels_for_panel can actually run.
    frame["fuel_model_code"] = 101
    frame["live_herbaceous_pct"] = rng.uniform(60, 120, size=n)
    frame["live_woody_pct"] = rng.uniform(90, 150, size=n)
    frame["wind_from_deg"] = rng.uniform(0, 360, size=n)
    return frame


class MeasureEmulationSpeedupTests(unittest.TestCase):
    def test_unavailable_when_panel_is_missing_label_input_columns(self):
        panel = _synthetic_labeled_panel().drop(columns=["fuel_model_code"])
        bundle = model_bundle.fit(_synthetic_labeled_panel())
        result = emulation_cost.measure_emulation_speedup(panel, bundle)
        self.assertFalse(result["available"])
        self.assertIn("fuel_model_code", result["reason"])

    @unittest.skipUnless(PYRETECHNICS_AVAILABLE, "pyretechnics is not importable in this environment")
    def test_measures_both_timings_and_a_speedup_factor(self):
        panel = _synthetic_labeled_panel(n=100)
        bundle = model_bundle.fit(panel)
        result = emulation_cost.measure_emulation_speedup(panel, bundle, sample_size=50)
        self.assertTrue(result["available"])
        self.assertEqual(result["sample_rows"], 50)
        self.assertGreater(result["model_seconds"], 0.0)
        self.assertGreater(result["physics_seconds"], 0.0)
        self.assertIsNotNone(result["speedup_factor"])

    @unittest.skipUnless(PYRETECHNICS_AVAILABLE, "pyretechnics is not importable in this environment")
    def test_samples_no_more_rows_than_the_panel_has(self):
        panel = _synthetic_labeled_panel(n=20)
        bundle = model_bundle.fit(panel)
        result = emulation_cost.measure_emulation_speedup(panel, bundle, sample_size=1000)
        self.assertEqual(result["sample_rows"], 20)


if __name__ == "__main__":
    unittest.main()
