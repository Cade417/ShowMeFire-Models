import unittest

import numpy as np
import pandas as pd

from fire_weather_ml import features


class KbdiStepTests(unittest.TestCase):
    def test_dry_hot_day_increases_kbdi(self):
        result = features.kbdi_step(previous_kbdi=200.0, daily_rain_in=0.0, max_temp_f=95.0,
                                    mean_annual_precip_in=40.0)
        self.assertGreater(result, 200.0)

    def test_heavy_rain_reduces_kbdi(self):
        result = features.kbdi_step(previous_kbdi=400.0, daily_rain_in=3.0, max_temp_f=70.0,
                                    mean_annual_precip_in=40.0)
        self.assertLess(result, 400.0)

    def test_rain_below_interception_threshold_has_no_direct_reduction_effect(self):
        no_rain = features.kbdi_step(200.0, 0.0, 70.0, 40.0)
        light_rain = features.kbdi_step(200.0, features.KBDI_RAIN_INTERCEPTION_IN, 70.0, 40.0)
        self.assertAlmostEqual(no_rain, light_rain, places=6)

    def test_stays_within_bounds(self):
        result = features.kbdi_step(previous_kbdi=799.0, daily_rain_in=0.0, max_temp_f=110.0,
                                    mean_annual_precip_in=20.0)
        self.assertLessEqual(result, features.KBDI_MAX)
        self.assertGreaterEqual(result, 0.0)


class KbdiSeriesTests(unittest.TestCase):
    def test_is_sequential_not_independent_per_row(self):
        rain = pd.Series([0.0, 0.0, 2.0, 0.0])
        temp = pd.Series([90.0, 92.0, 70.0, 91.0])
        series = features.kbdi_series(rain, temp, mean_annual_precip_in=40.0)
        # Two dry hot days in a row should accumulate higher than a single one.
        self.assertGreater(series.iloc[1], series.iloc[0])
        # The rain day should drop it back down relative to the day before.
        self.assertLess(series.iloc[2], series.iloc[1])

    def test_raises_on_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            features.kbdi_series(pd.Series([0.0, 0.0]), pd.Series([70.0]), mean_annual_precip_in=40.0)


class GddSeriesTests(unittest.TestCase):
    def test_accumulates_only_above_base_temp(self):
        temps = pd.Series([5.0, 15.0, 20.0])  # below, above, above base (10C)
        series = features.gdd_series(temps)
        self.assertEqual(series.iloc[0], 0.0)
        self.assertAlmostEqual(series.iloc[1], 5.0, places=6)
        self.assertAlmostEqual(series.iloc[2], 15.0, places=6)


class AssembleFeaturesTests(unittest.TestCase):
    def _sample_weather(self):
        return pd.DataFrame({
            "temp_c": [20.0, 25.0, 15.0],
            "rh_pct": [40.0, 30.0, 60.0],
            "wind_ms": [3.0, 5.0, 2.0],
            "precip_mm": [0.0, 0.0, 10.0],
            "fm1_pct": [8.0, 7.0, 12.0],
            "fm10_pct": [9.0, 8.0, 13.0],
            "fm100_pct": [12.0, 11.0, 14.0],
            "slope_deg": [10.0, 10.0, 10.0],
            "aspect_deg": [180.0, 180.0, 180.0],
            "canopy_cover_pct": [20.0, 20.0, 20.0],
            "canopy_height_m": [5.0, 5.0, 5.0],
        })

    def test_returns_exactly_the_feature_columns(self):
        result = features.assemble_features(self._sample_weather(), mean_annual_precip_in=40.0)
        self.assertEqual(list(result.columns), list(features.FEATURE_COLUMNS))
        self.assertEqual(len(result), 3)

    def test_raises_on_missing_required_column(self):
        incomplete = self._sample_weather().drop(columns=["wind_ms"])
        with self.assertRaises(ValueError):
            features.assemble_features(incomplete, mean_annual_precip_in=40.0)


if __name__ == "__main__":
    unittest.main()
