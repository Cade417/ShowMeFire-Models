import unittest

import numpy as np
import pandas as pd

from fire_weather_ml import features


class NelsonEmcTests(unittest.TestCase):
    def test_hot_dry_air_gives_low_emc(self):
        emc = features.nelson_emc(np.array([35.0]), np.array([15.0]))
        self.assertLess(emc[0], 10.0)

    def test_cool_humid_air_gives_high_emc(self):
        emc = features.nelson_emc(np.array([10.0]), np.array([90.0]))
        self.assertGreater(emc[0], 15.0)

    def test_stays_within_bounds(self):
        emc = features.nelson_emc(np.array([50.0, -10.0]), np.array([0.0, 100.0]))
        self.assertTrue((emc >= features.NELSON_EMC_MIN_PERCENT).all())
        self.assertTrue((emc <= features.NELSON_EMC_MAX_PERCENT).all())


class DeriveFm1Fm10Fm100Tests(unittest.TestCase):
    def _series(self, n=24, seed=0):
        rng = np.random.default_rng(seed)
        index = pd.RangeIndex(n)
        temp_c = pd.Series(rng.uniform(15, 30, n), index=index)
        rh = pd.Series(rng.uniform(20, 60, n), index=index)
        precip_mm = pd.Series(np.zeros(n), index=index)
        observed_fm10 = pd.Series(rng.uniform(8, 15, n), index=index)
        return temp_c, rh, precip_mm, observed_fm10

    def test_returns_expected_columns_and_passes_through_observed_fm10_unchanged(self):
        temp_c, rh, precip_mm, observed_fm10 = self._series()
        result = features.derive_fm1_fm10_fm100(temp_c, rh, precip_mm, observed_fm10)
        self.assertEqual(list(result.columns), ["fm1_pct", "fm10_pct", "fm100_pct"])
        pd.testing.assert_series_equal(result["fm10_pct"], observed_fm10, check_names=False)

    def test_stays_within_physical_bounds(self):
        temp_c, rh, precip_mm, observed_fm10 = self._series(n=100, seed=1)
        result = features.derive_fm1_fm10_fm100(temp_c, rh, precip_mm, observed_fm10)
        for column in ("fm1_pct", "fm100_pct"):
            self.assertTrue((result[column] >= features.NELSON_EMC_MIN_PERCENT).all())
            self.assertTrue((result[column] <= features.NELSON_EMC_MAX_PERCENT).all())

    def test_raises_on_mismatched_lengths(self):
        temp_c, rh, precip_mm, observed_fm10 = self._series()
        with self.assertRaises(ValueError):
            features.derive_fm1_fm10_fm100(temp_c, rh, precip_mm, observed_fm10.iloc[:-1])

    def test_hot_dry_weather_pushes_fm1_lower_than_cool_humid_weather_after_a_step_change(self):
        # Constant weather throughout would seed AND target the same EMC
        # every step, so nothing would ever move (a degenerate case, not a
        # useful test). Instead: mild conditions for the first hour, then a
        # step change to either hot/dry or cool/humid - fm1's fast (1hr) tau
        # should already reflect most of the new condition after just one
        # more hour, while fm10 (whose free-running lag sets the anchor
        # correction) hasn't caught up yet, so the weather difference shows
        # through in fm1 even though both scenarios anchor to the same
        # observed_fm10.
        index = pd.RangeIndex(3)
        temp_dry = pd.Series([20.0, 35.0, 35.0], index=index)
        rh_dry = pd.Series([50.0, 15.0, 15.0], index=index)
        temp_wet = pd.Series([20.0, 15.0, 15.0], index=index)
        rh_wet = pd.Series([50.0, 80.0, 80.0], index=index)
        precip = pd.Series([0.0, 0.0, 0.0], index=index)
        observed_fm10 = pd.Series([10.0, 10.0, 10.0], index=index)

        dry = features.derive_fm1_fm10_fm100(temp_dry, rh_dry, precip, observed_fm10)
        wet = features.derive_fm1_fm10_fm100(temp_wet, rh_wet, precip, observed_fm10)
        self.assertLess(dry["fm1_pct"].iloc[1], wet["fm1_pct"].iloc[1])


class GreenFactorTests(unittest.TestCase):
    def test_zero_below_greenup_start(self):
        self.assertEqual(features.green_factor(0.0), 0.0)

    def test_one_above_greenup_full(self):
        self.assertEqual(features.green_factor(features.GDD_GREENUP_FULL + 500.0), 1.0)

    def test_ramps_between(self):
        midpoint = (features.GDD_GREENUP_START + features.GDD_GREENUP_FULL) / 2.0
        self.assertAlmostEqual(features.green_factor(midpoint), 0.5, places=6)


class LiveMoisturePercentTests(unittest.TestCase):
    def test_cured_and_green_endpoints_match_scott_burgan_scenario_values(self):
        gdd = pd.Series([0.0, features.GDD_GREENUP_FULL + 100.0])
        result = features.live_moisture_percent(gdd)
        self.assertAlmostEqual(result["live_herbaceous_pct"].iloc[0], features.LIVE_HERBACEOUS_CURED_PCT)
        self.assertAlmostEqual(result["live_herbaceous_pct"].iloc[1], features.LIVE_HERBACEOUS_GREEN_PCT)
        self.assertAlmostEqual(result["live_woody_pct"].iloc[0], features.LIVE_WOODY_CURED_PCT)
        self.assertAlmostEqual(result["live_woody_pct"].iloc[1], features.LIVE_WOODY_GREEN_PCT)


if __name__ == "__main__":
    unittest.main()
