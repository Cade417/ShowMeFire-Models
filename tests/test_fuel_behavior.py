import unittest

from fuel_behavior import spread_index as si
from fuel_behavior.fuel_models import lookup


class FuelModelLookupTests(unittest.TestCase):
    def test_known_code_is_case_insensitive(self):
        self.assertEqual(lookup("gr1").code, "GR1")
        self.assertEqual(lookup("GR1").code, "GR1")

    def test_unknown_code_returns_none(self):
        self.assertIsNone(lookup("not_a_real_model"))

    def test_non_burnable_models_have_zero_bases(self):
        for code in ("NB1", "NB2", "NB3", "NB8", "NB9"):
            model = lookup(code)
            self.assertEqual(model.spread_base, 0.0)
            self.assertEqual(model.intensity_base, 0.0)


class SlopeFactorTests(unittest.TestCase):
    def test_flat_ground_is_neutral(self):
        self.assertAlmostEqual(si.slope_factor(0.0), 1.0)

    def test_increases_monotonically_with_slope(self):
        flat = si.slope_factor(0.0)
        gentle = si.slope_factor(10.0)
        steep = si.slope_factor(30.0)
        self.assertLess(flat, gentle)
        self.assertLess(gentle, steep)

    def test_capped_beyond_cap_degrees(self):
        at_cap = si.slope_factor(si.SLOPE_CAP_DEGREES)
        beyond_cap = si.slope_factor(si.SLOPE_CAP_DEGREES + 20.0)
        self.assertAlmostEqual(at_cap, beyond_cap)

    def test_negative_slope_clips_to_flat(self):
        self.assertAlmostEqual(si.slope_factor(-5.0), si.slope_factor(0.0))


class WindFactorTests(unittest.TestCase):
    def test_calm_wind_is_neutral(self):
        self.assertAlmostEqual(si.wind_factor(0.0, canopy_cover_pct=0.0), 1.0)

    def test_increases_monotonically_with_wind(self):
        calm = si.wind_factor(0.0, canopy_cover_pct=0.0)
        breezy = si.wind_factor(10.0, canopy_cover_pct=0.0)
        gusty = si.wind_factor(30.0, canopy_cover_pct=0.0)
        self.assertLess(calm, breezy)
        self.assertLess(breezy, gusty)

    def test_dense_canopy_shelters_surface_fuels_from_wind(self):
        open_ground = si.wind_factor(25.0, canopy_cover_pct=0.0)
        dense_canopy = si.wind_factor(25.0, canopy_cover_pct=90.0)
        self.assertGreater(open_ground, dense_canopy)

    def test_wind_reduction_factor_bounds(self):
        self.assertAlmostEqual(si.wind_reduction_factor(0.0), 1.0)
        self.assertAlmostEqual(si.wind_reduction_factor(100.0), 1.0 - si.CANOPY_MAX_REDUCTION)


class RelativeIndexTests(unittest.TestCase):
    def test_unrecognized_fuel_model_returns_none(self):
        self.assertIsNone(si.relative_spread_index("not_a_real_model", 0.0, 10.0, 0.0))
        self.assertIsNone(si.relative_intensity_index("not_a_real_model", 0.0, 10.0, 0.0))

    def test_non_burnable_is_zero_regardless_of_weather(self):
        # Steep + windy would maximize the environmental multiplier - a
        # non-burnable fuel model must still report zero, not just "low".
        self.assertEqual(si.relative_spread_index("NB1", 40.0, 40.0, 0.0), 0.0)
        self.assertEqual(si.relative_intensity_index("NB1", 40.0, 40.0, 0.0), 0.0)

    def test_high_load_grass_spreads_faster_than_compact_timber_litter_under_identical_weather(self):
        # GR9 (very high load grass, fine continuous fuel) vs TL1 (low
        # load, compact conifer litter) - a textbook contrast in Scott &
        # Burgan's own fuel model descriptions.
        grass = si.relative_spread_index("GR9", slope_degrees=5.0, wind_kts=15.0, canopy_cover_pct=0.0)
        timber_litter = si.relative_spread_index("TL1", slope_degrees=5.0, wind_kts=15.0, canopy_cover_pct=0.0)
        self.assertGreater(grass, timber_litter)

    def test_steep_and_windy_exceeds_flat_and_calm_for_the_same_fuel_model(self):
        calm_flat = si.relative_spread_index("GR2", slope_degrees=0.0, wind_kts=0.0, canopy_cover_pct=0.0)
        steep_windy = si.relative_spread_index("GR2", slope_degrees=30.0, wind_kts=25.0, canopy_cover_pct=0.0)
        self.assertGreater(steep_windy, calm_flat)

    def test_intensity_and_spread_can_diverge_for_the_same_fuel_model(self):
        # TL9 (very high load broadleaf litter) is rated low-moderate
        # spread but higher intensity than GR2 (low load grass, fast
        # spread but low intensity) - the two indices are not required to
        # move together, since spread and intensity reflect different
        # physical drivers (fine-fuel continuity vs. total fuel load).
        model_tl9 = lookup("TL9")
        model_gr2 = lookup("GR2")
        self.assertLess(model_tl9.spread_base, model_gr2.spread_base)
        self.assertGreater(model_tl9.intensity_base, model_gr2.intensity_base)


if __name__ == "__main__":
    unittest.main()
