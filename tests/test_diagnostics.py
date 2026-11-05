import unittest

import numpy as np

from risk_fusion import diagnostics


class PoissonBinaryConsistencyTests(unittest.TestCase):
    def test_well_specified_poisson_data_shows_low_dispersion_and_recommends_poisson(self):
        rng = np.random.default_rng(42)
        n = 5000
        lam = rng.uniform(0.001, 0.05, size=n)
        y = rng.poisson(lam)

        result = diagnostics.poisson_binary_consistency(y, lam)
        self.assertLess(result["max_decile_gap"], 0.05)
        self.assertLess(abs(result["pearson_dispersion"] - 1.0), 0.25)
        self.assertEqual(result["recommended_count_family"], "poisson")

    def test_overdispersed_gamma_poisson_mixture_recommends_negative_binomial(self):
        # Fires cluster within a county-day - simulate that as a Gamma-Poisson
        # mixture (the standard way to generate negative-binomial-like data):
        # each row's true rate is lam * Gamma(shape=1/alpha, scale=alpha),
        # which has mean lam and variance lam + alpha*lam^2 > lam.
        rng = np.random.default_rng(7)
        n = 5000
        lam = rng.uniform(0.01, 0.3, size=n)
        alpha = 2.0  # strong overdispersion
        mixing = rng.gamma(shape=1.0 / alpha, scale=alpha, size=n)
        y = rng.poisson(lam * mixing)

        result = diagnostics.poisson_binary_consistency(y, lam)
        self.assertTrue(
            result["max_decile_gap"] > diagnostics.MAX_DECILE_GAP_THRESHOLD
            or result["pearson_dispersion"] > diagnostics.PEARSON_DISPERSION_THRESHOLD
        )
        self.assertEqual(result["recommended_count_family"], "negative_binomial")

    def test_probabilities_are_bounded(self):
        rng = np.random.default_rng(1)
        lam = rng.uniform(0.001, 5.0, size=200)
        y = rng.poisson(lam)
        result = diagnostics.poisson_binary_consistency(y, lam)
        self.assertGreaterEqual(result["mean_abs_gap"], 0.0)
        self.assertLessEqual(result["mean_abs_gap"], 1.0)

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            diagnostics.poisson_binary_consistency(np.array([1, 2, 3]), np.array([0.1, 0.2]))

    def test_rejects_too_few_rows_for_requested_deciles(self):
        with self.assertRaises(ValueError):
            diagnostics.poisson_binary_consistency(np.array([1, 0, 1]), np.array([0.1, 0.2, 0.1]), n_deciles=10)

    def test_more_count_model_params_reduces_degrees_of_freedom(self):
        rng = np.random.default_rng(3)
        lam = rng.uniform(0.01, 0.1, size=100)
        y = rng.poisson(lam)
        result_1param = diagnostics.poisson_binary_consistency(y, lam, count_model_params=1)
        result_10param = diagnostics.poisson_binary_consistency(y, lam, count_model_params=10)
        # Same residuals, different df divisor -> different dispersion values.
        self.assertNotAlmostEqual(result_1param["pearson_dispersion"], result_10param["pearson_dispersion"])


class CalibrationDiagnosticsTests(unittest.TestCase):
    def test_well_calibrated_predictions_have_slope_near_one_and_small_bin_gap(self):
        rng = np.random.default_rng(11)
        n = 8000
        lam = rng.uniform(0.001, 0.2, size=n)
        y = rng.poisson(lam)
        result = diagnostics.calibration_diagnostics(y, lam)
        self.assertAlmostEqual(result["calibration_slope"], 1.0, delta=0.25)
        self.assertLess(result["reliability_max_bin_gap"], 0.05)

    def test_systematically_inflated_lam_is_overconfident_and_shows_a_reliability_gap(self):
        rng = np.random.default_rng(12)
        n = 8000
        lam_true = rng.uniform(0.001, 0.2, size=n)
        y = rng.poisson(lam_true)
        lam_reported = lam_true * 5.0  # model claims much higher risk than actually observed
        result = diagnostics.calibration_diagnostics(y, lam_reported)
        self.assertGreater(result["reliability_max_bin_gap"], 0.05)


if __name__ == "__main__":
    unittest.main()
