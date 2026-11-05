import unittest

import numpy as np
import pandas as pd

from risk_fusion import fit_glm
from risk_fusion.risk_fusion_contract import add_split_columns
from risk_fusion.risk_fusion_evidence import log_score


def _seasonal_plus_weather_panel(n=6000, seed=0, weather_effect=0.03):
    """True rate = strong seasonal cycle + a smaller day-to-day RH effect on top."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", "2020-12-31", freq="D")
    date_choices = rng.choice(dates, size=n)
    doy = pd.Series(date_choices).dt.dayofyear.to_numpy()
    doy_sin = np.sin(2 * np.pi * doy / 365.0)
    doy_cos = np.cos(2 * np.pi * doy / 365.0)

    frame = pd.DataFrame({
        "county_fips": rng.choice([f"290{i:02d}" for i in range(5)], size=n),
        "valid_local_date": pd.Series(date_choices).dt.strftime("%Y-%m-%d"),
        "valid_doy_sin": doy_sin,
        "valid_doy_cos": doy_cos,
        "rh_mean": rng.uniform(20, 90, size=n),
        "rh_min_afternoon": rng.uniform(10, 80, size=n),
        "wind_kts_max": rng.uniform(2, 30, size=n),
        "wind_kts_p90": rng.uniform(2, 25, size=n),
        "vpd_kpa_max": rng.uniform(0, 4, size=n),
        "precip_24h_mm": rng.exponential(2.0, size=n),
        "is_weekend": rng.integers(0, 2, size=n).astype(bool),
        "log_effort": rng.uniform(-4.0, -2.0, size=n),
    })
    # Strong seasonal signal (spring fire season peak) + a smaller,
    # genuinely fast day-to-day effect from rh_min_afternoon riding on top.
    true_log_lambda = (
        frame["log_effort"] + 1.5 * doy_sin + 0.5 * doy_cos
        - weather_effect * frame["rh_min_afternoon"]
    )
    frame["event_count"] = rng.poisson(np.exp(true_log_lambda.to_numpy()))
    return frame


class LinearPredictorAndResidualFitTests(unittest.TestCase):
    def test_linear_predictor_matches_manual_computation(self):
        panel = _seasonal_plus_weather_panel(n=1000)
        fit = fit_glm.fit_glm_with_covariates(panel, fit_glm.CLIMATOLOGY_FEATURES, alpha=0.01)
        eta = fit_glm.linear_predictor(fit, panel)
        lam = fit_glm.predict(fit, panel)
        np.testing.assert_allclose(np.exp(eta), lam)

    def test_residual_fit_offset_equals_base_models_eta(self):
        panel = _seasonal_plus_weather_panel(n=1000)
        base_fit = fit_glm.fit_glm_with_covariates(panel, fit_glm.CLIMATOLOGY_FEATURES, alpha=0.01)
        residual_fit = fit_glm.fit_residual_glm(panel, base_fit, fit_glm.FAST_WEATHER_FEATURES, alpha=1.0)
        self.assertEqual(residual_fit["offset_column"], "_base_eta")

    def test_predict_residual_round_trips_on_training_data(self):
        panel = _seasonal_plus_weather_panel(n=1000)
        base_fit = fit_glm.fit_glm_with_covariates(panel, fit_glm.CLIMATOLOGY_FEATURES, alpha=0.01)
        residual_fit = fit_glm.fit_residual_glm(panel, base_fit, fit_glm.FAST_WEATHER_FEATURES, alpha=1.0)
        predictions = fit_glm.predict_residual(residual_fit, base_fit, panel)
        self.assertTrue(np.all(predictions > 0))
        self.assertEqual(len(predictions), len(panel))


class CrossfitCompareResidualTests(unittest.TestCase):
    def test_residual_weather_correction_beats_climatology_alone_when_weather_is_real(self):
        # This is the exact question at stake: does fast day-to-day
        # weather add anything ON TOP OF a strong, correctly-specified
        # seasonal signal? Here it genuinely does (weather_effect=0.03
        # baked into the synthetic data), so the residual correction
        # must detect it out-of-fold.
        panel = _seasonal_plus_weather_panel(n=8000, seed=3, weather_effect=0.03)
        panel = add_split_columns(panel)
        result = fit_glm.crossfit_compare_residual(panel)

        climatology_score = log_score(result["event_count"], result["lam_climatological_doy"])
        combined_score = log_score(result["event_count"], result["lam_climatology_plus_weather"])
        self.assertLess(combined_score, climatology_score)

    def test_no_improvement_when_weather_has_no_true_effect(self):
        # weather_effect=0 - the residual correction should NOT invent a
        # large improvement out of pure noise (ridge-penalized, so this
        # should stay close to climatology alone, not run away).
        panel = _seasonal_plus_weather_panel(n=6000, seed=4, weather_effect=0.0)
        panel = add_split_columns(panel)
        result = fit_glm.crossfit_compare_residual(panel)

        climatology_score = log_score(result["event_count"], result["lam_climatological_doy"])
        combined_score = log_score(result["event_count"], result["lam_climatology_plus_weather"])
        self.assertLess(abs(combined_score - climatology_score), 0.02)

    def test_covers_every_row_exactly_once(self):
        panel = _seasonal_plus_weather_panel(n=2000)
        panel = add_split_columns(panel)
        result = fit_glm.crossfit_compare_residual(panel)
        self.assertEqual(len(result), len(panel))

    def test_includes_offset_only_baseline_beaten_by_the_seasonal_baseline(self):
        # Offset-only (reporting bias alone) has no seasonal or weather
        # information at all, so it should score worse than a baseline
        # that at least captures the strong seasonal signal baked into
        # this synthetic panel.
        panel = _seasonal_plus_weather_panel(n=6000, seed=7)
        panel = add_split_columns(panel)
        result = fit_glm.crossfit_compare_residual(panel)

        self.assertIn("lam_offset_only", result.columns)
        offset_score = log_score(result["event_count"], result["lam_offset_only"])
        climatology_score = log_score(result["event_count"], result["lam_climatological_doy"])
        self.assertLess(climatology_score, offset_score)


if __name__ == "__main__":
    unittest.main()
