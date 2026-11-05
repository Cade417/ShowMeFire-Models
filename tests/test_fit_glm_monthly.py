import unittest

import numpy as np
import pandas as pd

from risk_fusion import fit_glm
from risk_fusion.risk_fusion_contract import add_split_columns
from risk_fusion.risk_fusion_evidence import log_score


def _monthly_seasonal_panel(n=8000, seed=0, weather_effect=0.03):
    """
    True rate follows discrete per-month levels (a sharp spring peak that
    a 2-harmonic curve can't represent exactly), plus a smaller day-to-day
    RH effect on top - the same residual question as
    test_fit_glm_residual.py, but against a baseline whose true shape is
    itself discrete per-month rather than smooth.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", "2020-12-31", freq="D")
    date_choices = rng.choice(dates, size=n)
    month = pd.Series(date_choices).dt.month.to_numpy()

    # Sharp spring peak (Mar-Apr), quiet winter and midsummer - a step
    # pattern a smooth 2-harmonic curve fits only approximately.
    month_log_level = np.array([0.0, 0.2, 1.6, 1.8, 0.6, -0.3, -0.8, -0.8, -0.3, 0.4, 0.3, 0.1])
    seasonal_log_lambda = month_log_level[month - 1]

    frame = pd.DataFrame({
        "county_fips": rng.choice([f"290{i:02d}" for i in range(5)], size=n),
        "valid_local_date": pd.Series(date_choices).dt.strftime("%Y-%m-%d"),
        "rh_mean": rng.uniform(20, 90, size=n),
        "rh_min_afternoon": rng.uniform(10, 80, size=n),
        "wind_kts_max": rng.uniform(2, 30, size=n),
        "wind_kts_p90": rng.uniform(2, 25, size=n),
        "vpd_kpa_max": rng.uniform(0, 4, size=n),
        "precip_24h_mm": rng.exponential(2.0, size=n),
        "is_weekend": rng.integers(0, 2, size=n).astype(bool),
        "log_effort": rng.uniform(-4.0, -2.0, size=n),
    })
    true_log_lambda = (
        frame["log_effort"] + seasonal_log_lambda
        - weather_effect * frame["rh_min_afternoon"]
    )
    frame["event_count"] = rng.poisson(np.exp(true_log_lambda.to_numpy()))
    return frame


class AddMonthDummiesTests(unittest.TestCase):
    def test_creates_eleven_indicator_columns(self):
        panel = pd.DataFrame({"valid_local_date": ["2020-01-15", "2020-03-01", "2020-12-31"]})
        result = fit_glm.add_month_dummies(panel)
        for column in fit_glm.MONTH_DUMMY_COLUMNS:
            self.assertIn(column, result.columns)
        self.assertEqual(len(fit_glm.MONTH_DUMMY_COLUMNS), 11)

    def test_january_row_has_every_dummy_at_zero(self):
        panel = pd.DataFrame({"valid_local_date": ["2020-01-15"]})
        result = fit_glm.add_month_dummies(panel)
        self.assertEqual(result[fit_glm.MONTH_DUMMY_COLUMNS].iloc[0].sum(), 0.0)

    def test_march_row_has_only_month_3_set(self):
        panel = pd.DataFrame({"valid_local_date": ["2020-03-15"]})
        result = fit_glm.add_month_dummies(panel)
        row = result[fit_glm.MONTH_DUMMY_COLUMNS].iloc[0]
        self.assertEqual(row["month_3"], 1.0)
        self.assertEqual(row.drop("month_3").sum(), 0.0)

    def test_does_not_mutate_input_panel(self):
        panel = pd.DataFrame({"valid_local_date": ["2020-03-15"]})
        fit_glm.add_month_dummies(panel)
        self.assertNotIn("month_3", panel.columns)


class MonthlyBaselineRateMultipliersTests(unittest.TestCase):
    def test_january_reference_level_is_one(self):
        panel = _monthly_seasonal_panel(n=3000)
        panel = fit_glm.add_month_dummies(panel)
        fit = fit_glm.fit_glm_with_covariates(panel, fit_glm.MONTH_DUMMY_COLUMNS, alpha=0.01)
        multipliers = fit_glm.monthly_baseline_rate_multipliers(fit)
        self.assertEqual(list(multipliers.index), list(range(1, 13)))
        self.assertEqual(multipliers[1], 1.0)

    def test_spring_months_show_higher_multiplier_than_midsummer(self):
        # The synthetic panel's true rate peaks in Mar/Apr and troughs in
        # Jun/Jul - the fitted multipliers should recover that ordering.
        panel = _monthly_seasonal_panel(n=6000, seed=1)
        panel = fit_glm.add_month_dummies(panel)
        fit = fit_glm.fit_glm_with_covariates(panel, fit_glm.MONTH_DUMMY_COLUMNS, alpha=0.01)
        multipliers = fit_glm.monthly_baseline_rate_multipliers(fit)
        self.assertGreater(multipliers[4], multipliers[7])


class FitEffortExponentTests(unittest.TestCase):
    def test_recovers_a_coefficient_near_one_when_log_effort_truly_enters_at_one(self):
        panel = _monthly_seasonal_panel(n=6000, seed=21)
        result = fit_glm.fit_effort_exponent(panel)
        self.assertTrue(result["ci_low"] <= 1.0 <= result["ci_high"])

    def test_recovers_the_true_steeper_exponent_when_log_effort_enters_at_1_5(self):
        rng = np.random.default_rng(22)
        n = 20000
        dates = pd.date_range("2016-01-01", "2020-12-31", freq="D")
        date_choices = rng.choice(dates, size=n)
        month = pd.Series(date_choices).dt.month.to_numpy()
        month_log_level = np.array([0.0, 0.2, 1.6, 1.8, 0.6, -0.3, -0.8, -0.8, -0.3, 0.4, 0.3, 0.1])
        seasonal_log_lambda = month_log_level[month - 1]

        panel = pd.DataFrame({
            "valid_local_date": pd.Series(date_choices).dt.strftime("%Y-%m-%d"),
            "log_effort": rng.uniform(-4.0, -2.0, size=n),
        })
        true_log_lambda = seasonal_log_lambda + 1.5 * panel["log_effort"]
        panel["event_count"] = rng.poisson(np.exp(true_log_lambda.to_numpy()))

        result = fit_glm.fit_effort_exponent(panel)
        self.assertAlmostEqual(result["coefficient"], 1.5, delta=0.2)
        self.assertFalse(result["ci_low"] <= 1.0 <= result["ci_high"])

    def test_operates_on_an_arbitrary_offset_column_name(self):
        panel = _monthly_seasonal_panel(n=3000, seed=23)
        panel["log_effort_scaled"] = 1.5 * panel["log_effort"]
        result = fit_glm.fit_effort_exponent(panel, offset_column="log_effort_scaled")
        # Rescaling the covariate by a constant exactly divides its true
        # coefficient by that constant - fitting on the rescaled column
        # should recover ~1/1.5 of whatever fitting on log_effort gives.
        raw = fit_glm.fit_effort_exponent(panel, offset_column="log_effort")
        self.assertAlmostEqual(result["coefficient"] * 1.5, raw["coefficient"], places=6)


class CrossfitCompareResidualOffsetColumnTests(unittest.TestCase):
    def test_candidate_arm_uses_the_requested_offset_column_not_log_effort(self):
        panel = _monthly_seasonal_panel(n=4000, seed=31)
        panel = add_split_columns(panel)
        # A per-row RESCALING (not a constant shift, which a GLM's free
        # intercept could just absorb) - if offset_column were ignored,
        # this would have no effect on lam_climatology_plus_weather (or
        # would error looking up a column that isn't there).
        panel["log_effort_scaled"] = panel["log_effort"] * 3.0
        result = fit_glm.crossfit_compare_residual(panel, offset_column="log_effort_scaled")
        default_result = fit_glm.crossfit_compare_residual(panel)
        self.assertFalse(np.allclose(result["lam_climatology_plus_weather"],
                                     default_result["lam_climatology_plus_weather"]))
        # lam_offset_only always uses the raw log_effort regardless of offset_column.
        self.assertTrue(np.allclose(result["lam_offset_only"], default_result["lam_offset_only"]))

    def test_defaults_to_log_effort_for_backward_compatibility(self):
        panel = _monthly_seasonal_panel(n=2000, seed=32)
        panel = add_split_columns(panel)
        with_default = fit_glm.crossfit_compare_residual(panel)
        with_explicit = fit_glm.crossfit_compare_residual(panel, offset_column="log_effort")
        self.assertTrue(np.allclose(with_default["lam_climatology_plus_weather"],
                                    with_explicit["lam_climatology_plus_weather"]))


class CrossfitCompareResidualDefaultsToMonthlyBaselineTests(unittest.TestCase):
    def test_default_climatology_features_is_month_dummies(self):
        self.assertEqual(fit_glm.crossfit_compare_residual.__defaults__[1], fit_glm.MONTH_DUMMY_COLUMNS)

    def test_residual_weather_correction_beats_monthly_baseline_when_weather_is_real(self):
        panel = _monthly_seasonal_panel(n=8000, seed=3, weather_effect=0.03)
        panel = add_split_columns(panel)
        result = fit_glm.crossfit_compare_residual(panel)

        baseline_score = log_score(result["event_count"], result["lam_climatological_doy"])
        combined_score = log_score(result["event_count"], result["lam_climatology_plus_weather"])
        self.assertLess(combined_score, baseline_score)

    def test_monthly_baseline_beats_harmonic_baseline_on_a_sharp_seasonal_step(self):
        # The whole point of discrete per-month levels: a true rate with a
        # sharp Mar/Apr step is fit better by per-month dummies than by a
        # 2-harmonic sine/cosine curve, which can only approximate a step.
        from risk_fusion.risk_fusion_contract import crossfit_indices

        panel = _monthly_seasonal_panel(n=8000, seed=5, weather_effect=0.0)
        panel = add_split_columns(panel)
        doy = pd.to_datetime(panel["valid_local_date"]).dt.dayofyear
        panel["valid_doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
        panel["valid_doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
        panel = fit_glm.add_month_dummies(panel).reset_index(drop=True)

        monthly_lam = np.empty(len(panel))
        harmonic_lam = np.empty(len(panel))
        for train_idx, test_idx in crossfit_indices(panel):
            train_panel, test_panel = panel.loc[train_idx], panel.loc[test_idx]
            monthly_fit = fit_glm.fit_glm_with_covariates(train_panel, fit_glm.MONTH_DUMMY_COLUMNS, alpha=0.01)
            harmonic_fit = fit_glm.fit_glm_with_covariates(train_panel, fit_glm.CLIMATOLOGY_FEATURES, alpha=0.01)
            monthly_lam[test_idx] = fit_glm.predict(monthly_fit, test_panel)
            harmonic_lam[test_idx] = fit_glm.predict(harmonic_fit, test_panel)

        monthly_score = log_score(panel["event_count"], monthly_lam)
        harmonic_score = log_score(panel["event_count"], harmonic_lam)
        self.assertLess(monthly_score, harmonic_score)

    def test_covers_every_row_exactly_once(self):
        panel = _monthly_seasonal_panel(n=2000)
        panel = add_split_columns(panel)
        result = fit_glm.crossfit_compare_residual(panel)
        self.assertEqual(len(result), len(panel))


if __name__ == "__main__":
    unittest.main()
