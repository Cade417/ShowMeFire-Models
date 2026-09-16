import unittest

import numpy as np
import pandas as pd

from risk_fusion import fit_glm
from risk_fusion.risk_fusion_contract import add_split_columns


def _synthetic_panel(n=3000, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n // 10, freq="D")
    date_choices = rng.choice(dates, size=n)
    frame = pd.DataFrame({
        "county_fips": rng.choice([f"290{i:02d}" for i in range(5)], size=n),
        "valid_local_date": pd.Series(date_choices).dt.strftime("%Y-%m-%d"),
        "temp_max_c": rng.uniform(0, 40, size=n),
        "rh_mean": rng.uniform(10, 90, size=n),
        "rh_min_afternoon": rng.uniform(5, 80, size=n),
        "wind_kts_max": rng.uniform(2, 30, size=n),
        "wind_kts_p90": rng.uniform(2, 25, size=n),
        "vpd_kpa_max": rng.uniform(0, 4, size=n),
        "precip_24h_mm": rng.exponential(2.0, size=n),
        "kbdi": rng.uniform(0, 150, size=n),
        "kbdi_valid": True,
        "gdd_accum_since_mar1": rng.uniform(0, 2000, size=n),
        "valid_doy_sin": rng.uniform(-1, 1, size=n),
        "valid_doy_cos": rng.uniform(-1, 1, size=n),
        "is_weekend": rng.integers(0, 2, size=n).astype(bool),
        # Matches the real panel's observed scale (state_mean_rate ~2.2e-5
        # per km2-day x realistic burnable_area_km2 in the thousands ->
        # log_effort around -4 to -2, i.e. lam on the order of 0.02-0.1).
        # An unrealistically extreme offset (e.g. lam ~1e-5) makes almost
        # every row a hard zero and destabilizes the IRLS fit - not
        # something specific to fit_glm.py, just an unrepresentative test input.
        "log_effort": rng.uniform(-4.0, -2.0, size=n),
    })
    if informative:
        # True rate depends strongly on rh_min_afternoon (drier -> more fire).
        true_log_lambda = frame["log_effort"] - 0.03 * frame["rh_min_afternoon"] + 1.0
    else:
        true_log_lambda = frame["log_effort"]
    lam_true = np.exp(true_log_lambda.to_numpy())
    frame["event_count"] = rng.poisson(lam_true)
    return frame


class FitGlmTests(unittest.TestCase):
    def test_filter_valid_rows_drops_invalid_kbdi(self):
        panel = _synthetic_panel(n=100)
        panel.loc[0:9, "kbdi_valid"] = False
        filtered = fit_glm.filter_valid_rows(panel)
        self.assertEqual(len(filtered), 90)

    def test_fit_and_predict_produces_positive_rates(self):
        panel = _synthetic_panel(n=1000)
        fit = fit_glm.fit_glm_with_covariates(panel)
        predictions = fit_glm.predict(fit, panel)
        self.assertTrue(np.all(predictions > 0))
        self.assertEqual(len(predictions), len(panel))

    def test_offset_only_predictions_are_positive(self):
        panel = _synthetic_panel(n=500)
        fit = fit_glm.fit_offset_only(panel)
        predictions = fit_glm.predict_offset_only(fit, panel)
        self.assertTrue(np.all(predictions > 0))

    def test_offset_only_fit_reproduces_observed_mean(self):
        panel = _synthetic_panel(n=2000, informative=False)
        fit = fit_glm.fit_offset_only(panel)
        predictions = fit_glm.predict_offset_only(fit, panel)
        self.assertAlmostEqual(predictions.mean(), panel["event_count"].mean(), places=4)

    def test_crossfit_compare_covers_every_row_exactly_once(self):
        panel = _synthetic_panel(n=2000)
        panel = add_split_columns(panel)
        result = fit_glm.crossfit_compare(panel)
        self.assertEqual(len(result), len(panel))
        self.assertEqual(result["block"].nunique(), 5)

    def test_crossfit_compare_includes_climatology_baseline(self):
        panel = _synthetic_panel(n=2000)
        panel = add_split_columns(panel)
        result = fit_glm.crossfit_compare(panel)
        self.assertIn("lam_climatological_doy", result.columns)
        self.assertTrue((result["lam_climatological_doy"] > 0).all())

    def test_covariates_beat_climatology_when_weather_is_the_true_driver(self):
        # The synthetic data's true rate depends on rh_min_afternoon, not
        # on season at all - climatology should offer little, and the
        # full covariate model should still clearly beat it out-of-fold.
        from risk_fusion.risk_fusion_evidence import log_score

        panel = _synthetic_panel(n=6000, informative=True, seed=5)
        panel = add_split_columns(panel)
        result = fit_glm.crossfit_compare(panel)

        climatology_score = log_score(result["event_count"], result["lam_climatological_doy"])
        covariate_score = log_score(result["event_count"], result["lam_covariates"])
        self.assertLess(covariate_score, climatology_score)

    def test_informative_covariate_beats_offset_only_out_of_fold(self):
        # When a covariate genuinely drives the rate, the covariate GLM's
        # out-of-fold log_score should beat the offset-only baseline -
        # otherwise the whole apparatus couldn't detect real signal.
        from risk_fusion.risk_fusion_evidence import log_score

        panel = _synthetic_panel(n=6000, informative=True, seed=5)
        panel = add_split_columns(panel)
        result = fit_glm.crossfit_compare(panel)

        offset_score = log_score(result["event_count"], result["lam_offset_only"])
        covariate_score = log_score(result["event_count"], result["lam_covariates"])
        self.assertLess(covariate_score, offset_score)

    def test_uninformative_covariates_do_not_wildly_beat_offset_only(self):
        from risk_fusion.risk_fusion_evidence import log_score

        panel = _synthetic_panel(n=4000, informative=False, seed=9)
        panel = add_split_columns(panel)
        result = fit_glm.crossfit_compare(panel)

        offset_score = log_score(result["event_count"], result["lam_offset_only"])
        covariate_score = log_score(result["event_count"], result["lam_covariates"])
        # Ridge-penalized fit on noise should track the baseline closely,
        # not blow up - a large gap would mean the regularization isn't doing its job.
        self.assertLess(abs(covariate_score - offset_score), 0.05)


class FitLambdaUncertaintyTests(unittest.TestCase):
    def _crossfit_result(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        months = rng.integers(1, 13, size=n)
        dates = pd.to_datetime({"year": 2019, "month": months, "day": 1})
        lam = rng.uniform(0.02, 0.2, size=n)
        counts = rng.poisson(lam)
        return pd.DataFrame({
            "valid_local_date": dates.dt.strftime("%Y-%m-%d"),
            "event_count": counts,
            "lam_climatology_plus_weather": lam,
        })

    def test_widths_are_non_negative(self):
        uncertainty = fit_glm.fit_lambda_uncertainty(self._crossfit_result(), minimum_rows=50)
        self.assertGreaterEqual(uncertainty["global"], 0.0)
        for entry in uncertainty["regimes"].values():
            self.assertGreaterEqual(entry["half_width"], 0.0)

    def test_lambda_interval_brackets_lam_and_never_goes_negative(self):
        uncertainty = fit_glm.fit_lambda_uncertainty(self._crossfit_result(), minimum_rows=50)
        lam = np.array([0.01, 0.05, 0.5])
        month = np.array([1, 1, 1])
        bounds = fit_glm.lambda_interval(lam, month, uncertainty)
        self.assertTrue(np.all(bounds[:, 0] <= bounds[:, 1]))
        self.assertTrue(np.all(bounds[:, 1] <= bounds[:, 2]))
        self.assertTrue(np.all(bounds[:, 0] >= 0.0))

    def test_unseen_month_falls_back_to_global_width(self):
        result = self._crossfit_result()
        result = result[result["valid_local_date"].str.slice(5, 7) != "07"]
        uncertainty = fit_glm.fit_lambda_uncertainty(result, minimum_rows=50)
        bounds = fit_glm.lambda_interval([0.1], [7], uncertainty)
        self.assertAlmostEqual(bounds[0, 2] - bounds[0, 1], uncertainty["global"], places=6)


if __name__ == "__main__":
    unittest.main()
