import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from risk_fusion import model_bundle


def _synthetic_train_panel(n=3000, seed=0):
    """
    True rate: spring (Mar-May) elevated over the rest of the year, plus a
    day-to-day rh_min_afternoon effect - enough structure for the monthly
    baseline + weather-residual GLM pair to fit without singularity, and
    enough county-to-county area variation to exercise the effort offset.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", "2019-12-31", freq="D")
    date_choices = rng.choice(dates, size=n)
    counties = rng.choice(["29001", "29003", "29005"], size=n)
    area_by_county = {"29001": 1000.0, "29003": 500.0, "29005": 2000.0}

    frame = pd.DataFrame({
        "county_fips": counties,
        "valid_local_date": pd.Series(date_choices).dt.strftime("%Y-%m-%d"),
        "burnable_area_km2": pd.Series(counties).map(area_by_county).to_numpy(),
        "rh_mean": rng.uniform(20, 90, size=n),
        "rh_min_afternoon": rng.uniform(10, 80, size=n),
        "wind_kts_max": rng.uniform(2, 30, size=n),
        "wind_kts_p90": rng.uniform(2, 25, size=n),
        "vpd_kpa_max": rng.uniform(0, 4, size=n),
        "precip_24h_mm": rng.exponential(2.0, size=n),
        "is_weekend": rng.integers(0, 2, size=n).astype(bool),
        "log_effort": rng.uniform(-6.0, -3.0, size=n),
    })
    month = pd.to_datetime(frame["valid_local_date"]).dt.month
    seasonal = np.where(month.isin([3, 4, 5]), 1.0, -0.5)
    lam_true = np.exp(frame["log_effort"] + seasonal - 0.02 * frame["rh_min_afternoon"])
    frame["event_count"] = rng.poisson(lam_true)
    return frame


class LoadTrainingPanelTests(unittest.TestCase):
    def test_filters_to_window_and_normalizes_fips_to_string(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path = Path(root) / "panel.csv"
            pd.DataFrame({
                "county_fips": [29019, 29019, 29019],
                "valid_local_date": ["2014-08-31", "2014-09-01", "2021-01-01"],
                "event_count": [0, 1, 0],
                "burnable_area_km2": [100.0, 100.0, 100.0],
            }).to_csv(csv_path, index=False)
            panel = model_bundle.load_training_panel(csv_path)
        self.assertEqual(len(panel), 1)  # only 2014-09-01 is inside [2014-09-01, 2020-12-31]
        self.assertEqual(panel.iloc[0]["county_fips"], "29019")
        self.assertIsInstance(panel.iloc[0]["county_fips"], str)


class FitTests(unittest.TestCase):
    def test_returns_expected_keys(self):
        bundle = model_bundle.fit(_synthetic_train_panel())
        self.assertEqual(set(bundle), {"rate_table", "county_reference", "effort_exponent",
                                       "climatology_fit", "residual_fit", "uncertainty_fit"})

    def test_rate_table_has_a_row_for_every_county(self):
        bundle = model_bundle.fit(_synthetic_train_panel())
        self.assertEqual(set(bundle["rate_table"].index), {"29001", "29003", "29005"})

    def test_effort_exponent_is_a_finite_positive_number(self):
        bundle = model_bundle.fit(_synthetic_train_panel())
        self.assertGreater(bundle["effort_exponent"], 0.0)
        self.assertTrue(np.isfinite(bundle["effort_exponent"]))

    def test_climatology_fit_uses_the_scaled_offset(self):
        bundle = model_bundle.fit(_synthetic_train_panel())
        self.assertEqual(bundle["climatology_fit"]["offset_column"], "log_effort_scaled")


class UncertaintyTests(unittest.TestCase):
    def test_fit_produces_a_usable_uncertainty_fit(self):
        bundle = model_bundle.fit(_synthetic_train_panel(n=6000))
        uncertainty = bundle["uncertainty_fit"]
        self.assertIn("global", uncertainty)
        self.assertGreaterEqual(uncertainty["global"], 0.0)

    def test_score_attaches_lam_lo_hi_bracketing_lam(self):
        bundle = model_bundle.fit(_synthetic_train_panel(n=6000))
        target_panel = pd.DataFrame({
            "county_fips": ["29001"], "valid_local_date": ["2026-04-15"],
            "rh_mean": [50.0], "rh_min_afternoon": [40.0], "wind_kts_max": [12.0],
            "wind_kts_p90": [10.0], "vpd_kpa_max": [1.5], "precip_24h_mm": [0.0],
            "is_weekend": [False],
        })
        scored = model_bundle.score(target_panel, bundle)
        self.assertIn("lam_lo", scored.columns)
        self.assertIn("lam_hi", scored.columns)
        self.assertLessEqual(scored.iloc[0]["lam_lo"], scored.iloc[0]["lam"])
        self.assertGreaterEqual(scored.iloc[0]["lam_hi"], scored.iloc[0]["lam"])
        self.assertGreaterEqual(scored.iloc[0]["lam_lo"], 0.0)

    def test_uncertainty_asset_is_saved_and_reloadable(self):
        bundle = model_bundle.fit(_synthetic_train_panel(n=6000))
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "candidate"
            model_bundle.save(bundle, directory)
            reloaded = json.loads((directory / model_bundle.UNCERTAINTY_ASSET_FILENAME).read_text())
        self.assertEqual(reloaded, bundle["uncertainty_fit"])


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.bundle = model_bundle.fit(_synthetic_train_panel())

    def _target_panel(self, **overrides):
        base = {
            "county_fips": ["29001"], "valid_local_date": ["2026-04-15"],
            "rh_mean": [50.0], "rh_min_afternoon": [40.0], "wind_kts_max": [12.0],
            "wind_kts_p90": [10.0], "vpd_kpa_max": [1.5], "precip_24h_mm": [0.0],
            "is_weekend": [False],
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def test_produces_positive_lam_and_valid_probability(self):
        scored = model_bundle.score(self._target_panel(), self.bundle)
        self.assertTrue((scored["lam"] > 0).all())
        self.assertTrue(((scored["p_ge1_fire"] >= 0) & (scored["p_ge1_fire"] < 1)).all())
        self.assertIn("county_name", scored.columns)

    def test_sorted_by_lam_descending(self):
        target_panel = self._target_panel(
            county_fips=["29001", "29003", "29005"],
            valid_local_date=["2026-04-15"] * 3,
            rh_mean=[50.0, 60.0, 40.0], rh_min_afternoon=[40.0, 55.0, 20.0],
            wind_kts_max=[12.0, 8.0, 18.0], wind_kts_p90=[10.0, 6.0, 15.0],
            vpd_kpa_max=[1.5, 1.0, 2.5], precip_24h_mm=[0.0, 0.0, 0.0],
            is_weekend=[False, False, False],
        )
        scored = model_bundle.score(target_panel, self.bundle)
        self.assertEqual(list(scored["lam"]), sorted(scored["lam"], reverse=True))

    def test_ranks_dry_windy_conditions_above_wet_calm_ones(self):
        target_panel = self._target_panel(
            county_fips=["29001", "29001"], valid_local_date=["2026-04-15", "2026-04-15"],
            rh_mean=[30.0, 85.0], rh_min_afternoon=[15.0, 80.0],
            wind_kts_max=[20.0, 3.0], wind_kts_p90=[18.0, 2.0],
            vpd_kpa_max=[3.5, 0.3], precip_24h_mm=[0.0, 10.0],
            is_weekend=[False, False],
        )
        scored = model_bundle.score(target_panel, self.bundle)
        self.assertGreater(scored.iloc[0]["lam"], scored.iloc[1]["lam"])

    def test_accepts_int_county_fips_like_build_county_days_never_produces_but_defends_anyway(self):
        target_panel = self._target_panel(county_fips=[29001])
        scored = model_bundle.score(target_panel, self.bundle)
        self.assertEqual(scored.iloc[0]["county_fips"], "29001")


class SaveTests(unittest.TestCase):
    def test_writes_one_json_file_per_asset_role(self):
        bundle = model_bundle.fit(_synthetic_train_panel())
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "candidate"
            model_bundle.save(bundle, directory)
            written = {p.name for p in directory.iterdir()}
        # uncertainty is optional (not in BUNDLE_ASSET_FILENAMES - see
        # UNCERTAINTY_ASSET_FILENAME) but fit() always produces one, so a
        # freshly-saved bundle always includes it too.
        self.assertEqual(written, set(model_bundle.BUNDLE_ASSET_FILENAMES.values())
                          | {model_bundle.UNCERTAINTY_ASSET_FILENAME})

    def test_glm_asset_is_enough_to_reconstruct_predictions_by_hand(self):
        bundle = model_bundle.fit(_synthetic_train_panel())
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "candidate"
            model_bundle.save(bundle, directory)
            climatology = json.loads((directory / model_bundle.BUNDLE_ASSET_FILENAMES["climatology"]).read_text())

        self.assertEqual(climatology["feature_columns"], bundle["climatology_fit"]["feature_columns"])
        self.assertEqual(climatology["offset_column"], "log_effort_scaled")
        self.assertEqual(len(climatology["params"]), len(bundle["climatology_fit"]["result"].params))
        self.assertIn("month_3", climatology["means"])

    def test_effort_asset_records_shrinkage_parameters(self):
        bundle = model_bundle.fit(_synthetic_train_panel())
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "candidate"
            model_bundle.save(bundle, directory)
            effort_payload = json.loads((directory / model_bundle.BUNDLE_ASSET_FILENAMES["effort"]).read_text())

        self.assertEqual(effort_payload["shrinkage_k"], 5.0)
        self.assertEqual({row["county_fips"] for row in effort_payload["rate_table"]}, {"29001", "29003", "29005"})

    def test_effort_asset_records_the_fitted_effort_exponent(self):
        bundle = model_bundle.fit(_synthetic_train_panel())
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "candidate"
            model_bundle.save(bundle, directory)
            effort_payload = json.loads((directory / model_bundle.BUNDLE_ASSET_FILENAMES["effort"]).read_text())

        self.assertEqual(effort_payload["effort_exponent"], bundle["effort_exponent"])


if __name__ == "__main__":
    unittest.main()
