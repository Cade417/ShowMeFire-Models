import unittest

import numpy as np
import pandas as pd

from risk_fusion import effort


def _county_reference(fips_areas):
    return {fips: {"burnable_area_km2": area} for fips, area in fips_areas.items()}


class CountyEventTotalsTests(unittest.TestCase):
    def test_sums_events_per_county_across_days(self):
        counts = pd.DataFrame([
            {"county_fips": "29019", "valid_local_date": "2020-01-01", "event_count": 2},
            {"county_fips": "29019", "valid_local_date": "2020-01-05", "event_count": 3},
            {"county_fips": "29027", "valid_local_date": "2020-01-01", "event_count": 1},
        ])
        totals = effort.county_event_totals(counts)
        self.assertEqual(totals["29019"], 5)
        self.assertEqual(totals["29027"], 1)

    def test_empty_input_returns_empty_series(self):
        counts = pd.DataFrame(columns=["county_fips", "valid_local_date", "event_count"])
        totals = effort.county_event_totals(counts)
        self.assertEqual(len(totals), 0)


class FitReportingRateTests(unittest.TestCase):
    def setUp(self):
        # Two counties, wildly different exposure and event counts.
        self.reference = _county_reference({"29019": 1000.0, "29027": 10.0, "29099": 500.0})
        self.counts = pd.DataFrame([
            {"county_fips": "29019", "valid_local_date": "2020-01-01", "event_count": 50},
            {"county_fips": "29099", "valid_local_date": "2020-01-01", "event_count": 5},
            # 29027 has zero observed events entirely.
        ])

    def test_every_reference_county_gets_a_row_including_zero_event_counties(self):
        table = effort.fit_reporting_rate(self.counts, self.reference, n_days_observed=365)
        self.assertEqual(set(table.index), {"29019", "29027", "29099"})
        self.assertEqual(table.loc["29027", "events"], 0.0)

    def test_zero_event_county_gets_positive_shrunk_rate_not_zero(self):
        table = effort.fit_reporting_rate(self.counts, self.reference, n_days_observed=365)
        self.assertGreater(table.loc["29027", "reporting_rate_shrunk"], 0.0)

    def test_high_exposure_county_stays_close_to_its_raw_rate(self):
        # 29019 has huge exposure (1000 km2 x 365 days) relative to k=5 -
        # shrinkage should barely move it from the raw MLE rate.
        table = effort.fit_reporting_rate(self.counts, self.reference, n_days_observed=365, k=5.0)
        raw = table.loc["29019", "raw_rate"]
        shrunk = table.loc["29019", "reporting_rate_shrunk"]
        self.assertAlmostEqual(raw, shrunk, delta=raw * 0.05)

    def test_low_exposure_county_is_pulled_toward_the_statewide_mean(self):
        table = effort.fit_reporting_rate(self.counts, self.reference, n_days_observed=365, k=5.0)
        state_mean = table.attrs["state_mean_rate"]
        # 29027's exposure (10 km2 x 365 days = 3650) is tiny relative to
        # the prior strength (k=5 events worth) - its posterior should sit
        # much closer to the state mean than to its own raw rate of 0.
        shrunk = table.loc["29027", "reporting_rate_shrunk"]
        self.assertLess(abs(shrunk - state_mean), abs(shrunk - 0.0))

    def test_larger_k_shrinks_harder_toward_the_state_mean(self):
        table_k1 = effort.fit_reporting_rate(self.counts, self.reference, n_days_observed=365, k=1.0)
        table_k50 = effort.fit_reporting_rate(self.counts, self.reference, n_days_observed=365, k=50.0)
        state_mean = table_k1.attrs["state_mean_rate"]
        dist_k1 = abs(table_k1.loc["29099", "reporting_rate_shrunk"] - state_mean)
        dist_k50 = abs(table_k50.loc["29099", "reporting_rate_shrunk"] - state_mean)
        self.assertLess(dist_k50, dist_k1)

    def test_rejects_non_positive_days_observed(self):
        with self.assertRaises(ValueError):
            effort.fit_reporting_rate(self.counts, self.reference, n_days_observed=0)


class LogEffortOffsetTests(unittest.TestCase):
    def setUp(self):
        self.reference = _county_reference({"29019": 1000.0, "29027": 10.0})
        self.counts = pd.DataFrame([
            {"county_fips": "29019", "valid_local_date": "2020-01-01", "event_count": 50},
        ])
        self.table = effort.fit_reporting_rate(self.counts, self.reference, n_days_observed=365)

    def test_offset_matches_manual_log_sum(self):
        value = effort.log_effort_offset("29019", self.table, self.reference)
        expected = (
            np.log(self.reference["29019"]["burnable_area_km2"])
            + np.log(self.table.loc["29019", "reporting_rate_shrunk"])
        )
        self.assertAlmostEqual(value, expected, places=9)

    def test_coverage_fraction_below_one_lowers_the_offset(self):
        full = effort.log_effort_offset("29019", self.table, self.reference, source_coverage_fraction=1.0)
        partial = effort.log_effort_offset("29019", self.table, self.reference, source_coverage_fraction=0.5)
        self.assertLess(partial, full)

    def test_rejects_non_positive_coverage(self):
        with self.assertRaises(ValueError):
            effort.log_effort_offset("29019", self.table, self.reference, source_coverage_fraction=0.0)

    def test_rejects_unknown_county(self):
        with self.assertRaises(KeyError):
            effort.log_effort_offset("99999", self.table, self.reference)

    def test_batch_matches_scalar_for_every_row(self):
        series = pd.Series(["29019", "29019", "29019"])
        batch = effort.log_effort_offset_batch(series, self.table, self.reference)
        scalar = effort.log_effort_offset("29019", self.table, self.reference)
        self.assertTrue(np.allclose(batch.to_numpy(), scalar))

    def test_batch_rejects_non_positive_coverage(self):
        series = pd.Series(["29019"])
        coverage = pd.Series([0.0])
        with self.assertRaises(ValueError):
            effort.log_effort_offset_batch(series, self.table, self.reference, source_coverage_fraction=coverage)


if __name__ == "__main__":
    unittest.main()
