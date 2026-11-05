import unittest

import pandas as pd

from risk_fusion import join_panel


def _weather_row(county, date, temp=25.0):
    return {"county_fips": county, "valid_local_date": date, "temp_max_c": temp, "fm_features_available": False}


class BuildLabeledPanelTests(unittest.TestCase):
    def test_left_joins_and_fills_genuine_zeros(self):
        weather = pd.DataFrame([
            _weather_row("29019", "2020-07-01"),
            _weather_row("29019", "2020-07-02"),
            _weather_row("29027", "2020-07-01"),
        ])
        counts = pd.DataFrame([
            {"county_fips": "29019", "valid_local_date": "2020-07-01", "event_count": 2, "acres_sum": 5.0, "acres_max": 3.0},
        ])
        result = join_panel.build_labeled_panel(weather, counts)
        panel = result["panel"]
        self.assertEqual(len(panel), 3)  # every weather row survives
        matched = panel[(panel.county_fips == "29019") & (panel.valid_local_date == "2020-07-01")]
        self.assertEqual(matched.iloc[0]["event_count"], 2)
        zero_day = panel[(panel.county_fips == "29019") & (panel.valid_local_date == "2020-07-02")]
        self.assertEqual(zero_day.iloc[0]["event_count"], 0)
        self.assertEqual(zero_day.iloc[0]["acres_sum"], 0.0)

    def test_labels_with_no_weather_match_are_dropped_and_counted(self):
        weather = pd.DataFrame([_weather_row("29019", "2020-07-01")])
        counts = pd.DataFrame([
            {"county_fips": "29019", "valid_local_date": "2020-07-01", "event_count": 1, "acres_sum": 1.0, "acres_max": 1.0},
            {"county_fips": "29019", "valid_local_date": "2011-01-01", "event_count": 1, "acres_sum": 1.0, "acres_max": 1.0},
        ])
        result = join_panel.build_labeled_panel(weather, counts)
        self.assertEqual(result["labels_dropped_no_weather_match"], 1)
        self.assertEqual(len(result["panel"]), 1)  # weather panel size is unaffected by dropped labels

    def test_empty_labels_produces_all_zero_panel(self):
        weather = pd.DataFrame([_weather_row("29019", "2020-07-01"), _weather_row("29019", "2020-07-02")])
        counts = pd.DataFrame(columns=["county_fips", "valid_local_date", "event_count", "acres_sum", "acres_max"])
        result = join_panel.build_labeled_panel(weather, counts)
        self.assertTrue((result["panel"]["event_count"] == 0).all())
        self.assertEqual(result["weather_days_with_fire"], 0)

    def test_summary_counts_are_consistent(self):
        weather = pd.DataFrame([_weather_row("29019", d) for d in ["2020-07-01", "2020-07-02", "2020-07-03"]])
        counts = pd.DataFrame([
            {"county_fips": "29019", "valid_local_date": "2020-07-02", "event_count": 3, "acres_sum": 9.0, "acres_max": 4.0},
        ])
        result = join_panel.build_labeled_panel(weather, counts)
        self.assertEqual(result["weather_days_total"], 3)
        self.assertEqual(result["weather_days_with_fire"], 1)

    def test_does_not_mutate_no_matching_rows_county_across_counties(self):
        # A label for a DIFFERENT county on the same date must not leak
        # into another county's row via a bad join key.
        weather = pd.DataFrame([_weather_row("29019", "2020-07-01"), _weather_row("29027", "2020-07-01")])
        counts = pd.DataFrame([
            {"county_fips": "29027", "valid_local_date": "2020-07-01", "event_count": 5, "acres_sum": 1.0, "acres_max": 1.0},
        ])
        result = join_panel.build_labeled_panel(weather, counts)
        panel = result["panel"]
        boone = panel[panel.county_fips == "29019"]
        self.assertEqual(boone.iloc[0]["event_count"], 0)


if __name__ == "__main__":
    unittest.main()
