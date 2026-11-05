import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fire_weather_ml import occurrence_crosscheck as crosscheck


class HaversineKmTests(unittest.TestCase):
    def test_zero_distance_for_identical_points(self):
        self.assertAlmostEqual(crosscheck._haversine_km(np.array([38.5]), np.array([-92.5]),
                                                         np.array([38.5]), np.array([-92.5])).item(), 0.0, places=6)

    def test_roughly_matches_a_known_distance(self):
        # St. Louis (38.63, -90.20) to Kansas City (39.10, -94.58) is ~380 km.
        distance = crosscheck._haversine_km(np.array([38.63]), np.array([-90.20]),
                                            np.array([39.10]), np.array([-94.58])).item()
        self.assertTrue(370 <= distance <= 390, distance)


class CheckDateOverlapTests(unittest.TestCase):
    def test_reports_no_overlap_when_ranges_dont_touch(self):
        labels = pd.Series(pd.to_datetime(["2011-01-01", "2020-12-31"], utc=True))
        panel = pd.Series(pd.to_datetime(["2025-01-01", "2026-01-01"], utc=True))
        result = crosscheck.check_date_overlap(labels, panel)
        self.assertFalse(result["has_overlap"])
        self.assertIsNone(result["overlap_range"])

    def test_reports_overlap_when_ranges_touch(self):
        labels = pd.Series(pd.to_datetime(["2011-01-01", "2020-12-31"], utc=True))
        panel = pd.Series(pd.to_datetime(["2020-06-01", "2026-01-01"], utc=True))
        result = crosscheck.check_date_overlap(labels, panel)
        self.assertTrue(result["has_overlap"])
        self.assertIsNotNone(result["overlap_range"])


def _write_fire_labels(path: Path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


class ComputeOccurrenceRankingTests(unittest.TestCase):
    def _synthetic_panel(self, station_id="S1", lat=38.5, lon=-92.5, n=48):
        times = pd.date_range("2025-06-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame({
            "station_id": [station_id] * n, "lat": [lat] * n, "lon": [lon] * n,
            "valid_time": times.astype(str),
        })

    def test_unavailable_when_dates_dont_overlap(self):
        panel = self._synthetic_panel()
        predictions = pd.Series(np.linspace(0, 1, len(panel)), index=panel.index)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "labels.csv"
            _write_fire_labels(path, {
                "latitude": [38.5], "longitude": [-92.5],
                "occurred_at": ["2011-06-01T00:00:00Z"], "cause_category": ["wildfire"],
            })
            result = crosscheck.compute_occurrence_ranking(panel, predictions, path)
        self.assertFalse(result["available"])
        self.assertIn("no date overlap", result["reason"])

    def test_unavailable_when_no_station_is_near_any_fire(self):
        panel = self._synthetic_panel()
        predictions = pd.Series(np.linspace(0, 1, len(panel)), index=panel.index)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "labels.csv"
            _write_fire_labels(path, {
                "latitude": [50.0], "longitude": [-70.0],  # far from the station
                "occurred_at": ["2025-06-01T12:00:00Z"], "cause_category": ["wildfire"],
            })
            result = crosscheck.compute_occurrence_ranking(panel, predictions, path)
        self.assertFalse(result["available"])

    def test_excludes_filtered_cause_categories(self):
        panel = self._synthetic_panel()
        predictions = pd.Series(np.linspace(0, 1, len(panel)), index=panel.index)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "labels.csv"
            _write_fire_labels(path, {
                "latitude": [38.5], "longitude": [-92.5],
                "occurred_at": ["2025-06-01T12:00:00Z"], "cause_category": ["debris_burn"],
            })
            result = crosscheck.compute_occurrence_ranking(panel, predictions, path)
        self.assertFalse(result["available"])

    def test_available_and_matches_a_nearby_wildfire(self):
        panel = self._synthetic_panel()
        predictions = pd.Series(np.linspace(0, 1, len(panel)), index=panel.index)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "labels.csv"
            _write_fire_labels(path, {
                "latitude": [38.5], "longitude": [-92.5],
                "occurred_at": ["2025-06-01T12:00:00Z"], "cause_category": ["wildfire"],
            })
            result = crosscheck.compute_occurrence_ranking(panel, predictions, path)
        self.assertTrue(result["available"])
        self.assertEqual(result["matched_fire_station_pairs"], 1)
        self.assertIn("mean_prediction_percentile", result)


if __name__ == "__main__":
    unittest.main()
