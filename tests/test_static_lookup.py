import unittest

import numpy as np
import pandas as pd

from fire_weather_ml import static_lookup


def _synthetic_bundle():
    # A tiny 2x2 grid: valid cells at (0,0) and (1,1), invalid elsewhere.
    lat = np.array([[35.0, 35.0], [36.0, 36.0]])
    lon = np.array([[-92.0, -91.0], [-92.0, -91.0]])
    valid_mask = np.array([[True, False], [False, True]])
    fuel_model_code = np.array([[101, 91], [98, 165]], dtype=np.int32)
    slope_deg = np.array([[5.0, 0.0], [0.0, 20.0]])
    aspect_sin = np.array([[0.0, 0.0], [0.0, 1.0]])  # station near (36,-91) -> east-facing (90 deg)
    aspect_cos = np.array([[1.0, 0.0], [0.0, 0.0]])
    canopy_cover_pct = np.array([[10.0, 0.0], [0.0, 40.0]])
    canopy_height_m = np.array([[3.0, 0.0], [0.0, 8.0]])
    return {
        "lat": lat, "lon": lon, "valid_mask": valid_mask, "fuel_model_code": fuel_model_code,
        "slope_deg": slope_deg, "aspect_sin": aspect_sin, "aspect_cos": aspect_cos,
        "canopy_cover_pct": canopy_cover_pct, "canopy_height_m": canopy_height_m,
    }


class BuildStationLookupTests(unittest.TestCase):
    def test_matches_the_nearest_valid_cell_not_the_nearest_cell_overall(self):
        bundle = _synthetic_bundle()
        # Station sits essentially on top of the INVALID cell (35.0, -91.0),
        # so its nearest VALID cell should be (35.0, -92.0), not that one.
        stations = pd.DataFrame({"station_id": ["A"], "lat": [35.01], "lon": [-91.01]})
        result = static_lookup.build_station_lookup(bundle, stations, max_distance_deg=1.0)
        self.assertIn("A", result.index)
        self.assertEqual(result.loc["A", "fuel_model_code"], 101)

    def test_computes_aspect_degrees_from_sin_cos(self):
        bundle = _synthetic_bundle()
        stations = pd.DataFrame({"station_id": ["B"], "lat": [36.0], "lon": [-91.0]})
        result = static_lookup.build_station_lookup(bundle, stations, max_distance_deg=1.0)
        self.assertAlmostEqual(result.loc["B", "aspect_deg"], 90.0, places=3)

    def test_excludes_a_station_too_far_from_any_valid_cell(self):
        bundle = _synthetic_bundle()
        stations = pd.DataFrame({"station_id": ["FAR"], "lat": [50.0], "lon": [-70.0]})
        result = static_lookup.build_station_lookup(bundle, stations, max_distance_deg=1.0)
        self.assertNotIn("FAR", result.index)

    def test_returns_one_row_per_matched_station_indexed_by_station_id(self):
        bundle = _synthetic_bundle()
        stations = pd.DataFrame({"station_id": ["A", "B"], "lat": [35.01, 36.0], "lon": [-91.99, -91.0]})
        result = static_lookup.build_station_lookup(bundle, stations, max_distance_deg=1.0)
        self.assertEqual(sorted(result.index), ["A", "B"])
        for column in ("fuel_model_code", "slope_deg", "aspect_deg", "canopy_cover_pct", "canopy_height_m"):
            self.assertIn(column, result.columns)


if __name__ == "__main__":
    unittest.main()
