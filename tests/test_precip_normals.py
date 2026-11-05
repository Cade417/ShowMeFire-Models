import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from fire_weather_ml import precip_normals


def _write_synthetic_counties(path: Path):
    county_a = box(-92.0, 34.0, -91.0, 35.0)   # fips 00001
    county_b = box(-91.0, 34.0, -90.0, 35.0)   # fips 00002
    gdf = gpd.GeoDataFrame({"fips": ["00001", "00002"], "name": ["A", "B"]},
                           geometry=[county_a, county_b], crs="EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")


def _write_synthetic_normals(path: Path):
    payload = {
        "schema": "county-precip-normals-v1",
        "counties": {
            "00001": {"name": "A", "mean_annual_precip_mm": 1000.0},
            "00002": {"name": "B", "mean_annual_precip_mm": 900.0},
        },
    }
    path.write_text(json.dumps(payload))


class LoadCountyPrecipNormalsInTests(unittest.TestCase):
    def test_converts_mm_to_inches(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "normals.json"
            _write_synthetic_normals(path)
            normals = precip_normals.load_county_precip_normals_in(path)
        self.assertAlmostEqual(normals["00001"], 1000.0 / 25.4, places=4)


class StationToCountyFipsTests(unittest.TestCase):
    def test_matches_station_to_the_containing_county(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "counties.geojson"
            _write_synthetic_counties(path)
            stations = pd.DataFrame({"station_id": ["X", "Y"], "lat": [34.5, 34.5], "lon": [-91.5, -90.5]})
            result = precip_normals.station_to_county_fips(stations, path)
        self.assertEqual(result["X"], "00001")
        self.assertEqual(result["Y"], "00002")

    def test_excludes_a_station_outside_every_county(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "counties.geojson"
            _write_synthetic_counties(path)
            stations = pd.DataFrame({"station_id": ["OUTSIDE"], "lat": [50.0], "lon": [-70.0]})
            result = precip_normals.station_to_county_fips(stations, path)
        self.assertNotIn("OUTSIDE", result.index)


class BuildStationPrecipNormalsTests(unittest.TestCase):
    def test_combines_county_join_and_normals_lookup(self):
        with tempfile.TemporaryDirectory() as root:
            counties_path = Path(root) / "counties.geojson"
            normals_path = Path(root) / "normals.json"
            _write_synthetic_counties(counties_path)
            _write_synthetic_normals(normals_path)
            stations = pd.DataFrame({"station_id": ["X"], "lat": [34.5], "lon": [-91.5]})
            result = precip_normals.build_station_precip_normals(stations, counties_path, normals_path)
        self.assertAlmostEqual(result["X"], 1000.0 / 25.4, places=4)

    def test_excludes_a_station_whose_county_has_no_recorded_normal(self):
        with tempfile.TemporaryDirectory() as root:
            counties_path = Path(root) / "counties.geojson"
            normals_path = Path(root) / "normals.json"
            _write_synthetic_counties(counties_path)
            path = Path(root) / "partial_normals.json"
            path.write_text(json.dumps({"counties": {"00001": {"mean_annual_precip_mm": 1000.0}}}))
            stations = pd.DataFrame({"station_id": ["Y"], "lat": [34.5], "lon": [-90.5]})  # falls in county 00002
            result = precip_normals.build_station_precip_normals(stations, counties_path, path)
        self.assertNotIn("Y", result)


if __name__ == "__main__":
    unittest.main()
