import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from fire_weather_ml import wind_direction


class WindFromDegreesTests(unittest.TestCase):
    def test_matches_known_quadrant(self):
        # u negative (wind vector points west) -> wind FROM the east, 90 degrees.
        self.assertAlmostEqual(wind_direction.wind_from_degrees(-5.0, 0.0), 90.0, places=3)


class LookupHourWindTests(unittest.TestCase):
    def _write_rtma_file(self, directory: Path, hour: pd.Timestamp) -> Path:
        lat = np.array([[35.0, 35.0], [36.0, 36.0]])
        lon = np.array([[-92.0, -91.0], [-92.0, -91.0]])
        u10 = np.array([[0.0, -5.0], [0.0, 0.0]])
        v10 = np.array([[0.0, 0.0], [0.0, 0.0]])
        ds = xr.Dataset(
            {"u10": (("y", "x"), u10), "v10": (("y", "x"), v10)},
            coords={"latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon)},
        )
        path = directory / f"rtma_{hour:%Y%m%d}_{hour:%H}z.nc"
        ds.to_netcdf(path)
        return path

    def test_returns_none_when_cache_file_is_missing(self):
        with tempfile.TemporaryDirectory() as root:
            stations = pd.DataFrame({"station_id": ["A"], "lat": [35.0], "lon": [-91.0]})
            result = wind_direction.lookup_hour_wind(Path(root), pd.Timestamp("2025-01-01 00:00"), stations)
        self.assertIsNone(result)

    def test_returns_wind_for_each_station_when_cache_file_exists(self):
        with tempfile.TemporaryDirectory() as root:
            hour = pd.Timestamp("2025-01-01 00:00")
            self._write_rtma_file(Path(root), hour)
            stations = pd.DataFrame({"station_id": ["A"], "lat": [35.0], "lon": [-91.0]})
            result = wind_direction.lookup_hour_wind(Path(root), hour, stations)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["A"], 90.0, places=3)


class BuildWindDirectionLookupTests(unittest.TestCase):
    def test_reads_each_unique_hour_once_and_skips_missing_hours(self):
        with tempfile.TemporaryDirectory() as root:
            present_hour = pd.Timestamp("2025-01-01 00:00")
            LookupHourWindTests()._write_rtma_file(Path(root), present_hour)
            stations = pd.DataFrame({"station_id": ["A"], "lat": [35.0], "lon": [-91.0]})
            hours = pd.Series([present_hour, present_hour, pd.Timestamp("2025-01-02 00:00")])
            result = wind_direction.build_wind_direction_lookup(Path(root), hours, stations)
        self.assertEqual(len(result), 1)  # only the present hour produced a row
        self.assertEqual(result.iloc[0]["station_id"], "A")

    def test_returns_empty_frame_when_no_hours_have_cache_files(self):
        with tempfile.TemporaryDirectory() as root:
            stations = pd.DataFrame({"station_id": ["A"], "lat": [35.0], "lon": [-91.0]})
            result = wind_direction.build_wind_direction_lookup(
                Path(root), pd.Series([pd.Timestamp("2025-01-01 00:00")]), stations)
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ["valid_time_hour", "station_id", "wind_from_deg"])


if __name__ == "__main__":
    unittest.main()
