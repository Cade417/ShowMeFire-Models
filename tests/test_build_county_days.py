import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from risk_fusion import build_county_days as bcd


def _synthetic_hrrr_dataset(n_steps=12, shape=(2, 2)):
    """A tiny 2x2-grid, 12-step dataset shaped like a real HRRR cache file."""
    steps = np.arange(4, 4 + n_steps).astype("timedelta64[h]")
    t2m = 273.15 + 25.0 + np.random.default_rng(0).normal(0, 1, size=(n_steps, *shape))
    r2 = np.full((n_steps, *shape), 40.0)
    u10 = np.full((n_steps, *shape), 3.0)
    v10 = np.full((n_steps, *shape), 4.0)  # |wind| = 5 m/s
    tp = np.cumsum(np.full((n_steps, *shape), 1.0), axis=0)  # 1mm/step cumulative
    return xr.Dataset(
        {
            "t2m": (["step", "y", "x"], t2m),
            "r2": (["step", "y", "x"], r2),
            "u10": (["step", "y", "x"], u10),
            "v10": (["step", "y", "x"], v10),
            "tp": (["step", "y", "x"], tp, {"units": "mm", "GRIB_stepType": "accum"}),
        },
        coords={"step": steps, "time": np.datetime64("2026-07-01T12:00:00")},
    )


class RunFilenameParsingTests(unittest.TestCase):
    def test_parses_valid_filename(self):
        result = bcd._run_date_from_filename(Path("hrrr_20260701_12z_f04-15.nc"))
        self.assertEqual(result, pd.Timestamp("2026-07-01T12:00:00Z"))

    def test_returns_none_for_unrecognized_filename(self):
        self.assertIsNone(bcd._run_date_from_filename(Path("not_a_hrrr_file.nc")))


class GridShapeGuardTests(unittest.TestCase):
    def test_raises_when_expected_grid_shape_does_not_match(self):
        # Regression test for a real, observed failure mode: an
        # uncropped/full-CONUS HRRR file landing in the cache under a
        # filename that collides with this repo's own naming convention -
        # county_cells.json's cell indices would silently read the wrong
        # geographic cells against a grid of a different shape.
        ds = _synthetic_hrrr_dataset(shape=(2, 2))
        cell_to_fips = {"0,0": "29019"}

        with patch("xarray.open_dataset") as mock_open, \
             patch.object(bcd, "_load_county_reference", return_value={}):
            mock_open.return_value = ds
            with self.assertRaises(ValueError) as context:
                bcd.process_one_run(Path("hrrr_20260701_12z_f04-15.nc"), cell_to_fips, expected_grid_shape=(273, 267))
        self.assertIn("grid shape", str(context.exception))

    def test_passes_when_expected_grid_shape_matches(self):
        ds = _synthetic_hrrr_dataset(shape=(2, 2))
        cell_to_fips = {"0,0": "29019"}

        with patch("xarray.open_dataset") as mock_open, \
             patch.object(bcd, "_load_county_reference", return_value={}):
            mock_open.return_value = ds
            frame = bcd.process_one_run(Path("hrrr_20260701_12z_f04-15.nc"), cell_to_fips, expected_grid_shape=(2, 2))
        self.assertEqual(len(frame), 1)

    def test_none_expected_grid_shape_skips_the_check(self):
        ds = _synthetic_hrrr_dataset(shape=(2, 2))
        cell_to_fips = {"0,0": "29019"}

        with patch("xarray.open_dataset") as mock_open, \
             patch.object(bcd, "_load_county_reference", return_value={}):
            mock_open.return_value = ds
            frame = bcd.process_one_run(Path("hrrr_20260701_12z_f04-15.nc"), cell_to_fips, expected_grid_shape=None)
        self.assertEqual(len(frame), 1)


class ProcessOneRunTests(unittest.TestCase):
    def test_produces_one_row_per_county_with_plausible_values(self):
        ds = _synthetic_hrrr_dataset()
        cell_to_fips = {"0,0": "29019", "0,1": "29019", "1,0": "29027", "1,1": "29027"}

        with patch("xarray.open_dataset") as mock_open, \
             patch.object(bcd, "_load_county_reference", return_value={}):
            mock_open.return_value = ds
            frame = bcd.process_one_run(Path("hrrr_20260701_12z_f04-15.nc"), cell_to_fips)

        self.assertEqual(len(frame), 2)
        self.assertEqual(set(frame["county_fips"]), {"29019", "29027"})
        for _, row in frame.iterrows():
            self.assertAlmostEqual(row["temp_max_c"], 25.0, delta=5.0)
            self.assertAlmostEqual(row["rh_mean"], 40.0, places=3)
            # |wind| = 5 m/s -> knots -> x0.8 fire-reduction factor
            expected_wind_kts = 5.0 * bcd.rff.MPS_TO_KNOTS * bcd.rff.FIRE_WIND_REDUCTION_FACTOR
            self.assertAlmostEqual(row["wind_kts_max"], expected_wind_kts, places=3)
            self.assertFalse(row["fm_features_available"])
            self.assertFalse(row["gust_available"])
            self.assertIsNone(row["gust_kts_max"])

    def test_precipitation_is_spatially_averaged_not_summed_across_cells(self):
        # Regression test for the bug caught during manual verification:
        # summing interval_mm across a county's grid cells inflates the
        # daily total by the county's cell count (one county produced an
        # 8614mm/day reading before this was fixed to nanmean).
        ds = _synthetic_hrrr_dataset()
        cell_to_fips = {"0,0": "29019", "0,1": "29019", "1,0": "29019", "1,1": "29019"}

        with patch("xarray.open_dataset") as mock_open, \
             patch.object(bcd, "_load_county_reference", return_value={}):
            mock_open.return_value = ds
            frame = bcd.process_one_run(Path("hrrr_20260701_12z_f04-15.nc"), cell_to_fips)

        # 1mm/step accumulation differenced across 12 steps of the fixed
        # 1mm/step cumulative series -> ~1mm/step interval, summed over
        # 12 full-day steps -> plausible single-digit-to-low-double-digit
        # daily total, not thousands of mm.
        self.assertLess(frame.iloc[0]["precip_24h_mm"], 50.0)


class AddStatefulFeaturesTests(unittest.TestCase):
    def test_kbdi_and_gdd_are_computed_independently_per_county(self):
        dates = pd.date_range("2026-06-01", periods=5, freq="D").strftime("%Y-%m-%d")
        panel = pd.DataFrame({
            "county_fips": ["29019"] * 5 + ["29027"] * 5,
            "valid_local_date": list(dates) * 2,
            "temp_max_c": [30.0] * 5 + [10.0] * 5,   # hot county vs cold county
            "precip_24h_mm": [0.0] * 5 + [0.0] * 5,
            "run_id": ["r"] * 10,
        })
        result = bcd.add_stateful_features(panel)

        hot = result[result.county_fips == "29019"].sort_values("valid_local_date")
        cold = result[result.county_fips == "29027"].sort_values("valid_local_date")
        self.assertGreater(hot["kbdi"].iloc[-1], cold["kbdi"].iloc[-1])
        self.assertGreater(hot["gdd_accum_since_mar1"].iloc[-1], cold["gdd_accum_since_mar1"].iloc[-1])

    def test_adds_calendar_columns(self):
        panel = pd.DataFrame({
            "county_fips": ["29019"],
            "valid_local_date": ["2026-01-01"],
            "temp_max_c": [20.0],
            "precip_24h_mm": [0.0],
            "run_id": ["r"],
        })
        result = bcd.add_stateful_features(panel)
        self.assertTrue(bool(result["is_holiday_us"].iloc[0]))


class BuildSinceFilterTests(unittest.TestCase):
    def test_since_excludes_runs_before_that_date(self):
        stub_row = lambda run_name: pd.DataFrame([{
            "county_fips": "29019", "valid_local_date": run_name[5:13], "run_id": run_name,
            "temp_max_c": 20.0, "precip_24h_mm": 0.0,
        }])
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root)
            for name in ["hrrr_20260601_12z_f04-15.nc", "hrrr_20260701_12z_f04-15.nc", "hrrr_20260801_12z_f04-15.nc"]:
                (cache / name).touch()
            with patch.object(bcd.paths, "CACHE_HRRR_DIR", cache), \
                 patch.object(bcd, "_load_county_cells", return_value={"cell_to_fips": {}, "grid_shape": [2, 2]}), \
                 patch.object(bcd, "process_one_run",
                              side_effect=lambda path, cell_to_fips, expected_grid_shape=None: stub_row(path.name)) as mock_process:
                bcd.build(since=date(2026, 7, 1))
            processed_names = sorted(call.args[0].name for call in mock_process.call_args_list)
        self.assertEqual(processed_names, ["hrrr_20260701_12z_f04-15.nc", "hrrr_20260801_12z_f04-15.nc"])

    def test_since_none_processes_every_run(self):
        stub_row = lambda run_name: pd.DataFrame([{
            "county_fips": "29019", "valid_local_date": run_name[5:13], "run_id": run_name,
            "temp_max_c": 20.0, "precip_24h_mm": 0.0,
        }])
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root)
            for name in ["hrrr_20260601_12z_f04-15.nc", "hrrr_20260701_12z_f04-15.nc"]:
                (cache / name).touch()
            with patch.object(bcd.paths, "CACHE_HRRR_DIR", cache), \
                 patch.object(bcd, "_load_county_cells", return_value={"cell_to_fips": {}, "grid_shape": [2, 2]}), \
                 patch.object(bcd, "process_one_run",
                              side_effect=lambda path, cell_to_fips, expected_grid_shape=None: stub_row(path.name)) as mock_process:
                bcd.build()
        self.assertEqual(mock_process.call_count, 2)


if __name__ == "__main__":
    unittest.main()
