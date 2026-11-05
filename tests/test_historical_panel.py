import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from fire_weather_ml import historical_panel

PYRETECHNICS_AVAILABLE = importlib.util.find_spec("pyretechnics") is not None

# Moniteau County, MO (fips 29135) - a real point verified (in this test
# suite's own setup, see docs/fire_weather_ml_plan.md) to fall inside a real
# county polygon in risk_fusion/county_boundaries.geojson with a recorded
# normal in risk_fusion/county_precip_normals.json, so integration tests can
# use the project's REAL, checked-in precip-normals data (small, static,
# no network) rather than faking county geometry too.
REAL_MO_LAT, REAL_MO_LON = 38.5, -92.5


class LoadAlignedWeatherTests(unittest.TestCase):
    def _write_csv(self, path: Path):
        pd.DataFrame({
            "station_id": ["IN_DOMAIN", "OUT_OF_DOMAIN"],
            "lat": [REAL_MO_LAT, 42.0],  # 42.0 is north of the MO_BBOX
            "lon": [REAL_MO_LON, -103.0],
            "valid_time": ["2025-07-14T16:00:00+00:00", "2025-07-14T16:00:00+00:00"],
            "rtma_temp_c": [25.0, 25.0],
            "rtma_rh": [40.0, 40.0],
            "rtma_wind_ms": [5.0, 5.0],
            "hrrr_precip_increment_mm": [0.0, np.nan],
            "initial_fm": [10.0, 10.0],
        }).to_csv(path, index=False)

    def test_filters_to_the_bbox_and_renames_columns(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "aligned.csv"
            self._write_csv(path)
            result = historical_panel.load_aligned_weather(path)
        self.assertEqual(list(result["station_id"]), ["IN_DOMAIN"])
        for column in ("temp_c", "rh_pct", "wind_ms", "precip_mm", "fm10_observed_pct"):
            self.assertIn(column, result.columns)

    def test_fills_missing_precip_with_zero_rather_than_dropping_the_row(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "aligned.csv"
            pd.DataFrame({
                "station_id": ["A"], "lat": [REAL_MO_LAT], "lon": [REAL_MO_LON],
                "valid_time": ["2025-07-14T16:00:00+00:00"], "rtma_temp_c": [25.0], "rtma_rh": [40.0],
                "rtma_wind_ms": [5.0], "hrrr_precip_increment_mm": [np.nan], "initial_fm": [10.0],
            }).to_csv(path, index=False)
            result = historical_panel.load_aligned_weather(path)
        self.assertEqual(result.loc[0, "precip_mm"], 0.0)


@unittest.skipUnless(PYRETECHNICS_AVAILABLE, "pyretechnics is not importable in this environment")
class BuildHistoricalPanelIntegrationTests(unittest.TestCase):
    """
    End-to-end wiring test with a small synthetic static bundle + RTMA
    cache + aligned CSV, but the REAL risk_fusion county-boundary/precip-
    normals data (see REAL_MO_LAT/REAL_MO_LON above) - proves the whole
    Phase 2 pipeline (static lookup -> wind direction -> precip normals ->
    derived moisture -> Rothermel labels) actually produces a labeled
    panel, without needing the real multi-year RTMA archive.
    """

    def _write_static_bundle(self, path: Path):
        lat = np.full((2, 2), REAL_MO_LAT)
        lon = np.full((2, 2), REAL_MO_LON)
        ds = xr.Dataset(
            {
                "slope_degrees": (("y", "x"), np.full((2, 2), 5.0)),
                "aspect_sin": (("y", "x"), np.zeros((2, 2))),
                "aspect_cos": (("y", "x"), np.ones((2, 2))),
                "canopy_cover_pct": (("y", "x"), np.full((2, 2), 10.0)),
                "canopy_height_m": (("y", "x"), np.full((2, 2), 3.0)),
                "fuel_model_fbfm40": (("y", "x"), np.full((2, 2), 101.0)),
                "static_valid_mask": (("y", "x"), np.ones((2, 2))),
            },
            coords={"latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon)},
        )
        ds.to_netcdf(path)

    def _write_rtma_cache(self, directory: Path, hours):
        for hour in hours:
            lat = np.full((2, 2), REAL_MO_LAT)
            lon = np.full((2, 2), REAL_MO_LON)
            ds = xr.Dataset(
                {"u10": (("y", "x"), np.full((2, 2), 2.0)), "v10": (("y", "x"), np.full((2, 2), 3.0))},
                coords={"latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon)},
            )
            ds.to_netcdf(directory / f"rtma_{hour:%Y%m%d}_{hour:%H}z.nc")

    def _write_aligned_csv(self, path: Path, hours):
        rng = np.random.default_rng(0)
        n = len(hours)
        pd.DataFrame({
            "station_id": ["STN1"] * n,
            "lat": [REAL_MO_LAT] * n,
            "lon": [REAL_MO_LON] * n,
            "valid_time": [h.isoformat() for h in hours],
            "rtma_temp_c": rng.uniform(20, 32, n),
            "rtma_rh": rng.uniform(20, 50, n),
            "rtma_wind_ms": rng.uniform(2, 8, n),
            "hrrr_precip_increment_mm": np.zeros(n),
            "initial_fm": rng.uniform(8, 14, n),
        }).to_csv(path, index=False)

    def test_produces_a_labeled_panel_end_to_end(self):
        hours = pd.date_range("2025-07-01", periods=6, freq="h", tz="UTC")
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            static_bundle_path = root_path / "bundle.nc"
            rtma_cache_dir = root_path / "rtma"
            rtma_cache_dir.mkdir()
            aligned_path = root_path / "aligned.csv"

            self._write_static_bundle(static_bundle_path)
            self._write_rtma_cache(rtma_cache_dir, hours)
            self._write_aligned_csv(aligned_path, hours)

            panel, coverage = historical_panel.build_historical_panel(
                aligned_weather_path=aligned_path,
                static_bundle_path=static_bundle_path,
                rtma_cache_dir=rtma_cache_dir,
            )

        self.assertEqual(coverage["usable_stations"], 1)
        self.assertEqual(coverage["rows_dropped_missing_wind_direction"], 0)
        self.assertEqual(len(panel), len(hours))
        for column in ("ros_ch_per_h", "kbdi", "gdd_accum", "fm1_pct", "fm100_pct"):
            self.assertIn(column, panel.columns)
        self.assertGreater(panel["ros_ch_per_h"].notna().sum(), 0)


if __name__ == "__main__":
    unittest.main()
