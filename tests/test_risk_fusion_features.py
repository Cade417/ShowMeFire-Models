import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import paths
from risk_fusion import features as rff

COUNTY_CELLS_PATH = Path(__file__).resolve().parent.parent / "risk_fusion" / "county_cells.json"


class RiskFusionFeaturesSmokeTests(unittest.TestCase):
    """
    Lightweight smoke coverage on the training side - the exhaustive
    correctness tests (KBDI physical sanity, GDD reset logic, calendar
    edge cases) live in api/tests/test_risk_fusion_features.py against
    the byte-identical mirror; verify_contract_mirrors.py is what
    guarantees this copy behaves identically without duplicating every
    assertion here.
    """

    def test_calendar_features_runs(self):
        dates = pd.date_range("2026-03-01", periods=5, freq="D")
        result = rff.calendar_features(dates)
        self.assertEqual(len(result), 5)

    def test_kbdi_runs_and_stays_bounded(self):
        result = rff.keetch_byram_drought_index(
            np.full(100, 30.0), np.zeros(100), mean_annual_precip_mm=900.0
        )
        self.assertTrue(np.all(result["kbdi"] >= 0.0))
        self.assertTrue(np.all(result["kbdi"] <= 203.2 + 1e-6))


class RiskFusionFeaturesRealDataIntegrationTests(unittest.TestCase):
    """Exercises reduce_cells_to_county against a real cached HRRR file and the real county-cell index."""

    @classmethod
    def setUpClass(cls):
        if not COUNTY_CELLS_PATH.exists():
            raise unittest.SkipTest("county_cells.json not built - run scripts/build_county_cells.py first")
        cls.cell_index = json.loads(COUNTY_CELLS_PATH.read_text(encoding="utf-8"))

        samples = sorted(paths.CACHE_HRRR_DIR.glob("hrrr_*.nc"))
        if not samples:
            raise unittest.SkipTest("no cached HRRR file available")
        with xr.open_dataset(samples[0], decode_cf=False) as ds:
            cls.temp_k = np.asarray(ds["t2m"].values)[0]  # first lead step
            cls.rh = np.asarray(ds["r2"].values)[0]

    def test_grid_shape_matches_county_cell_index(self):
        self.assertEqual(list(self.temp_k.shape), self.cell_index["grid_shape"])

    def test_reduces_real_temperature_grid_to_plausible_per_county_values(self):
        temp_c = self.temp_k - 273.15
        by_county = rff.reduce_cells_to_county(temp_c, self.cell_index["cell_to_fips"], reducer=np.nanmean)
        self.assertEqual(len(by_county), self.cell_index["counties_covered"])
        # Missouri summer/spring HRRR 2m temps should be within a sane
        # meteorological range for every county, not just on average.
        for fips, value in by_county.items():
            self.assertGreater(value, -40.0, msg=f"county {fips} temp implausibly low: {value}")
            self.assertLess(value, 55.0, msg=f"county {fips} temp implausibly high: {value}")

    def test_reduces_real_rh_grid_within_valid_percent_range(self):
        by_county = rff.reduce_cells_to_county(self.rh, self.cell_index["cell_to_fips"], reducer=np.nanmean)
        for fips, value in by_county.items():
            self.assertGreaterEqual(value, 0.0, msg=f"county {fips}")
            self.assertLessEqual(value, 100.0, msg=f"county {fips}")


if __name__ == "__main__":
    unittest.main()
