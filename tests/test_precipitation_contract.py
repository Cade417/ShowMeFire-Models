import unittest

import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd
import tempfile

from spatial.validate_precip_rebuild import compare

from spatial.precipitation import decode_forecast_precipitation, final_accumulation_mm, normalize_to_mm


class PrecipitationContractTests(unittest.TestCase):
    def test_contract_matches_api_copy(self):
        training = Path(__file__).resolve().parents[1] / "spatial" / "precipitation_contract.json"
        api = Path(__file__).resolve().parents[2] / "api" / "core" / "precipitation_contract.json"
        self.assertEqual(training.read_bytes(), api.read_bytes())

    def dataset(self, values=(1.0, 3.0, 6.0), units="kg m**-2", step_type="accum"):
        step = np.asarray([1, 2, 3], dtype="timedelta64[h]")
        data = xr.DataArray(np.asarray(values)[:, None, None], dims=("step", "y", "x"),
                            coords={"step": step}, attrs={"units": units, "GRIB_stepType": step_type})
        return xr.Dataset({"tp": data})

    def test_supported_unit_conversions(self):
        np.testing.assert_allclose(normalize_to_mm(np.array([2.0]), "kg m-2"), [2.0])
        np.testing.assert_allclose(normalize_to_mm(np.array([0.002]), "m"), [2.0])
        np.testing.assert_allclose(normalize_to_mm(np.array([1.0]), "inches"), [25.4])

    def test_unknown_and_missing_units_fail(self):
        with self.assertRaises(ValueError): normalize_to_mm(np.array([1.0]), None)
        with self.assertRaises(ValueError): normalize_to_mm(np.array([1.0]), "furlongs")

    def test_cumulative_values_are_differenced(self):
        decoded = decode_forecast_precipitation(self.dataset())
        np.testing.assert_allclose(decoded.interval_mm.values[:, 0, 0], [1, 2, 3])
        self.assertEqual(float(final_accumulation_mm(self.dataset()).values[0, 0]), 6.0)

    def test_negative_change_is_zeroed_and_flagged(self):
        decoded = decode_forecast_precipitation(self.dataset((2.0, 1.0, 4.0)))
        np.testing.assert_allclose(decoded.interval_mm.values[:, 0, 0], [2, 0, 3])
        self.assertTrue(bool(decoded.reset_flag.values[1, 0, 0]))

    def test_rebuild_comparison_ignores_only_precipitation_changes(self):
        base = pd.DataFrame({"run_id":["1"],"station_id":["A"],"valid_time":["2026-01-01"],
                             "lead_hour":[4],"target_fm":[8.0],"hrrr_precip_mm":[0.0]})
        rebuilt = base.assign(hrrr_precip_mm=2.0,hrrr_precip_accum_mm=2.0,
                              hrrr_precip_increment_mm=1.0)
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory)/"old.csv", Path(directory)/"new.csv"
            base.to_csv(old,index=False); rebuilt.to_csv(new,index=False)
            self.assertTrue(compare(old,new)["pass"])


if __name__ == "__main__":
    unittest.main()
