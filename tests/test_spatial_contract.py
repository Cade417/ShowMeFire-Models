import unittest

import numpy as np
import pandas as pd

from spatial.grid import idw_initial_analysis, nearest_mapping
from spatial.observations import causal_initial_and_targets
from spatial.physics import evolve_fm
from pipelines.unpack_archive_zip import _destination_for
import paths


class SpatialContractTests(unittest.TestCase):
    def test_initial_observation_is_causal_and_stale_values_are_masked(self):
        obs = pd.DataFrame({"time": pd.to_datetime(["2026-07-12T08:00Z", "2026-07-12T12:05Z", "2026-07-12T13:00Z"]),
                            "fuel_moisture": [8.0, 99.0, 7.0]})
        initial, age, targets = causal_initial_and_targets(obs, pd.Timestamp("2026-07-12T12:00Z"), [pd.Timestamp("2026-07-12T13:00Z")], max_age_hours=3)
        self.assertTrue(np.isnan(initial))
        self.assertEqual(targets[0][0], 7.0)

    def test_idw_returns_coverage_layers(self):
        lat = np.array([[38.0, 38.0], [39.0, 39.0]])
        lon = np.array([[-94.0, -93.0], [-94.0, -93.0]])
        analysis, distance, effective = idw_initial_analysis([38.0, 39.0], [-94.0, -93.0], [5.0, 15.0], lat, lon)
        self.assertEqual(analysis.shape, lat.shape)
        self.assertTrue(np.all(distance >= 0))
        self.assertTrue(np.all(effective >= 1))

    def test_physics_moves_toward_equilibrium(self):
        result = evolve_fm(20.0, [30.0, 30.0], [10.0, 10.0])
        self.assertLess(result[-1], result[0])
        self.assertLess(result[0], 20.0)

    def test_rtma_archive_members_route_before_generic_netcdf(self):
        destination, name = _destination_for("rtma/rtma_20260712_12z.nc")
        self.assertEqual(destination, paths.CACHE_RTMA_DIR)
        self.assertEqual(name, "rtma_20260712_12z.nc")


if __name__ == "__main__":
    unittest.main()
