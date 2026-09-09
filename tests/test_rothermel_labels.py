"""
Tests rothermel_labels.py, including the cross-check against
api/services/spread_rate.py's own physics logic (the concrete proof this
module computes the real Rothermel calculation, not an approximation of
it - see fire_weather_ml/rothermel_labels.py's module docstring).

The cross-check is skipped, not failed, when `pyretechnics` cannot be
imported - this Windows dev environment cannot currently build it from
source against NumPy 2.x (a real, discovered incompatibility; see
requirements-pyretechnics.txt). Non-physics unit conversions/helpers are
tested unconditionally since they don't need the package.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fire_weather_ml import rothermel_labels as labels

PYRETECHNICS_AVAILABLE = importlib.util.find_spec("pyretechnics") is not None
API_SPREAD_RATE_PATH = Path(__file__).resolve().parents[2] / "api" / "services" / "spread_rate.py"


class WindFromDegreesTests(unittest.TestCase):
    def test_north_wind_reports_from_the_north(self):
        # u=0, v negative -> wind blowing FROM the north (blowing southward)
        self.assertAlmostEqual(labels.wind_from_degrees(0.0, -5.0), 0.0, places=3)

    def test_matches_known_quadrant(self):
        # u negative -> wind vector points west (blowing toward the west),
        # meaning it originates FROM the east: 90 degrees.
        self.assertAlmostEqual(labels.wind_from_degrees(-5.0, 0.0), 90.0, places=3)


class SpreadDirectionDegreesTests(unittest.TestCase):
    def test_pure_x_direction_is_90_degrees(self):
        self.assertAlmostEqual(labels.spread_direction_degrees((1.0, 0.0, 0.0)), 90.0, places=3)

    def test_pure_y_direction_is_0_degrees(self):
        self.assertAlmostEqual(labels.spread_direction_degrees((0.0, 1.0, 0.0)), 0.0, places=3)


class ComputeRowLabelUnavailableFuelModelTests(unittest.TestCase):
    def test_returns_none_for_out_of_range_fuel_code_without_importing_pyretechnics(self):
        # fuel_model_code range-checking happens before pyretechnics is
        # imported, so this must work even where pyretechnics can't build.
        result = labels.compute_row_label(
            fuel_model_code=0, fm1_pct=10, fm10_pct=10, fm100_pct=10,
            live_herbaceous_pct=60, live_woody_pct=90, wind_ms=5, wind_from_deg=180,
            slope_deg=10, aspect_deg=180, canopy_cover_pct=20, canopy_height_m=5,
        )
        self.assertIsNone(result)


@unittest.skipUnless(PYRETECHNICS_AVAILABLE, "pyretechnics is not importable in this environment")
class ComputeLabelsForPanelTests(unittest.TestCase):
    def test_appends_expected_label_columns(self):
        panel = pd.DataFrame([{
            "fuel_model_code": 101, "fm1_pct": 8.0, "fm10_pct": 9.0, "fm100_pct": 12.0,
            "live_herbaceous_pct": 90.0, "live_woody_pct": 120.0, "wind_ms": 6.0,
            "wind_from_deg": 200.0, "slope_deg": 15.0, "aspect_deg": 180.0,
            "canopy_cover_pct": 10.0, "canopy_height_m": 3.0,
        }])
        result = labels.compute_labels_for_panel(panel)
        for column in ("ros_ch_per_h", "spread_direction_deg", "fireline_intensity_kw_per_m", "flame_length_m"):
            self.assertIn(column, result.columns)


@unittest.skipUnless(PYRETECHNICS_AVAILABLE, "pyretechnics is not importable in this environment")
@unittest.skipUnless(API_SPREAD_RATE_PATH.is_file(), "sibling api/ repo checkout not found")
class CrossCheckAgainstApiSpreadRateTests(unittest.TestCase):
    """
    Loads api/services/spread_rate.py's `_compute_cell_ros` directly from
    its file path (not a package import - api/ isn't on sys.path and this
    repo must never import from it at runtime; this test-only load is
    strictly for numerical comparison) and asserts it agrees with this
    module's compute_row_label on identical sample inputs.
    """

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("_api_spread_rate_for_test", API_SPREAD_RATE_PATH)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - api module has its own heavy import chain
            raise unittest.SkipTest(f"could not load api/services/spread_rate.py standalone: {exc}")
        cls._api_compute_cell_ros = module._compute_cell_ros

    def test_matches_api_service_on_sample_inputs(self):
        sample = dict(
            fuel_code=101, fm1=8.0, fm10=9.0, fm100=12.0, live_herb=90.0, live_woody=120.0,
            wind_ms=6.0, wind_from_deg=200.0, slope_deg=15.0, aspect_deg=180.0,
            canopy_cover_pct=10.0, canopy_height_m=3.0,
        )
        api_result = self._api_compute_cell_ros(**sample)
        self.assertIsNotNone(api_result)
        api_ros_ch_h, api_direction = api_result

        ours = labels.compute_row_label(
            fuel_model_code=sample["fuel_code"], fm1_pct=sample["fm1"], fm10_pct=sample["fm10"],
            fm100_pct=sample["fm100"], live_herbaceous_pct=sample["live_herb"],
            live_woody_pct=sample["live_woody"], wind_ms=sample["wind_ms"],
            wind_from_deg=sample["wind_from_deg"], slope_deg=sample["slope_deg"],
            aspect_deg=sample["aspect_deg"], canopy_cover_pct=sample["canopy_cover_pct"],
            canopy_height_m=sample["canopy_height_m"],
        )
        self.assertIsNotNone(ours)
        self.assertAlmostEqual(ours.ros_ch_per_h, api_ros_ch_h, places=6)
        self.assertAlmostEqual(ours.spread_direction_deg, api_direction, places=6)


if __name__ == "__main__":
    unittest.main()
