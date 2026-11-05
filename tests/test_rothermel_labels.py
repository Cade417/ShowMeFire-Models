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
import unittest.mock
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

    api/services/spread_rate.py's own module-level imports (core.executors,
    services.beta_products, etc.) assume it runs from inside the api/
    package - not the case here. `_compute_cell_ros` itself never calls any
    of those names (they're used by other functions in that file), so
    stubbing them as empty placeholder modules is enough to let the module
    body execute without pulling in api/'s full runtime.
    """

    # spread_rate.py's module BODY (not just _compute_cell_ros) evaluates
    # some of these at import time (e.g. `BETA_ROOT / "spread_rate"`,
    # `spread_rate_poll_minutes()`), so stub values need to behave like the
    # real thing structurally (a Path, a zero-arg callable) even though
    # they're never functionally exercised by this test.
    _NOOP = staticmethod(lambda *a, **kw: None)
    _ZERO = staticmethod(lambda *a, **kw: 0)
    _STUB_MODULES = {
        "core": {"run_in_process_pool_async": _NOOP},
        "core.executors": {"run_in_process_pool_async": _NOOP},
        "services": {},
        "services.beta_products": {"BETA_ROOT": Path("."), "load_manifest": _NOOP, "save_manifest": _NOOP},
        "services.fire_behavior_static": {"diagnostics": _NOOP, "load_static_fields": _NOOP},
        "services.rtma_capture": {
            "ensure_latest_analysis_cached": _NOOP, "is_analysis_hour_cached": _NOOP,
            "latest_complete_hour": _NOOP, "spread_rate_poll_minutes": _ZERO, "warmup_rtma_cache": _NOOP,
        },
        "services.spread_rate_moisture": {
            "MIN_CONDITIONING_HOURS": 0, "TARGET_CONDITIONING_HOURS": 0, "condition_moisture": _NOOP,
        },
    }

    @classmethod
    def setUpClass(cls):
        import types

        cls._sys_modules_patch = {}
        for name, attributes in cls._STUB_MODULES.items():
            stub = types.ModuleType(name)
            for attribute, value in attributes.items():
                setattr(stub, attribute, value)
            cls._sys_modules_patch[name] = stub
        cls._patcher = unittest.mock.patch.dict(sys.modules, cls._sys_modules_patch)
        cls._patcher.start()

        spec = importlib.util.spec_from_file_location("_api_spread_rate_for_test", API_SPREAD_RATE_PATH)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - api module has its own heavy import chain
            cls._patcher.stop()
            raise unittest.SkipTest(f"could not load api/services/spread_rate.py standalone: {exc}")
        # staticmethod() prevents `self._api_compute_cell_ros(...)` from
        # binding self as an implicit first positional arg (it's a plain
        # function, not a method - without this it would silently collide
        # with the real first parameter, fuel_code).
        cls._api_compute_cell_ros = staticmethod(module._compute_cell_ros)

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()

    def test_matches_api_service_on_sample_inputs(self):
        # Percentages, as compute_row_label's own public contract expects
        # (its docstring/param names are *_pct - it converts to fractions
        # internally via _percent_to_fraction).
        sample_pct = dict(
            fuel_code=101, fm1=8.0, fm10=9.0, fm100=12.0, live_herb=90.0, live_woody=120.0,
            wind_ms=6.0, wind_from_deg=200.0, slope_deg=15.0, aspect_deg=180.0,
            canopy_cover_pct=10.0, canopy_height_m=3.0,
        )
        # api/services/spread_rate.py's own compute_spread_rate_grid already
        # converts fm1/fm10/fm100/live_herb/live_woody to FRACTIONS before
        # calling _compute_cell_ros (see its `_percent_to_fraction(...)`
        # calls) - _compute_cell_ros itself takes fractions, not percent, so
        # the fair comparison converts those five fields here to match what
        # the real caller would actually pass it.
        sample_fraction = {
            **sample_pct,
            "fm1": sample_pct["fm1"] / 100.0, "fm10": sample_pct["fm10"] / 100.0,
            "fm100": sample_pct["fm100"] / 100.0, "live_herb": sample_pct["live_herb"] / 100.0,
            "live_woody": sample_pct["live_woody"] / 100.0,
        }
        api_result = self._api_compute_cell_ros(**sample_fraction)
        self.assertIsNotNone(api_result)
        api_ros_ch_h, api_direction = api_result

        ours = labels.compute_row_label(
            fuel_model_code=sample_pct["fuel_code"], fm1_pct=sample_pct["fm1"], fm10_pct=sample_pct["fm10"],
            fm100_pct=sample_pct["fm100"], live_herbaceous_pct=sample_pct["live_herb"],
            live_woody_pct=sample_pct["live_woody"], wind_ms=sample_pct["wind_ms"],
            wind_from_deg=sample_pct["wind_from_deg"], slope_deg=sample_pct["slope_deg"],
            aspect_deg=sample_pct["aspect_deg"], canopy_cover_pct=sample_pct["canopy_cover_pct"],
            canopy_height_m=sample_pct["canopy_height_m"],
        )
        self.assertIsNotNone(ours)
        self.assertAlmostEqual(ours.ros_ch_per_h, api_ros_ch_h, places=6)
        self.assertAlmostEqual(ours.spread_direction_deg, api_direction, places=6)


if __name__ == "__main__":
    unittest.main()
