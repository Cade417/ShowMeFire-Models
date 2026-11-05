import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from risk_fusion import blend_station_forecast as bsf
from risk_fusion import rule_uncertainty as ru
from risk_fusion.county_geometry import load_counties
from spatial.rule_contract import RULE_SPEC


class StationToCountyTests(unittest.TestCase):
    def setUp(self):
        self.counties = load_counties()

    def test_a_real_boone_county_station_resolves_correctly(self):
        # ASLM7's real coordinates from station_forecasts_beta_20260711_12.json -
        # already confirmed to land in Boone county (29019) when this script
        # was run against the real archive.
        fips = bsf.station_to_county(38.81194, -92.25694, self.counties)
        self.assertEqual(fips, "29019")

    def test_a_point_outside_missouri_returns_none(self):
        fips = bsf.station_to_county(39.0, -100.0, self.counties)  # western Kansas
        self.assertIsNone(fips)


class StationPeakProbabilityTests(unittest.TestCase):
    def test_matches_direct_sample_category_probabilities_call(self):
        station_hours = pd.DataFrame({
            "fuel_moisture": [20.0, 20.0, 6.0, 20.0],   # one dangerous hour
            "rh": [60.0, 60.0, 15.0, 60.0],
            "wind_kts": [5.0, 5.0, 30.0, 5.0],
            "fire_danger": [0, 0, 3, 1],
        })
        result = bsf.station_peak_probability(station_hours, RULE_SPEC["thresholds"])

        expected = ru.sample_category_probabilities(
            fm=station_hours["fuel_moisture"].to_numpy(), rh=station_hours["rh"].to_numpy(),
            wind_kts=station_hours["wind_kts"].to_numpy(),
            fm_sigma=np.full(4, bsf.FM_SIGMA_FALLBACK), rh_sigma=np.full(4, bsf.RH_SIGMA_FALLBACK),
            wind_sigma_log=np.full(4, bsf.WIND_SIGMA_LOG_FALLBACK), thresholds=RULE_SPEC["thresholds"],
        )
        self.assertAlmostEqual(
            result["peak_probability_at_or_above_elevated"],
            float(np.max(expected["probability_at_or_above_elevated"])),
        )

    def test_peak_reported_category_is_the_max_fire_danger(self):
        station_hours = pd.DataFrame({
            "fuel_moisture": [20.0, 6.0], "rh": [60.0, 15.0], "wind_kts": [5.0, 30.0],
            "fire_danger": [0, 3],
        })
        result = bsf.station_peak_probability(station_hours, RULE_SPEC["thresholds"])
        self.assertEqual(result["peak_reported_category"], 3)


class BlendTests(unittest.TestCase):
    def test_clips_at_one_when_the_product_exceeds_one(self):
        station_table = pd.DataFrame({
            "station_id": ["X"], "county_fips": ["29019"],
            "peak_probability_at_or_above_elevated": [0.9],
        })
        blended = bsf.blend(station_table, {"29019": 2.0})
        self.assertEqual(blended.loc[0, "blended_probability"], 1.0)

    def test_ordinary_case_is_a_plain_product(self):
        station_table = pd.DataFrame({
            "station_id": ["X"], "county_fips": ["29019"],
            "peak_probability_at_or_above_elevated": [0.1],
        })
        blended = bsf.blend(station_table, {"29019": 0.8})
        self.assertAlmostEqual(blended.loc[0, "blended_probability"], 0.08)


class LoadStationArchiveTests(unittest.TestCase):
    def test_flattens_stations_and_hours_and_converts_wind_to_knots(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "archive.json"
            path.write_text(json.dumps({
                "run_date": "2026-07-11",
                "stations": {
                    "AAA": {"lat": 38.0, "lon": -92.0, "forecasts": [
                        {"time": "2026-07-11T16:00:00Z", "fuel_moisture": 15.0, "rh": 50.0,
                         "wind_speed_ms": 5.0, "fire_danger": 1},
                    ]},
                },
            }))
            frame = bsf.load_station_archive(path)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.attrs["run_date"], "2026-07-11")
        expected_kts = 5.0 * bsf.features.MPS_TO_KNOTS
        self.assertAlmostEqual(frame.loc[0, "wind_kts"], expected_kts)


def _write_station_archive(root: Path, filename: str, run_date: str, stations: dict) -> Path:
    path = Path(root) / filename
    path.write_text(json.dumps({"run_date": run_date, "stations": stations}))
    return path


_ASLM7 = {"ASLM7": {"lat": 38.81194, "lon": -92.25694, "forecasts": [
    {"time": "2026-07-11T16:00:00Z", "fuel_moisture": 20.0, "rh": 60.0, "wind_speed_ms": 2.0, "fire_danger": 0},
]}}


class ArchiveDateTests(unittest.TestCase):
    def test_parses_the_date_out_of_the_beta_filename_pattern(self):
        self.assertEqual(bsf.archive_date(Path("station_forecasts_beta_20260711_12.json")),
                         bsf.datetime(2026, 7, 11).date())

    def test_parses_the_date_out_of_the_plain_filename_pattern(self):
        self.assertEqual(bsf.archive_date(Path("station_forecasts_20260327_12.json")),
                         bsf.datetime(2026, 3, 27).date())

    def test_returns_none_for_an_unrecognized_filename(self):
        self.assertIsNone(bsf.archive_date(Path("not_a_station_archive.json")))


class SelectArchivesDedupTests(unittest.TestCase):
    def test_prefers_one_file_when_both_naming_conventions_cover_the_same_date(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / "station_forecasts_20260327_12.json").write_text("{}")
            (root_path / "station_forecasts_beta_20260327_12.json").write_text("{}")
            with patch.object(bsf.paths, "ARCHIVE_FORECASTS_DIR", root_path):
                selected = bsf._select_archives(None, "2026-03-27", "2026-03-27")
        self.assertEqual(len(selected), 1)

    def test_uses_the_plain_name_when_only_it_covers_a_date(self):
        # 2026-03-30 only exists under the plain naming convention in the
        # real archive - confirms the union (not just the beta glob) is used.
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / "station_forecasts_20260330_12.json").write_text("{}")
            with patch.object(bsf.paths, "ARCHIVE_FORECASTS_DIR", root_path):
                selected = bsf._select_archives(None, "2026-03-30", "2026-03-30")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].name, "station_forecasts_20260330_12.json")


class MainEndToEndTests(unittest.TestCase):
    def _run_main(self, argv, mocked_ratios):
        with patch.object(bsf, "county_weather_multiplier", return_value=mocked_ratios), \
             patch.object(bsf.model_bundle, "fit", return_value={}), \
             patch.object(bsf.model_bundle, "load_training_panel", return_value=pd.DataFrame()), \
             patch.object(sys, "argv", argv):
            bsf.main()

    def test_writes_a_blended_table_using_a_small_synthetic_archive(self):
        with tempfile.TemporaryDirectory() as root:
            archive_path = _write_station_archive(root, "station_forecasts_beta_20260711_12.json",
                                                   "2026-07-11", _ASLM7)
            output_path = Path(root) / "output.csv"
            self._run_main(["blend_station_forecast.py", "--archive", str(archive_path),
                            "--output", str(output_path)], {"29019": 0.95})

            self.assertTrue(output_path.exists())
            result = pd.read_csv(output_path, dtype={"county_fips": str})
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "county_fips"], "29019")
        self.assertEqual(result.loc[0, "county_name"], "Boone")
        self.assertEqual(result.loc[0, "run_date"], "2026-07-11")
        self.assertAlmostEqual(result.loc[0, "county_weather_multiplier"], 0.95)

    def test_unresolved_station_is_dropped_with_a_warning_not_a_crash(self):
        with tempfile.TemporaryDirectory() as root:
            stations = {**_ASLM7, "OUTSIDE": {"lat": 39.0, "lon": -100.0, "forecasts": [
                {"time": "2026-07-11T16:00:00Z", "fuel_moisture": 20.0, "rh": 60.0,
                 "wind_speed_ms": 2.0, "fire_danger": 0},
            ]}}
            archive_path = _write_station_archive(root, "station_forecasts_beta_20260711_12.json",
                                                   "2026-07-11", stations)
            output_path = Path(root) / "output.csv"
            self._run_main(["blend_station_forecast.py", "--archive", str(archive_path),
                            "--output", str(output_path)], {"29019": 1.0})

            result = pd.read_csv(output_path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "station_id"], "ASLM7")

    def test_since_until_processes_every_matching_day_and_combines_them(self):
        with tempfile.TemporaryDirectory() as root:
            _write_station_archive(root, "station_forecasts_beta_20260709_12.json", "2026-07-09", _ASLM7)
            _write_station_archive(root, "station_forecasts_beta_20260710_12.json", "2026-07-10", _ASLM7)
            _write_station_archive(root, "station_forecasts_beta_20260711_12.json", "2026-07-11", _ASLM7)
            output_path = Path(root) / "combined.csv"

            with patch.object(bsf.paths, "ARCHIVE_FORECASTS_DIR", Path(root)):
                self._run_main(["blend_station_forecast.py", "--since", "2026-07-10", "--until", "2026-07-11",
                                "--output", str(output_path)], {"29019": 1.0})

            result = pd.read_csv(output_path)
        # since/until is inclusive on both ends and excludes 07-09 - exactly the 2 matching days, one station each.
        self.assertEqual(sorted(result["run_date"].tolist()), ["2026-07-10", "2026-07-11"])

    def test_archive_and_since_together_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            archive_path = _write_station_archive(root, "station_forecasts_beta_20260711_12.json",
                                                   "2026-07-11", _ASLM7)
            with self.assertRaises(SystemExit):
                self._run_main(["blend_station_forecast.py", "--archive", str(archive_path),
                                "--since", "2026-07-01"], {"29019": 1.0})


if __name__ == "__main__":
    unittest.main()
