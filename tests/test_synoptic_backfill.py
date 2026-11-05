import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import backfill_synoptic as backfill


def args(**overrides):
    values = {"start": "2026-07-01", "end": "2026-07-03", "days_back": 365, "chunk_days": 2,
              "states": None, "networks": None, "limit": None, "archive_dir": None,
              "workers": 1, "token": "test-token", "dry_run": False, "force": False}
    values.update(overrides); return SimpleNamespace(**values)


def fake_response(day_strs, station_id="ASLM7"):
    times = [f"{day}T15:13:00Z" for day in day_strs]
    return {
        "STATION": [{
            "STID": station_id, "LATITUDE": "38.0", "LONGITUDE": "-92.0", "STATE": "MO",
            "OBSERVATIONS": {"date_time": times, "fuel_moisture_set_1": [10.0] * len(times),
                              "air_temp_set_1": [70.0] * len(times), "relative_humidity_set_1": [50.0] * len(times)},
        }]
    }


class SynopticBackfillTests(unittest.TestCase):
    def test_dry_run_performs_no_fetch_or_manifest_write(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "raw_data"; manifest = archive / "manifest.json"
            with patch.object(backfill, "_fetch_chunk", side_effect=AssertionError("dry-run fetched")):
                result = backfill.run(args(archive_dir=str(archive), dry_run=True), manifest_path=manifest)
            self.assertEqual(result, 0); self.assertFalse(manifest.exists())

    def test_backfill_writes_one_file_per_day_and_resumes(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "raw_data"; archive.mkdir(); manifest = archive / "backfill_manifest.json"
            calls = []
            def fake_fetch_chunk(chunk_start, chunk_end, token, states, networks, ignored):
                calls.append((chunk_start, chunk_end))
                days = [(chunk_start + __import__("datetime").timedelta(days=i)).date().isoformat()
                        for i in range((chunk_end - chunk_start).days)]
                from spatial.synoptic_capture import split_by_day
                return split_by_day(fake_response(days))
            with patch.object(backfill, "_fetch_chunk", side_effect=fake_fetch_chunk):
                self.assertEqual(backfill.run(args(archive_dir=str(archive)), manifest), 0)
                self.assertEqual(backfill.run(args(archive_dir=str(archive)), manifest), 0)
            self.assertEqual(len(calls), 2)  # 3-day range / 2-day chunks = 2 chunks, not refetched on resume
            for day in ("20260701", "20260702", "20260703"):
                self.assertTrue((archive / f"raw_data_{day}.json").exists())
            document = json.loads(manifest.read_text())
            self.assertEqual(len(document["days"]), 3)
            self.assertTrue(all(r["status"] == "complete" for r in document["days"].values()))

    def test_ignored_stations_are_filtered_out(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "raw_data"; archive.mkdir(); manifest = archive / "backfill_manifest.json"
            def fake_fetch_chunk(chunk_start, chunk_end, token, states, networks, ignored):
                from spatial.synoptic_capture import split_by_day
                response = fake_response(["2026-07-01"], station_id="IGNOREME")
                return split_by_day(response, ignored_stations=ignored)
            with patch.object(backfill, "_fetch_chunk", side_effect=fake_fetch_chunk), \
                 patch.object(backfill, "get_ignored_stations", return_value={"IGNOREME"}):
                backfill.run(args(archive_dir=str(archive), start="2026-07-01", end="2026-07-01"), manifest)
            document = json.loads((archive / "raw_data_20260701.json").read_text())
            self.assertEqual(document["STATION"], [])

    def test_unavailable_chunk_is_recorded_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "raw_data"; archive.mkdir(); manifest = archive / "backfill_manifest.json"
            with patch.object(backfill, "_fetch_chunk", side_effect=RuntimeError("404 not found")):
                result = backfill.run(args(archive_dir=str(archive), start="2026-07-01", end="2026-07-01"), manifest)
            self.assertEqual(result, 1)
            record = json.loads(manifest.read_text())["days"]["2026-07-01"]
            self.assertEqual(record["error_class"], "unavailable")


if __name__ == "__main__": unittest.main()
