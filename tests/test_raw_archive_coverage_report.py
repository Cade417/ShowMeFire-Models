import json
import tempfile
import unittest
from pathlib import Path

from scripts import raw_archive_coverage_report as report


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


class TimestampedSummaryTests(unittest.TestCase):
    def test_empty_manifest_reports_no_entries(self):
        self.assertEqual(report._timestamped_summary({}, "hrrr")["status"], "no_manifest_entries")

    def test_counts_complete_and_failed(self):
        entries = {
            "2026-01-01T12:00:00+00:00": {"status": "complete"},
            "2026-01-02T12:00:00+00:00": {"status": "failed"},
            "2026-01-03T12:00:00+00:00": {"status": "complete"},
        }
        summary = report._timestamped_summary(entries, "hrrr")
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["complete"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertAlmostEqual(summary["failed_fraction"], 1 / 3, places=4)

    def test_healthy_flag_respects_threshold(self):
        entries = {f"2026-01-{day:02d}T12:00:00+00:00": {"status": "failed"} for day in range(1, 3)}
        entries["2026-01-03T12:00:00+00:00"] = {"status": "complete"}
        summary = report._timestamped_summary(entries, "hrrr")
        self.assertGreater(summary["failed_fraction"], report.FAILED_FRACTION_WARN_THRESHOLD)
        self.assertFalse(summary["healthy"])

    def test_gap_days_detected_within_range(self):
        entries = {
            "2026-01-01T12:00:00+00:00": {"status": "complete"},
            "2026-01-05T12:00:00+00:00": {"status": "complete"},
        }
        summary = report._timestamped_summary(entries, "hrrr")
        self.assertEqual(summary["covered_calendar_days"], 2)
        self.assertEqual(summary["gap_days_within_range"], 3)  # Jan 2,3,4 missing

    def test_no_gap_when_contiguous(self):
        entries = {f"2026-01-0{day}T12:00:00+00:00": {"status": "complete"} for day in range(1, 4)}
        summary = report._timestamped_summary(entries, "hrrr")
        self.assertEqual(summary["gap_days_within_range"], 0)


class RawDataSummaryTests(unittest.TestCase):
    def test_zero_station_days_reported_separately_from_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = _write(Path(tmp) / "manifest.json", {
                "days": {
                    "2026-01-01": {"status": "complete", "stations": 0},
                    "2026-01-02": {"status": "complete", "stations": 114},
                }
            })
            summary = report.raw_data_summary(manifest_path)
        self.assertEqual(summary["zero_station_days"], 1)
        self.assertEqual(summary["failed"], 0)  # a zero-station day is a legitimate complete outcome, not a failure
        self.assertTrue(summary["healthy"])

    def test_missing_manifest_reports_not_found(self):
        summary = report.raw_data_summary(Path("/definitely/not/a/real/manifest.json"))
        self.assertEqual(summary["status"], "no_manifest_found")


class StaticRasterSummaryTests(unittest.TestCase):
    def test_reports_not_yet_acquired_when_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            import paths
            original = paths.STATIC_SOURCE_DIR
            try:
                paths.STATIC_SOURCE_DIR = Path(tmp)
                summary = report.static_raster_summary()
            finally:
                paths.STATIC_SOURCE_DIR = original
        self.assertEqual(summary["status"], "not_yet_acquired")


class GenerateTests(unittest.TestCase):
    def test_writes_report_and_computes_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            hrrr_manifest = _write(Path(tmp) / "hrrr_manifest.json", {
                "runs": {f"2026-01-{day:02d}T12:00:00+00:00": {"status": "complete"} for day in range(1, 32)}
            })
            rtma_manifest = _write(Path(tmp) / "rtma_manifest.json", {
                "analyses": {f"2026-01-{day:02d}T12:00:00+00:00": {"status": "complete"} for day in range(1, 32)}
            })
            raw_data_manifest = _write(Path(tmp) / "raw_data_manifest.json", {
                "days": {f"2026-01-{day:02d}": {"status": "complete", "stations": 100} for day in range(1, 32)}
            })

            original_hrrr, original_rtma, original_raw = report.hrrr_summary, report.rtma_summary, report.raw_data_summary

            def fake_hrrr():
                return report._timestamped_summary(json.loads(hrrr_manifest.read_text())["runs"], "hrrr")

            def fake_rtma():
                return report._timestamped_summary(json.loads(rtma_manifest.read_text())["analyses"], "rtma")

            def fake_raw_data():
                return original_raw(raw_data_manifest)

            report.hrrr_summary, report.rtma_summary, report.raw_data_summary = fake_hrrr, fake_rtma, fake_raw_data
            try:
                output_path = Path(tmp) / "raw_archive_coverage.json"
                result = report.generate(output_path)
            finally:
                report.hrrr_summary, report.rtma_summary, report.raw_data_summary = original_hrrr, original_rtma, original_raw

            self.assertTrue(output_path.exists())
            # 31 covered days is below the 180-day gate threshold - the gate must fail, not silently pass.
            self.assertFalse(result["raw_archive_ready"]["pass"])
            self.assertTrue(result["raw_archive_ready"]["failing_reasons"])


if __name__ == "__main__":
    unittest.main()
