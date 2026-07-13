import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr

from scripts import backfill_rtma_for_hrrr as backfill


def args(hrrr_dir, **overrides):
    values = {"start": None, "end": None, "limit": None, "hrrr_dir": str(hrrr_dir), "dry_run": False, "force": False, "window": "anchor"}
    values.update(overrides); return SimpleNamespace(**values)


def valid_rtma(path):
    xr.Dataset({name: (("y", "x"), np.ones((2, 2), dtype="float32")) for name in ("t2m", "r2", "u10", "v10")}).to_netcdf(path)
    return path


class RTMABackfillTests(unittest.TestCase):
    def test_discovery_deduplicates_runs_and_ignores_malformed_files(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name in ("hrrr_20260712_12z_f04-15.nc", "hrrr_20260712_12z_other.nc", "hrrr_20260712_06z_f04-15.nc", "bad.nc"):
                (root / name).touch()
            runs = backfill.discover_hrrr_runs(root)
            self.assertEqual(len(runs), 2); self.assertEqual(sorted(len(files) for files in runs.values()), [1, 2])

    def test_dry_run_performs_no_fetch_or_manifest_write(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); hrrr = root / "hrrr"; hrrr.mkdir(); (hrrr / "hrrr_20260712_12z_f04-15.nc").touch()
            manifest = root / "manifest.json"
            result = backfill.run(args(hrrr, dry_run=True), fetcher=lambda *_: self.fail("dry-run fetched"), manifest_path=manifest)
            self.assertEqual(result, 0); self.assertFalse(manifest.exists())

    def test_backfill_resumes_without_refetching_complete_anchors(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); hrrr = root / "hrrr"; cache = root / "rtma"; hrrr.mkdir(); cache.mkdir()
            for hour in (6, 12): (hrrr / f"hrrr_20260712_{hour:02d}z_f04-15.nc").touch()
            calls = []
            def fetcher(run_time, cache_dir):
                calls.append(run_time); return valid_rtma(cache_dir / f"rtma_{run_time:%Y%m%d_%H}z.nc")
            manifest = cache / "backfill_manifest.json"
            with patch.object(backfill.paths, "CACHE_RTMA_DIR", cache):
                self.assertEqual(backfill.run(args(hrrr), fetcher, manifest), 0)
                self.assertEqual(backfill.run(args(hrrr), fetcher, manifest), 0)
            self.assertEqual(len(calls), 2)
            document = json.loads(manifest.read_text()); records = document["analyses"]
            self.assertEqual(document["version"], 2)
            self.assertEqual(len(records), 2); self.assertTrue(all(record["status"] == "complete" for record in records.values()))

    def test_unavailable_anchor_is_recorded_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); hrrr = root / "hrrr"; cache = root / "rtma"; hrrr.mkdir(); cache.mkdir()
            (hrrr / "hrrr_20260712_12z_f04-15.nc").touch(); manifest = cache / "backfill_manifest.json"
            def unavailable(*_): raise FileNotFoundError("404 not found")
            with patch.object(backfill.paths, "CACHE_RTMA_DIR", cache):
                self.assertEqual(backfill.run(args(hrrr), unavailable, manifest), 1)
            record = next(iter(json.loads(manifest.read_text())["analyses"].values()))
            self.assertEqual(record["error_class"], "unavailable"); self.assertEqual(record["attempts"], 1)

    def test_teacher_window_has_28_hourly_analyses_for_f04_f15(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); path = root / "hrrr_20260712_12z_f04-15.nc"; path.touch()
            requirement = next(iter(backfill.derive_requirements(backfill.discover_hrrr_runs(root), "teacher").values()))
            self.assertEqual(len(requirement["antecedent"]), 13)
            self.assertEqual(len(requirement["realized"]), 15)
            self.assertEqual(len(requirement["analyses"]), 28)
            self.assertEqual(requirement["analyses"][0].hour, 0)
            self.assertEqual(requirement["analyses"][-1].hour, 3)

    def test_v1_manifest_migrates_complete_anchor_to_analysis(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "manifest.json"
            path.write_text(json.dumps({"version": 1, "runs": {"2026-07-12T12:00:00+00:00": {"status": "complete", "attempts": 1}}}))
            value = backfill.read_manifest(path)
            self.assertEqual(value["version"], 2)
            self.assertEqual(value["analyses"]["2026-07-12T12:00:00+00:00"]["status"], "complete")


if __name__ == "__main__": unittest.main()
