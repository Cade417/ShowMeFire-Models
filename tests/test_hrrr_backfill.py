import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import xarray as xr

from scripts import backfill_hrrr as backfill


def args(**overrides):
    values = {"start": "2026-07-01", "end": "2026-07-03", "days_back": 365, "init_hour": 12,
              "lead_start": 4, "lead_end": 5, "limit": None, "cache_dir": None,
              "workers": 1, "dry_run": False, "force": False,
              "existing_runs_only": False}
    values.update(overrides); return SimpleNamespace(**values)


def valid_hrrr(path, lead_hours=(4, 5)):
    variables = {name: (("step", "y", "x"), np.ones((len(lead_hours), 2, 2), dtype="float32"))
                 for name in ("t2m", "r2", "u10", "v10", "tp")}
    ds = xr.Dataset(variables,
                     coords={"step": list(lead_hours)})
    ds["tp"].attrs.update({"units": "kg m**-2", "GRIB_stepType": "accum"})
    ds.to_netcdf(path)
    return path


class HRRRBackfillTests(unittest.TestCase):
    def test_daily_runs_covers_inclusive_date_range(self):
        start = datetime(2026, 7, 1, tzinfo=timezone.utc); end = datetime(2026, 7, 3, tzinfo=timezone.utc)
        runs = list(backfill.daily_runs(start, end, 12))
        self.assertEqual([run.date().isoformat() for run in runs], ["2026-07-01", "2026-07-02", "2026-07-03"])
        self.assertTrue(all(run.hour == 12 for run in runs))

    def test_dry_run_performs_no_fetch_or_manifest_write(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); cache = root / "hrrr"; manifest = cache / "manifest.json"
            result = backfill.run(args(cache_dir=str(cache), dry_run=True),
                                   fetcher=lambda *_ , **__: self.fail("dry-run fetched"), manifest_path=manifest)
            self.assertEqual(result, 0); self.assertFalse(manifest.exists())

    def test_backfill_resumes_without_refetching_complete_runs(self):
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "hrrr"; cache.mkdir(); manifest = cache / "backfill_manifest.json"
            calls = []
            def fetcher(run_dt, cache_dir, lead_hours=(4, 5)):
                calls.append(run_dt)
                target = Path(cache_dir) / f"hrrr_{run_dt:%Y%m%d_%H}z_f{lead_hours[0]:02d}-{lead_hours[-1]:02d}.nc"
                return valid_hrrr(target, lead_hours)
            with patch.object(backfill.paths, "CACHE_HRRR_DIR", cache):
                self.assertEqual(backfill.run(args(cache_dir=str(cache)), fetcher, manifest), 0)
                self.assertEqual(backfill.run(args(cache_dir=str(cache)), fetcher, manifest), 0)
            self.assertEqual(len(calls), 3)
            document = json.loads(manifest.read_text())
            self.assertEqual(len(document["runs"]), 3)
            self.assertTrue(all(record["status"] == "complete" for record in document["runs"].values()))

    def test_unavailable_run_is_recorded_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "hrrr"; cache.mkdir(); manifest = cache / "backfill_manifest.json"
            def unavailable(*_, **__): raise FileNotFoundError("404 not found")
            result = backfill.run(args(cache_dir=str(cache), end="2026-07-01", start="2026-07-01"), unavailable, manifest)
            self.assertEqual(result, 1)
            record = next(iter(json.loads(manifest.read_text())["runs"].values()))
            self.assertEqual(record["error_class"], "unavailable"); self.assertEqual(record["attempts"], 1)

    def test_poisoned_herbie_cache_filenotfounderror_is_retried_not_abandoned(self):
        # Regression test for a real failure observed live: a transient
        # network hiccup mid-download can leave Herbie's own local cache
        # poisoned (.idx cached, .grib2 subset never written), so retrying
        # the fetch raises a bare FileNotFoundError with a generic OS
        # message (no "404"/"not found" wording) on every attempt unless
        # the poisoned directory is cleared first. This must be classified
        # "transient" (retried up to max_attempts), not "unavailable"
        # (which the sibling test above confirms still applies when the
        # message explicitly says so).
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "hrrr"; cache.mkdir(); manifest = cache / "backfill_manifest.json"
            def poisoned_cache(*_, **__):
                raise FileNotFoundError("[Errno 2] No such file or directory: 'subset_c94bb1fe__hrrr.t12z.wrfsfcf04.grib2'")
            with patch.object(backfill, "clear_local_cache") as mock_clear:
                result = backfill.run(args(cache_dir=str(cache), end="2026-07-01", start="2026-07-01"), poisoned_cache, manifest)
            self.assertEqual(result, 1)
            record = next(iter(json.loads(manifest.read_text())["runs"].values()))
            self.assertEqual(record["error_class"], "transient")
            self.assertEqual(record["attempts"], 3)  # max_attempts for non-context fetches
            self.assertEqual(mock_clear.call_count, 3)  # cleared before every attempt, not just retries

    def test_classify_failure_distinguishes_message_content_from_exception_type(self):
        self.assertEqual(backfill.classify_fetch_failure(FileNotFoundError("404 not found")), "unavailable")
        self.assertEqual(
            backfill.classify_fetch_failure(FileNotFoundError("[Errno 2] No such file or directory: 'x'")),
            "transient",
        )
        self.assertEqual(backfill.classify_fetch_failure(ValueError("no index file was found")), "deterministic")

    def test_precip_context_writes_compact_f00_f03_sidecar(self):
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "hrrr"; cache.mkdir(); manifest = cache / "context_manifest.json"
            def fetcher(run_dt, cache_dir, lead_hours):
                target = Path(cache_dir) / f"hrrr_precip_context_{run_dt:%Y%m%d_%H}z_f00-03.nc"
                data = xr.DataArray(np.arange(4, dtype="float32")[:, None, None], dims=("step", "y", "x"),
                                    coords={"step": np.arange(4)},
                                    attrs={"units": "kg m**-2", "GRIB_stepType": "accum"})
                xr.Dataset({"tp": data}).to_netcdf(target); return target
            result = backfill.run(args(start="2026-07-01", end="2026-07-01", cache_dir=str(cache),
                                       precip_context=True), fetcher, manifest)
            self.assertEqual(result, 0)
            self.assertTrue((cache / "hrrr_precip_context_20260701_12z_f00-03.nc").exists())

    def test_existing_runs_only_applies_to_normal_backfill(self):
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root) / "hrrr"; cache.mkdir(); manifest = cache / "manifest.json"
            valid_hrrr(cache / "hrrr_20260702_12z_f04-15.nc", range(4, 16))
            manifest.write_text(json.dumps({"version": 1, "runs": {
                datetime(2026, 7, 1, 12, tzinfo=timezone.utc).isoformat(): {"status": "failed"}
            }}))
            result = backfill.run(
                args(cache_dir=str(cache), lead_end=15, existing_runs_only=True),
                fetcher=lambda *_, **__: self.fail("existing-runs-only fetched"),
                manifest_path=manifest,
            )
            self.assertEqual(result, 0)
            document = json.loads(manifest.read_text())
            self.assertEqual(list(document["runs"]), [datetime(2026, 7, 2, 12, tzinfo=timezone.utc).isoformat()])


if __name__ == "__main__": unittest.main()
