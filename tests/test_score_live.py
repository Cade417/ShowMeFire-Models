import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from risk_fusion import score_live


class ResolveTargetRunTests(unittest.TestCase):
    def test_succeeds_on_first_try_uses_today(self):
        today = datetime.now(timezone.utc).date()
        with patch.object(score_live, "fetch_hrrr", return_value=Path("dummy.nc")), \
             patch.object(score_live, "clear_local_cache"):
            target_date, run_dt = score_live.resolve_target_run()
        self.assertEqual(target_date, today)
        self.assertEqual(run_dt.hour, 12)

    def test_steps_back_a_day_on_unavailable_failure(self):
        calls = []

        def fake_fetch(run_dt, cache_dir=None):
            calls.append(run_dt)
            if len(calls) == 1:
                raise RuntimeError("404 not found - run not published yet")
            return Path("dummy.nc")

        with patch.object(score_live, "fetch_hrrr", side_effect=fake_fetch), \
             patch.object(score_live, "clear_local_cache"):
            target_date, run_dt = score_live.resolve_target_run(max_lookback_days=2)

        self.assertEqual(len(calls), 2)  # today failed once (not retried - "unavailable"), yesterday succeeded
        self.assertEqual(target_date, calls[1].date())
        self.assertEqual(calls[1].date(), calls[0].date() - timedelta(days=1))

    def test_retries_transient_failures_within_the_same_day(self):
        calls = []

        def fake_fetch(run_dt, cache_dir=None):
            calls.append(run_dt)
            if len(calls) == 1:
                raise FileNotFoundError("[Errno 2] No such file or directory: 'x'")
            return Path("dummy.nc")

        with patch.object(score_live, "fetch_hrrr", side_effect=fake_fetch), \
             patch.object(score_live, "clear_local_cache"), \
             patch.object(score_live.time, "sleep"):
            target_date, run_dt = score_live.resolve_target_run(max_lookback_days=2)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].date(), calls[1].date())  # same day, retried in place - not stepped back

    def test_raises_after_exhausting_all_lookback_days(self):
        with patch.object(score_live, "fetch_hrrr", side_effect=RuntimeError("404 not found")), \
             patch.object(score_live, "clear_local_cache") as mock_clear:
            with self.assertRaises(RuntimeError):
                score_live.resolve_target_run(max_lookback_days=2)
        self.assertEqual(mock_clear.call_count, 3)  # today + 2 lookback days, one attempt each ("unavailable")


class TargetRunFileTests(unittest.TestCase):
    def test_matches_fetch_hrrr_naming_convention(self):
        run_dt = datetime(2026, 8, 8, 12, tzinfo=timezone.utc)
        path = score_live.target_run_file(run_dt, cache_dir=Path("/tmp/cache"))
        self.assertEqual(path.name, "hrrr_20260808_12z_f04-15.nc")


class BuildTargetPanelTests(unittest.TestCase):
    def test_passes_exactly_one_target_file(self):
        run_dt = datetime(2026, 8, 8, 12, tzinfo=timezone.utc)
        with patch.object(score_live.build_county_days, "build", return_value=pd.DataFrame({"a": [1]})) as mock_build:
            score_live.build_target_panel(run_dt)
        _, kwargs = mock_build.call_args
        self.assertEqual(len(kwargs["run_files"]), 1)
        self.assertEqual(kwargs["run_files"][0].name, "hrrr_20260808_12z_f04-15.nc")


class MainArgValidationTests(unittest.TestCase):
    def test_no_fetch_without_date_errors_before_any_network_or_disk_work(self):
        with patch.object(sys, "argv", ["score_live.py", "--no-fetch"]):
            with self.assertRaises(SystemExit):
                score_live.main()


if __name__ == "__main__":
    unittest.main()
