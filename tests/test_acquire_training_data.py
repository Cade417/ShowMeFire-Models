import argparse
import unittest
from unittest.mock import MagicMock, patch

from scripts import acquire_training_data as orchestrator


def _args(**overrides):
    defaults = dict(start=None, end=None, days_back=None, workers=None, limit=None,
                    dry_run=False, force=False, skip_doctor=True, skip=[], include_static_rasters=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class BuildStepArgsTests(unittest.TestCase):
    def test_passes_through_common_flags(self):
        args = _args(start="2026-01-01", end="2026-01-31", workers=4, limit=10, dry_run=True, force=True)
        step_args = orchestrator._build_step_args("synoptic", args)
        self.assertIn("--start", step_args)
        self.assertIn("2026-01-01", step_args)
        self.assertIn("--workers", step_args)
        self.assertIn("--dry-run", step_args)
        self.assertIn("--force", step_args)

    def test_days_back_included_for_synoptic_and_hrrr(self):
        args = _args(days_back=30)
        self.assertIn("--days-back", orchestrator._build_step_args("synoptic", args))
        self.assertIn("--days-back", orchestrator._build_step_args("hrrr", args))

    def test_days_back_excluded_for_rtma(self):
        # backfill_rtma_for_hrrr.py has no --days-back flag - passing one would crash it.
        args = _args(days_back=30)
        self.assertNotIn("--days-back", orchestrator._build_step_args("rtma", args))


class RunStaticRastersTests(unittest.TestCase):
    def test_skipped_by_default(self):
        result = orchestrator.run_static_rasters(_args())
        self.assertTrue(result.skipped)
        self.assertIn("not yet configured", result.reason)

    def test_still_skipped_when_opted_in_without_source_files(self):
        # download_sources.py needs real --*-file/--*-url args this orchestrator cannot invent.
        result = orchestrator.run_static_rasters(_args(include_static_rasters=True))
        self.assertTrue(result.skipped)


class RunAcquisitionTests(unittest.TestCase):
    def test_runs_all_three_steps_in_order_when_doctor_skipped(self):
        calls = []

        def fake_run_step(step, args):
            calls.append(step)
            return orchestrator.StepResult(step=step, skipped=False, exit_code=0, reason=None)

        with patch.object(orchestrator, "run_step", side_effect=fake_run_step), \
             patch.object(orchestrator.raw_archive_coverage_report, "generate", return_value={}):
            results = orchestrator.run_acquisition(_args())

        self.assertEqual(calls, ["synoptic", "hrrr", "rtma"])
        self.assertEqual([r.step for r in results], ["synoptic", "hrrr", "rtma", "static_rasters"])

    def test_skip_flag_excludes_a_step_without_running_it(self):
        calls = []

        def fake_run_step(step, args):
            calls.append(step)
            return orchestrator.StepResult(step=step, skipped=False, exit_code=0, reason=None)

        with patch.object(orchestrator, "run_step", side_effect=fake_run_step), \
             patch.object(orchestrator.raw_archive_coverage_report, "generate", return_value={}):
            results = orchestrator.run_acquisition(_args(skip=["rtma"]))

        self.assertNotIn("rtma", calls)
        rtma_result = next(r for r in results if r.step == "rtma")
        self.assertTrue(rtma_result.skipped)

    def test_one_step_failing_does_not_abort_the_rest(self):
        calls = []

        def fake_run_step(step, args):
            calls.append(step)
            exit_code = 1 if step == "hrrr" else 0
            return orchestrator.StepResult(step=step, skipped=False, exit_code=exit_code, reason=None)

        with patch.object(orchestrator, "run_step", side_effect=fake_run_step), \
             patch.object(orchestrator.raw_archive_coverage_report, "generate", return_value={}):
            results = orchestrator.run_acquisition(_args())

        # All three still attempted even though hrrr "failed".
        self.assertEqual(calls, ["synoptic", "hrrr", "rtma"])
        hrrr_result = next(r for r in results if r.step == "hrrr")
        self.assertEqual(hrrr_result.exit_code, 1)

    def test_coverage_report_runs_even_after_a_step_failure(self):
        def fake_run_step(step, args):
            return orchestrator.StepResult(step=step, skipped=False, exit_code=1, reason=None)

        with patch.object(orchestrator, "run_step", side_effect=fake_run_step), \
             patch.object(orchestrator.raw_archive_coverage_report, "generate", return_value={}) as mock_generate:
            orchestrator.run_acquisition(_args())

        mock_generate.assert_called_once()

    def test_doctor_failure_aborts_before_any_step_runs(self):
        calls = []

        def fake_run_step(step, args):
            calls.append(step)
            return orchestrator.StepResult(step=step, skipped=False, exit_code=0, reason=None)

        with patch.object(orchestrator.check_environment, "run_all", return_value=[]), \
             patch.object(orchestrator.check_environment, "summarize", return_value=1), \
             patch.object(orchestrator, "run_step", side_effect=fake_run_step):
            with self.assertRaises(SystemExit):
                orchestrator.run_acquisition(_args(skip_doctor=False))

        self.assertEqual(calls, [])

    def test_doctor_pass_allows_steps_to_run(self):
        with patch.object(orchestrator.check_environment, "run_all", return_value=[]), \
             patch.object(orchestrator.check_environment, "summarize", return_value=0), \
             patch.object(orchestrator, "run_step",
                         return_value=orchestrator.StepResult("synoptic", False, 0, None)) as mock_step, \
             patch.object(orchestrator.raw_archive_coverage_report, "generate", return_value={}):
            orchestrator.run_acquisition(_args(skip_doctor=False))

        self.assertTrue(mock_step.called)


class SummarizeStepsTests(unittest.TestCase):
    def test_all_ok_returns_zero(self):
        results = [orchestrator.StepResult("a", False, 0, None), orchestrator.StepResult("b", True, None, "skipped")]
        self.assertEqual(orchestrator.summarize_steps(results), 0)

    def test_any_nonzero_exit_returns_one(self):
        results = [orchestrator.StepResult("a", False, 0, None), orchestrator.StepResult("b", False, 1, None)]
        self.assertEqual(orchestrator.summarize_steps(results), 1)


if __name__ == "__main__":
    unittest.main()
