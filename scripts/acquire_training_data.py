"""
Unified data-acquisition orchestrator. Does not reimplement any backfill
logic - scripts/backfill_synoptic.py, backfill_hrrr.py, and
backfill_rtma_for_hrrr.py already have their own resumable manifests and
real content validation (see docs/spatial_fuel_moisture_runbook.md). This
script's job is sequencing (the runbook's own order: Synoptic obs -> HRRR
-> RTMA-for-HRRR, since RTMA backfill discovers its analysis window from
local HRRR valid times), an environment preflight, and a final aggregate
report - nothing here should duplicate a sub-script's own logic.

Each step runs as a real subprocess invocation of that script's own CLI,
so every step remains exactly as independently runnable and resumable as
before this orchestrator existed. A failed step does not abort the
sequence - each step is already designed to tolerate partial failure and
resume - but its exit code is recorded and surfaced in the final summary.

Static raster acquisition (static_features/download_sources.py) is
skipped by default: it needs real user-supplied source file/URL
selections this orchestrator cannot invent (see docs/
spatial_fuel_moisture_runbook.md section 6). --include-static-rasters
exists for when those selections are ready; until then this prints a
clear "skipped: not yet configured" line rather than silently omitting it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from scripts import check_environment
from scripts import raw_archive_coverage_report

SCRIPTS_DIR = Path(__file__).resolve().parent
STEP_ORDER = ("synoptic", "hrrr", "rtma")
STEP_SCRIPTS = {
    "synoptic": SCRIPTS_DIR / "backfill_synoptic.py",
    "hrrr": SCRIPTS_DIR / "backfill_hrrr.py",
    "rtma": SCRIPTS_DIR / "backfill_rtma_for_hrrr.py",
}
# backfill_rtma_for_hrrr.py has no --days-back (its window is derived from
# local HRRR valid times, not a standalone lookback) - never pass it one.
STEPS_SUPPORTING_DAYS_BACK = {"synoptic", "hrrr"}


class StepResult(NamedTuple):
    step: str
    skipped: bool
    exit_code: Optional[int]
    reason: Optional[str]


def _build_step_args(step: str, args: argparse.Namespace) -> List[str]:
    step_args = []
    if args.start:
        step_args += ["--start", args.start]
    if args.end:
        step_args += ["--end", args.end]
    if args.days_back is not None and step in STEPS_SUPPORTING_DAYS_BACK:
        step_args += ["--days-back", str(args.days_back)]
    if args.workers is not None:
        step_args += ["--workers", str(args.workers)]
    if args.limit is not None:
        step_args += ["--limit", str(args.limit)]
    if args.dry_run:
        step_args.append("--dry-run")
    if args.force:
        step_args.append("--force")
    return step_args


def run_step(step: str, args: argparse.Namespace) -> StepResult:
    script = STEP_SCRIPTS[step]
    command = [sys.executable, str(script)] + _build_step_args(step, args)
    print(f"\n=== {step}: {' '.join(command)} ===")
    completed = subprocess.run(command)
    return StepResult(step=step, skipped=False, exit_code=completed.returncode, reason=None)


def run_static_rasters(args: argparse.Namespace) -> StepResult:
    if not args.include_static_rasters:
        return StepResult(step="static_rasters", skipped=True, exit_code=None,
                          reason="not yet configured - needs real DEM/NLCD/LANDFIRE source selections; pass --include-static-rasters once ready")
    script = SCRIPTS_DIR.parent / "static_features" / "download_sources.py"
    print(f"\n=== static_rasters: {sys.executable} {script} ===")
    print("--include-static-rasters was passed, but this orchestrator does not know which source "
          "files/URLs to use - run static_features/download_sources.py directly with --*-file/--*-url "
          "arguments per docs/spatial_fuel_moisture_runbook.md section 6.")
    return StepResult(step="static_rasters", skipped=True, exit_code=None,
                      reason="requires manually-supplied --*-file/--*-url arguments; run download_sources.py directly")


def run_acquisition(args: argparse.Namespace) -> List[StepResult]:
    if not args.skip_doctor:
        print("=== environment preflight ===")
        doctor_results = check_environment.run_all()
        doctor_exit = check_environment.summarize(doctor_results)
        if doctor_exit != 0:
            print("\nEnvironment preflight failed - aborting before touching the network. Pass --skip-doctor to override.")
            sys.exit(doctor_exit)

    results = []
    for step in STEP_ORDER:
        if step in args.skip:
            results.append(StepResult(step=step, skipped=True, exit_code=None, reason="excluded via --skip"))
            continue
        results.append(run_step(step, args))
    results.append(run_static_rasters(args))

    print("\n=== raw archive coverage report ===")
    raw_archive_coverage_report.generate()

    return results


def summarize_steps(results: List[StepResult]) -> int:
    print("\n=== acquisition summary ===")
    failed = False
    for result in results:
        if result.skipped:
            print(f"  {result.step}: skipped ({result.reason})")
        elif result.exit_code == 0:
            print(f"  {result.step}: ok")
        else:
            print(f"  {result.step}: FAILED (exit {result.exit_code})")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequence the Synoptic/HRRR/RTMA backfill scripts with an environment preflight and a final coverage report.")
    parser.add_argument("--start", help="Passed through to each step that supports it")
    parser.add_argument("--end", help="Passed through to each step that supports it")
    parser.add_argument("--days-back", type=int, help="Passed through to synoptic/hrrr (rtma derives its window from local HRRR files instead)")
    parser.add_argument("--workers", type=int, help="Passed through to each step")
    parser.add_argument("--limit", type=int, help="Passed through to each step")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-doctor", action="store_true", help="Skip the environment preflight")
    parser.add_argument("--skip", action="append", choices=STEP_ORDER, default=[],
                        help="Exclude a step (repeatable), e.g. --skip rtma")
    parser.add_argument("--include-static-rasters", action="store_true",
                        help="Attempt the static-raster step (still requires manual source-file/URL setup)")
    parsed_args = parser.parse_args()

    step_results = run_acquisition(parsed_args)
    sys.exit(summarize_steps(step_results))
