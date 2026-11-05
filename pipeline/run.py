#!/usr/bin/env python3
"""Single sequencing entrypoint for all data-source pullers.

Replaces the logic that used to be duplicated between
scripts/sync_training_data.py and scripts/acquire_training_data.py. Step
order rule (unchanged from both of those): Synoptic obs -> HRRR -> RTMA-for-
HRRR, since RTMA backfill discovers its analysis window from local HRRR
valid times. RRFS/FV3-HIRES/archive-zip-rsync are independent and run after.
A failed step does not abort the run - every underlying puller already
tolerates partial failure and resumes on the next run; this just records
and surfaces the exit code.

Usage:
    python -m pipeline.run
    python -m pipeline.run --dry-run
    python -m pipeline.run --status
    python -m pipeline.run --only synoptic,rrfs
    python -m pipeline.run --skip fv3hires --skip archives
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.jobs import JobManager
from pipeline.registry import DATA_SOURCES, DATA_SOURCES_BY_KEY, INDEPENDENT_STEPS, SEQUENCED_STEPS
from pipeline.status import all_source_status
from scripts import check_environment


def ordered_keys():
    return list(SEQUENCED_STEPS) + list(INDEPENDENT_STEPS)


def run_sync(selected_keys, *, dry_run=False, start=None, end=None, days_back=None, limit=None, workers=None,
             force=False, skip_doctor=False):
    if not skip_doctor:
        print("=== environment preflight ===")
        results = check_environment.run_all()
        if check_environment.summarize(results) != 0:
            print("\nEnvironment preflight failed - aborting before touching the network. Pass --skip-doctor to override.")
            return 1

    manager = JobManager()
    outcomes = {}
    for key in selected_keys:
        source = DATA_SOURCES_BY_KEY[key]
        if not source.available():
            outcomes[key] = "skipped (not available - check script path / SMF_SSH_TARGET & SMF_REMOTE_ARCHIVE_DIR)"
            print(f"\n=== {source.label} ===\n   skipped: {outcomes[key]}")
            continue
        command = source.build_command(dry_run=dry_run, start=start, end=end,
                                        days_back=days_back or source.default_days_back,
                                        limit=limit, workers=workers, force=force)
        print(f"\n=== {source.label} ===")
        print("   $", " ".join(command))
        job = manager.start(source.label, command, source.script.parent.parent)
        manager.wait(job)
        outcomes[key] = "dry-run" if dry_run else ("ok" if job.status == "ok" else f"failed (exit {job.returncode})")

    print("\n=== Sync summary ===")
    for key, status in outcomes.items():
        print(f"   {key}: {status}")
    return 1 if any(status.startswith("failed") for status in outcomes.values()) else 0


def report_status():
    print("=== Data freshness ===")
    for status in all_source_status():
        if not status["available"]:
            print(f"   {status['label']}: not available ({status['note']})")
            continue
        print(f"   {status['label']}: {status['complete']} complete, {status['pending']} pending, "
              f"{status['failed']} failed, latest: {status['latest'] or 'none'} "
              f"(updated_at={status['updated_at'] or 'unknown'})" + (f" - {status['note']}" if status["note"] else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="Only print current data freshness and exit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", help="Comma-separated source keys to run (default: all)")
    parser.add_argument("--start", help="Passed through to sources that accept it")
    parser.add_argument("--end", help="Passed through to sources that accept it")
    parser.add_argument("--skip", action="append", default=[], choices=[s.key for s in DATA_SOURCES],
                         help="Exclude a source (repeatable)")
    parser.add_argument("--days-back", type=int, help="Override each source's default lookback window")
    parser.add_argument("--limit", type=int, help="Limit units of work fetched per source this run")
    parser.add_argument("--workers", type=int, help="Concurrent workers, passed through where supported")
    parser.add_argument("--force", action="store_true", help="Re-fetch even where the manifest says complete")
    parser.add_argument("--skip-doctor", action="store_true", help="Skip the environment preflight")
    args = parser.parse_args()

    if args.status:
        report_status()
        return

    keys = args.only.split(",") if args.only else ordered_keys()
    keys = [k for k in keys if k not in args.skip]
    code = run_sync(keys, dry_run=args.dry_run, start=args.start, end=args.end, days_back=args.days_back,
                     limit=args.limit, workers=args.workers, force=args.force, skip_doctor=args.skip_doctor)
    print()
    report_status()
    sys.exit(code)


if __name__ == "__main__":
    main()
