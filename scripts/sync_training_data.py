#!/usr/bin/env python3
"""One-command entrypoint for pulling fresh training data before a retrain.

Thin wrapper around pipeline/run.py, which now owns the sequencing and
freshness-reporting logic this script and scripts/acquire_training_data.py
used to each duplicate a copy of. Flags below map 1:1 onto pipeline.run's
generic --only/--skip/--days-back/--limit/--workers so existing commands
keep working unchanged.

Usage:
    python scripts/sync_training_data.py
    python scripts/sync_training_data.py --dry-run
    python scripts/sync_training_data.py --status
    python scripts/sync_training_data.py --skip-archives --synoptic-days-back 30
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.registry import DATA_SOURCES
from pipeline.run import ordered_keys, report_status, run_sync


def main():
    parser = argparse.ArgumentParser(description="Pull fresh training data from all sources in one command")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be fetched without fetching")
    parser.add_argument("--status", action="store_true",
                         help="Only print current data freshness (no fetching) and exit")
    parser.add_argument("--skip-archives", action="store_true", help="Skip the server rsync pull")
    parser.add_argument("--skip-synoptic", action="store_true", help="Skip Synoptic station backfill")
    parser.add_argument("--skip-rtma", action="store_true", help="Skip RTMA backfill for cached HRRR runs")
    parser.add_argument("--skip-rrfs", action="store_true", help="Skip RRFS cycle backfill")
    parser.add_argument("--skip-fv3hires", action="store_true", help="Skip FV3-HIRES cycle capture")
    parser.add_argument("--rrfs-days-back", type=int, default=21,
                         help="How many days back to check for missing RRFS cycles (default: 21, clamped to the "
                              "2026-08-12 operational archive start)")
    parser.add_argument("--fv3hires-days-back", type=int, default=3,
                         help="How many days back to sweep for FV3-HIRES cycles (default: 3, matching NOMADS' "
                              "short rolling retention - going further back finds nothing)")
    parser.add_argument("--synoptic-days-back", type=int, default=14,
                         help="How many days back to check for missing Synoptic data "
                              "(default: 14, matches the biweekly retrain cadence)")
    parser.add_argument("--synoptic-limit", type=int, help="Limit Synoptic chunks fetched this run")
    parser.add_argument("--rtma-limit", type=int, help="Limit RTMA analyses fetched this run")
    parser.add_argument("--rtma-days-back", type=int, default=21,
                         help="Only backfill RTMA for HRRR runs initialized in the last N days (default: 21)")
    parser.add_argument("--rtma-full-history", action="store_true",
                         help="Ignore --rtma-days-back and backfill RTMA for the entire cached HRRR history")
    args = parser.parse_args()

    if args.status:
        report_status()
        return

    # This script's historical scope: synoptic/rtma/rrfs/fv3hires/archives
    # (not hrrr - that's scripts/acquire_training_data.py's job).
    skip_map = {"synoptic": args.skip_synoptic, "rtma": args.skip_rtma,
                "rrfs": args.skip_rrfs, "fv3hires": args.skip_fv3hires, "archives": args.skip_archives}
    keys = [k for k in ordered_keys() if k != "hrrr" and not skip_map.get(k)]

    # Each source gets its own historical --*-days-back flag rather than one
    # shared value - preserve that by running sources individually instead
    # of a single run_sync call with one days_back.
    per_source_days_back = {
        "synoptic": args.synoptic_days_back,
        "rtma": None if not args.rtma_full_history else None,  # rtma never takes --days-back (see registry.py)
        "rrfs": args.rrfs_days_back,
        "fv3hires": args.fv3hires_days_back,
    }
    per_source_limit = {"synoptic": args.synoptic_limit, "rtma": args.rtma_limit, "rrfs": args.rtma_limit}

    overall_failed = False
    for key in keys:
        code = run_sync([key], dry_run=args.dry_run, days_back=per_source_days_back.get(key),
                         limit=per_source_limit.get(key), skip_doctor=True)
        overall_failed = overall_failed or code != 0

    print()
    report_status()
    if overall_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
