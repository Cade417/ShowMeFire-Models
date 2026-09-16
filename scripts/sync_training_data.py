#!/usr/bin/env python3
"""
One-command entrypoint for pulling fresh training data before a retrain.

This wraps the existing pullers rather than reinventing them - each is
already independently incremental/resumable with its own manifest:

    pull_archives.sh          rsync of server-bundled zips (requires SSH target)
    backfill_synoptic.py       Synoptic station obs -> archive/raw_data/backfill_manifest.json
    backfill_rtma_for_hrrr.py  RTMA analyses for cached HRRR runs -> cache/rtma/backfill_manifest.json

The actual friction being solved here isn't "we lack versioning" (each
puller above already tracks exactly what's fresh) - it's "which of three
scripts do I run, with what arguments, in what order." This script answers
that with one command and one summary, without changing what any puller
does.

Usage:
    python scripts/sync_training_data.py
    python scripts/sync_training_data.py --dry-run
    python scripts/sync_training_data.py --status
    python scripts/sync_training_data.py --skip-archives --synoptic-days-back 30
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd, label):
    print(f"\n=== {label} ===")
    print("   $", " ".join(str(c) for c in cmd))
    # PYTHONUTF8/PYTHONIOENCODING are required, not optional: Herbie prints a
    # checkmark/emoji on every fetch attempt, and that print raises
    # UnicodeEncodeError under the ANSI codepage a non-interactive session
    # (Task Scheduler, or this script's own subprocess here) falls back to -
    # same documented incident as scripts/run_daily_score_live.ps1's header
    # comment, just hit again via a different launcher (confirmed live
    # 2026-09-14 running capture_fv3hires.py this same way).
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        print(f"   WARNING: {label} exited with code {result.returncode} (see output above)")
    return result.returncode


def sync(dry_run=False, skip_archives=False, skip_synoptic=False, skip_rtma=False, skip_rrfs=False, skip_fv3hires=False,
         synoptic_days_back=14, synoptic_limit=None, rtma_limit=None, rtma_days_back=21,
         rtma_full_history=False, rrfs_days_back=21, fv3hires_days_back=3):
    """Run the pullers in sequence. Returns a {source: status} dict."""
    results = {}

    if skip_archives:
        results["archives"] = "skipped (--skip-archives)"
    else:
        ssh_target = os.getenv("SMF_SSH_TARGET")
        remote_dir = os.getenv("SMF_REMOTE_ARCHIVE_DIR")
        print("\n=== Archive zip pull (server rsync) ===")
        if not ssh_target or not remote_dir:
            print("   Skipped: SMF_SSH_TARGET / SMF_REMOTE_ARCHIVE_DIR not set - only relevant when this "
                  "machine pulls bundles from a remote server; fine to skip on the training machine itself.")
            results["archives"] = "skipped (no SSH target configured)"
        elif dry_run:
            print(f"   Would rsync {ssh_target}:{remote_dir} -> data/archive_zips/ then unpack")
            results["archives"] = "dry-run"
        else:
            code = _run(["bash", str(REPO_ROOT / "scripts" / "pull_archives.sh")], "Archive zip pull (server rsync)")
            results["archives"] = "ok" if code == 0 else "failed"

    if skip_synoptic:
        results["synoptic"] = "skipped (--skip-synoptic)"
    else:
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "backfill_synoptic.py"),
               "--days-back", str(synoptic_days_back)]
        if dry_run:
            cmd.append("--dry-run")
        if synoptic_limit:
            cmd += ["--limit", str(synoptic_limit)]
        code = _run(cmd, "Synoptic station observations")
        results["synoptic"] = "dry-run" if dry_run else ("ok" if code == 0 else "failed")

    if skip_rtma:
        results["rtma"] = "skipped (--skip-rtma)"
    else:
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "backfill_rtma_for_hrrr.py")]
        if dry_run:
            cmd.append("--dry-run")
        if rtma_limit:
            cmd += ["--limit", str(rtma_limit)]
        if not rtma_full_history:
            # backfill_rtma_for_hrrr.py discovers requirements from EVERY
            # HRRR file in the cache, not just recent ones - without a
            # --start filter, a routine sync rescans the entire cached
            # history (which can go back years, e.g. from building the
            # risk_fusion historical panel) instead of just what's new
            # since the last sync. --rtma-full-history opts back into that
            # for a genuine one-time historical backfill.
            start = (datetime.now(timezone.utc) - timedelta(days=rtma_days_back)).strftime("%Y-%m-%d")
            cmd += ["--start", start]
        code = _run(cmd, "RTMA analyses for cached HRRR runs")
        results["rtma"] = "dry-run" if dry_run else ("ok" if code == 0 else "failed")

    if skip_rrfs:
        results["rrfs"] = "skipped (--skip-rrfs)"
    else:
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "backfill_rrfs.py"), "--days-back", str(rrfs_days_back)]
        if dry_run:
            cmd.append("--dry-run")
        if rtma_limit:  # reuse the same --limit knob users already know from RTMA
            cmd += ["--limit", str(rtma_limit)]
        code = _run(cmd, "RRFS cycles (noaa-rrfs-ops-pds)")
        results["rrfs"] = "dry-run" if dry_run else ("ok" if code == 0 else "failed")

    if skip_fv3hires:
        results["fv3hires"] = "skipped (--skip-fv3hires)"
    else:
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "capture_fv3hires.py"), "--days-back", str(fv3hires_days_back)]
        if dry_run:
            cmd.append("--dry-run")
        code = _run(cmd, "FV3-HIRES cycles (NOMADS, short rolling window)")
        results["fv3hires"] = "dry-run" if dry_run else ("ok" if code == 0 else "failed")

    print("\n=== Sync summary ===")
    for source, status in results.items():
        print(f"   {source}: {status}")
    return results


def report_freshness():
    """Read each puller's own manifest and print one freshness summary,
    without maintaining a separate/competing manifest of our own."""
    print("=== Data freshness (from each puller's own manifest) ===")

    synoptic_manifest = paths.ARCHIVE_RAW_DATA_DIR / "backfill_manifest.json"
    if synoptic_manifest.exists():
        data = json.loads(synoptic_manifest.read_text())
        days = data.get("days", {})
        complete = [d for d, rec in days.items() if rec.get("status") == "complete"]
        failed = [d for d, rec in days.items() if rec.get("status") == "failed"]
        latest = max(complete) if complete else None
        print(f"   Synoptic observations: {len(complete)} complete days, {len(failed)} failed, "
              f"latest complete day: {latest or 'none'} (updated_at={data.get('updated_at', 'unknown')})")
    else:
        print("   Synoptic observations: no backfill_manifest.json yet - run scripts/backfill_synoptic.py")

    rtma_manifest = paths.CACHE_RTMA_DIR / "backfill_manifest.json"
    if rtma_manifest.exists():
        data = json.loads(rtma_manifest.read_text())
        analyses = data.get("analyses", {})
        complete = [k for k, rec in analyses.items() if rec.get("status") == "complete"]
        pending = [k for k, rec in analyses.items() if rec.get("status") != "complete"]
        latest = max(complete) if complete else None
        print(f"   RTMA analyses: {len(complete)} complete, {len(pending)} pending, "
              f"latest complete analysis: {latest or 'none'} (updated_at={data.get('updated_at', 'unknown')})")
    else:
        print("   RTMA analyses: no backfill_manifest.json yet - run scripts/backfill_rtma_for_hrrr.py")

    rrfs_manifest = paths.CACHE_RRFS_DIR / "backfill_manifest.json"
    if rrfs_manifest.exists():
        data = json.loads(rrfs_manifest.read_text())
        cycles = data.get("cycles", {})
        complete = [k for k, rec in cycles.items() if rec.get("status") == "complete"]
        pending = [k for k, rec in cycles.items() if rec.get("status") != "complete"]
        latest = max(complete) if complete else None
        print(f"   RRFS cycles: {len(complete)} complete, {len(pending)} pending, "
              f"latest complete cycle: {latest or 'none'} (updated_at={data.get('updated_at', 'unknown')})")
    else:
        print("   RRFS cycles: no backfill_manifest.json yet - run scripts/backfill_rrfs.py")

    fv3hires_manifest = paths.CACHE_FV3HIRES_DIR / "capture_manifest.json"
    if fv3hires_manifest.exists():
        data = json.loads(fv3hires_manifest.read_text())
        cycles = data.get("cycles", {})
        complete = [k for k, rec in cycles.items() if rec.get("status") == "complete"]
        latest = max(complete) if complete else None
        print(f"   FV3-HIRES cycles: {len(complete)} currently cached, "
              f"latest complete cycle: {latest or 'none'} (updated_at={data.get('updated_at', 'unknown')}) "
              f"- NOMADS rolling window, older cycles roll off and cannot be recovered")
    else:
        print("   FV3-HIRES cycles: no capture_manifest.json yet - run scripts/capture_fv3hires.py")

    zips_dir = paths.ARCHIVE_ZIPS_DIR
    zip_files = sorted(zips_dir.glob("*.zip")) if zips_dir.exists() else []
    if zip_files:
        newest = max(zip_files, key=lambda p: p.stat().st_mtime)
        print(f"   Archive zips: {len(zip_files)} on disk, newest: {newest.name}")
    else:
        print("   Archive zips: none on disk (nothing pulled via pull_archives.sh yet, or already unpacked/cleaned up)")


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
                         help="Only backfill RTMA for HRRR runs initialized in the last N days "
                              "(default: 21). Without this cutoff, RTMA backfill rescans EVERY "
                              "cached HRRR run regardless of age - including years-old runs left "
                              "over from building the risk_fusion historical panel.")
    parser.add_argument("--rtma-full-history", action="store_true",
                         help="Ignore --rtma-days-back and backfill RTMA for the entire cached HRRR "
                              "history. Only intended for a genuine one-time historical backfill, "
                              "not routine biweekly syncs.")
    args = parser.parse_args()

    if args.status:
        report_freshness()
        return

    sync(dry_run=args.dry_run, skip_archives=args.skip_archives, skip_synoptic=args.skip_synoptic,
         skip_rtma=args.skip_rtma, skip_rrfs=args.skip_rrfs, skip_fv3hires=args.skip_fv3hires,
         synoptic_days_back=args.synoptic_days_back,
         synoptic_limit=args.synoptic_limit, rtma_limit=args.rtma_limit,
         rtma_days_back=args.rtma_days_back, rtma_full_history=args.rtma_full_history,
         rrfs_days_back=args.rrfs_days_back, fv3hires_days_back=args.fv3hires_days_back)
    print()
    report_freshness()


if __name__ == "__main__":
    main()
