#!/usr/bin/env python3
"""Capture ensemble fire danger member / ensprod runs into the training data store.

Resumable and manifest-tracked (CACHE_ENSEMBLE_DIR/capture_manifest.json),
same shape as scripts/backfill_rrfs.py. See ensemble_fire_danger/capture.py
for which sources can be backfilled (HRRR, RRFS, REFS) and which
are forward-only (HiResW, HREF - NOMADS keeps ~2 days, so run this at least
daily; the recurring capture task does it every 4 hours).

Examples:
    python scripts/capture_ensemble_members.py --days-back 2           # routine / scheduled
    python scripts/capture_ensemble_members.py --start 2026-08-12 --end 2026-09-21 --workers 6
    python scripts/capture_ensemble_members.py --days-back 1 --dry-run # show the resolved plan only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ensemble_fire_danger import capture  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, help="first forecast day (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, help="last forecast day (default: yesterday)")
    parser.add_argument("--days-back", type=int, default=2, help="when --start is omitted (default 2)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="re-fetch even if cached/complete")
    parser.add_argument("--no-emulate-live", action="store_true",
                        help="use the freshest archived cycles instead of emulating live publish latency")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    today = datetime.now(timezone.utc).date()
    end = args.end or today
    start = args.start or (today - timedelta(days=args.days_back))
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    summary = capture.capture_days(days, workers=args.workers, force=args.force,
                                   emulate_live=not args.no_emulate_live, dry_run=args.dry_run)
    print(f"ensemble capture {start}..{end}: {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
