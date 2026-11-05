#!/usr/bin/env python3
"""Capture recent FV3-HIRES (HiResW FV3) cycles from NOMADS into the training data store.

No backfill is possible here (see spatial/fv3hires_capture.py's module
docstring) - NOMADS only serves a short rolling window of recent cycles, so
this just sweeps whatever is CURRENTLY available. Meant to run frequently
(every few hours, via a scheduled job) rather than as part of the twice-daily
sync_training_data.py habit - a run not captured before it rolls off NOMADS
is lost permanently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.fv3hires_capture import classify_fetch_failure, fetch_fv3hires, validate_fv3hires

MANIFEST_PATH = paths.CACHE_FV3HIRES_DIR / "capture_manifest.json"
CYCLE_HOURS = (0, 6, 12, 18)  # HiResW's 4 daily cycles


def sha256(path: Path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path = MANIFEST_PATH):
    if not path.exists():
        return {"version": 1, "cycles": {}}
    try:
        value = json.loads(path.read_text())
        if value.get("version") != 1 or not isinstance(value.get("cycles"), dict):
            raise ValueError("unsupported manifest schema")
        return value
    except Exception as exc:
        raise RuntimeError(f"Cannot read FV3-HIRES capture manifest {path}: {exc}") from exc


def write_manifest(manifest, path: Path = MANIFEST_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["version"] = 1
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(path)


def recent_cycles(days_back: int):
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days_back)).replace(minute=0, second=0, microsecond=0)
    cycles = []
    day = start.replace(hour=0)
    while day <= now:
        for hour in CYCLE_HOURS:
            cycle = day.replace(hour=hour)
            if start <= cycle <= now:
                cycles.append(cycle)
        day += timedelta(days=1)
    return cycles


def run(args, manifest_path: Path = MANIFEST_PATH):
    lead_hours = range(args.lead_start, args.lead_end + 1)
    cycles = recent_cycles(args.days_back)
    manifest = read_manifest(manifest_path)
    complete, failed, skipped = 0, 0, 0

    for hour in cycles:
        naive_hour = hour.astimezone(timezone.utc).replace(tzinfo=None)
        target = paths.CACHE_FV3HIRES_DIR / f"fv3hires_{naive_hour:%Y%m%d_%H}z_f{lead_hours[0]:02d}-{lead_hours[-1]:02d}.nc"
        key = hour.isoformat()
        valid, reason = validate_fv3hires(target) if target.exists() else (False, "missing")
        if valid and not args.force:
            skipped += 1
            continue
        if args.dry_run:
            continue
        record = manifest["cycles"].get(key, {})
        last_exc = None
        for attempt in range(2):  # NOMADS is a live, short window - not worth the 3-attempt backoff a long archive fetch gets
            record["attempts"] = int(record.get("attempts") or 0) + 1
            try:
                output = fetch_fv3hires(naive_hour, paths.CACHE_FV3HIRES_DIR, lead_hours)
                record.update(status="complete", error=None, error_class=None, output=str(output),
                              sha256=sha256(output), size=output.stat().st_size,
                              updated_at=datetime.now(timezone.utc).isoformat())
                complete += 1
                break
            except Exception as exc:
                last_exc = exc
                category = classify_fetch_failure(exc)
                if category != "transient" or attempt == 1:
                    record.update(status="failed", error=str(exc), error_class=category,
                                  updated_at=datetime.now(timezone.utc).isoformat())
                    failed += 1
                    logging.info("%s %s: %s", key, category, exc)
                    break
                time.sleep(2)
        manifest["cycles"][key] = record
        write_manifest(manifest, manifest_path)

    print(f"cycles_checked={len(cycles)} newly_complete={complete} failed_or_unavailable={failed} "
          f"already_cached={skipped} manifest={manifest_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Capture recent FV3-HIRES cycles from NOMADS")
    parser.add_argument("--days-back", type=int, default=3, help="How far back to sweep (default: 3 - NOMADS' typical retention)")
    parser.add_argument("--lead-start", type=int, default=4)
    parser.add_argument("--lead-end", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
