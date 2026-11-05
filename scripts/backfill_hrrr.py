#!/usr/bin/env python3
"""Backfill historical HRRR runs (cropped to Missouri) for a date range.

Mirrors scripts/backfill_rtma_for_hrrr.py's resumable-manifest / process-pool
design: fetch_hrrr's netCDF write and GRIB decode aren't thread-safe, so real
concurrency needs process isolation, not threads (confirmed by an actual
segfault when this was first tried with threads on the RTMA backfill).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.hrrr_capture import (DEFAULT_LEAD_HOURS, PRECIP_CONTEXT_LEAD_HOURS, classify_fetch_failure,
                                  clear_local_cache, fetch_hrrr, fetch_precip_context, validate_hrrr,
                                  validate_precip_context)

MANIFEST_PATH = paths.CACHE_HRRR_DIR / "backfill_manifest.json"
PRECIP_CONTEXT_MANIFEST_PATH = paths.CACHE_HRRR_DIR / "precip_context_backfill_manifest.json"


def daily_runs(start: datetime, end: datetime, init_hour: int):
    day = start.date()
    while day <= end.date():
        yield datetime.combine(day, datetime.min.time()).replace(hour=init_hour, tzinfo=timezone.utc)
        day += timedelta(days=1)


def sha256(path: Path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path = MANIFEST_PATH):
    if not path.exists():
        return {"version": 1, "runs": {}}
    try:
        value = json.loads(path.read_text())
        if value.get("version") != 1 or not isinstance(value.get("runs"), dict):
            raise ValueError("unsupported manifest schema")
        return value
    except Exception as exc:
        raise RuntimeError(f"Cannot read HRRR backfill manifest {path}: {exc}") from exc


def write_manifest(manifest, path: Path = MANIFEST_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(path)


def _fetch_worker(run_dt, cache_dir, lead_hours, context_mode=False):
    """Runs in a separate process; owns its own retry loop and returns a
    plain, picklable outcome dict."""
    last_exc = None
    max_attempts = 5 if context_mode else 3
    for attempt in range(max_attempts):
        try:
            clear_local_cache(run_dt)
            worker_fetch = fetch_precip_context if context_mode else fetch_hrrr
            worker_validate = validate_precip_context if context_mode else validate_hrrr
            output = worker_fetch(run_dt, cache_dir, lead_hours=lead_hours)
            valid, validation = worker_validate(output)
            if not valid:
                raise RuntimeError(f"fetched HRRR failed validation: {validation}")
            return {"ok": True, "output": str(output), "sha256": sha256(output),
                    "size": output.stat().st_size, "attempts": attempt + 1}
        except Exception as exc:
            last_exc = exc
            category = classify_fetch_failure(exc)
            if category != "transient" or attempt == max_attempts - 1:
                return {"ok": False, "error": str(exc), "error_class": category, "attempts": attempt + 1}
            time.sleep(2 ** (attempt + 1))
    return {"ok": False, "error": str(last_exc), "error_class": classify_fetch_failure(last_exc), "attempts": max_attempts}


def run(args, fetcher=fetch_hrrr, manifest_path: Path = MANIFEST_PATH):
    end = datetime.now(timezone.utc) if args.end is None else datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    start = end - timedelta(days=args.days_back) if args.start is None else datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    context_mode = bool(getattr(args, "precip_context", False))
    lead_hours = list(PRECIP_CONTEXT_LEAD_HOURS if context_mode else range(args.lead_start, args.lead_end + 1))
    if context_mode and fetcher is fetch_hrrr:
        fetcher = fetch_precip_context
    validator = validate_precip_context if context_mode else validate_hrrr
    cache_dir = Path(args.cache_dir) if args.cache_dir else paths.CACHE_HRRR_DIR

    requested_runs = list(daily_runs(start, end, args.init_hour))
    runs = requested_runs
    if getattr(args, "existing_runs_only", False):
        runs = [run_dt for run_dt in runs
                if (cache_dir / f"hrrr_{run_dt:%Y%m%d_%H}z_f{DEFAULT_LEAD_HOURS[0]:02d}-{DEFAULT_LEAD_HOURS[-1]:02d}.nc").exists()]
    print(f"Scanning {len(runs)} candidate runs for already-cached files (this can take a while)...")
    states = []
    for index, run_dt in enumerate(runs, 1):
        prefix = "hrrr_precip_context" if context_mode else "hrrr"
        target = cache_dir / f"{prefix}_{run_dt:%Y%m%d_%H}z_f{lead_hours[0]:02d}-{lead_hours[-1]:02d}.nc"
        valid, reason = validator(target) if target.exists() else (False, "missing")
        states.append((run_dt, target, valid, reason))
        if index % 25 == 0 or index == len(runs):
            print(f"  scanned {index}/{len(runs)}")
    missing = [state for state in states if args.force or not state[2]]
    print(f"unique_runs={len(states)} valid={len(states)-len(missing)} required_fetches={len(missing)}")
    if args.dry_run:
        return 0

    manifest = read_manifest(manifest_path)
    if getattr(args, "existing_runs_only", False):
        selected_keys = {run_dt.isoformat() for run_dt in runs}
        for run_dt in requested_runs:
            key = run_dt.isoformat()
            if key not in selected_keys:
                manifest["runs"].pop(key, None)
    for run_dt, target, valid, reason in states:
        key = run_dt.isoformat(); previous = manifest["runs"].get(key, {})
        record = {**previous, "output": str(target), "attempts": int(previous.get("attempts") or 0)}
        if valid and not args.force:
            record.update(status="complete", error=None, error_class=None, sha256=sha256(target), size=target.stat().st_size,
                          completed_at=previous.get("completed_at") or datetime.now(timezone.utc).isoformat())
        else:
            record.update(status="pending", error=reason, error_class=None)
        manifest["runs"][key] = record
    if states:
        write_manifest(manifest, manifest_path)

    work = missing[:args.limit] if args.limit else missing
    unresolved = []
    workers = max(1, args.workers)

    if fetcher in (fetch_hrrr, fetch_precip_context) and workers > 1 and work:
        for run_dt, target, _, _ in work:
            if args.force and target.exists():
                target.unlink()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_worker, run_dt, cache_dir, lead_hours, context_mode): run_dt for run_dt, _, _, _ in work}
            for future in as_completed(futures):
                run_dt = futures[future]; key = run_dt.isoformat(); record = manifest["runs"][key]
                result = future.result()
                record["attempts"] = int(record.get("attempts") or 0) + result["attempts"]
                if result["ok"]:
                    record.update(status="complete", error=None, error_class=None, output=result["output"],
                                  sha256=result["sha256"], size=result["size"], completed_at=datetime.now(timezone.utc).isoformat())
                else:
                    record.update(status="failed", error=result["error"], error_class=result["error_class"],
                                  completed_at=datetime.now(timezone.utc).isoformat())
                    unresolved.append(key)
                    logging.error("%s %s failure: %s", key, result["error_class"], result["error"])
                write_manifest(manifest, manifest_path)
    else:
        for run_dt, target, _, reason in work:
            key = run_dt.isoformat(); record = manifest["runs"][key]
            if args.force and target.exists():
                target.unlink()
            max_attempts = 5 if context_mode else 3
            for attempt in range(max_attempts):
                record["attempts"] = int(record.get("attempts") or 0) + 1
                try:
                    clear_local_cache(run_dt)
                    output = fetcher(run_dt, cache_dir, lead_hours=lead_hours)
                    valid, validation = validator(output)
                    if not valid:
                        raise RuntimeError(f"fetched HRRR failed validation: {validation}")
                    record.update(status="complete", error=None, error_class=None, output=str(output), sha256=sha256(output),
                                  size=output.stat().st_size, completed_at=datetime.now(timezone.utc).isoformat())
                    break
                except Exception as exc:
                    category = classify_fetch_failure(exc)
                    record.update(status="failed", error=str(exc), error_class=category, completed_at=datetime.now(timezone.utc).isoformat())
                    if category != "transient" or attempt == max_attempts - 1:
                        unresolved.append(key); logging.error("%s %s failure: %s", key, category, exc); break
                    time.sleep(2 ** (attempt + 1))
            write_manifest(manifest, manifest_path)
    print(f"processed={len(work)} complete={len(work)-len(unresolved)} unresolved={len(unresolved)} manifest={manifest_path}")
    return 1 if unresolved else 0


def main():
    parser = argparse.ArgumentParser(description="Backfill historical HRRR runs, cropped to Missouri")
    parser.add_argument("--start", help="Earliest run date, ISO date/time (default: --days-back before --end)")
    parser.add_argument("--end", help="Latest run date, ISO date/time (default: now)")
    parser.add_argument("--days-back", type=int, default=365, help="Days back from --end when --start is omitted (default: 365)")
    parser.add_argument("--init-hour", type=int, default=12, help="HRRR init hour UTC (default: 12, matching existing cache)")
    parser.add_argument("--lead-start", type=int, default=DEFAULT_LEAD_HOURS[0], help="First forecast lead hour (default: 4)")
    parser.add_argument("--lead-end", type=int, default=DEFAULT_LEAD_HOURS[-1], help="Last forecast lead hour (default: 15)")
    parser.add_argument("--limit", type=int, help="Process at most N missing/forced runs")
    parser.add_argument("--cache-dir", help="Override HRRR cache directory")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent HRRR fetches (default: 6)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--precip-context", action="store_true",
                        help="Fetch only compact F00-F03 precipitation sidecars for existing F04-F15 runs")
    parser.add_argument("--existing-runs-only", action="store_true",
                        help="Target only dates with an existing F04-F15 HRRR file")
    arguments = parser.parse_args()
    manifest_path = PRECIP_CONTEXT_MANIFEST_PATH if arguments.precip_context else MANIFEST_PATH
    raise SystemExit(run(arguments, manifest_path=manifest_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
