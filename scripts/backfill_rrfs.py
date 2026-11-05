#!/usr/bin/env python3
"""Backfill RRFS cycles from the operational bucket (noaa-rrfs-ops-pds) into the training data store.

The "na" product this fetches only exists at the 4 synoptic cycles
(00/06/12/18 UTC) - see spatial/rrfs_capture.py's NA_PRODUCT_CYCLE_HOURS
and its module docstring for why (every other hour publishes a different,
unsupported product instead). The operational bucket only exists from
RRFS_OPS_ARCHIVE_START (2026-08-12) onward - there is nothing to backfill
before that date. See spatial/rrfs_capture.py's module docstring for why
this bucket/product, not Herbie's built-in (stale) `model="rrfs"` template.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.rrfs_capture import NA_PRODUCT_CYCLE_HOURS, RRFS_OPS_ARCHIVE_START, classify_fetch_failure, fetch_rrfs, validate_rrfs

MANIFEST_PATH = paths.CACHE_RRFS_DIR / "backfill_manifest.json"


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
        raise RuntimeError(f"Cannot read RRFS backfill manifest {path}: {exc}") from exc


def write_manifest(manifest, path: Path = MANIFEST_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["version"] = 1
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(path)


def synoptic_cycles(start: datetime, end: datetime):
    """Only the 4 cycle hours the 'na' product actually publishes at - see
    spatial/rrfs_capture.py's NA_PRODUCT_CYCLE_HOURS. Every other hour was
    previously attempted here too and deterministically 404'd (confirmed
    live 2026-09-15) - this generator just never produces those anymore."""
    cycle = start.replace(minute=0, second=0, microsecond=0)
    while cycle <= end:
        if cycle.hour in NA_PRODUCT_CYCLE_HOURS:
            yield cycle
        cycle += timedelta(hours=1)


def _fetch_worker(hour, lead_hours):
    """Runs in a separate process - fetch_rrfs's netCDF/GRIB decode isn't
    thread-safe, mirroring backfill_rtma_for_hrrr.py's process-isolation
    approach for real concurrency. Owns its own retry loop."""
    last_exc = None
    for attempt in range(3):
        try:
            output = fetch_rrfs(hour, paths.CACHE_RRFS_DIR, lead_hours)
            valid, reason = validate_rrfs(output)
            if not valid:
                raise RuntimeError(f"downloaded RRFS failed validation: {reason}")
            return {"ok": True, "output": str(output), "sha256": sha256(output),
                    "size": output.stat().st_size, "attempts": attempt + 1}
        except Exception as exc:
            last_exc = exc
            category = classify_fetch_failure(exc)
            if category != "transient" or attempt == 2:
                return {"ok": False, "error": str(exc), "error_class": category, "attempts": attempt + 1}
            time.sleep(2 ** attempt)
    return {"ok": False, "error": str(last_exc), "error_class": classify_fetch_failure(last_exc), "attempts": 3}


def _parse_boundary(value, end=False):
    if value is None:
        return None
    parsed = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return parsed.replace(hour=23, minute=59, second=59) if end and len(value) == 10 else parsed


def run(args, manifest_path: Path = MANIFEST_PATH):
    now = datetime.now(timezone.utc)
    start = _parse_boundary(args.start) or (RRFS_OPS_ARCHIVE_START.replace(tzinfo=timezone.utc)
                                             if args.full_history else now - timedelta(days=args.days_back))
    start = max(start, RRFS_OPS_ARCHIVE_START.replace(tzinfo=timezone.utc))
    end = _parse_boundary(args.end, end=True) or now
    lead_hours = range(args.lead_start, args.lead_end + 1)

    cycles = list(synoptic_cycles(start, end))
    states = []
    for hour in cycles:
        naive_hour = hour.astimezone(timezone.utc).replace(tzinfo=None)
        target = paths.CACHE_RRFS_DIR / f"rrfs_{naive_hour:%Y%m%d_%H}z_f{lead_hours[0]:02d}-{lead_hours[-1]:02d}.nc"
        valid, reason = validate_rrfs(target) if target.exists() else (False, "missing")
        states.append((hour, target, valid, reason))
    missing = [state for state in states if args.force or not state[2]]
    sizes = [target.stat().st_size for _, target, valid, _ in states if valid]
    median_size = int(np.median(sizes)) if sizes else 0
    estimate = median_size * len(missing) if median_size else None
    print(f"cycles={len(cycles)} valid={len(states) - len(missing)} required_fetches={len(missing)} "
          f"median_bytes={median_size or 'unknown'} estimated_additional_bytes={estimate if estimate is not None else 'unknown'}")
    if args.dry_run:
        return 0

    manifest = read_manifest(manifest_path)
    for hour, target, valid, reason in states:
        key = hour.isoformat(); previous = manifest["cycles"].get(key, {})
        record = {**previous, "requested_cycle": key, "output": str(target), "attempts": int(previous.get("attempts") or 0)}
        if valid and not args.force:
            record.update(status="complete", error=None, error_class=None, sha256=sha256(target), size=target.stat().st_size,
                          updated_at=datetime.now(timezone.utc).isoformat())
        else:
            record.update(status="pending", error=reason, error_class=None)
        manifest["cycles"][key] = record
    if states:
        write_manifest(manifest, manifest_path)

    work = missing[:args.limit] if args.limit else missing
    unresolved = []
    workers = max(1, args.workers or 1)

    if workers > 1 and work:
        for hour, target, _, _ in work:
            if args.force and target.exists():
                target.unlink()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_worker, hour, lead_hours): hour for hour, _, _, _ in work}
            for future in as_completed(futures):
                hour = futures[future]
                key = hour.isoformat()
                record = manifest["cycles"][key]
                result = future.result()
                record["attempts"] = int(record.get("attempts") or 0) + result["attempts"]
                if result["ok"]:
                    record.update(status="complete", error=None, error_class=None, output=result["output"],
                                  sha256=result["sha256"], size=result["size"], updated_at=datetime.now(timezone.utc).isoformat())
                else:
                    record.update(status="failed", error=result["error"], error_class=result["error_class"],
                                  updated_at=datetime.now(timezone.utc).isoformat())
                    unresolved.append(key)
                    logging.error("%s %s failure: %s", key, result["error_class"], result["error"])
                write_manifest(manifest, manifest_path)
    else:
        for hour, target, _, _ in work:
            key = hour.isoformat(); record = manifest["cycles"][key]
            if args.force and target.exists():
                target.unlink()
            for attempt in range(3):
                record["attempts"] = int(record.get("attempts") or 0) + 1
                try:
                    output = fetch_rrfs(hour, paths.CACHE_RRFS_DIR, lead_hours)
                    valid, reason = validate_rrfs(output)
                    if not valid:
                        raise RuntimeError(f"downloaded RRFS failed validation: {reason}")
                    record.update(status="complete", error=None, error_class=None, output=str(output), sha256=sha256(output),
                                  size=output.stat().st_size, updated_at=datetime.now(timezone.utc).isoformat())
                    break
                except Exception as exc:
                    category = classify_fetch_failure(exc)
                    record.update(status="failed", error=str(exc), error_class=category, updated_at=datetime.now(timezone.utc).isoformat())
                    if category != "transient" or attempt == 2:
                        unresolved.append(key); logging.error("%s %s failure: %s", key, category, exc); break
                    time.sleep(2 ** attempt)
            write_manifest(manifest, manifest_path)
    print(f"processed={len(work)} complete={len(work) - len(unresolved)} unresolved={len(unresolved)} manifest={manifest_path}")
    return 1 if unresolved else 0


def main():
    parser = argparse.ArgumentParser(description="Backfill RRFS cycles into the training data store")
    parser.add_argument("--start", help="Earliest RRFS cycle, ISO date/time (clamped to the operational archive start)")
    parser.add_argument("--end", help="Latest RRFS cycle, ISO date/time")
    parser.add_argument("--days-back", type=int, default=21, help="Default window when --start is omitted (default: 21)")
    parser.add_argument("--full-history", action="store_true", help="Backfill the entire operational archive (from 2026-08-12)")
    parser.add_argument("--lead-start", type=int, default=4)
    parser.add_argument("--lead-end", type=int, default=15)
    parser.add_argument("--limit", type=int, help="Process at most N missing/forced cycles")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent RRFS fetches (default: 4)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
