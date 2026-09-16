#!/usr/bin/env python3
"""Backfill historical Synoptic station observations for a date range.

Fetches in day-chunked windows (default 7 days/request) and splits each
response into the same per-day raw_data_YYYYMMDD.json files ingest_obs.py
already expects. Pure HTTP + JSON work, so thread-based concurrency is
safe here (unlike the HRRR/RTMA backfills, which need process isolation
because of non-thread-safe netCDF/GRIB decode libraries).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from core.database import get_ignored_stations
from spatial.synoptic_capture import DEFAULT_NETWORKS, DEFAULT_STATES, daily_chunks, fetch_timeseries, split_by_day, validate_raw_data

MANIFEST_PATH = paths.ARCHIVE_RAW_DATA_DIR / "backfill_manifest.json"


def read_manifest(path: Path = MANIFEST_PATH):
    if not path.exists():
        return {"version": 1, "days": {}}
    try:
        value = json.loads(path.read_text())
        if value.get("version") != 1 or not isinstance(value.get("days"), dict):
            raise ValueError("unsupported manifest schema")
        return value
    except Exception as exc:
        raise RuntimeError(f"Cannot read Synoptic backfill manifest {path}: {exc}") from exc


def write_manifest(manifest, path: Path = MANIFEST_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(path)


def classify_failure(exc: Exception):
    message = str(exc).lower()
    if any(token in message for token in ("401", "403", "invalid token", "unauthorized")):
        return "deterministic"
    if any(token in message for token in ("404", "not found")):
        return "unavailable"
    return "transient"


def _fetch_chunk(chunk_start, chunk_end, token, states, networks, ignored):
    """Fetch and split one chunk; returns {day_str: response_dict}. Runs in
    a worker thread - safe since this is pure HTTP/JSON, no native libs."""
    last_exc = None
    for attempt in range(3):
        try:
            body = fetch_timeseries(chunk_start, chunk_end, token, states=states, networks=networks)
            return split_by_day(body, ignored_stations=ignored)
        except Exception as exc:
            last_exc = exc
            category = classify_failure(exc)
            if category != "transient" or attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise last_exc


def run(args, manifest_path: Path = MANIFEST_PATH):
    end = datetime.now(timezone.utc) if args.end is None else datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    start = end - timedelta(days=args.days_back) if args.start is None else datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    archive_dir = Path(args.archive_dir) if args.archive_dir else paths.ARCHIVE_RAW_DATA_DIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    states = args.states.split(",") if args.states else DEFAULT_STATES
    networks = [int(n) for n in args.networks.split(",")] if args.networks else DEFAULT_NETWORKS

    target_days = [(start + timedelta(days=i)).date() for i in range((end.date() - start.date()).days + 1)]
    day_states = {}
    for day in target_days:
        target = archive_dir / f"raw_data_{day:%Y%m%d}.json"
        valid, reason = validate_raw_data(target) if target.exists() else (False, "missing")
        day_states[day.isoformat()] = (target, valid, reason)

    chunks = list(daily_chunks(start, end + timedelta(days=1), args.chunk_days))
    missing_chunks = [c for c in chunks if args.force or any(
        not day_states.get((c[0] + timedelta(days=i)).date().isoformat(), (None, False, ""))[1]
        for i in range((c[1] - c[0]).days))]

    valid_count = sum(1 for _, valid, _ in day_states.values() if valid)
    print(f"target_days={len(target_days)} valid_days={valid_count} required_chunks={len(missing_chunks)}/{len(chunks)}")
    if args.dry_run:
        return 0

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    token = args.token or os.getenv("SYNOPTIC_API_TOKEN")
    if not token:
        print("ERROR: no Synoptic API token. Set SYNOPTIC_API_TOKEN in .env or pass --token.", file=sys.stderr)
        return 1

    ignored = get_ignored_stations()
    manifest = read_manifest(manifest_path)
    unresolved = []
    work = missing_chunks[:args.limit] if args.limit else missing_chunks

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_fetch_chunk, c[0], c[1], token, states, networks, ignored): c for c in work}
        for future in as_completed(futures):
            chunk_start, chunk_end = futures[future]
            chunk_days = [chunk_start + timedelta(days=i) for i in range((chunk_end - chunk_start).days)]
            chunk_days = [d.date() for d in chunk_days if d.date() in target_days]
            try:
                by_day = future.result()
            except Exception as exc:
                category = classify_failure(exc)
                logging.error("chunk %s-%s %s failure: %s", chunk_start.date(), chunk_end.date(), category, exc)
                for day in chunk_days:
                    key = day.isoformat()
                    manifest["days"][key] = {**manifest["days"].get(key, {}), "status": "failed",
                                              "error": str(exc), "error_class": category,
                                              "updated_at": datetime.now(timezone.utc).isoformat()}
                    unresolved.append(key)
                write_manifest(manifest, manifest_path)
                continue

            # Write every day in this chunk, even ones with zero surviving
            # stations (e.g. all ignore-filtered) - a written, empty file is
            # a legitimate "we tried" result, not a gap to retry forever.
            for day in chunk_days:
                day_response = by_day.get(day.isoformat(), {"STATION": []})
                target = archive_dir / f"raw_data_{day:%Y%m%d}.json"
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(day_response, indent=2))
                temporary.replace(target)
                valid, reason = validate_raw_data(target)
                key = day.isoformat()
                if valid:
                    manifest["days"][key] = {"status": "complete", "error": None, "error_class": None,
                                              "stations": len(day_response.get("STATION", [])),
                                              "output": str(target), "updated_at": datetime.now(timezone.utc).isoformat()}
                else:
                    manifest["days"][key] = {"status": "failed", "error": reason, "error_class": "deterministic",
                                              "updated_at": datetime.now(timezone.utc).isoformat()}
                    unresolved.append(key)
            write_manifest(manifest, manifest_path)

    print(f"chunks_processed={len(work)} days_written={len(target_days)-len(unresolved)} unresolved={len(unresolved)} manifest={manifest_path}")
    return 1 if unresolved else 0


def main():
    parser = argparse.ArgumentParser(description="Backfill historical Synoptic station observations")
    parser.add_argument("--start", help="Earliest date, ISO date/time (default: --days-back before --end)")
    parser.add_argument("--end", help="Latest date, ISO date/time (default: now)")
    parser.add_argument("--days-back", type=int, default=365, help="Days back from --end when --start is omitted (default: 365)")
    parser.add_argument("--chunk-days", type=int, default=7, help="Days fetched per API request (default: 7)")
    parser.add_argument("--states", help="Comma-separated state list (default: MO + 8 surrounding states)")
    parser.add_argument("--networks", help="Comma-separated Synoptic network IDs (default: 2)")
    parser.add_argument("--limit", type=int, help="Process at most N missing/forced chunks")
    parser.add_argument("--archive-dir", help="Override raw_data archive directory")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent chunk fetches (default: 4)")
    parser.add_argument("--token", help="Synoptic API token (default: SYNOPTIC_API_TOKEN from .env)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
