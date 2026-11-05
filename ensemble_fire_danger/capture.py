"""
Historical / forward capture of every ensemble member and ensprod run the
Day-1 product needs, one forecast day at a time.

Uses the exact same cycle-resolution rules as the live product
(members.resolve_plan) with two capture-specific switches:

- use_archive=True: read each source's archive_url when it has one (no current
  source does; NAM nest, the one that did, was removed in 2026-10).
- emulate_live_at=anchor + live_run_offset_hours: an archive has every
  cycle, but the live run (~14:45-15:00Z) only ever sees cycles already
  published by then. Without this, the training panel would be built from a
  systematically fresher member mix (12z HREF, 12z HiResW...) than the
  live product - and calibration fitted on it would be over-confident.

What can be backfilled (confirmed 2026-09-22): HRRR (AWS, years),
RRFS + REFS ensprod (noaa-rrfs-ops-pds, from 2026-08-12).
Forward-only (NOMADS rolling ~2 days, no archive anywhere): HiResW
ARW/FV3/ARW-mem2 and HREF. That is why the recurring capture task must run
at least daily - a missed day is gone for those sources.

Output: per-run NetCDF files under CACHE_ENSEMBLE_DIR/runs (same naming
as the live cache, members.cache_path) and a manifest
CACHE_ENSEMBLE_DIR/capture_manifest.json keyed by forecast day, recording
each member's resolved cycle, status and file.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

import paths
from spatial.domain import crop

from . import members

logger = logging.getLogger(__name__)

RUNS_DIR = paths.CACHE_ENSEMBLE_DIR / "runs"
TMP_DIR = paths.CACHE_ENSEMBLE_DIR / "tmp"
MANIFEST_PATH = paths.CACHE_ENSEMBLE_DIR / "capture_manifest.json"


def read_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.exists():
        return {"version": 1, "days": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("version") != 1 or not isinstance(value.get("days"), dict):
        raise RuntimeError(f"unsupported ensemble capture manifest schema: {path}")
    return value


def write_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["version"] = 1
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def anchor_for(day: date, config: dict) -> datetime:
    return datetime(day.year, day.month, day.day, int(config["window"]["anchor_cycle_hour_utc"]))


def plan_day(day: date, config: dict, session: requests.Session, emulate_live: bool = True
             ) -> List[members.ResolvedMember]:
    anchor = anchor_for(day, config)
    emulate = anchor + timedelta(hours=float(config.get("live_run_offset_hours", 3.0))) if emulate_live else None
    return members.resolve_plan(config, anchor, members.live_availability(session),
                                now=datetime.now(timezone.utc).replace(tzinfo=None),
                                emulate_live_at=emulate, use_archive=True)


def capture_day(day: date, config: Optional[dict] = None, *, workers: int = 4, force: bool = False,
                emulate_live: bool = True, dry_run: bool = False) -> Dict[str, dict]:
    """Resolve + fetch one forecast day. Returns {member_id: record}."""
    config = config or members.load_config()
    with requests.Session() as session:
        plan = plan_day(day, config, session, emulate_live=emulate_live)
    records: Dict[str, dict] = {item.member_id: item.as_dict() for item in plan}
    if dry_run:
        return records

    def one(item: members.ResolvedMember):
        started = time.time()
        try:
            with requests.Session() as worker_session:
                path = members.fetch_resolved(config, item, RUNS_DIR, crop, TMP_DIR, session=worker_session,
                                              force=force, use_archive=True)
            return item.member_id, {"status": "fetched", "path": str(path),
                                    "seconds": round(time.time() - started, 1)}
        except Exception as error:
            logger.warning("%s %s %s failed: %s", day, item.member_id, item.cycle, error)
            return item.member_id, {"status": "failed", "error": f"{type(error).__name__}: {error}"}

    todo = [item for item in plan if item.status == "resolved"]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for member_id, update in pool.map(one, todo):
            records[member_id].update(update)
    return records


def capture_days(days: List[date], *, workers: int = 4, force: bool = False, emulate_live: bool = True,
                 dry_run: bool = False, skip_complete: bool = True) -> dict:
    config = members.load_config()
    manifest = read_manifest()
    summary = {"days": 0, "fetched": 0, "failed": 0, "unavailable": 0, "skipped_days": 0}
    for day in days:
        key = day.isoformat()
        previous = manifest["days"].get(key, {})
        if skip_complete and not force and previous and all(
                r.get("status") in ("fetched", "unavailable") and (r.get("status") != "fetched" or
                                                                   Path(r.get("path", "")).exists())
                for r in previous.values()) and not _forward_only_retry_possible(day, previous, config):
            summary["skipped_days"] += 1
            continue
        records = capture_day(day, config, workers=workers, force=force, emulate_live=emulate_live, dry_run=dry_run)
        summary["days"] += 1
        for record in records.values():
            summary[record["status"]] = summary.get(record["status"], 0) + 1
        if not dry_run:
            manifest["days"][key] = records
            write_manifest(manifest)
        logger.info("%s: %s", key, {mid: r["status"] for mid, r in records.items()})
    return summary


def _forward_only_retry_possible(day: date, previous: dict, config: dict) -> bool:
    """A NOMADS-only member recorded 'unavailable' is worth retrying while
    the day is still inside NOMADS' ~2-day window (it may simply not have
    been published yet on the earlier pass)."""
    if (datetime.now(timezone.utc).date() - day).days > 2:
        return False
    return any(r.get("status") == "unavailable" for r in previous.values())
