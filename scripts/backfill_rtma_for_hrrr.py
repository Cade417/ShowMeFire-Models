#!/usr/bin/env python3
"""Backfill causal and realized RTMA analyses required by local HRRR runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.rtma_capture import fetch_rtma, validate_rtma

HRRR_FILE_RE = re.compile(r"^hrrr_(\d{8})_(\d{2})z_(.*)\.nc$")
LEAD_RANGE_RE = re.compile(r"f(\d{2})-(\d{2})")
MANIFEST_PATH = paths.CACHE_RTMA_DIR / "backfill_manifest.json"


def discover_hrrr_runs(directory: Path = paths.CACHE_HRRR_DIR):
    runs = {}
    if not directory.is_dir():
        return runs
    for path in sorted(directory.iterdir()):
        match = HRRR_FILE_RE.fullmatch(path.name) if path.is_file() else None
        if match:
            run = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)
            runs.setdefault(run, []).append(path)
    return runs


def _as_utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _valid_times(path: Path, init: datetime):
    """Read valid times from HRRR; filename lead ranges are a compatibility fallback."""
    try:
        with xr.open_dataset(path) as ds:
            if "valid_time" in ds:
                values = np.asarray(ds.valid_time.values).reshape(-1)
            elif "time" in ds and "step" in ds:
                values = (np.asarray(ds.time.values).reshape(-1)[0] + np.asarray(ds.step.values).reshape(-1)).reshape(-1)
            elif "step" in ds:
                values = (np.datetime64(init.replace(tzinfo=None)) + np.asarray(ds.step.values).reshape(-1)).reshape(-1)
            else:
                values = []
        times = sorted({_as_utc(value).to_pydatetime() for value in values})
        if times:
            return times
    except Exception:
        pass
    match = LEAD_RANGE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot determine HRRR valid times from {path}")
    first, last = map(int, match.groups())
    return [init + timedelta(hours=lead) for lead in range(first, last + 1)]


def derive_requirements(discovered, window="teacher"):
    runs = {}
    for init, files in sorted(discovered.items()):
        valid = sorted({value for path in files for value in _valid_times(path, init)})
        if not valid or max(valid) <= init:
            raise ValueError(f"HRRR run {init.isoformat()} has no future valid times")
        antecedent = [init - timedelta(hours=hour) for hour in range(12, -1, -1)]
        realized = [init + timedelta(hours=hour) for hour in range(1, int((max(valid) - init).total_seconds() // 3600) + 1)]
        analyses = [init] if window == "anchor" else antecedent + realized
        runs[init] = {"files": files, "valid": valid, "antecedent": antecedent, "realized": realized,
                      "analyses": sorted(set(analyses))}
    return runs


def sha256(path: Path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path = MANIFEST_PATH):
    if not path.exists():
        return {"version": 2, "runs": {}, "analyses": {}}
    try:
        value = json.loads(path.read_text())
        if value.get("version") == 1 and isinstance(value.get("runs"), dict):
            old = value["runs"]
            return {"version": 2, "runs": {}, "analyses": {
                key: {field: record.get(field) for field in ("status", "attempts", "error", "error_class", "output",
                                                              "output_sha256", "output_size", "completed_at")}
                for key, record in old.items()}}
        if value.get("version") != 2 or not isinstance(value.get("runs"), dict) or not isinstance(value.get("analyses"), dict):
            raise ValueError("unsupported manifest schema")
        return value
    except Exception as exc:
        raise RuntimeError(f"Cannot read RTMA backfill manifest {path}: {exc}") from exc


def write_manifest(manifest, path: Path = MANIFEST_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["version"] = 2
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temporary.replace(path)


def classify_failure(exc: Exception):
    message = str(exc).lower()
    if isinstance(exc, (KeyError, ValueError)) or "decode variable" in message or "serialization" in message:
        return "deterministic"
    if isinstance(exc, FileNotFoundError) or any(token in message for token in ("404", "not found", "no grib", "unavailable")):
        return "unavailable"
    return "transient"


def _parse_boundary(value, end=False):
    if value is None:
        return None
    parsed = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return parsed.replace(hour=23, minute=59, second=59) if end and len(value) == 10 else parsed


def _iso(values):
    return [value.isoformat() for value in values]


def run(args, fetcher=fetch_rtma, manifest_path: Path = MANIFEST_PATH):
    discovered = discover_hrrr_runs(Path(args.hrrr_dir) if args.hrrr_dir else paths.CACHE_HRRR_DIR)
    start, end = _parse_boundary(args.start), _parse_boundary(args.end, end=True)
    discovered = {key: value for key, value in discovered.items()
                  if (start is None or key >= start) and (end is None or key <= end)}
    requirements = derive_requirements(discovered, getattr(args, "window", "teacher"))
    analyses = sorted({hour for requirement in requirements.values() for hour in requirement["analyses"]})
    states = []
    for hour in analyses:
        target = paths.CACHE_RTMA_DIR / f"rtma_{hour:%Y%m%d_%H}z.nc"
        valid, reason = validate_rtma(target) if target.exists() else (False, "missing")
        states.append((hour, target, valid, reason))
    missing = [state for state in states if args.force or not state[2]]
    sizes = [target.stat().st_size for _, target, valid, _ in states if valid]
    median_size = int(np.median(sizes)) if sizes else 0
    estimate = median_size * len(missing) if median_size else None
    print(f"HRRR files={sum(len(item['files']) for item in requirements.values())} unique_runs={len(requirements)} "
          f"unique_analyses={len(analyses)} valid_rtma={len(states)-len(missing)} required_fetches={len(missing)} "
          f"median_bytes={median_size or 'unknown'} estimated_additional_bytes={estimate if estimate is not None else 'unknown'}")
    if args.dry_run:
        return 0

    manifest = read_manifest(manifest_path)
    for init, item in requirements.items():
        manifest["runs"][init.isoformat()] = {"hrrr_files": [path.name for path in item["files"]],
            "antecedent_times": _iso(item["antecedent"]), "realized_times": _iso(item["realized"]),
            "valid_times": _iso(item["valid"])}
    for hour, target, valid, reason in states:
        key = hour.isoformat(); previous = manifest["analyses"].get(key, {})
        record = {**previous, "requested_analysis_time": key, "output": str(target),
                  "attempts": int(previous.get("attempts") or 0)}
        if valid and not args.force:
            record.update(status="complete", error=None, error_class=None, sha256=sha256(target), size=target.stat().st_size,
                          updated_at=datetime.now(timezone.utc).isoformat())
        else:
            record.update(status="pending", error=reason, error_class=None)
        manifest["analyses"][key] = record
    if states:
        write_manifest(manifest, manifest_path)

    work = missing[:args.limit] if args.limit else missing
    unresolved = []
    for hour, target, _, reason in work:
        key = hour.isoformat(); record = manifest["analyses"][key]
        if args.force and target.exists():
            target.unlink()
        for attempt in range(3):
            record["attempts"] = int(record.get("attempts") or 0) + 1
            try:
                output = fetcher(hour, paths.CACHE_RTMA_DIR)
                valid, validation = validate_rtma(output)
                if not valid:
                    raise RuntimeError(f"downloaded RTMA failed validation: {validation}")
                record.update(status="complete", error=None, error_class=None, output=str(output), sha256=sha256(output),
                              size=output.stat().st_size, updated_at=datetime.now(timezone.utc).isoformat())
                break
            except Exception as exc:
                category = classify_failure(exc)
                record.update(status="failed", error=str(exc), error_class=category, updated_at=datetime.now(timezone.utc).isoformat())
                if category != "transient" or attempt == 2:
                    unresolved.append(key); logging.error("%s %s failure: %s", key, category, exc); break
                time.sleep(2 ** attempt)
        write_manifest(manifest, manifest_path)
    print(f"processed={len(work)} complete={len(work)-len(unresolved)} unresolved={len(unresolved)} manifest={manifest_path}")
    return 1 if unresolved else 0


def main():
    parser = argparse.ArgumentParser(description="Backfill RTMA history and realized weather required by local HRRR files")
    parser.add_argument("--start", help="Earliest HRRR initialization, ISO date/time")
    parser.add_argument("--end", help="Latest HRRR initialization, ISO date/time")
    parser.add_argument("--limit", type=int, help="Process at most N missing/forced analyses")
    parser.add_argument("--hrrr-dir", help="Override HRRR discovery directory")
    parser.add_argument("--window", choices=("anchor", "teacher"), default="teacher")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
