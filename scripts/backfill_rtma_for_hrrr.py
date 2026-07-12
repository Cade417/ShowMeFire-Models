#!/usr/bin/env python3
"""Backfill exactly one RTMA anchor for every local HRRR initialization."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.rtma_capture import fetch_rtma, validate_rtma

HRRR_FILE_RE = re.compile(r"^hrrr_(\d{8})_(\d{2})z_.*\.nc$")
MANIFEST_PATH = paths.CACHE_RTMA_DIR / "backfill_manifest.json"


def discover_hrrr_runs(directory: Path = paths.CACHE_HRRR_DIR):
    runs = {}
    if not directory.is_dir():
        return runs
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = HRRR_FILE_RE.fullmatch(path.name)
        if not match:
            continue
        run = datetime.strptime("".join(match.groups()), "%Y%m%d%H").replace(tzinfo=timezone.utc)
        runs.setdefault(run, []).append(path)
    return runs


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
        raise RuntimeError(f"Cannot read RTMA backfill manifest {path}: {exc}") from exc


def write_manifest(manifest, path: Path = MANIFEST_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def run(args, fetcher=fetch_rtma, manifest_path: Path = MANIFEST_PATH):
    discovered = discover_hrrr_runs(Path(args.hrrr_dir) if args.hrrr_dir else paths.CACHE_HRRR_DIR)
    start, end = _parse_boundary(args.start), _parse_boundary(args.end, end=True)
    selected = [(run_time, files) for run_time, files in sorted(discovered.items())
                if (start is None or run_time >= start) and (end is None or run_time <= end)]
    states = []
    for run_time, files in selected:
        target = paths.CACHE_RTMA_DIR / f"rtma_{run_time:%Y%m%d_%H}z.nc"
        valid, reason = validate_rtma(target) if target.exists() else (False, "missing")
        states.append((run_time, files, target, valid, reason))
    missing = [state for state in states if args.force or not state[3]]
    print(f"HRRR files={sum(len(files) for _, files in selected)} unique_runs={len(selected)} valid_rtma={len(states)-len(missing)} required_fetches={len(missing)}")
    if args.dry_run:
        for run_time, files, _, valid, reason in states:
            print(f"{run_time.isoformat()} {'VALID' if valid and not args.force else 'FETCH'} hrrr={len(files)}{'' if valid else f' reason={reason}'}")
        return 0
    work = missing[:args.limit] if args.limit else missing
    manifest = read_manifest(manifest_path)
    for run_time, files, target, valid, validation_reason in states:
        key = run_time.isoformat()
        previous = manifest["runs"].get(key, {})
        base_record = {**previous, "hrrr_files": [path.name for path in files],
            "requested_analysis_time": key, "output": str(target), "status": "complete",
            "error": None, "error_class": None, "attempts": int(previous.get("attempts", 0)),
            "completed_at": previous.get("completed_at")}
        if valid and not args.force:
            base_record.update({"output_sha256": sha256(target), "output_size": target.stat().st_size,
                                "completed_at": previous.get("completed_at") or datetime.now(timezone.utc).isoformat()})
        else:
            base_record.update({"status": "pending", "error": validation_reason, "error_class": None})
        manifest["runs"][key] = base_record
    if states:
        write_manifest(manifest, manifest_path)
    unresolved = []
    for run_time, files, target, _, validation_reason in work:
        key = run_time.isoformat(); previous = manifest["runs"].get(key, {})
        record = {**previous, "hrrr_files": [path.name for path in files], "requested_analysis_time": key,
                  "output": str(target), "status": "pending", "error": None, "error_class": None,
                  "attempts": int(previous.get("attempts", 0))}
        if args.force and target.exists():
            target.unlink()
        for attempt in range(1, 4):
            record["attempts"] += 1
            try:
                output = fetcher(run_time, paths.CACHE_RTMA_DIR)
                valid, reason = validate_rtma(output)
                if not valid:
                    raise RuntimeError(f"downloaded RTMA failed validation: {reason}")
                record.update({"status": "complete", "error": None, "error_class": None,
                               "output_sha256": sha256(output), "output_size": output.stat().st_size,
                               "completed_at": datetime.now(timezone.utc).isoformat()})
                break
            except Exception as exc:
                category = classify_failure(exc)
                record.update({"status": "failed", "error": str(exc), "error_class": category})
                if category != "transient" or attempt == 3:
                    unresolved.append(key)
                    logging.error("%s %s failure: %s", key, category, exc)
                    break
                time.sleep(2 ** (attempt - 1))
        manifest["runs"][key] = record
        write_manifest(manifest, manifest_path)
    print(f"processed={len(work)} complete={len(work)-len(unresolved)} unresolved={len(unresolved)} manifest={manifest_path}")
    return 1 if unresolved else 0


def main():
    parser = argparse.ArgumentParser(description="Backfill RTMA anchors required by local HRRR files")
    parser.add_argument("--start", help="Earliest HRRR initialization, ISO date/time")
    parser.add_argument("--end", help="Latest HRRR initialization, ISO date/time")
    parser.add_argument("--limit", type=int, help="Process at most N missing/forced anchors")
    parser.add_argument("--hrrr-dir", help="Override HRRR discovery directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
