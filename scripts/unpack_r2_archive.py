#!/usr/bin/env python3
"""Unpack data-v2/objects/<date>/<source>/<sha256>/<filename> (pulled by
scripts/migration/r2_pull.py from the production R2 archive, see
api/docs/r2_archive_runbook.md) into the training pipeline's expected
locations - the same job unpack_archive_zip.py does for the legacy
data_archive_day zips, for the newer continuously-archived window.

Multiple capture dates can reference the same logical filename (re-verified
without a content change creates no new version, but a real change does) -
this picks the newest capture date per (source, filename) as the
authoritative "latest revision", mirroring the production archive's own
semantics (see r2_archive_runbook.md: "the downloader selects the newest
revision").

HRRR/RRFS members here are the server's own full-CONUS/full-domain grids,
never cropped (same situation as the legacy zip archive - see
unpack_archive_zip.py's MAX_CROPPED_NC_BYTES comment for the incident this
guards against: this repo's cache must only ever hold Missouri-cropped
grids). Crop them with the same spatial.domain.crop() the live pipeline
uses before ever writing to CACHE_HRRR_DIR/CACHE_RRFS_DIR. Forecasts and
observations JSON need no transform - copy directly.

"forecasts_beta"/"hrrr_beta" sources (a production-side beta-model concept)
have no corresponding destination in this repo's paths.py and are skipped,
not treated as an error.

Usage:
    python scripts/unpack_r2_archive.py
    python scripts/unpack_r2_archive.py --dry-run
"""
import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.domain import crop
from spatial.hrrr_capture import validate_hrrr
from spatial.rrfs_capture import validate_rrfs

OBJECTS_ROOT = paths.DATA_ROOT / "r2_pull" / "data-v2" / "objects"
SKIPPED_SOURCES = {"forecasts_beta", "hrrr_beta"}


def latest_versions():
    """{(source, filename): Path} - the newest capture-date's copy of each logical file."""
    best = {}
    if not OBJECTS_ROOT.exists():
        return best
    for date_dir in sorted(p for p in OBJECTS_ROOT.iterdir() if p.is_dir()):
        for source_dir in date_dir.iterdir():
            if not source_dir.is_dir():
                continue
            source = source_dir.name
            for hash_dir in source_dir.iterdir():
                if not hash_dir.is_dir():
                    continue
                for file_path in hash_dir.iterdir():
                    if not file_path.is_file():
                        continue
                    # date_dir names sort lexicographically as YYYY-MM-DD, so
                    # later iteration order = later capture date; keep newest.
                    best[(source, file_path.name)] = file_path
    return best


def place_json(src: Path, dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        return "already present"
    shutil.copy2(src, dest)
    return "copied"


def place_cropped_nc(src: Path, dest_dir: Path, target_name: str, validator):
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / target_name
    if target.exists():
        valid, _ = validator(target)
        if valid:
            return "already present", ""
    # Same decode workaround validate_hrrr()/validate_rrfs() rely on: these
    # archived grids carry a stale "dtype" attr that collides with xarray's
    # own CF-decoding bookkeeping if decoded eagerly.
    raw = xr.open_dataset(src, decode_cf=False)
    for name in raw.variables:
        raw[name].attrs.pop("dtype", None)
    with xr.decode_cf(raw, decode_timedelta=True) as ds:
        cropped = crop(ds).load()
    temp_target = target.with_suffix(".nc.tmp")
    try:
        cropped.to_netcdf(temp_target, engine="netcdf4")
        valid, reason = validator(temp_target)
        if not valid:
            return "failed", reason
        temp_target.replace(target)
    finally:
        if temp_target.exists():
            temp_target.unlink()
    return "cropped", ""


def main():
    parser = argparse.ArgumentParser(description="Unpack the pulled data-v2 R2 archive into the training pipeline's expected locations")
    parser.add_argument("--dry-run", action="store_true", help="List what would happen without writing anything")
    args = parser.parse_args()

    versions = latest_versions()
    print(f"{len(versions)} logical files found across all capture dates")

    counts = defaultdict(int)
    for (source, filename), src in sorted(versions.items()):
        if source in SKIPPED_SOURCES:
            counts["skipped (" + source + ")"] += 1
            continue
        if args.dry_run:
            print(f"  [{source}] {filename}")
            continue

        try:
            if source == "forecasts":
                counts[place_json(src, paths.ARCHIVE_FORECASTS_DIR)] += 1
            elif source == "observations":
                counts[place_json(src, paths.ARCHIVE_RAW_DATA_DIR)] += 1
            elif source == "hrrr":
                status, reason = place_cropped_nc(src, paths.CACHE_HRRR_DIR, filename.lower(), validate_hrrr)
                counts[status] += 1
                if status == "failed":
                    print(f"  FAILED {filename}: {reason}")
            elif source == "rrfs":
                # Server names these "RRFS_..."; this repo's own fetch_rrfs()
                # writes lowercase "rrfs_..." targets - normalize so downstream
                # code (which globs the lowercase pattern) actually finds it.
                status, reason = place_cropped_nc(src, paths.CACHE_RRFS_DIR, filename.replace("RRFS_", "rrfs_"), validate_rrfs)
                counts[status] += 1
                if status == "failed":
                    print(f"  FAILED {filename}: {reason}")
            else:
                print(f"  WARNING: unrecognized source {source!r} for {filename} - not placed")
                counts["unrecognized"] += 1
        except Exception as exc:
            # A single incompatible/corrupt archive member (e.g. a grid from
            # a different capture pipeline, confirmed live 2026-10-03 with an
            # orphaned native-grid RRFS file) must not abort the whole run.
            print(f"  FAILED {source}/{filename}: {exc}")
            counts["failed"] += 1

    print(dict(counts))
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
