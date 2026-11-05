#!/usr/bin/env python3
"""Crop the oversized full-CONUS HRRR .nc members already sitting in
data/archive_zips/*.zip - the server's own uncropped archive, refused by
unpack_archive_zip.py's MAX_CROPPED_NC_BYTES guard (see that file's comment
for why: extracting them as-is would silently overwrite a correctly-cropped
cache file with the wrong grid) - down to Missouri using the exact same
crop() the live pipeline uses, and write them into CACHE_HRRR_DIR under the
same filename fetch_hrrr() would have produced. No network fetch: the data
is already on disk, just not yet cropped.

Usage:
    python scripts/crop_archived_hrrr.py             # all zips in archive_zips
    python scripts/crop_archived_hrrr.py --zip 20260806.zip
"""
import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from pipelines.unpack_archive_zip import MAX_CROPPED_NC_BYTES
from spatial.domain import crop
from spatial.hrrr_capture import validate_hrrr

ARCHIVE_ZIPS_DIR = paths.ARCHIVE_ZIPS_DIR
CACHE_HRRR_DIR = paths.CACHE_HRRR_DIR


def oversized_hrrr_members(zf):
    for info in zf.infolist():
        basename = Path(info.filename).name
        if basename.endswith(".nc") and not basename.startswith("rtma_") and info.file_size > MAX_CROPPED_NC_BYTES:
            yield info


def crop_member(zf, info, target: Path):
    if target.exists():
        valid, _ = validate_hrrr(target)
        if valid:
            return True, "already present"
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(zf.extract(info, path=tmp))
        ds = xr.open_dataset(raw_path)
        try:
            cropped = crop(ds).load()
        finally:
            ds.close()
        CACHE_HRRR_DIR.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_suffix(".nc.tmp")
        try:
            cropped.to_netcdf(temp_target, engine="netcdf4")
            valid, reason = validate_hrrr(temp_target)
            if not valid:
                return False, reason
            temp_target.replace(target)
        finally:
            if temp_target.exists():
                temp_target.unlink()
    return True, "cropped"


def main():
    parser = argparse.ArgumentParser(description="Crop already-downloaded full-CONUS HRRR archive members to Missouri")
    parser.add_argument("--zip", default=None, help="Process a single zip by filename instead of all zips in archive_zips")
    args = parser.parse_args()

    zip_paths = [ARCHIVE_ZIPS_DIR / args.zip] if args.zip else sorted(ARCHIVE_ZIPS_DIR.glob("*.zip"))
    done = skipped = failed = 0
    for zip_path in zip_paths:
        if not zip_path.exists():
            print(f"SKIP: {zip_path} does not exist")
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                members = list(oversized_hrrr_members(zf))
                if not members:
                    continue
                print(f"{zip_path.name}: {len(members)} oversized member(s)")
                for info in members:
                    basename = Path(info.filename).name
                    target = CACHE_HRRR_DIR / basename
                    ok, reason = crop_member(zf, info, target)
                    if ok and reason == "already present":
                        skipped += 1
                    elif ok:
                        print(f"  cropped {basename} -> {target.stat().st_size / 1e6:.1f} MB")
                        done += 1
                    else:
                        print(f"  FAILED {basename}: {reason}")
                        failed += 1
        except zipfile.BadZipFile as exc:
            print(f"SKIP: {zip_path.name} is not a valid zip yet ({exc})")

    print(f"\ncropped={done} already_present={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
