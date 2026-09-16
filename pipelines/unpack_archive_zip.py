"""Unpack per-day zip bundles (produced by the server's
api/services/archive_bundler.py, landing in data_archive_day/) into the
locations the training pipeline expects, under SMF_DATA_ROOT.

Routes each entry by filename pattern into:

    rtma_*.nc                -> cache/rtma/
    other *.nc               -> cache/hrrr/
    raw_data_*.json          -> archive/raw_data/
    station_forecasts_*.json -> archive/forecasts/

Anything that doesn't match a known pattern is logged, never silently
dropped. Safe to re-run - files already present at the destination (same
name and size) are skipped.

Usage:
    python pipelines/unpack_archive_zip.py                # all zips in paths.ARCHIVE_ZIPS_DIR
    python pipelines/unpack_archive_zip.py --zip 20260129.zip
"""
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

ARCHIVE_ZIPS_DIR = paths.ARCHIVE_ZIPS_DIR
DESTINATIONS = [
    (lambda name: name.startswith("rtma_") and name.endswith(".nc"), paths.CACHE_RTMA_DIR),
    (lambda name: name.endswith(".nc"), paths.CACHE_HRRR_DIR),
    (lambda name: name.startswith("raw_data_") and name.endswith(".json"), paths.ARCHIVE_RAW_DATA_DIR),
    (lambda name: name.startswith("station_forecasts_") and name.endswith(".json"), paths.ARCHIVE_FORECASTS_DIR),
]

# The server's own cache/hrrr archive stores the full-CONUS HRRR grid
# (~488MB), never cropped - unlike this repo's fetch_hrrr(), which crops to
# Missouri (~25MB) before ever writing to CACHE_HRRR_DIR. Both sides use the
# same "hrrr_YYYYMMDD_12z_f04-15.nc" filename, so extracting a server zip's
# member here would silently overwrite a correctly-cropped cache file with
# an uncropped one carrying the wrong grid - a real incident this guard is a
# regression test for. RTMA members are well under this too (~3.5MB), so one
# generous threshold on the zip's own recorded size (ZIP_STORED for .nc, so
# this is the true uncompressed size - no extraction needed to check it)
# covers both destinations.
MAX_CROPPED_NC_BYTES = 100 * 1024 * 1024


def _destination_for(entry_name):
    basename = Path(entry_name).name
    for matches, dest_dir in DESTINATIONS:
        if matches(basename):
            return dest_dir, basename
    return None, basename


def unpack_zip(zip_path):
    print(f"Unpacking {zip_path.name}...")
    unrecognized = []
    oversized = []
    extracted = 0
    skipped = 0

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            dest_dir, basename = _destination_for(info.filename)
            if dest_dir is None:
                unrecognized.append(info.filename)
                continue

            if basename.endswith(".nc") and info.file_size > MAX_CROPPED_NC_BYTES:
                oversized.append((info.filename, info.file_size))
                continue

            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / basename

            if dest_path.exists() and dest_path.stat().st_size == info.file_size:
                skipped += 1
                continue

            with zf.open(info) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

    print(f"  extracted {extracted}, skipped {skipped} (already present)")
    if oversized:
        print(f"  REFUSED {len(oversized)} oversized .nc member(s) (not Missouri-cropped - see MAX_CROPPED_NC_BYTES):")
        for name, size in oversized:
            print(f"    - {name} ({size / 1e6:.1f} MB)")
    if unrecognized:
        print(f"  WARNING: {len(unrecognized)} unrecognized entries not extracted:")
        for name in unrecognized:
            print(f"    - {name}")

    return extracted, skipped, unrecognized, oversized


def main():
    parser = argparse.ArgumentParser(description="Unpack archive_zips bundles into the pipeline's expected data locations")
    parser.add_argument("--zip", default=None, help="Unpack a single zip by filename instead of all of data/archive_zips")
    args = parser.parse_args()

    if args.zip:
        zip_paths = [ARCHIVE_ZIPS_DIR / args.zip]
    else:
        zip_paths = sorted(ARCHIVE_ZIPS_DIR.glob("*.zip"))

    if not zip_paths:
        print(f"No zip files found in {ARCHIVE_ZIPS_DIR}")
        return

    total_unrecognized = []
    total_oversized = []
    for zip_path in zip_paths:
        if not zip_path.exists():
            print(f"SKIP: {zip_path} does not exist")
            continue
        try:
            _, _, unrecognized, oversized = unpack_zip(zip_path)
            total_unrecognized.extend(unrecognized)
            total_oversized.extend(oversized)
        except zipfile.BadZipFile as e:
            print(f"SKIP: {zip_path.name} is not a valid zip yet ({e}) - server may still be writing it")

    if total_oversized:
        print(f"\n{len(total_oversized)} total oversized .nc member(s) refused across all zips - "
              f"these are the server's own uncropped archive, not this repo's Missouri-cropped cache.")
    if total_unrecognized:
        print(f"\n{len(total_unrecognized)} total unrecognized entries across all zips - review DESTINATIONS in this script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
