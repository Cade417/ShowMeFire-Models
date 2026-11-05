#!/usr/bin/env python3
"""Materialize already-pulled R2 archive objects into the exact cache/
archive filenames the backfill scripts expect - zero network calls, zero R2
credentials needed, since scripts/migration/r2_pull.py already downloaded
everything to <SMF_DATA_ROOT>/r2_pull/<bucket key layout> on this machine.

R2 archive keys look like data-v2/objects/<date>/<source>/<sha256>/<name>
(see api/services/archive_store.py's ArchiveStore.ingest/SOURCES) - the
sha256 is baked into the path itself, so verifying a copy is free. This
only covers the three sources archive_store.py actually archives that
model-training also consumes: hrrr, rrfs, observations (Synoptic raw_data).
RTMA and FV3-HIRES are not archived to R2 at all (missing from SOURCES) -
those stay internet-only regardless.

Existing files already in the destination cache dirs are left untouched
(this never overwrites - the backfill scripts' own validators are the
source of truth for whether a file is good); this only fills in what's
missing there but present in the local R2 mirror.

Usage:
    python -m pipeline.r2_restore
    python -m pipeline.r2_restore --dry-run
    python -m pipeline.r2_restore --only hrrr,observations
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

PULL_ROOT = paths.DATA_ROOT / "r2_pull"
DATA_PREFIX = os.getenv("R2_DATA_PREFIX", "data-v2")
OBJECTS_DIR = PULL_ROOT / DATA_PREFIX / "objects"
MANIFEST_PATH = paths.DATA_ROOT / "r2_restore_manifest.json"

# R2 source name -> local cache/archive dir this repo's backfill scripts
# actually read from (see paths.py). "forecasts"/"_beta" variants are
# server-side forecast output, not training input - intentionally excluded.
SOURCE_DIRS = {
    "hrrr": paths.CACHE_HRRR_DIR,
    "rrfs": paths.CACHE_RRFS_DIR,
    "observations": paths.ARCHIVE_RAW_DATA_DIR,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest():
    if not MANIFEST_PATH.exists():
        return {"version": 1, "restored": {}}
    try:
        value = json.loads(MANIFEST_PATH.read_text())
        if value.get("version") != 1:
            raise ValueError("unsupported manifest schema")
        return value
    except Exception:
        return {"version": 1, "restored": {}}


def write_manifest(manifest):
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    temp = MANIFEST_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    temp.replace(MANIFEST_PATH)


def iter_candidates(sources):
    """Yield (source, expected_sha256, src_path, dest_path) for every object
    under the local R2 mirror matching the requested sources."""
    if not OBJECTS_DIR.is_dir():
        return
    for date_dir in sorted(OBJECTS_DIR.iterdir()):
        if not date_dir.is_dir():
            continue
        for source_dir in date_dir.iterdir():
            if source_dir.name not in sources or not source_dir.is_dir():
                continue
            dest_root = SOURCE_DIRS[source_dir.name]
            for sha_dir in source_dir.iterdir():
                if not sha_dir.is_dir():
                    continue
                for file_path in sha_dir.iterdir():
                    if file_path.is_file():
                        yield source_dir.name, sha_dir.name, file_path, dest_root / file_path.name


def restore(sources=None, dry_run=False):
    sources = sources or list(SOURCE_DIRS)
    unknown = [s for s in sources if s not in SOURCE_DIRS]
    if unknown:
        raise ValueError(f"Unknown source(s) for R2 restore: {unknown} (known: {list(SOURCE_DIRS)})")

    manifest = read_manifest()
    restored = manifest.setdefault("restored", {})
    counts = {"copied": 0, "already_present": 0, "mismatched": 0, "scanned": 0}

    for source, expected_sha, src_path, dest_path in iter_candidates(sources):
        counts["scanned"] += 1
        if dest_path.exists():
            counts["already_present"] += 1
            continue
        if dry_run:
            counts["copied"] += 1
            continue
        actual_sha = sha256(src_path)
        if actual_sha != expected_sha:
            counts["mismatched"] += 1
            restored[str(dest_path)] = {"status": "failed", "source": source,
                                         "expected": expected_sha, "actual": actual_sha}
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp = dest_path.with_suffix(dest_path.suffix + ".restore.tmp")
        shutil.copy2(src_path, temp)
        temp.replace(dest_path)
        restored[str(dest_path)] = {"status": "complete", "source": source, "sha256": actual_sha,
                                     "restored_at": datetime.now(timezone.utc).isoformat()}
        counts["copied"] += 1

    if not dry_run:
        write_manifest(manifest)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help=f"Comma-separated sources to restore (default: all of {list(SOURCE_DIRS)})")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not OBJECTS_DIR.is_dir():
        print(f"No local R2 mirror found at {OBJECTS_DIR} - nothing to restore "
              f"(run scripts/migration/r2_pull.py first if you want one).")
        return 0

    sources = args.only.split(",") if args.only else None
    counts = restore(sources=sources, dry_run=args.dry_run)
    print(f"scanned={counts['scanned']} copied={counts['copied']} "
          f"already_present={counts['already_present']} mismatched={counts['mismatched']}")
    if counts["mismatched"]:
        print(f"WARNING: {counts['mismatched']} local object(s) had a checksum mismatch against "
              f"their R2 key - see {MANIFEST_PATH} for details")
    return 0


if __name__ == "__main__":
    sys.exit(main())
