#!/usr/bin/env python3
"""Step 1: full inventory of everything in the R2 bucket, across every
prefix/scheme currently in use - versioned archive objects+manifests
(cache/hrrr, cache/rrfs, archive/raw_data, archive/forecasts), legacy daily
zips, NGFS raw zips, CDN image/graphics uploads, and post media.

Lists ONLY - no downloads, no deletes. inventory.json is the one shared
source of truth r2_pull.py and r2_delete.py both read from.

Versioned-archive object keys already embed their own sha256
(data-v2/objects/{date}/{source}/{sha256}/{filename} - see
api/services/archive_store.py), so no separate manifest fetch is needed
just to know the expected checksum.

Usage:
    python scripts/migration/r2_inventory.py
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[2]))
import paths
import status as mstatus
from r2_client import r2_bucket, r2_client, r2_data_prefix

INVENTORY_PATH = paths.DATA_ROOT / "migration" / "inventory.json"


def classify(key, data_prefix):
    if key.startswith(f"{data_prefix}/objects/"):
        parts = key.split("/")
        sha256 = parts[-2] if len(parts) >= 6 else None
        return {"scheme": "versioned-object", "sha256": sha256}
    if key.startswith(f"{data_prefix}/manifests/"):
        return {"scheme": "versioned-manifest", "sha256": None}
    if key.startswith("data-archive/") and key.endswith(".zip"):
        return {"scheme": "legacy-zip", "sha256": None}
    if key.startswith("ngfs-ogc-raw/") and key.endswith(".zip"):
        return {"scheme": "ngfs-zip", "sha256": None}
    if key.startswith("assets/posts/"):
        return {"scheme": "post-media", "sha256": None}
    return {"scheme": "cdn-media", "sha256": None}


def build_inventory():
    bucket = r2_bucket()
    data_prefix = r2_data_prefix()
    s3 = r2_client()

    mstatus.update_phase("inventory", status="running", message=f"Listing bucket {bucket}",
                          objects_found=0, bytes_found=0)
    objects = []
    by_scheme = {}
    count = 0
    total_bytes = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for item in page.get("Contents", []):
            key = item["Key"]
            meta = classify(key, data_prefix)
            objects.append({"key": key, "size": item["Size"], **meta})
            by_scheme[meta["scheme"]] = by_scheme.get(meta["scheme"], 0) + 1
            count += 1
            total_bytes += item["Size"]
            if count % 500 == 0:
                mstatus.update_phase("inventory", status="running",
                                      message=f"Listed {count} objects so far",
                                      objects_found=count, bytes_found=total_bytes, by_scheme=by_scheme)

    inventory = {
        "bucket": bucket, "data_prefix": data_prefix, "generated_at": mstatus.utc_now(),
        "object_count": count, "total_bytes": total_bytes,
        "by_scheme": by_scheme, "objects": objects,
    }
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY_PATH.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    mstatus.update_phase("inventory", status="completed",
                          message=f"Inventory complete: {count} objects, {total_bytes} bytes",
                          objects_found=count, bytes_found=total_bytes, by_scheme=by_scheme)
    return inventory


def main():
    inventory = build_inventory()
    print(f"Bucket: {inventory['bucket']}")
    print(f"Objects: {inventory['object_count']}")
    print(f"Total size: {inventory['total_bytes'] / (1024 ** 3):.2f} GB")
    for scheme, n in sorted(inventory["by_scheme"].items()):
        print(f"  {scheme}: {n}")
    print(f"\nWrote {INVENTORY_PATH}")


if __name__ == "__main__":
    main()
