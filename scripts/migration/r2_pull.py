#!/usr/bin/env python3
"""Step 2: download every object in inventory.json into a local mirror of
the bucket's own key layout (<SMF_DATA_ROOT>/r2_pull/<key>), verifying
content as it goes.

For versioned-archive objects the sha256 is baked into the key itself
(see r2_inventory.py's classify()) - after download the hash is recomputed
and MUST match, or the file is discarded and logged as failed rather than
recorded as verified. Every other scheme has no reference hash to check
against; verification there means the download completed, the byte count
matches what R2 reported, and the computed hash is recorded for the record
(and for r2_delete.py to key off of).

Resumable: a key already present in verified.jsonl with a matching
recorded size is skipped outright - safe to re-run after an interruption.

Usage:
    python scripts/migration/r2_pull.py
    python scripts/migration/r2_pull.py --limit 50     # smoke test
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[2]))
import paths
import status as mstatus
from r2_client import r2_client

INVENTORY_PATH = paths.DATA_ROOT / "migration" / "inventory.json"
PULL_ROOT = paths.DATA_ROOT / "r2_pull"
VERIFIED_LOG = paths.DATA_ROOT / "migration" / "verified.jsonl"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_verified():
    verified = {}
    if VERIFIED_LOG.exists():
        for line in VERIFIED_LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                verified[record["key"]] = record
    return verified


def append_verified(record):
    VERIFIED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with VERIFIED_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def pull(limit=None):
    if not INVENTORY_PATH.exists():
        raise SystemExit("No inventory.json found - run r2_inventory.py first")
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    bucket = inventory["bucket"]
    objects = inventory["objects"][:limit] if limit else inventory["objects"]
    already = load_verified()
    s3 = r2_client()

    total = len(objects)
    total_bytes = sum(o["size"] for o in objects)
    done = sum(1 for o in objects if already.get(o["key"], {}).get("size") == o["size"])
    bytes_done = sum(o["size"] for o in objects if already.get(o["key"], {}).get("size") == o["size"])
    errors = []

    mstatus.update_phase("r2_pull", status="running", files_total=total, files_done=done,
                          bytes_total=total_bytes, bytes_done=bytes_done, current_file=None)

    for obj in objects:
        key = obj["key"]
        if already.get(key, {}).get("size") == obj["size"]:
            continue
        local_path = PULL_ROOT / key
        temp_path = local_path.with_name(local_path.name + ".part")
        mstatus.update_phase("r2_pull", status="running", current_file=key)
        try:
            # R2's flat namespace allows a bare key (e.g. "assets") to
            # coexist with "assets/whatever" - impossible to mirror as a
            # real filesystem path, since "assets" can't be both a file
            # and a directory at once. Confirmed live 2026-10-03: a prior
            # object already landed at a path another object needs as its
            # parent directory. Move the conflicting file aside rather
            # than losing it or crashing the whole pull over one key.
            for ancestor in local_path.parents:
                if ancestor == PULL_ROOT:
                    break
                if ancestor.is_file():
                    ancestor.rename(ancestor.with_name(ancestor.name + ".r2-file-conflict"))
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(temp_path))
            actual_size = temp_path.stat().st_size
            if actual_size != obj["size"]:
                raise RuntimeError(f"size mismatch: expected {obj['size']}, got {actual_size}")
            sha_actual = digest(temp_path)
            expected_sha = obj.get("sha256")
            if expected_sha and sha_actual != expected_sha:
                raise RuntimeError(f"checksum mismatch: expected {expected_sha}, got {sha_actual}")
            temp_path.replace(local_path)
            record = {
                "key": key, "size": obj["size"], "scheme": obj["scheme"],
                "sha256_expected": expected_sha, "sha256_computed": sha_actual,
                "checksum_source": "key" if expected_sha else "computed-only",
                "local_path": str(local_path), "verified_at": mstatus.utc_now(),
            }
            append_verified(record)
            already[key] = record
            done += 1
            bytes_done += obj["size"]
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            errors.append({"key": key, "error": str(exc)})
            mstatus.log_event("r2_pull", f"FAILED {key}: {exc}", level="error")
        mstatus.update_phase("r2_pull", status="running", files_total=total, files_done=done,
                              bytes_total=total_bytes, bytes_done=bytes_done, current_file=key,
                              errors=errors[-50:])

    final_status = "completed" if not errors else "completed_with_errors"
    mstatus.update_phase("r2_pull", status=final_status, files_total=total, files_done=done,
                          bytes_total=total_bytes, bytes_done=bytes_done, current_file=None, errors=errors[-50:])
    return {"total": total, "done": done, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Download everything in inventory.json to local disk, verified")
    parser.add_argument("--limit", type=int, help="Only pull the first N objects (smoke test)")
    args = parser.parse_args()
    result = pull(limit=args.limit)
    print(f"Pulled {result['done']}/{result['total']} objects, {len(result['errors'])} errors")
    if result["errors"]:
        print("See migration_status.json / verified.jsonl for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
