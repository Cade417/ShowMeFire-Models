#!/usr/bin/env python3
"""Step 3 (manual, separate from the pull): delete from R2 anything
r2_pull.py has already downloaded and verified locally.

Default is a dry run - prints what would be deleted and how many bytes
would be reclaimed. Nothing is actually deleted until --confirm-delete is
passed, and every key is appended to delete_audit.jsonl *before* the
delete_object call fires, so there is a paper trail even if the run is
interrupted mid-way. Already-deleted keys (per the audit log) are skipped
on a re-run, so this is safe to run more than once.

This script only ever reads from verified.jsonl (written by r2_pull.py) -
it will never delete something that was not itself verified after
download.

Usage:
    python scripts/migration/r2_delete.py                 # dry run
    python scripts/migration/r2_delete.py --confirm-delete
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[2]))
import paths
import status as mstatus
from r2_client import r2_bucket, r2_client

VERIFIED_LOG = paths.DATA_ROOT / "migration" / "verified.jsonl"
DELETE_AUDIT_LOG = paths.DATA_ROOT / "migration" / "delete_audit.jsonl"


def load_verified():
    records = []
    if VERIFIED_LOG.exists():
        for line in VERIFIED_LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def already_deleted_keys():
    keys = set()
    if DELETE_AUDIT_LOG.exists():
        for line in DELETE_AUDIT_LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys.add(json.loads(line)["key"])
    return keys


def append_audit(record):
    DELETE_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DELETE_AUDIT_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def run(confirm_delete=False):
    verified = load_verified()
    done = already_deleted_keys()
    pending = [r for r in verified if r["key"] not in done]
    total_bytes = sum(r["size"] for r in pending)

    if not confirm_delete:
        print(f"DRY RUN: {len(pending)} verified objects would be deleted from R2 "
              f"({total_bytes / (1024 ** 3):.2f} GB reclaimed).")
        print("Re-run with --confirm-delete to actually delete.")
        return {"dry_run": True, "would_delete": len(pending), "bytes": total_bytes}

    bucket = r2_bucket()
    s3 = r2_client()
    mstatus.update_phase("r2_delete", status="running", files_total=len(pending), files_done=0,
                          bytes_total=total_bytes, bytes_done=0, current_file=None)
    deleted = 0
    bytes_deleted = 0
    errors = []
    for record in pending:
        key = record["key"]
        # Logged before the delete call, on purpose: if this process dies
        # mid-call we want the audit log to still show intent, not silence.
        append_audit({"key": key, "sha256": record.get("sha256_computed"),
                       "size": record["size"], "deleted_at": mstatus.utc_now()})
        try:
            s3.delete_object(Bucket=bucket, Key=key)
            deleted += 1
            bytes_deleted += record["size"]
        except Exception as exc:
            errors.append({"key": key, "error": str(exc)})
            mstatus.log_event("r2_delete", f"FAILED to delete {key}: {exc}", level="error")
        mstatus.update_phase("r2_delete", status="running", files_total=len(pending), files_done=deleted,
                              bytes_total=total_bytes, bytes_done=bytes_deleted, current_file=key,
                              errors=errors[-50:])

    final_status = "completed" if not errors else "completed_with_errors"
    mstatus.update_phase("r2_delete", status=final_status, files_total=len(pending), files_done=deleted,
                          bytes_total=total_bytes, bytes_done=bytes_deleted, current_file=None, errors=errors[-50:])
    return {"dry_run": False, "deleted": deleted, "bytes": bytes_deleted, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Delete from R2 what has already been verified locally")
    parser.add_argument("--confirm-delete", action="store_true",
                         help="Actually delete. Without this flag, only reports what would happen.")
    args = parser.parse_args()
    result = run(confirm_delete=args.confirm_delete)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
