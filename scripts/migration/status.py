"""Shared progress-status file for the R2/server -> local HDD migration tool.

One JSON file, one phase per key ("inventory", "r2_pull", "server_pull",
"r2_delete"), atomic write-then-replace - the same pattern
api/services/forecast_v1_job.py uses for its admin-job.json, just shared
across several independent CLI scripts instead of one worker thread.
dashboard.py only ever reads this file; it never writes to it.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import paths

STATUS_PATH = paths.DATA_ROOT / "migration" / "migration_status.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_status():
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"phases": {}}


def write_status(status):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = STATUS_PATH.with_name(STATUS_PATH.name + ".tmp")
    temp.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, STATUS_PATH)


def update_phase(phase, **fields):
    """Atomically merge fields into one phase's status block."""
    status = read_status()
    block = status.setdefault("phases", {}).setdefault(phase, {})
    block.update(fields)
    block["updated_at"] = utc_now()
    status["updated_at"] = utc_now()
    write_status(status)
    return status


def log_event(phase, message, level="info"):
    status = read_status()
    block = status.setdefault("phases", {}).setdefault(phase, {})
    events = list(block.get("events") or [])
    events.append({"time": utc_now(), "level": level, "message": message})
    block["events"] = events[-200:]
    block["updated_at"] = utc_now()
    status["updated_at"] = utc_now()
    write_status(status)
    return status
