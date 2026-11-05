"""Structured status for data sources and trainable models.

Reads each puller's own manifest / the existing model registry and returns
plain dicts - no printing. scripts/sync_training_data.py's old
report_freshness() did this as print statements only; this is that same
logic, returned as data so both a CLI and the dashboard API can use it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from models.model_types import KNOWN_MODEL_TYPES
from models.versioning import get_model_entry
from pipelines.experiment_log import read_experiments
from pipeline.registry import DATA_SOURCES, TRAIN_TARGETS_BY_TYPE, detection_confidence_target


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def source_status(source) -> Dict:
    """One source's freshness, as a plain dict."""
    if not source.manifest_key:
        # Archive zips: no JSON manifest, just files on disk (see
        # scripts/sync_training_data.py's original ad-hoc handling).
        files = sorted(source.manifest_path.glob("*.zip")) if source.manifest_path.exists() else []
        newest = max(files, key=lambda p: p.stat().st_mtime) if files else None
        return {
            "key": source.key, "label": source.label, "available": source.available(),
            "complete": len(files), "pending": 0, "failed": 0,
            "latest": newest.name if newest else None,
            "updated_at": None, "note": source.note,
        }

    data = _load_json(source.manifest_path)
    if data is None:
        return {
            "key": source.key, "label": source.label, "available": source.available(),
            "complete": 0, "pending": 0, "failed": 0, "latest": None,
            "updated_at": None, "note": source.note or "no manifest yet - never run",
        }

    records = data.get(source.manifest_key, {}) or {}
    complete = [k for k, rec in records.items() if rec.get("status") == "complete"]
    failed = [k for k, rec in records.items() if rec.get("status") == "failed"]
    pending = [k for k in records if k not in complete and k not in failed]
    return {
        "key": source.key, "label": source.label, "available": source.available(),
        "complete": len(complete), "pending": len(pending), "failed": len(failed),
        "latest": max(complete) if complete else None,
        "updated_at": data.get("updated_at"), "note": source.note,
    }


def all_source_status() -> list:
    return [source_status(source) for source in DATA_SOURCES]


def model_status(model_type: str) -> Dict:
    entry = get_model_entry(model_type)
    target = TRAIN_TARGETS_BY_TYPE.get(model_type)
    return {
        "model_type": model_type,
        "label": target.label if target else model_type,
        "trainable": target is not None and target.available(),
        "train_note": target.note if target else "no single-command retrain wired up yet",
        "stable": entry.get("stable"),
        "beta": entry.get("beta"),
        "history_count": len(entry.get("history", [])),
    }


def all_model_status() -> list:
    statuses = [model_status(model_type) for model_type in KNOWN_MODEL_TYPES]
    for extra in ("detection_dataset", "detection_confidence_v2"):
        if not any(s["model_type"] == extra for s in statuses):
            statuses.append(model_status(extra))
    detection = detection_confidence_target()
    statuses.append({
        "model_type": "detection_confidence",
        "label": "Detection confidence (XGBoost)",
        "trainable": detection is not None,
        "train_note": detection.note if detection else "SMF_API_REPO_ROOT not set or sibling api/ repo not found",
        "stable": None, "beta": None, "history_count": 0,
    })
    return statuses


def experiments_for(model_type: str) -> list:
    return read_experiments(model_type)
