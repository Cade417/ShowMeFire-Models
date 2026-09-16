"""Immutable data/split contract for the V3 hybrid station model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from spatial.station_contract import ROW_KEY, sha256_file

SPLIT_VERSION = "station-split-v2"
FEATURE_SCHEMA_VERSION = "station-hybrid-causal-v3"


def _manifest_hash(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def create_manifest(frame: pd.DataFrame, dataset_path: Path, feature_names: list[str]) -> dict:
    runs = frame[["run_id", "forecast_init_time"]].drop_duplicates().sort_values("forecast_init_time")
    run_ids = runs.run_id.astype(str).tolist()
    if len(run_ids) < 30:
        raise RuntimeError("At least 30 initialization runs are required")
    development_at = max(1, int(len(run_ids) * 0.8))
    calibration_at = max(development_at + 1, int(len(run_ids) * 0.9))
    development = run_ids[:development_at]
    calibration = run_ids[development_at:calibration_at]
    locked_test = run_ids[calibration_at:]
    if not calibration or not locked_test:
        raise RuntimeError("Calibration and locked-test partitions must be non-empty")
    fold_size = max(1, len(development) // 10)
    starts = [len(development) - fold_size * count for count in (3, 2, 1)]
    folds = []
    for index, start in enumerate(starts, 1):
        stop = min(len(development), start + fold_size)
        folds.append({"name": f"fold-{index}", "train_runs": development[:start],
                      "validation_runs": development[start:stop]})
    manifest = {
        "version": SPLIT_VERSION,
        "dataset_sha256": sha256_file(dataset_path),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": feature_names,
        "row_key": ROW_KEY,
        "policy": "chronological-80-development-10-calibration-10-historical-relock",
        "historical_relock": True,
        "development_runs": development,
        "calibration_runs": calibration,
        "locked_test_runs": locked_test,
        "folds": folds,
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    return manifest


def load_or_create_manifest(path: Path, frame: pd.DataFrame, dataset_path: Path,
                            feature_names: list[str], create=True) -> dict:
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text())
    elif create:
        manifest = create_manifest(frame, dataset_path, feature_names)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(path)
    else:
        raise FileNotFoundError(path)
    expected = {
        "version": SPLIT_VERSION,
        "dataset_sha256": sha256_file(dataset_path),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": feature_names,
        "row_key": ROW_KEY,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        mismatches.append("manifest_sha256")
    partitions = [set(manifest[key]) for key in ("development_runs", "calibration_runs", "locked_test_runs")]
    if partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2]:
        mismatches.append("partition_overlap")
    locked = partitions[2]
    for fold in manifest.get("folds", []):
        train, validation = set(fold["train_runs"]), set(fold["validation_runs"])
        if train & validation or locked & (train | validation):
            mismatches.append(f"{fold['name']}_leakage")
    if mismatches:
        raise RuntimeError(f"Hybrid split contract mismatch: {', '.join(dict.fromkeys(mismatches))}")
    return manifest
