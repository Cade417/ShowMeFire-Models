"""Immutable evidence contract for guarded station model V4."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from spatial.rule_contract import RULE_SPEC_SHA256, RULE_SPEC_VERSION
from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
from spatial.station_contract import ROW_KEY, sha256_file

SPLIT_VERSION = "station-split-v3"
FEATURE_SCHEMA_VERSION = "station-guarded-causal-v4.2-precipitation-v1"
PROSPECTIVE_AFTER = "2026-08-01T23:59:59+00:00"
MIN_PROSPECTIVE_DAYS = 30


def _hash(value):
    clean = {key: item for key, item in value.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _partition_hash(frame, run_ids):
    selected = frame[frame.run_id.astype(str).isin(set(map(str, run_ids)))].copy()
    sort_columns = [name for name in ROW_KEY if name in selected]
    selected = selected.sort_values(sort_columns).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(selected, index=False).values.tobytes()).hexdigest()


def create_manifest(frame, dataset_path: Path, v3_manifest: dict, feature_names: list[str]):
    development = list(map(str, v3_manifest["development_runs"]))
    calibration = list(map(str, v3_manifest["calibration_runs"]))
    forbidden = list(map(str, v3_manifest["locked_test_runs"]))
    midpoint = max(1, len(calibration) // 2)
    runs = frame[["run_id", "forecast_init_time"]].drop_duplicates().copy()
    runs["run_id"] = runs.run_id.astype(str)
    prospective = runs[pd.to_datetime(runs.forecast_init_time, utc=True) > pd.Timestamp(PROSPECTIVE_AFTER)]
    prospective_runs = prospective.sort_values("forecast_init_time").run_id.tolist()
    ordered_development = runs[runs.run_id.isin(set(development))].sort_values("forecast_init_time")
    august = ordered_development[pd.to_datetime(ordered_development.forecast_init_time, utc=True).dt.month == 8]
    if not august.empty:
        first_summer_validation = pd.to_datetime(august.forecast_init_time, utc=True).min()
        summer_train = ordered_development[pd.to_datetime(ordered_development.forecast_init_time, utc=True) < first_summer_validation].run_id.tolist()
        summer_valid = august.run_id.tolist()
    else:
        summer_train, summer_valid = [], []
    inherited = v3_manifest["folds"]
    if len(summer_train) >= 10 and len(summer_valid) >= 10 and len(inherited) >= 3:
        folds = [
            {"name": "fold-summer", "train_runs": summer_train, "validation_runs": summer_valid},
            {"name": "fold-winter", "train_runs": inherited[0]["train_runs"], "validation_runs": inherited[0]["validation_runs"]},
            {"name": "fold-spring", "train_runs": inherited[-1]["train_runs"], "validation_runs": inherited[-1]["validation_runs"]},
        ]
        fold_policy = "three-expanding-chronological-folds-with-august-development-validation"
    else:
        folds = inherited
        fold_policy = "inherited-v3-folds-no-supported-development-summer"
    if (set(development) | set(calibration)) & set(forbidden):
        raise RuntimeError("V3 forbidden runs overlap reusable V4 partitions")
    manifest = {
        "version": SPLIT_VERSION, "dataset_sha256_at_creation": sha256_file(dataset_path),
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_names": feature_names,
        "row_key": ROW_KEY, "rule_spec_version": RULE_SPEC_VERSION,
        "rule_spec_sha256": RULE_SPEC_SHA256,
        "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
        "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
        "development_runs": development, "calibration_runs": calibration,
        "calibration_fit_runs": calibration[:midpoint],
        "calibration_validation_runs": calibration[midpoint:],
        "forbidden_v3_test_runs": forbidden, "folds": folds, "fold_policy": fold_policy,
        "prospective_after": PROSPECTIVE_AFTER, "prospective_runs_at_creation": prospective_runs,
        "prospective_requirements": {"minimum_days": MIN_PROSPECTIVE_DAYS,
                                     "elevated_period_required": True},
    }
    manifest["forbidden_runs_sha256"] = hashlib.sha256("\n".join(forbidden).encode()).hexdigest()
    manifest["frozen_partition_sha256"] = _partition_hash(frame, development + calibration + forbidden)
    manifest["manifest_sha256"] = _hash(manifest)
    return manifest


def validate_manifest(manifest, frame, dataset_path, feature_names):
    expected = {"version": SPLIT_VERSION,
                "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_names": feature_names,
                "row_key": ROW_KEY, "rule_spec_sha256": RULE_SPEC_SHA256,
                "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
                "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256}
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if manifest.get("manifest_sha256") != _hash(manifest): mismatches.append("manifest_sha256")
    frozen = manifest.get("development_runs", []) + manifest.get("calibration_runs", []) + manifest.get("forbidden_v3_test_runs", [])
    if manifest.get("frozen_partition_sha256") != _partition_hash(frame, frozen): mismatches.append("frozen_partition_sha256")
    reusable = set(manifest.get("development_runs", [])) | set(manifest.get("calibration_runs", []))
    forbidden = set(manifest.get("forbidden_v3_test_runs", []))
    if reusable & forbidden: mismatches.append("forbidden_overlap")
    for fold in manifest.get("folds", []):
        if forbidden & (set(fold["train_runs"]) | set(fold["validation_runs"])):
            mismatches.append(f"{fold['name']}_forbidden")
    if mismatches: raise RuntimeError(f"V4 evidence contract mismatch: {', '.join(mismatches)}")
    return manifest


def prospective_runs(manifest, frame):
    runs = frame[["run_id", "forecast_init_time"]].drop_duplicates().copy(); runs["run_id"] = runs.run_id.astype(str)
    selected = runs[pd.to_datetime(runs.forecast_init_time, utc=True) > pd.Timestamp(manifest["prospective_after"])]
    result = selected.sort_values("forecast_init_time").run_id.tolist()
    reject_forbidden(manifest, result, "prospective discovery")
    return result


def load_or_create(path, frame, dataset_path, v3_path, feature_names, create=True):
    path, v3_path = Path(path), Path(v3_path)
    if path.exists(): manifest = json.loads(path.read_text())
    elif create:
        manifest = create_manifest(frame, dataset_path, json.loads(v3_path.read_text()), feature_names)
        path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2)); temporary.replace(path)
    else: raise FileNotFoundError(path)
    return validate_manifest(manifest, frame, dataset_path, feature_names)


def reject_forbidden(manifest, run_ids, stage):
    overlap = set(map(str, run_ids)) & set(manifest["forbidden_v3_test_runs"])
    if overlap: raise RuntimeError(f"{stage} attempted to access {len(overlap)} forbidden V3 test runs")
