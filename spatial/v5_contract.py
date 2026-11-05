"""Frozen evidence contract for the conservative summer-guarded V5 model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
from spatial.rule_contract import RULE_SPEC_SHA256, RULE_SPEC_VERSION
from spatial.station_contract import ROW_KEY, sha256_file

SPLIT_VERSION = "station-split-v4"
FEATURE_SCHEMA_VERSION = "station-summer-guarded-v5.0-precipitation-v1"
PROSPECTIVE_AFTER = "2026-08-01T23:59:59+00:00"
MIN_PROSPECTIVE_DAYS = 30


def _hash(value):
    clean = {key: item for key, item in value.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _partition_hash(frame, run_ids):
    selected = frame[frame.run_id.astype(str).isin(set(map(str, run_ids)))].copy()
    selected = selected.sort_values([name for name in ROW_KEY if name in selected]).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(selected, index=False).values.tobytes()).hexdigest()


def create_manifest(frame, dataset_path: Path, v4_manifest: dict, feature_names: list[str]):
    development = list(map(str, v4_manifest["development_runs"]))
    # V4 calibration and locked-test metrics have been observed. Both are permanently
    # unavailable to V5 fitting and model-selection decisions.
    exposed = list(map(str, v4_manifest.get("calibration_runs", [])))
    locked = list(map(str, v4_manifest.get("forbidden_v3_test_runs", [])))
    forbidden = sorted(set(exposed + locked))
    folds = v4_manifest["folds"]
    used = set(development)
    if used & set(forbidden):
        raise RuntimeError("V5 development overlaps exposed or forbidden evidence")
    for fold in folds:
        fold_runs = set(map(str, fold["train_runs"] + fold["validation_runs"]))
        if not fold_runs <= used or fold_runs & set(forbidden):
            raise RuntimeError(f"Invalid inherited V5 fold {fold['name']}")
    manifest = {
        "version": SPLIT_VERSION,
        "dataset_sha256_at_creation": sha256_file(dataset_path),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": feature_names,
        "row_key": ROW_KEY,
        "policy": "development-only-selection; exposed-v4-evidence-forbidden; prospective-final",
        "rule_spec_version": RULE_SPEC_VERSION,
        "rule_spec_sha256": RULE_SPEC_SHA256,
        "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
        "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
        "development_runs": development,
        "folds": folds,
        "forbidden_runs": forbidden,
        "forbidden_sources": {"v4_calibration_runs": exposed, "v3_v4_locked_test_runs": locked},
        "prospective_after": PROSPECTIVE_AFTER,
        "prospective_requirements": {
            "minimum_days": MIN_PROSPECTIVE_DAYS,
            "elevated_period_required": True,
            "adequate_summer_rain_critical_support_required": True,
        },
    }
    manifest["forbidden_runs_sha256"] = hashlib.sha256("\n".join(forbidden).encode()).hexdigest()
    manifest["development_partition_sha256"] = _partition_hash(frame, development)
    manifest["manifest_sha256"] = _hash(manifest)
    return manifest


def validate_manifest(manifest, frame, dataset_path, feature_names):
    expected = {
        "version": SPLIT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": feature_names,
        "row_key": ROW_KEY,
        "rule_spec_sha256": RULE_SPEC_SHA256,
        "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
        "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if manifest.get("manifest_sha256") != _hash(manifest):
        mismatches.append("manifest_sha256")
    if manifest.get("development_partition_sha256") != _partition_hash(frame, manifest.get("development_runs", [])):
        mismatches.append("development_partition_sha256")
    development, forbidden = set(manifest.get("development_runs", [])), set(manifest.get("forbidden_runs", []))
    if development & forbidden:
        mismatches.append("forbidden_overlap")
    for fold in manifest.get("folds", []):
        fold_runs = set(map(str, fold["train_runs"] + fold["validation_runs"]))
        if not fold_runs <= development or fold_runs & forbidden:
            mismatches.append(f"{fold['name']}_contract")
    if mismatches:
        raise RuntimeError(f"V5 evidence contract mismatch: {', '.join(mismatches)}")
    return manifest


def load_or_create(path, frame, dataset_path, v4_path, feature_names, create=True):
    path, v4_path = Path(path), Path(v4_path)
    if path.exists():
        manifest = json.loads(path.read_text())
    elif create:
        manifest = create_manifest(frame, dataset_path, json.loads(v4_path.read_text()), feature_names)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(path)
    else:
        raise FileNotFoundError(path)
    return validate_manifest(manifest, frame, dataset_path, feature_names)


def reject_forbidden(manifest, run_ids, stage):
    overlap = set(map(str, run_ids)) & set(manifest["forbidden_runs"])
    if overlap:
        raise RuntimeError(f"{stage} attempted to access {len(overlap)} forbidden V5 evidence runs")


def prospective_runs(manifest, frame):
    runs = frame[["run_id", "forecast_init_time"]].drop_duplicates().copy()
    runs["run_id"] = runs.run_id.astype(str)
    result = runs[pd.to_datetime(runs.forecast_init_time, utc=True) > pd.Timestamp(manifest["prospective_after"])]
    ids = result.sort_values("forecast_init_time").run_id.tolist()
    reject_forbidden(manifest, ids, "V5 prospective discovery")
    return ids
