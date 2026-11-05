"""Backfill promotion metadata for an existing beta without retraining.

This is intended for legacy candidates registered before the metadata contract
was wired through the training/release pipeline. It never changes the model
artifact, version, performance metrics, or checksum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
from models import versioning
from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repair(model_type: str = "fuel_moisture", version: str | None = None) -> dict:
    config = versioning._load_config()
    entry = config.get(model_type) or {}
    beta = entry.get("beta")
    if not beta:
        raise RuntimeError(f"No beta candidate registered for {model_type!r}")
    if version and beta.get("version") != version:
        raise RuntimeError(f"Requested {version!r}, current beta is {beta.get('version')!r}")
    if beta.get("metadata"):
        raise RuntimeError("Beta already has metadata; refusing to overwrite it")

    artifact = paths.DATA_ROOT / beta["file"]
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    actual_sha = _sha256(artifact)
    if beta.get("sha256") and beta["sha256"].lower() != actual_sha.lower():
        raise RuntimeError("Registered beta checksum does not match its artifact")

    booster = xgb.Booster()
    booster.load_model(str(artifact))
    feature_columns = list(booster.feature_names or [])
    if not feature_columns:
        raise RuntimeError("Artifact does not expose feature names")

    data_path = paths.TRAINING_DATA_DIR / "final_training_data.csv"
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    frame = pd.read_csv(data_path, parse_dates=["obs_time"]).sort_values("obs_time")
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Training data cannot describe artifact features: {missing}")
    split = int(len(frame) * 0.8)
    train = frame.iloc[:split]
    match_meta_path = paths.DATA_ROOT / "training_set_mo_meta.json"
    match_meta = json.loads(match_meta_path.read_text()) if match_meta_path.is_file() else {}

    metadata = {
        "feature_schema_version": "2.0.0",
        "feature_columns": feature_columns,
        "feature_ranges": {
            name: {"min": float(train[name].min()), "max": float(train[name].max())}
            for name in feature_columns
        },
        "rule_spec_version": "1.0.0",
        "training_window": {
            "start": frame["obs_time"].min().isoformat(),
            "end": frame["obs_time"].max().isoformat(),
            "holdout_start": frame["obs_time"].iloc[split].isoformat(),
            "holdout_end": frame["obs_time"].max().isoformat(),
        },
        "data_match_policy": match_meta.get("data_match_policy", {}),
        # The legacy .6 run used a chronological holdout, not validation folds.
        "validation_folds": [],
        "class_support": {},
        "imputation_policy": {"hours_since_rain_without_history": 24.0},
        "max_feature_age_minutes": 60,
        "promotion_gates": {},
        "shadow_required": True,
        "ground_truth_shadow_required": True,
    }
    if any(name.startswith("precip_") or name == "hours_since_rain" for name in feature_columns):
        metadata.update({
            "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
            "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
        })

    backup = versioning.CONFIG_PATH.with_suffix(".json.bak")
    shutil.copy2(versioning.CONFIG_PATH, backup)
    beta["metadata"] = metadata
    versioning._save_config(config)
    return {"version": beta["version"], "artifact": str(artifact), "sha256": actual_sha,
            "backup": str(backup),
            "feature_columns": feature_columns, "metadata": metadata}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="fuel_moisture")
    parser.add_argument("--version")
    args = parser.parse_args()
    result = repair(args.model, args.version)
    print(json.dumps(result, indent=2))
