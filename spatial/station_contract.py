"""Dataset, split, scoring, and artifact contracts for station experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from spatial.station_features import FEATURE_SCHEMA_VERSION, FEATURES

SPLIT_VERSION = "station-split-v1"
ROW_KEY = ["run_id", "station_id", "valid_time"]
MIN_GROUP_SAMPLES = 30
MIN_CRITICAL_SAMPLES = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_split_manifest(frame: pd.DataFrame, dataset_path: Path) -> dict:
    runs = frame[["run_id", "forecast_init_time"]].drop_duplicates().sort_values("forecast_init_time")
    run_ids = runs.run_id.astype(str).tolist()
    holdout_at = max(1, int(len(run_ids) * 0.8))
    development, holdout = run_ids[:holdout_at], run_ids[holdout_at:]
    if len(development) < 10 or not holdout:
        raise RuntimeError("Not enough chronological runs for development and locked holdout")
    fold_size = max(1, len(development) // 10)
    starts = [max(1, len(development) - fold_size * n) for n in (3, 2, 1)]
    folds = []
    for index, start in enumerate(starts):
        stop = min(len(development), start + fold_size)
        folds.append({"name": f"fold-{index + 1}", "train_runs": development[:start], "validation_runs": development[start:stop]})
    stations = sorted(frame.station_id.dropna().astype(str).unique())
    held_stations = [station for station in stations if sum(map(ord, station)) % 5 == 0]
    return {
        "version": SPLIT_VERSION,
        "dataset_sha256": sha256_file(dataset_path),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURES,
        "row_key": ROW_KEY,
        "policy": "chronological-80pct-development-20pct-locked-holdout",
        "development_runs": development,
        "holdout_runs": holdout,
        "folds": folds,
        "held_stations": held_stations,
        "region_lon_cutoff": float(frame.lon.median()),
    }


def load_or_create_manifest(path: Path, frame: pd.DataFrame, dataset_path: Path, create=True) -> dict:
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text())
    elif create:
        manifest = create_split_manifest(frame, dataset_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))
    else:
        raise FileNotFoundError(path)
    expected = {
        "version": SPLIT_VERSION,
        "dataset_sha256": sha256_file(dataset_path),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURES,
        "row_key": ROW_KEY,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise RuntimeError(f"Split manifest contract mismatch: {', '.join(mismatches)}")
    return manifest


def sequence_arrays(frame: pd.DataFrame, run_keys) -> tuple[tuple[np.ndarray, ...], list[tuple], np.ndarray]:
    features, physics, targets, masks, keys, row_indices = [], [], [], [], [], []
    selected = frame[frame.run_id.astype(str).isin(set(map(str, run_keys)))]
    if selected.empty:
        return tuple(np.empty((0,)) for _ in range(4)), [], np.empty((0, 0), dtype=int)
    expected = int(frame.groupby(["run_id", "station_id"]).size().mode().iloc[0])
    for key, group in selected.sort_values("lead_hour").groupby(["run_id", "station_id"]):
        if len(group) != expected or group[FEATURES + ["physics_fm"]].isna().any().any():
            continue
        features.append(group[FEATURES].to_numpy(dtype="float32"))
        physics.append(group.physics_fm.to_numpy(dtype="float32"))
        targets.append(group.target_fm.fillna(0).to_numpy(dtype="float32"))
        masks.append(group.target_mask.to_numpy(dtype="float32"))
        keys.append(tuple(map(str, key)))
        row_indices.append(group.index.to_numpy(dtype=int))
    arrays = tuple(np.asarray(value) for value in (features, physics, targets, masks))
    return arrays, keys, np.asarray(row_indices)


def basic_metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if len(actual) < 2:
        return None
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "bias": float(np.mean(predicted - actual)),
        "r2": float(r2_score(actual, predicted)),
        "samples": int(len(actual)),
    }


def detailed_metrics(scored: pd.DataFrame, quantiles=None) -> dict:
    report = basic_metrics(scored.target_fm, scored.prediction)
    report["by_lead"] = {str(key): basic_metrics(group.target_fm, group.prediction)
                         for key, group in scored.groupby("lead_hour") if len(group) >= MIN_GROUP_SAMPLES}
    drying = scored.target_fm < scored.initial_fm
    critical = scored.target_fm <= 6
    report["regimes"] = {
        "drying": basic_metrics(scored.target_fm[drying], scored.prediction[drying]) if drying.sum() >= MIN_GROUP_SAMPLES else None,
        "wetting": basic_metrics(scored.target_fm[~drying], scored.prediction[~drying]) if (~drying).sum() >= MIN_GROUP_SAMPLES else None,
        "critical_low_fm": basic_metrics(scored.target_fm[critical], scored.prediction[critical]) if critical.sum() >= MIN_CRITICAL_SAMPLES else None,
    }
    report["by_month"] = {str(key): basic_metrics(group.target_fm, group.prediction)
                          for key, group in scored.groupby("month") if len(group) >= MIN_GROUP_SAMPLES}
    report["by_station"] = {str(key): basic_metrics(group.target_fm, group.prediction)
                            for key, group in scored.groupby("station_id") if len(group) >= MIN_GROUP_SAMPLES}
    predicted_critical = scored.prediction <= 6
    tp, fn = int((critical & predicted_critical).sum()), int((critical & ~predicted_critical).sum())
    fp, tn = int((~critical & predicted_critical).sum()), int((~critical & ~predicted_critical).sum())
    report["critical_threshold"] = {
        "threshold": 6, "true_positive": tp, "false_negative": fn, "false_positive": fp, "true_negative": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "false_negative_rate": fn / (tp + fn) if tp + fn else None,
        "support": int(critical.sum()),
    }
    if quantiles is not None:
        q10, q50, q90 = np.moveaxis(np.asarray(quantiles), -1, 0)
        report["interval_coverage"] = float(np.mean((scored.target_fm.to_numpy() >= q10) & (scored.target_fm.to_numpy() <= q90)))
        report["quantile_order_violation_rate"] = float(np.mean((q10 > q50) | (q50 > q90)))
    return report
