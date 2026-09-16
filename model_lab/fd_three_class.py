"""Train the staged three-class Fire Danger model.

Classes are Low (0), Moderate (1), and Elevated+ (2). Critical/Extreme
severity remains available from the operational rule layer.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

import paths
from model_lab.dataset_builder import FD_DATASET_PATH

FEATURES = [
    "temp_c", "rel_humidity", "wind_speed_ms", "hour", "month",
    "emc_baseline", "temp_mean_3h", "rh_mean_3h", "temp_mean_6h", "rh_mean_6h",
]


def _thresholds(predicted: np.ndarray, actual: np.ndarray) -> list[float]:
    centers = []
    for category in range(3):
        values = predicted[actual == category]
        centers.append(float(np.median(values)) if len(values) else float(category))
    centers = np.maximum.accumulate(np.asarray(centers))
    return [float((centers[index] + centers[index + 1]) / 2) for index in range(2)]


def train_three_class_candidate(
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    dataset_path: str | Path = FD_DATASET_PATH,
) -> dict:
    frame = pd.read_csv(dataset_path)
    missing = [column for column in FEATURES + ["obs_time", "fire_danger_category"] if column not in frame]
    if missing:
        raise ValueError(f"FD dataset missing columns: {missing}")
    frame["obs_time"] = pd.to_datetime(frame["obs_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=FEATURES + ["fire_danger_category", "obs_time"]).sort_values("obs_time")
    frame["target_category"] = frame["fire_danger_category"].astype(int).clip(0, 2)
    frame["target_score"] = frame["target_category"].astype(float)
    # Hold out complete calendar days. The ordinary last-20% split can contain
    # almost no Elevated+ events even when the full dataset has them.
    frame["_date"] = frame["obs_time"].dt.date
    dates = sorted(frame["_date"].unique())
    test_dates = set(dates[max(1, int(len(dates) * 0.8)):])
    high_dates = [
        day for day in dates
        if day not in test_dates and int((frame.loc[frame["_date"] == day, "target_category"] == 2).sum()) > 0
    ]
    for day in reversed(high_dates):
        test_dates.add(day)
        if int((frame.loc[frame["_date"].isin(test_dates), "target_category"] == 2).sum()) >= 25:
            break
    test = frame[frame["_date"].isin(test_dates)]
    train = frame[~frame["_date"].isin(test_dates)]
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(train[FEATURES], train["target_score"])
    train_pred = model.predict(train[FEATURES])
    test_pred = model.predict(test[FEATURES])
    thresholds = _thresholds(train_pred, train["target_category"].to_numpy())
    train_cat = np.clip(np.digitize(train_pred, thresholds), 0, 2)
    test_cat = np.clip(np.digitize(test_pred, thresholds), 0, 2)
    labels = [0, 1, 2]
    support = {str(category): int((test["target_category"] == category).sum()) for category in labels}
    full_support = {str(category): int((frame["target_category"] == category).sum()) for category in labels}
    report = {
        "class_mode": "three_class",
        "class_labels": ["Low", "Moderate", "Elevated+"],
        "dataset_path": str(dataset_path),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_first": test["obs_time"].min().isoformat(),
        "test_last": test["obs_time"].max().isoformat(),
        "split_strategy": "calendar_day_holdout_with_elevated_support",
        "class_support": support,
        "full_class_support": full_support,
        "regression": {
            "mae": float(mean_absolute_error(test["target_score"], test_pred)),
            "rmse": float(np.sqrt(mean_squared_error(test["target_score"], test_pred))),
            "r2": float(r2_score(test["target_score"], test_pred)),
        },
        "macro_f1": float(f1_score(test["target_category"], test_cat, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test["target_category"], test_cat, labels=labels, average="weighted", zero_division=0)),
        "thresholds": thresholds,
        "ready_for_evaluation": all(support[str(category)] >= 25 for category in labels),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = paths.MODELS_DIR / "experiments"
    report_dir = paths.REPORTS_DIR / "model_lab" / "workshop"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"fd3_{stamp}.json"
    meta_path = output_dir / f"fd3_{stamp}_meta.json"
    report_path = report_dir / f"fd3_{stamp}_evaluation.json"
    model.save_model(model_path)
    meta = {
        "model_type": "xgboost_regressor",
        "class_mode": "three_class",
        "class_count": 3,
        "class_labels": report["class_labels"],
        "feature_columns": FEATURES,
        "target_score_col": "target_score",
        "target_category_col": "target_category",
        "category_thresholds": thresholds,
        "created_utc": stamp,
        "versioned_model_path": str(model_path),
        "training_metrics": report,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report.update({"checkpoint": str(model_path), "meta": str(meta_path), "eval_report": str(report_path)})
    return report
