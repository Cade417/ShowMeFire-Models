"""
The shared definition of "the fitted fire_weather_ml model": fit, score,
save, load - all in one place, mirroring risk_fusion/model_bundle.py's
shape. Unlike risk_fusion's GLM (deliberately conservative because real
fire-occurrence labels are scarce - ~300 primary-tier events), this model's
label volume is bounded only by how much historical weather/fuel-moisture
data the Rothermel label generator (rothermel_labels.py) is run over, not
by scarce fire reports - so a gradient-boosted tree model (xgboost, already
a project dependency) is a reasonable starting point rather than something
that needs justifying against a label-scarcity ceiling.

Predicts `ros_ch_per_h` (rate of spread) by default - the primary physical
target. fireline_intensity_kw_per_m/flame_length_m are available in the
same panel (see rothermel_labels.py) for a future multi-output model; kept
single-output for Phase 1 to keep the fit/evaluate/register contract
simple to reason about first.

Trains on `features.MODEL_FEATURE_COLUMNS` (weather/fuel-moisture/terrain),
not the full `FEATURE_COLUMNS` panel schema - see that constant's own
docstring for the real Phase 3 finding that motivated excluding kbdi/
gdd_accum from what the model actually sees.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from fire_weather_ml.features import MODEL_FEATURE_COLUMNS

DEFAULT_LABEL_COLUMN = "ros_ch_per_h"
MODEL_ASSET_FILENAME = "fire_weather_ml_model.json"
METADATA_ASSET_FILENAME = "fire_weather_ml_metadata.json"
RISK_CALIBRATION_ASSET_FILENAME = "fire_weather_ml_risk_calibration.json"

# 0, 1, 2, ..., 100 - a value at every integer percentile.
RISK_CALIBRATION_PERCENTILES = list(range(101))

DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit(
    train_panel: pd.DataFrame,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    label_column: str = DEFAULT_LABEL_COLUMN,
    xgb_params: Dict | None = None,
) -> Dict:
    """
    Fits an XGBoost regressor against the Rothermel-computed label. Rows
    with a NaN label (fuel model didn't produce a Rothermel result - see
    rothermel_labels.py) are dropped before fitting - the model should
    never be asked to predict a physically-undefined target.
    """
    usable = train_panel.dropna(subset=[label_column])
    if usable.empty:
        raise ValueError(f"no rows with a non-null {label_column!r} label to fit against")

    missing = [c for c in feature_columns if c not in usable.columns]
    if missing:
        raise ValueError(f"train_panel is missing required feature columns: {missing}")

    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    model = xgb.XGBRegressor(**params)
    model.fit(usable[list(feature_columns)], usable[label_column])

    return {
        "model": model,
        "feature_columns": list(feature_columns),
        "label_column": label_column,
        "xgb_params": params,
        "training_row_count": int(len(usable)),
        "dropped_null_label_rows": int(len(train_panel) - len(usable)),
    }


def score(panel: pd.DataFrame, bundle: Dict) -> pd.Series:
    missing = [c for c in bundle["feature_columns"] if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing required feature columns: {missing}")
    predictions = bundle["model"].predict(panel[bundle["feature_columns"]])
    return pd.Series(predictions, index=panel.index, name=f"predicted_{bundle['label_column']}")


def calibrate_risk_score(predictions) -> Dict:
    """
    Builds the percentile lookup table that turns a raw ch/h prediction
    into a 0-100 "ML Fire Weather Risk Score" - calibrated against this
    model's OWN prediction distribution on real data, not an arbitrary or
    physics-derived scale. This matters concretely: this real distribution
    is heavily right-skewed (on the actual historical panel, p50 is only
    ~0.07 ch/h and p90 is only ~1.08 ch/h) - reusing a fixed physics-style
    scale (like api/services/spread_rate.py's own 0-150 ch/h ROS classes,
    built for more fire-prone climates) would collapse almost everything in
    Missouri into the bottom bucket, the exact same failure mode the rule-
    based public category already has. A percentile-rank score sidesteps
    that entirely: it's evenly distributed across 0-100 by construction,
    regardless of how skewed the underlying raw values are.

    `predictions` should be this model's own predictions (already clipped
    at zero) on a real, representative dataset - ideally the same panel it
    was fit on, so the calibration reflects what this model actually
    outputs, not an assumption about what it should output.
    """
    values = np.clip(np.asarray(predictions, dtype=float), 0.0, None)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite predictions to calibrate a risk score against")
    percentile_values = np.percentile(values, RISK_CALIBRATION_PERCENTILES)
    # percentile_values must be non-decreasing for interpolation at score
    # time to behave (np.percentile already guarantees this, but a repeat
    # value at both ends of a flat run is fine - np.interp handles ties).
    return {
        "percentiles": RISK_CALIBRATION_PERCENTILES,
        "values_ch_per_h": [float(v) for v in percentile_values],
        "sample_size": int(values.size),
        "min_ch_per_h": float(values.min()),
        "max_ch_per_h": float(values.max()),
    }


def risk_score_0_100(predicted_ch_per_h, calibration: Dict) -> np.ndarray:
    """Maps raw ch/h prediction(s) to the calibrated 0-100 risk score via linear interpolation against the percentile table."""
    values = np.clip(np.asarray(predicted_ch_per_h, dtype=float), 0.0, None)
    return np.interp(values, calibration["values_ch_per_h"], calibration["percentiles"])


def save(bundle: Dict, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bundle["model"].save_model(directory / MODEL_ASSET_FILENAME)
    metadata = {
        "feature_columns": bundle["feature_columns"],
        "label_column": bundle["label_column"],
        "xgb_params": bundle["xgb_params"],
        "training_row_count": bundle["training_row_count"],
        "dropped_null_label_rows": bundle["dropped_null_label_rows"],
    }
    (directory / METADATA_ASSET_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (directory / RISK_CALIBRATION_ASSET_FILENAME).write_text(
        json.dumps(bundle["risk_calibration"], indent=2), encoding="utf-8")


def load(directory: Path) -> Dict:
    directory = Path(directory)
    metadata = json.loads((directory / METADATA_ASSET_FILENAME).read_text(encoding="utf-8"))
    model = xgb.XGBRegressor()
    model.load_model(directory / MODEL_ASSET_FILENAME)
    risk_calibration = json.loads((directory / RISK_CALIBRATION_ASSET_FILENAME).read_text(encoding="utf-8"))
    return {"model": model, "risk_calibration": risk_calibration, **metadata}


BUNDLE_ASSET_FILENAMES = {
    "model": MODEL_ASSET_FILENAME,
    "metadata": METADATA_ASSET_FILENAME,
    "risk_calibration": RISK_CALIBRATION_ASSET_FILENAME,
}
