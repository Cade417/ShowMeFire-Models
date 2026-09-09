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


def load(directory: Path) -> Dict:
    directory = Path(directory)
    metadata = json.loads((directory / METADATA_ASSET_FILENAME).read_text(encoding="utf-8"))
    model = xgb.XGBRegressor()
    model.load_model(directory / MODEL_ASSET_FILENAME)
    return {"model": model, **metadata}


BUNDLE_ASSET_FILENAMES = {
    "model": MODEL_ASSET_FILENAME,
    "metadata": METADATA_ASSET_FILENAME,
}
