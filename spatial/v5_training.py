"""Training primitives for the rain-aware base and shallow V5 specialist."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from spatial.evaluate_baselines import add_physics
from spatial.evaluate_baselines import FEATURES as INCUMBENT_FEATURES
from spatial.v5_features import BASE_FEATURES, SPECIALIST_FEATURES, add_v5_features, regime_labels


def load_frame(path):
    frame = pd.read_csv(path, parse_dates=["forecast_init_time", "valid_time", "target_time"])
    required = {"hrrr_precip_accum_mm", "hrrr_precip_increment_mm", "precip_interval_hours", "precip_available"}
    if missing := required.difference(frame.columns):
        raise RuntimeError(f"V5 precipitation contract is incomplete: {sorted(missing)}")
    if not (frame.hrrr_precip_accum_mm.fillna(0) > 0).any():
        raise RuntimeError("V5 precipitation archive is constant zero")
    frame["run_id"] = frame.run_id.astype(str)
    frame["physics_fm"] = add_physics(frame)
    for column in ("target_rh", "target_wind_ms", "target_rh_mask", "target_wind_mask", "target_match_age_minutes"):
        if column not in frame: frame[column] = np.nan if "mask" not in column else 0
    return frame


def xgb_execution_device():
    requested = os.getenv("SMF_XGB_DEVICE", "auto").strip().lower()
    cuda_built = bool(xgb.build_info().get("USE_CUDA"))
    if requested == "auto": return "cuda" if cuda_built else "cpu"
    if requested not in {"cpu", "cuda"}: raise ValueError("SMF_XGB_DEVICE must be auto, cpu, or cuda")
    if requested == "cuda" and not cuda_built: raise RuntimeError("XGBoost has no CUDA support")
    return requested


def base_configs():
    profiles = (
        {"min_child_weight": 1, "subsample": .9, "colsample_bytree": .9, "reg_alpha": 0, "reg_lambda": 1},
        {"min_child_weight": 5, "subsample": .8, "colsample_bytree": .8, "reg_alpha": .1, "reg_lambda": 5},
    )
    return [dict(profile, max_depth=depth, learning_rate=rate, objective=objective)
            for profile, depth, rate, objective in itertools.product(
                profiles, (3, 5), (.03, .05), ("reg:squarederror", "reg:pseudohubererror"))]


def specialist_configs():
    return [{"max_depth": depth, "min_child_weight": child, "learning_rate": rate,
             "subsample": .85, "colsample_bytree": .9, "reg_alpha": .1, "reg_lambda": 5}
            for depth, child, rate in itertools.product((1, 2), (10, 30), (.03, .05))]


def prepare(frame, base_prediction=None, physics_variant="rain25_scale2", static=None):
    """Prepare features, optionally reusing a base-independent full-frame cache."""
    if static is None:
        result = add_v5_features(frame, np.zeros(len(frame)), physics_variant)
    else:
        result = static.loc[frame.index].copy()
    if base_prediction is not None:
        prediction = np.asarray(base_prediction, float)
        if len(prediction) != len(result): raise ValueError("V5 base prediction is not row-aligned")
        result["incumbent_base_fm"] = prediction
        result["physics_minus_incumbent"] = result.physics_fm - prediction
        result["rain_physics_minus_base"] = result.rain_physics_fm - prediction
    return result


def load_or_prepare_static(frame, cache_path, contract_sha256, physics_variant="rain25_scale2"):
    """Persist expensive base-independent features behind an evidence checksum."""
    cache_path = Path(cache_path); metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    expected = {"contract_sha256": contract_sha256, "physics_variant": physics_variant,
                "rows": int(len(frame)), "index_min": int(frame.index.min()), "index_max": int(frame.index.max())}
    if cache_path.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()) == expected:
        print(f"Loading cached V5 static features from {cache_path}", flush=True)
        cached = pd.read_pickle(cache_path)
        if cached.index.equals(frame.index): return cached
        raise RuntimeError("V5 static feature cache row contract mismatch")
    print("Building one-time V5 static feature cache...", flush=True)
    prepared = prepare(frame, physics_variant=physics_variant)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    prepared.to_pickle(temporary); temporary.replace(cache_path)
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(json.dumps(expected, indent=2)); temporary_metadata.replace(metadata_path)
    return prepared


def _regressor(config, validation=False, seed=417, n_estimators=None):
    parameters = {
        "n_estimators": n_estimators or 800, "random_state": seed, "tree_method": "hist",
        "device": xgb_execution_device(), "early_stopping_rounds": 30 if validation else None,
        **config,
    }
    return xgb.XGBRegressor(**parameters)


def fit_base(frame, run_ids, config, physics_variant="rain25_scale2", validation_runs=None, prepared=None):
    prepared = prepare(frame, physics_variant=physics_variant) if prepared is None else prepared
    training = prepared[prepared.run_id.isin(set(map(str, run_ids))) & (prepared.target_mask == 1)].dropna(subset=["target_fm"])
    model = _regressor(config, validation=bool(validation_runs))
    fit_args = {}
    if validation_runs:
        valid = prepared[prepared.run_id.isin(set(map(str, validation_runs))) & (prepared.target_mask == 1)].dropna(subset=["target_fm"])
        fit_args = {"eval_set": [(valid[BASE_FEATURES], valid.target_fm)], "verbose": False}
    model.fit(training[BASE_FEATURES], training.target_fm, **fit_args)
    return model


def predict_base(model, frame, physics_variant="rain25_scale2", prepared=None):
    values = prepare(frame, physics_variant=physics_variant) if prepared is None else prepared.loc[frame.index]
    return model.predict(values[BASE_FEATURES])


def fit_incumbent(frame, run_ids):
    training = frame[frame.run_id.isin(set(map(str, run_ids))) & (frame.target_mask == 1)].dropna(subset=["target_fm"])
    model = xgb.XGBRegressor(n_estimators=300, learning_rate=.05, max_depth=5,
                             objective="reg:squarederror", random_state=417,
                             tree_method="hist", device=xgb_execution_device())
    model.fit(training[INCUMBENT_FEATURES], training.target_fm)
    return model


def predict_incumbent(model, frame):
    return model.predict(frame[INCUMBENT_FEATURES])


def crossfit_incumbent(frame, run_ids, blocks=5):
    ordered = frame[frame.run_id.isin(set(map(str, run_ids)))][["run_id", "forecast_init_time"]].drop_duplicates().sort_values("forecast_init_time").run_id.tolist()
    result = pd.Series(index=frame.index, dtype=float); provenance = []
    for number, block in enumerate(np.array_split(np.asarray(ordered, object), blocks), 1):
        scored = set(map(str, block)); fitted = [run for run in ordered if run not in scored]
        model = fit_incumbent(frame, fitted); index = frame.index[frame.run_id.isin(scored)]
        result.loc[index] = predict_incumbent(model, frame.loc[index])
        provenance.append({"block": number, "fit_runs": fitted, "scored_runs": sorted(scored)})
    index = frame.index[frame.run_id.isin(set(ordered))]
    if result.loc[index].isna().any(): raise RuntimeError("Incomplete grouped OOF incumbent predictions")
    return result, provenance


def crossfit_base(frame, run_ids, config, physics_variant="rain25_scale2", blocks=5, prepared=None):
    prepared = prepare(frame, physics_variant=physics_variant) if prepared is None else prepared
    ordered = frame[frame.run_id.isin(set(map(str, run_ids)))][["run_id", "forecast_init_time"]].drop_duplicates().sort_values("forecast_init_time").run_id.tolist()
    result = pd.Series(index=frame.index, dtype=float); provenance = []
    for number, block in enumerate(np.array_split(np.asarray(ordered, object), blocks), 1):
        scored = set(map(str, block)); fitted = [run for run in ordered if run not in scored]
        model = fit_base(frame, fitted, config, physics_variant, prepared=prepared)
        index = frame.index[frame.run_id.isin(scored)]
        result.loc[index] = predict_base(model, frame.loc[index], physics_variant, prepared)
        provenance.append({"block": number, "fit_runs": fitted, "scored_runs": sorted(scored)})
    index = frame.index[frame.run_id.isin(set(ordered))]
    if result.loc[index].isna().any(): raise RuntimeError("Incomplete grouped OOF V5 base predictions")
    return result, provenance


def fit_specialist(prepared, run_ids, config, validation_runs=None):
    selected = prepared[prepared.run_id.isin(set(map(str, run_ids))) & (prepared.target_mask == 1)].dropna(subset=["target_fm", "incumbent_base_fm"])
    target = selected.target_fm - selected.incumbent_base_fm
    model = _regressor(config, validation=bool(validation_runs), n_estimators=500)
    fit_args = {}
    if validation_runs:
        valid = prepared[prepared.run_id.isin(set(map(str, validation_runs))) & (prepared.target_mask == 1)].dropna(subset=["target_fm", "incumbent_base_fm"])
        fit_args = {"eval_set": [(valid[SPECIALIST_FEATURES], valid.target_fm - valid.incumbent_base_fm)], "verbose": False}
    model.fit(selected[SPECIALIST_FEATURES], target, **fit_args)
    return model


def predict_specialist(model, prepared):
    return model.predict(prepared[SPECIALIST_FEATURES])


def prepare_fold(frame, train_runs, validation_runs, base_config, specialist_config,
                 physics_variant="rain25_scale2", base_prepared=None):
    base_prepared = prepare(frame, physics_variant=physics_variant) if base_prepared is None else base_prepared
    oof, provenance = crossfit_base(frame, train_runs, base_config, physics_variant, prepared=base_prepared)
    base_model = fit_base(frame, train_runs, base_config, physics_variant, validation_runs, base_prepared)
    train_index = frame.index[frame.run_id.isin(set(map(str, train_runs)))]
    valid_index = frame.index[frame.run_id.isin(set(map(str, validation_runs)))]
    train = prepare(frame.loc[train_index], oof.loc[train_index], physics_variant, prepared)
    valid_base = predict_base(base_model, frame.loc[valid_index], physics_variant, base_prepared)
    valid = prepare(frame.loc[valid_index], valid_base, physics_variant, prepared)
    specialist = fit_specialist(train, train_runs, specialist_config, validation_runs=None)
    valid["raw_correction"] = predict_specialist(specialist, valid)
    valid["regime"] = regime_labels(valid)
    return train, valid, base_model, specialist, provenance


def best_iterations(model, default):
    value = getattr(model, "best_iteration", None)
    return default if value is None else int(value) + 1
