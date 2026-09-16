"""Shared enhanced-XGBoost and guarded-GRU training for V4."""
from __future__ import annotations

import random
import time
import os

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset

from spatial.station_contract import basic_metrics
from spatial.v4_features import FEATURES, TREE_FEATURES, add_v4_features
from spatial.v4_model import GuardedQuantileGRU, v4_loss
from spatial.evaluate_baselines import add_physics


def load_frame(path):
    frame = pd.read_csv(path, parse_dates=["forecast_init_time", "valid_time", "target_time"])
    required_precip = {"hrrr_precip_accum_mm", "hrrr_precip_increment_mm", "precip_interval_hours", "precip_available"}
    if missing := required_precip.difference(frame.columns):
        raise RuntimeError(f"V4 precipitation contract is incomplete: {sorted(missing)}")
    if not (frame.hrrr_precip_accum_mm.fillna(0) > 0).any():
        raise RuntimeError("V4 precipitation archive is constant zero")
    frame["run_id"] = frame.run_id.astype(str)
    frame["physics_fm"] = add_physics(frame)
    for column in ("target_rh", "target_wind_ms", "target_rh_mask", "target_wind_mask",
                   "target_match_age_minutes"):
        if column not in frame: frame[column] = np.nan if "mask" not in column else 0
    return frame


def base_configs():
    import itertools
    return [{"max_depth": depth, "min_child_weight": child, "learning_rate": rate, "objective": objective}
            for depth, child, rate, objective in itertools.product((3, 5), (1, 5), (.03, .05),
                ("reg:squarederror", "reg:pseudohubererror"))]


def xgb_execution_device():
    requested = os.getenv("SMF_XGB_DEVICE", "auto").strip().lower()
    cuda_built = bool(xgb.build_info().get("USE_CUDA"))
    if requested == "auto":
        return "cuda" if cuda_built else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("SMF_XGB_DEVICE must be auto, cpu, or cuda")
    if requested == "cuda" and not cuda_built:
        raise RuntimeError("SMF_XGB_DEVICE=cuda but this XGBoost build has no CUDA support")
    return requested


def residual_configs():
    import itertools
    return [{"hidden_size": hidden, "gate_regularization": gate, "residual_cap": cap}
            for hidden, gate, cap in itertools.product((32, 64, 96), (.01, .05), (2.0, 4.0))]


def prebase(frame, physics_variant):
    return add_v4_features(frame, np.zeros(len(frame)), physics_variant)


def fit_base(frame, run_ids, config, physics_variant, validation_runs=None, prepared=None):
    prepared = prebase(frame, physics_variant) if prepared is None else prepared
    train = prepared.run_id.isin(set(map(str, run_ids))) & (prepared.target_mask == 1)
    training = prepared.loc[train].dropna(subset=TREE_FEATURES + ["target_fm"])
    parameters = {"n_estimators": 1000, "random_state": 417, "tree_method": "hist",
                  "device": xgb_execution_device(),
                  "early_stopping_rounds": 30 if validation_runs else None, **config}
    model = xgb.XGBRegressor(**parameters)
    fit_args = {}
    if validation_runs:
        valid = prepared[prepared.run_id.isin(set(map(str, validation_runs))) & (prepared.target_mask == 1)]
        valid = valid.dropna(subset=TREE_FEATURES + ["target_fm"])
        fit_args["eval_set"] = [(valid[TREE_FEATURES], valid.target_fm)]; fit_args["verbose"] = False
    model.fit(training[TREE_FEATURES], training.target_fm, **fit_args)
    return model


def predict_base(model, frame, physics_variant, prepared=None):
    prepared = prebase(frame, physics_variant) if prepared is None else prepared.loc[frame.index]
    return model.predict(prepared[TREE_FEATURES])


def crossfit_base(frame, run_ids, config, physics_variant, blocks=5, prepared=None):
    prepared = prebase(frame, physics_variant) if prepared is None else prepared
    selected = frame[frame.run_id.isin(set(map(str, run_ids)))][["run_id", "forecast_init_time"]]
    ordered = selected.drop_duplicates().sort_values("forecast_init_time").run_id.tolist()
    prediction = pd.Series(index=frame.index, dtype=float); provenance = []
    for number, block in enumerate(np.array_split(np.asarray(ordered, object), blocks), 1):
        scored = set(map(str, block)); fitted = [run for run in ordered if run not in scored]
        model = fit_base(frame, fitted, config, physics_variant, prepared=prepared)
        index = frame.index[frame.run_id.isin(scored)]
        prediction.loc[index] = predict_base(model, frame.loc[index], physics_variant, prepared=prepared)
        provenance.append({"block": number, "fit_runs": fitted, "scored_runs": sorted(scored)})
    index = frame.index[frame.run_id.isin(set(ordered))]
    if prediction.loc[index].isna().any(): raise RuntimeError("Incomplete V4 OOF base predictions")
    return prediction, provenance


def prepare_fold(frame, train_runs, validation_runs, base_config, physics_variant, prepared=None):
    prepared = prebase(frame, physics_variant) if prepared is None else prepared
    oof, provenance = crossfit_base(frame, train_runs, base_config, physics_variant, prepared=prepared)
    model = fit_base(frame, train_runs, base_config, physics_variant, validation_runs, prepared=prepared)
    train_index = frame.index[frame.run_id.isin(set(map(str, train_runs)))]
    validation_index = frame.index[frame.run_id.isin(set(map(str, validation_runs)))]
    train = add_v4_features(frame.loc[train_index], oof.loc[train_index], physics_variant)
    valid = add_v4_features(
        frame.loc[validation_index],
        predict_base(model, frame.loc[validation_index], physics_variant, prepared=prepared),
        physics_variant,
    )
    return train, valid, model, provenance


def sequence_arrays(frame):
    values, base, target, mask, leads, indices, keys = [], [], [], [], [], [], []
    if frame.empty: return tuple(np.empty((0,)) for _ in range(5)), np.empty((0, 0), int), []
    expected = int(frame.groupby(["run_id", "station_id"]).size().mode().iloc[0])
    for key, group in frame.sort_values("lead_hour").groupby(["run_id", "station_id"]):
        if len(group) != expected or group[FEATURES].isna().any().any(): continue
        values.append(group[FEATURES].to_numpy("float32")); base.append(group.incumbent_base_fm.to_numpy("float32"))
        target.append(group.target_fm.fillna(0).to_numpy("float32")); mask.append(group.target_mask.to_numpy("float32"))
        leads.append(group.lead_hour.to_numpy("float32")); indices.append(group.index.to_numpy(int)); keys.append(tuple(map(str, key)))
    return tuple(np.asarray(item) for item in (values, base, target, mask, leads)), np.asarray(indices), keys


def _seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def train_residual(train_arrays, validation_arrays=None, hidden_size=64, gate_regularization=.01,
                   residual_cap=4.0, epochs=25, patience=5, fixed_epochs=None, seed=417,
                   batch_size=32, learning_rate=.001, weight_decay=1e-4, device=None):
    started = time.perf_counter(); _seed(seed); x, base, target, mask, _ = train_arrays
    summer_weight = np.where(x[..., FEATURES.index("summer_indicator")] > 0, 1.25, 1.0).astype("float32")
    mean = x.reshape(-1, x.shape[-1]).mean(0); std = x.reshape(-1, x.shape[-1]).std(0); std[std < 1e-6] = 1
    x = ((x - mean) / std).astype("float32"); tensors = [torch.from_numpy(item) for item in (x, base, target, mask, summer_weight)]
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = GuardedQuantileGRU(len(FEATURES), hidden_size, residual_cap).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best, best_mae, best_epoch, stale, history = None, float("inf"), 0, 0, []
    for epoch in range(fixed_epochs or epochs):
        model.train(); losses = []
        for batch in loader:
            bx, bb, by, bm, bweight = (item.to(device) for item in batch); optimizer.zero_grad(set_to_none=True)
            prediction, gate = model(bx, bb, return_gate=True)
            loss = v4_loss(prediction, gate, by, bm, gate_regularization, bweight); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses))}
        if validation_arrays is not None:
            prediction = predict(model, validation_arrays, mean, std, device); observed = validation_arrays[3].astype(bool)
            score = basic_metrics(validation_arrays[2][observed], prediction[..., 3][observed]); record.update(score)
            print(f"epoch={epoch+1} loss={record['loss']:.4f} val_mae={score['mae']:.4f} bias={score['bias']:.4f}", flush=True)
            if score["mae"] < best_mae:
                best_mae, best_epoch, stale = score["mae"], epoch + 1, 0
                best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if fixed_epochs is None and stale >= patience: history.append(record); break
        else:
            best_epoch = epoch + 1; best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(record)
    model.load_state_dict(best)
    checkpoint = {"state_dict": best, "mean": mean, "std": std, "features": FEATURES,
                  "model_config": {"hidden_size": hidden_size, "residual_cap": residual_cap,
                                   "gate_regularization": gate_regularization}, "best_epoch": best_epoch,
                  "quantiles": [5, 10, 25, 50, 75, 90, 95]}
    return model, checkpoint, {"best_epoch": best_epoch, "history": history,
                               "runtime_seconds": time.perf_counter() - started}


def predict(model, arrays, mean, std, device="cpu", lead_weights=None):
    x, base, _, _, _ = arrays; normalized = ((x - mean) / std).astype("float32")
    model.eval()
    with torch.no_grad():
        weights = None if lead_weights is None else torch.as_tensor(lead_weights, dtype=torch.float32, device=device)
        return model(torch.from_numpy(normalized).to(device), torch.from_numpy(base).to(device), weights).cpu().numpy()


def load_residual(checkpoint, device="cpu"):
    config = checkpoint["model_config"]; model = GuardedQuantileGRU(len(FEATURES), config["hidden_size"], config["residual_cap"])
    model.load_state_dict(checkpoint["state_dict"]); return model.to(device).eval()
