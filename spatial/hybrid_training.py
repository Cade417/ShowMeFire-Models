"""Shared cross-fitting and GPU training implementation for hybrid V3."""
from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset

from spatial.evaluate_baselines import FEATURES as BASE_FEATURES, add_physics
from spatial.hybrid_contract import FEATURE_SCHEMA_VERSION
from spatial.hybrid_features import BASE_FEATURE, FEATURES, add_hybrid_features
from spatial.hybrid_model import HybridStationModel, hybrid_loss
from spatial.station_contract import basic_metrics

BASE_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5,
               "objective": "reg:squarederror", "random_state": 417}


def load_frame(dataset_path):
    frame = pd.read_csv(dataset_path, parse_dates=["forecast_init_time", "valid_time"])
    frame["run_id"] = frame.run_id.astype(str)
    frame["physics_fm"] = add_physics(frame)
    return frame


def fit_base_model(frame, run_ids):
    runs = set(map(str, run_ids))
    training = frame[frame.run_id.isin(runs) & (frame.target_mask == 1)]
    training = training.dropna(subset=BASE_FEATURES + ["target_fm"])
    if len(training) < 100:
        raise RuntimeError("Insufficient rows for XGBoost base model")
    model = xgb.XGBRegressor(**BASE_PARAMS)
    model.fit(training[BASE_FEATURES], training.target_fm)
    return model


def cross_fitted_base_predictions(frame, run_ids, blocks=5):
    """OOF predictions where the scored initialization group is excluded."""
    ordered = frame[frame.run_id.isin(set(map(str, run_ids)))][["run_id", "forecast_init_time"]]
    ordered = ordered.drop_duplicates().sort_values("forecast_init_time").run_id.tolist()
    if len(ordered) < blocks * 2:
        raise RuntimeError("Insufficient runs for grouped cross-fitting")
    prediction = pd.Series(index=frame.index, dtype=float)
    provenance = []
    for index, block in enumerate(np.array_split(np.asarray(ordered, dtype=object), blocks), 1):
        scored_runs = set(map(str, block.tolist()))
        fit_runs = [run for run in ordered if run not in scored_runs]
        if scored_runs & set(fit_runs):
            raise RuntimeError("Cross-fit initialization leakage")
        model = fit_base_model(frame, fit_runs)
        scored_index = frame.index[frame.run_id.isin(scored_runs)]
        values = frame.loc[scored_index, BASE_FEATURES]
        # XGBoost has deterministic native handling for NaN. Infinities are not
        # part of that contract and indicate corrupt source data.
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise RuntimeError("Infinite base feature during cross-fitting")
        prediction.loc[scored_index] = model.predict(values)
        provenance.append({"block": index, "fit_runs": fit_runs, "scored_runs": sorted(scored_runs)})
    selected = frame.index[frame.run_id.isin(set(ordered))]
    if prediction.loc[selected].isna().any():
        raise RuntimeError("Cross-fit predictions are incomplete")
    return prediction, provenance


def prepare_fold(frame, train_runs, validation_runs):
    train_runs, validation_runs = list(map(str, train_runs)), list(map(str, validation_runs))
    if set(train_runs) & set(validation_runs):
        raise RuntimeError("Training and validation runs overlap")
    oof, provenance = cross_fitted_base_predictions(frame, train_runs)
    base_model = fit_base_model(frame, train_runs)
    train_index = frame.index[frame.run_id.isin(set(train_runs))]
    validation_index = frame.index[frame.run_id.isin(set(validation_runs))]
    train_frame = add_hybrid_features(frame.loc[train_index], oof.loc[train_index])
    validation_prediction = base_model.predict(frame.loc[validation_index, BASE_FEATURES])
    validation_frame = add_hybrid_features(frame.loc[validation_index], validation_prediction)
    return train_frame, validation_frame, provenance


def prepare_final_fit(frame, development_runs):
    development_runs = list(map(str, development_runs))
    oof, provenance = cross_fitted_base_predictions(frame, development_runs)
    index = frame.index[frame.run_id.isin(set(development_runs))]
    prepared = add_hybrid_features(frame.loc[index], oof.loc[index])
    return prepared, fit_base_model(frame, development_runs), provenance


def prepare_with_base(frame, run_ids, base_model):
    index = frame.index[frame.run_id.isin(set(map(str, run_ids)))]
    values = frame.loc[index, BASE_FEATURES]
    return add_hybrid_features(frame.loc[index], base_model.predict(values))


def sequence_arrays(frame):
    values, base, targets, masks, row_indices, keys = [], [], [], [], [], []
    if frame.empty:
        return tuple(np.empty((0,)) for _ in range(4)), np.empty((0, 0), dtype=int), []
    expected = int(frame.groupby(["run_id", "station_id"]).size().mode().iloc[0])
    for key, group in frame.sort_values("lead_hour").groupby(["run_id", "station_id"]):
        if len(group) != expected or group[FEATURES].isna().any().any():
            continue
        values.append(group[FEATURES].to_numpy(dtype="float32"))
        base.append(group[BASE_FEATURE].to_numpy(dtype="float32"))
        targets.append(group.target_fm.fillna(0).to_numpy(dtype="float32"))
        masks.append(group.target_mask.to_numpy(dtype="float32"))
        row_indices.append(group.index.to_numpy(dtype=int))
        keys.append(tuple(map(str, key)))
    return tuple(np.asarray(item) for item in (values, base, targets, masks)), np.asarray(row_indices), keys


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict(model, x, base, device):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x).to(device), torch.from_numpy(base).to(device)).cpu().numpy()


def train_model(train_arrays, validation_arrays=None, *, hidden_size=64, num_layers=2, dropout=0.1,
                learning_rate=1e-3, p50_loss="mae", batch_size=32, weight_decay=1e-4,
                epochs=25, patience=5, seed=417, fixed_epochs=None, device=None):
    started = time.perf_counter(); _seed(seed)
    x_train, base_train, y_train, mask_train = train_arrays
    if not len(x_train):
        raise RuntimeError("No hybrid training sequences")
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0); std[std < 1e-6] = 1
    x_train = (x_train - mean) / std
    has_validation = validation_arrays is not None and len(validation_arrays[0])
    if has_validation:
        x_val, base_val, y_val, mask_val = validation_arrays
        x_val = (x_val - mean) / std
    dataset = TensorDataset(*[torch.from_numpy(item) for item in (x_train, base_train, y_train, mask_train)])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(seed))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HybridStationModel(len(FEATURES), hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state, best_epoch, best_mae, stale, history = None, 0, float("inf"), 0, []
    for epoch in range(fixed_epochs or epochs):
        model.train(); losses = []
        for x, base, target, mask in loader:
            x, base, target, mask = (item.to(device) for item in (x, base, target, mask))
            optimizer.zero_grad(set_to_none=True)
            loss = hybrid_loss(model(x, base), target, mask, p50_loss=p50_loss)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        if has_validation:
            prediction = _predict(model, x_val, base_val, device)
            observed = mask_val.astype(bool)
            score = basic_metrics(y_val[observed], prediction[..., 1][observed])
            record.update({"validation_mae": score["mae"], "validation_rmse": score["rmse"],
                           "validation_bias": score["bias"]})
            print(f"epoch={epoch + 1} loss={record['train_loss']:.4f} val_mae={score['mae']:.4f} bias={score['bias']:.4f}", flush=True)
            if score["mae"] < best_mae:
                best_mae, best_epoch, stale = score["mae"], epoch + 1, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if fixed_epochs is None and stale >= patience:
                    history.append(record); print(f"early stopping epoch={epoch + 1} best={best_mae:.4f}", flush=True); break
        else:
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"epoch={epoch + 1} loss={record['train_loss']:.4f}", flush=True)
        history.append(record)
    model.load_state_dict(best_state)
    validation_metrics = None
    if has_validation:
        prediction = _predict(model, x_val, base_val, device); observed = mask_val.astype(bool)
        validation_metrics = basic_metrics(y_val[observed], prediction[..., 1][observed])
    config = {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout,
              "learning_rate": learning_rate, "p50_loss": p50_loss, "batch_size": batch_size,
              "weight_decay": weight_decay, "epochs": epochs, "patience": patience, "seed": seed,
              "fixed_epochs": fixed_epochs}
    checkpoint = {"state_dict": best_state, "features": FEATURES,
                  "feature_schema_version": FEATURE_SCHEMA_VERSION, "mean": mean, "std": std,
                  "model_config": config, "best_epoch": best_epoch, "sequence_steps": x_train.shape[1]}
    report = {"config": config, "best_epoch": best_epoch, "validation_metrics": validation_metrics,
              "history": history, "runtime_seconds": time.perf_counter() - started,
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "device": str(device)}
    return model, checkpoint, report


def load_model(checkpoint, device="cpu"):
    config = checkpoint["model_config"]
    model = HybridStationModel(len(FEATURES), config["hidden_size"], config["num_layers"], config["dropout"])
    model.load_state_dict(checkpoint["state_dict"]); model.to(device); model.eval()
    return model


def predict_checkpoint(checkpoint, arrays, device="cpu"):
    x, base, _, _ = arrays
    normalized = (x - np.asarray(checkpoint["mean"])) / np.asarray(checkpoint["std"])
    return _predict(load_model(checkpoint, device), normalized.astype("float32"), base, torch.device(device))
