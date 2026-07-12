from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from models.versioning import register_trained_model
from spatial.evaluate_baselines import add_physics
from spatial.station_model import StationSequenceModel, pinball_loss

FEATURES = ["initial_fm", "initial_age_hours", "rtma_temp_c", "rtma_rh", "rtma_wind_ms", "hrrr_temp_c", "hrrr_rh", "hrrr_wind_ms", "hrrr_precip_mm", "lead_hour", "lat", "lon"]


def _sequences(frame, run_keys):
    features, physics, targets, masks, keys = [], [], [], [], []
    expected = int(frame.groupby(["run_id", "station_id"]).size().mode().iloc[0])
    for key, group in frame[frame.run_id.isin(run_keys)].sort_values("lead_hour").groupby(["run_id", "station_id"]):
        if len(group) != expected or group[FEATURES].isna().any().any():
            continue
        features.append(group[FEATURES].values.astype("float32"))
        physics.append(group.physics_fm.values.astype("float32"))
        targets.append(group.target_fm.fillna(0).values.astype("float32"))
        masks.append(group.target_mask.values.astype("float32"))
        keys.append(key)
    return tuple(np.asarray(value) for value in (features, physics, targets, masks)), keys


def train(epochs=40, batch_size=32):
    torch.manual_seed(417)
    frame = pd.read_csv(paths.ALIGNED_DIR / "station_leads.csv", parse_dates=["forecast_init_time"])
    frame["physics_fm"] = add_physics(frame)
    runs = frame[["run_id", "forecast_init_time"]].drop_duplicates().sort_values("forecast_init_time")
    cutoff = max(1, int(len(runs) * .8))
    train_runs, test_runs = set(runs.run_id.iloc[:cutoff]), set(runs.run_id.iloc[cutoff:])
    (train_arrays, _), (test_arrays, test_keys) = _sequences(frame, train_runs), _sequences(frame, test_runs)
    x_train, p_train, y_train, m_train = train_arrays
    x_test, p_test, y_test, m_test = test_arrays
    if not len(x_train) or not len(x_test):
        raise RuntimeError("Not enough complete run/station sequences for chronological training and testing")
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    dataset = TensorDataset(*[torch.from_numpy(v) for v in (x_train, p_train, y_train, m_train)])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StationSequenceModel(len(FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        for x, physics, target, mask in loader:
            x, physics, target, mask = (value.to(device) for value in (x, physics, target, mask))
            optimizer.zero_grad(set_to_none=True)
            loss = pinball_loss(model(x, physics), target, mask)
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch + 1} loss={loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(x_test).to(device), torch.from_numpy(p_test).to(device)).cpu().numpy()
    observed = m_test.astype(bool)
    p50 = prediction[..., 1][observed]
    actual = y_test[observed]
    persistence = np.repeat(x_test[:, :1, FEATURES.index("initial_fm")], x_test.shape[1], axis=1)  # normalized; replaced below
    raw_test, _ = _sequences(frame, test_runs)
    raw_x = raw_test[0]
    persistence = np.repeat(raw_x[:, :1, FEATURES.index("initial_fm")], raw_x.shape[1], axis=1)[observed]
    physics_values = p_test[observed]
    metrics = {
        "mae": float(mean_absolute_error(actual, p50)),
        "persistence_mae": float(mean_absolute_error(actual, persistence)),
        "physics_mae": float(mean_absolute_error(actual, physics_values)),
        "interval_coverage": float(np.mean((actual >= prediction[..., 0][observed]) & (actual <= prediction[..., 2][observed]))),
        "samples": int(len(actual)), "test_sequences": len(test_keys),
    }
    metrics["improvement_over_persistence"] = 1 - metrics["mae"] / metrics["persistence_mae"]
    artifact = paths.MODELS_DIR / ".scratch_fuel_moisture_station_sequence.pt"
    torch.save({"state_dict": model.state_dict(), "features": FEATURES, "mean": mean, "std": std, "metrics": metrics, "sequence_steps": x_train.shape[1]}, artifact)
    version = register_trained_model("fuel_moisture_station_sequence", artifact, performance=metrics, channel="beta")
    artifact.unlink()
    metrics["registered_version"] = version
    (paths.REPORTS_DIR / "sequence_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
