from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.evaluate_baselines import add_physics
from spatial.station_contract import detailed_metrics, load_or_create_manifest, sequence_arrays, sha256_file
from spatial.station_features import FEATURE_SCHEMA_VERSION, FEATURES, add_causal_features
from spatial.station_model import StationSequenceModel, pinball_loss


def load_frame(dataset_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(dataset_path, parse_dates=["forecast_init_time", "valid_time"])
    frame["run_id"] = frame.run_id.astype(str)
    frame = add_causal_features(frame)
    frame["physics_fm"] = add_physics(frame)
    return frame


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict(model, x, physics, device):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x).to(device), torch.from_numpy(physics).to(device)).cpu().numpy()


def _validation_report(frame, prediction, targets, masks, row_indices):
    observed = masks.astype(bool)
    indices = row_indices[observed]
    scored = frame.loc[indices, ["target_fm", "lead_hour", "initial_fm", "station_id", "valid_time"]].copy()
    scored["prediction"] = prediction[..., 1][observed]
    scored["month"] = pd.to_datetime(scored.valid_time, utc=True).dt.month
    return detailed_metrics(scored, prediction[observed])


def train_experiment(frame, manifest, train_runs, validation_runs, *, epochs=40, batch_size=32, patience=5,
                     learning_rate=1e-3, hidden_size=16, num_layers=1, dropout=0.0,
                     weight_decay=1e-4, seed=417, fixed_epochs=None, device=None):
    started = time.perf_counter()
    _seed_everything(seed)
    train_arrays, _, _ = sequence_arrays(frame, train_runs)
    validation_arrays, _, validation_indices = sequence_arrays(frame, validation_runs) if validation_runs else ((None,) * 4, [], None)
    x_train, p_train, y_train, m_train = train_arrays
    if not len(x_train):
        raise RuntimeError("No complete training sequences")
    has_validation = validation_runs is not None and len(validation_arrays[0])
    if validation_runs is not None and not has_validation:
        raise RuntimeError("No complete validation sequences")
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1
    x_train = (x_train - mean) / std
    if has_validation:
        x_val, p_val, y_val, m_val = validation_arrays
        x_val = (x_val - mean) / std
    dataset = TensorDataset(*[torch.from_numpy(value) for value in (x_train, p_train, y_train, m_train)])
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        pin_memory=torch.cuda.is_available(), generator=loader_generator)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = StationSequenceModel(len(FEATURES), hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_epochs = fixed_epochs or epochs
    best_state, best_epoch, best_mae, without_improvement = None, 0, float("inf"), 0
    history = []
    for epoch in range(total_epochs):
        model.train()
        losses = []
        for x, physics, target, mask in loader:
            x, physics, target, mask = (value.to(device) for value in (x, physics, target, mask))
            optimizer.zero_grad(set_to_none=True)
            loss = pinball_loss(model(x, physics), target, mask)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        if has_validation:
            prediction = _predict(model, x_val, p_val, device)
            observed = m_val.astype(bool)
            val_mae = float(np.mean(np.abs(y_val[observed] - prediction[..., 1][observed])))
            record["validation_mae"] = val_mae
            print(f"epoch={epoch + 1} loss={record['train_loss']:.4f} val_mae={val_mae:.4f}", flush=True)
            if val_mae < best_mae:
                best_mae, best_epoch = val_mae, epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                without_improvement = 0
            else:
                without_improvement += 1
                if fixed_epochs is None and without_improvement >= patience:
                    print(f"early stopping at epoch={epoch + 1}, best val_mae={best_mae:.4f}", flush=True)
                    history.append(record)
                    break
        else:
            print(f"epoch={epoch + 1} loss={record['train_loss']:.4f}", flush=True)
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(record)
    model.load_state_dict(best_state)
    validation_metrics = None
    if has_validation:
        prediction = _predict(model, x_val, p_val, device)
        validation_metrics = _validation_report(frame, prediction, y_val, m_val, validation_indices)
    config = {
        "epochs": epochs, "fixed_epochs": fixed_epochs, "batch_size": batch_size, "patience": patience,
        "learning_rate": learning_rate, "hidden_size": hidden_size, "num_layers": num_layers,
        "dropout": dropout, "weight_decay": weight_decay, "seed": seed,
    }
    checkpoint = {
        "state_dict": best_state, "features": FEATURES, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "mean": mean, "std": std, "sequence_steps": x_train.shape[1], "model_config": config,
        "split_version": manifest["version"], "dataset_sha256": manifest["dataset_sha256"],
        "best_epoch": best_epoch, "validation_metrics": validation_metrics,
    }
    report = {
        "status": "experiment", "model_config": config, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": FEATURES, "split_version": manifest["version"], "dataset_sha256": manifest["dataset_sha256"],
        "best_epoch": best_epoch, "validation_metrics": validation_metrics, "history": history,
        "runtime_seconds": time.perf_counter() - started, "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "device": str(device),
    }
    return model, checkpoint, report


def main():
    parser = argparse.ArgumentParser(description="Train a non-registering station-sequence experiment")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--fixed-epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, choices=[16, 32, 64], default=16)
    parser.add_argument("--num-layers", type=int, choices=[1, 2], default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=417)
    parser.add_argument("--fold", choices=["fold-1", "fold-2", "fold-3"], default="fold-3")
    parser.add_argument("--dataset", type=Path, default=paths.ALIGNED_DIR / "station_leads.csv")
    parser.add_argument("--split-manifest", type=Path, default=paths.REPORTS_DIR / "station_split_manifest.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--register", action="store_true", help="Rejected: registration is a separate gated command")
    args = parser.parse_args()
    if args.register:
        parser.error("Training cannot register models; use register_station_candidate.py after the final gate")
    if args.num_layers == 1 and args.dropout:
        parser.error("--dropout must be 0 for a one-layer GRU")
    frame = load_frame(args.dataset)
    manifest = load_or_create_manifest(args.split_manifest, frame, args.dataset)
    fold = next(item for item in manifest["folds"] if item["name"] == args.fold)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_path = args.checkpoint or paths.MODELS_DIR / "experiments" / f"station_{stamp}.pt"
    metrics_path = args.metrics or paths.REPORTS_DIR / "experiments" / f"station_{stamp}.json"
    _, checkpoint, report = train_experiment(
        frame, manifest, fold["train_runs"], fold["validation_runs"], epochs=args.epochs,
        fixed_epochs=args.fixed_epochs, batch_size=args.batch_size, patience=args.patience,
        learning_rate=args.learning_rate, hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, weight_decay=args.weight_decay, seed=args.seed,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    report.update({"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
                   "created_at": datetime.now(timezone.utc).isoformat()})
    metrics_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"checkpoint": str(checkpoint_path), "metrics": str(metrics_path),
                      "best_epoch": report["best_epoch"], "validation_mae": report["validation_metrics"]["mae"]}, indent=2))


if __name__ == "__main__":
    main()
