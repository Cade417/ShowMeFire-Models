from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.model import SpatialQuantileModel, masked_spatial_pinball
from spatial.static_inputs import FEATURE_SETS, load_static_inputs


class TensorFiles(Dataset):
    def __init__(self, files, mean, std, static_continuous, static_categorical):
        self.files, self.mean, self.std = files, mean, std
        self.static_continuous, self.static_categorical = static_continuous, static_categorical
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        with np.load(self.files[index]) as item:
            dynamic = (item["dynamic_sequence"] - self.mean) / self.std
            values = (dynamic.astype("float32"), self.static_continuous, self.static_categorical,
                      item["physics"].astype("float32"), item["target"].astype("float32"), item["train_mask"].astype("float32"),
                      item["station_holdout_mask"].astype("float32"), item["region_holdout_mask"].astype("float32"))
            return tuple(torch.from_numpy(value) for value in values)


def _dynamic_stats(files):
    total = total2 = None; count = 0
    for path in files:
        with np.load(path) as item:
            values = item["dynamic_sequence"].astype("float64"); current = values.sum(axis=(0, 2, 3)); current2 = np.square(values).sum(axis=(0, 2, 3))
            total = current if total is None else total + current; total2 = current2 if total2 is None else total2 + current2
            count += values.shape[0] * values.shape[2] * values.shape[3]
    mean = total / count; std = np.sqrt(np.maximum(total2 / count - mean ** 2, 1e-6))
    return mean.reshape(1, -1, 1, 1).astype("float32"), std.reshape(1, -1, 1, 1).astype("float32")


def _gate():
    coverage = json.loads((paths.REPORTS_DIR / "coverage.json").read_text()); sequence = json.loads((paths.REPORTS_DIR / "sequence_metrics.json").read_text())
    baseline = json.loads((paths.REPORTS_DIR / "baseline_metrics.json").read_text()); incumbent = baseline["temporal"]["incumbent_control"]["mae"]
    return coverage["spatial_model_data_gate"]["pass"] and sequence["mae"] <= .95 * min(sequence["persistence_mae"], incumbent)


def _evaluate(model, loader, device, mask_index):
    model.eval(); actuals, medians, lowers, uppers = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            dynamic, static_cont, static_cat, physics, target = batch[:5]; mask = batch[mask_index]
            prediction = model(dynamic.to(device), static_cont.to(device), static_cat.to(device), physics.to(device)).cpu().numpy()
            observed = mask.numpy().astype(bool); target_values = np.broadcast_to(target.numpy(), prediction.shape)
            actuals.extend(target_values[:, :, 1:2][observed].tolist()); medians.extend(prediction[:, :, 1:2][observed].tolist())
            lowers.extend(prediction[:, :, 0:1][observed].tolist()); uppers.extend(prediction[:, :, 2:3][observed].tolist())
    actual, median, lower, upper = map(np.asarray, (actuals, medians, lowers, uppers))
    if not len(actual): return {"mae": None, "interval_coverage": None, "quantile_order_valid": True, "samples": 0}
    low = actual <= 6
    return {"mae": float(np.mean(np.abs(actual - median))), "interval_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
            "quantile_order_valid": bool(np.all(lower <= median) and np.all(median <= upper)), "samples": int(len(actual)),
            "critical_low_fm_mae": float(np.mean(np.abs(actual[low] - median[low]))) if low.any() else None,
            "critical_low_fm_samples": int(low.sum())}


def train(static_bundle: Path, feature_set="all", epochs=20, batch_size=2, force=False):
    if not force and not _gate(): raise RuntimeError("Spatial training gate did not pass")
    files = sorted((paths.ALIGNED_DIR / "spatial_tensors").glob("spatial_*.npz")); cutoff = max(1, int(len(files) * .8))
    train_files, test_files = files[:cutoff], files[cutoff:]
    if not train_files or not test_files: raise RuntimeError("Need chronological train and test tensor files")
    static_cont, static_cat, static_contract = load_static_inputs(static_bundle, feature_set)
    mean, std = _dynamic_stats(train_files); train_set = TensorFiles(train_files, mean, std, static_cont, static_cat); test_set = TensorFiles(test_files, mean, std, static_cont, static_cat)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()); test_loader = DataLoader(test_set, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(train_files[0]) as first: dynamic_channels = first["dynamic_sequence"].shape[1]
    model = SpatialQuantileModel(dynamic_channels, static_cont.shape[0], static_contract["category_sizes"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4); scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for epoch in range(epochs):
        model.train()
        for dynamic, scont, scat, physics, target, train_mask, _, _ in loader:
            dynamic, scont, scat, physics, target, train_mask = (value.to(device) for value in (dynamic, scont, scat, physics, target, train_mask)); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = masked_spatial_pinball(model(dynamic, scont, scat, physics), target, train_mask, physics)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        print(f"{feature_set} epoch={epoch + 1} loss={loss.item():.4f}")
    metrics = {"temporal": _evaluate(model, test_loader, device, 5), "station": _evaluate(model, test_loader, device, 6),
               "region": _evaluate(model, test_loader, device, 7), "feature_set": feature_set, "training_files": len(train_files)}
    artifact = paths.MODELS_DIR / f".scratch_fuel_moisture_spatial_{feature_set}.pt"
    torch.save({"state_dict": model.state_dict(), "dynamic_channels": dynamic_channels, "static_continuous_channels": static_cont.shape[0],
                "category_sizes": static_contract["category_sizes"], "embedding_dims": model.embedding_dims, "dynamic_mean": mean, "dynamic_std": std,
                "static_contract": static_contract, "metrics": metrics, "sequence_steps": np.load(train_files[0])["dynamic_sequence"].shape[0]}, artifact)
    report = paths.REPORTS_DIR / f"spatial_{feature_set}.json"; report.write_text(json.dumps(metrics, indent=2))
    final = paths.MODELS_DIR / f"fuel_moisture_spatial_{feature_set}.pt"; artifact.replace(final); artifact = final
    print(json.dumps(metrics, indent=2)); return artifact, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--static-bundle", required=True, type=Path); parser.add_argument("--feature-set", choices=FEATURE_SETS, default="all")
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); train(args.static_bundle, args.feature_set, args.epochs, args.batch_size, args.force)
