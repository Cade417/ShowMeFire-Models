from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.model import TeacherStudentSpatialModel, teacher_student_loss
from spatial.static_inputs import FEATURE_SETS, load_static_inputs


def _stats(files, key, preserve_availability=False):
    total = total2 = None; count = 0
    for path in files:
        with np.load(path) as item:
            values = item[key].astype("float64"); current = values.sum(axis=(0, 2, 3)); current2 = np.square(values).sum(axis=(0, 2, 3))
            total = current if total is None else total + current; total2 = current2 if total2 is None else total2 + current2
            count += values.shape[0] * values.shape[2] * values.shape[3]
    mean = total / count; std = np.sqrt(np.maximum(total2 / count - mean ** 2, 1e-6))
    if preserve_availability:
        mean[-1], std[-1] = 0, 1
    return mean.reshape(1, -1, 1, 1).astype("float32"), std.reshape(1, -1, 1, 1).astype("float32")


class TensorFiles(Dataset):
    def __init__(self, files, normalizers, static_continuous, static_categorical, augment=False):
        self.files, self.normalizers = files, normalizers; self.static_continuous = static_continuous
        self.static_categorical, self.augment = static_categorical, augment
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        with np.load(self.files[index]) as item:
            antecedent = item["antecedent_rtma"].astype("float32").copy()
            if self.augment:
                # Keep the oldest frame available so every synthetic gap has a causal fill.
                count = random.choice((0, 1, 2)); choices = random.sample(range(1, antecedent.shape[0]), count)
                for frame in sorted(choices):
                    antecedent[frame, :3] = antecedent[frame - 1, :3]; antecedent[frame, 3] = 0
            antecedent = (antecedent - self.normalizers["antecedent"][0]) / self.normalizers["antecedent"][1]
            realized = (item["realized_rtma_future"] - self.normalizers["realized"][0]) / self.normalizers["realized"][1]
            hrrr = (item["hrrr_forecast"] - self.normalizers["hrrr"][0]) / self.normalizers["hrrr"][1]
            arrays = (antecedent, realized, hrrr, item["current_fm_state"], self.static_continuous, self.static_categorical,
                      item["physics_trajectory"], item["target"], item["train_mask"], item["station_holdout_mask"], item["region_holdout_mask"])
            return tuple(torch.from_numpy(np.asarray(value, dtype="int64" if position == 5 else "float32")) for position, value in enumerate(arrays))


def _gate():
    coverage = json.loads((paths.REPORTS_DIR / "coverage.json").read_text()); sequence = json.loads((paths.REPORTS_DIR / "sequence_metrics.json").read_text())
    baseline = json.loads((paths.REPORTS_DIR / "baseline_metrics.json").read_text()); incumbent = baseline["temporal"]["incumbent_control"]["mae"]
    return coverage["spatial_model_data_gate"]["pass"] and sequence["mae"] <= .95 * min(sequence["persistence_mae"], incumbent)


def _evaluate(model, loader, device, mask_index):
    model.eval(); actuals, medians, lowers, uppers = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            antecedent, _, hrrr, current, scont, scat, physics, target = batch[:8]; mask = batch[mask_index]
            prediction = model.student_forward(*(value.to(device) for value in (antecedent, hrrr, current, scont, scat, physics)))[0].cpu().numpy()
            observed = mask.numpy().astype(bool); expanded = np.broadcast_to(target.numpy(), prediction.shape)
            actuals.extend(expanded[:, :, 1:2][observed].tolist()); medians.extend(prediction[:, :, 1:2][observed].tolist())
            lowers.extend(prediction[:, :, 0:1][observed].tolist()); uppers.extend(prediction[:, :, 2:3][observed].tolist())
    actual, median, lower, upper = map(np.asarray, (actuals, medians, lowers, uppers))
    if not len(actual): return {"mae": None, "interval_coverage": None, "quantile_order_valid": True, "samples": 0}
    low = actual <= 6
    return {"mae": float(np.mean(np.abs(actual - median))), "interval_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
            "quantile_order_valid": bool(np.all(lower <= median) and np.all(median <= upper)), "samples": int(len(actual)),
            "critical_low_fm_mae": float(np.mean(np.abs(actual[low] - median[low]))) if low.any() else None,
            "critical_low_fm_samples": int(low.sum())}


def distillation_gate(distilled, control):
    temporal_ok = distilled["temporal"]["mae"] <= .98 * control["temporal"]["mae"] or (
        distilled["temporal"]["mae"] <= control["temporal"]["mae"] and
        abs(distilled["temporal"]["interval_coverage"] - .8) < abs(control["temporal"]["interval_coverage"] - .8))
    regressions = all(distilled[name]["mae"] <= 1.01 * control[name]["mae"] for name in ("station", "region"))
    critical = all(not distilled[name].get("critical_low_fm_mae") or not control[name].get("critical_low_fm_mae") or
                   distilled[name]["critical_low_fm_mae"] <= 1.01 * control[name]["critical_low_fm_mae"]
                   for name in ("temporal", "station", "region"))
    return {"pass": bool(temporal_ok and regressions and critical), "temporal_requirement": temporal_ok,
            "holdout_regressions_valid": regressions, "critical_low_fm_valid": critical}


def train(static_bundle: Path, feature_set="all", epochs=20, batch_size=2, force=False, distilled=True):
    if not force and not _gate(): raise RuntimeError("Spatial training gate did not pass")
    files = sorted((paths.ALIGNED_DIR / "spatial_tensors").glob("spatial_*.npz")); cutoff = max(1, int(len(files) * .8))
    train_files, test_files = files[:cutoff], files[cutoff:]
    if not train_files or not test_files: raise RuntimeError("Need chronological train and test tensor files")
    static_cont, static_cat, static_contract = load_static_inputs(static_bundle, feature_set)
    normalizers = {"antecedent": _stats(train_files, "antecedent_rtma", True),
                   "realized": _stats(train_files, "realized_rtma_future", True), "hrrr": _stats(train_files, "hrrr_forecast")}
    train_set = TensorFiles(train_files, normalizers, static_cont, static_cat, True); test_set = TensorFiles(test_files, normalizers, static_cont, static_cat)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TeacherStudentSpatialModel(static_cont.shape[0], static_contract["category_sizes"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4); scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            antecedent, realized, hrrr, current, scont, scat, physics, target, mask = (value.to(device) for value in batch[:9])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(antecedent, realized, hrrr, current, scont, scat, physics)
                loss = teacher_student_loss(*outputs, target, mask, physics, distilled=distilled)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        print(f"{feature_set} distilled={distilled} epoch={epoch + 1} loss={loss.item():.4f}")
    metrics = {"temporal": _evaluate(model, test_loader, device, 8), "station": _evaluate(model, test_loader, device, 9),
               "region": _evaluate(model, test_loader, device, 10), "feature_set": feature_set, "training_files": len(train_files),
               "distilled": distilled, "findings_are_associations_not_causation": True}
    name = f"fuel_moisture_spatial_{feature_set}_{'distilled' if distilled else 'control'}"
    artifact = paths.MODELS_DIR / f".{name}.pt"
    serialized_normalizers = {key: {"mean": value[0], "std": value[1]} for key, value in normalizers.items()}
    torch.save({"state_dict": model.state_dict(), "static_continuous_channels": static_cont.shape[0],
                "category_sizes": static_contract["category_sizes"], "embedding_dims": model.embedding_dims,
                "hidden_channels": model.hidden_channels, "normalizers": serialized_normalizers, "static_contract": static_contract,
                "metrics": metrics, "antecedent_length": 13, "allowed_missing_antecedent": 2, "hrrr_leads": list(range(4, 16)),
                "teacher_used_for_training": distilled, "teacher_exported": False}, artifact)
    report = paths.REPORTS_DIR / f"spatial_{feature_set}_{'distilled' if distilled else 'control'}.json"; report.write_text(json.dumps(metrics, indent=2))
    final = paths.MODELS_DIR / f"{name}.pt"; artifact.replace(final)
    print(json.dumps(metrics, indent=2)); return final, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--static-bundle", required=True, type=Path); parser.add_argument("--feature-set", choices=FEATURE_SETS, default="all")
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-distillation", action="store_true", help="Train the identical HRRR student control without teacher losses")
    args = parser.parse_args(); train(args.static_bundle, args.feature_set, args.epochs, args.batch_size, args.force, not args.no_distillation)
