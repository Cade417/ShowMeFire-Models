"""One-shot locked-holdout evaluation for a selected station checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.evaluate_baselines import FEATURES as INCUMBENT_FEATURES
from spatial.station_contract import MIN_CRITICAL_SAMPLES, ROW_KEY, detailed_metrics, load_or_create_manifest, sequence_arrays, sha256_file
from spatial.station_features import FEATURE_SCHEMA_VERSION, FEATURES
from spatial.station_model import StationSequenceModel
from spatial.train_station_sequence import load_frame


def _scored_frame(frame, indices, predictions):
    columns = list(dict.fromkeys([*ROW_KEY, "target_fm", "initial_fm", "lead_hour", "station_id", "valid_time"]))
    scored = frame.loc[indices, columns].copy()
    scored["prediction"] = predictions
    scored["month"] = pd.to_datetime(scored.valid_time, utc=True).dt.month
    return scored


def evaluate(checkpoint_path: Path, dataset_path: Path, split_path: Path) -> dict:
    frame = load_frame(dataset_path)
    manifest = load_or_create_manifest(split_path, frame, dataset_path, create=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "dataset_sha256": manifest["dataset_sha256"], "split_version": manifest["version"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "features": FEATURES,
    }
    mismatches = [key for key, value in required.items() if checkpoint.get(key) != value]
    if mismatches:
        raise RuntimeError(f"Checkpoint contract mismatch: {', '.join(mismatches)}")
    arrays, keys, row_indices = sequence_arrays(frame, manifest["holdout_runs"])
    x, physics, target, mask = arrays
    if not len(x):
        raise RuntimeError("Locked holdout has no complete sequences")
    normalized = (x - np.asarray(checkpoint["mean"])) / np.asarray(checkpoint["std"])
    config = checkpoint["model_config"]
    model = StationSequenceModel(len(FEATURES), config["hidden_size"], config["num_layers"], config["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        quantiles = model(torch.from_numpy(normalized).float(), torch.from_numpy(physics).float()).numpy()
    observed = mask.astype(bool)
    indices = row_indices[observed]
    actual = target[observed]
    candidate_values = quantiles[..., 1][observed]

    incumbent_train = frame[frame.run_id.astype(str).isin(set(manifest["development_runs"]))]
    incumbent_train = incumbent_train[incumbent_train.target_mask == 1].dropna(subset=INCUMBENT_FEATURES + ["target_fm"])
    incumbent = xgb.XGBRegressor(n_estimators=300, learning_rate=.05, max_depth=5,
                                 objective="reg:squarederror", random_state=417)
    incumbent.fit(incumbent_train[INCUMBENT_FEATURES], incumbent_train.target_fm)
    holdout_rows = frame.loc[indices]
    if holdout_rows[INCUMBENT_FEATURES].isna().any().any():
        raise RuntimeError("Incumbent features are missing on candidate-scored rows")
    predictions = {
        "candidate": candidate_values,
        "incumbent_control": incumbent.predict(holdout_rows[INCUMBENT_FEATURES]),
        "physics": physics[observed],
        "persistence": holdout_rows.initial_fm.to_numpy(),
    }
    row_keys = holdout_rows[ROW_KEY].astype(str).agg("|".join, axis=1).tolist()
    row_key_sha256 = __import__("hashlib").sha256("\n".join(row_keys).encode()).hexdigest()
    metrics = {}
    for name, values in predictions.items():
        scored = _scored_frame(frame, indices, values)
        metrics[name] = detailed_metrics(scored, quantiles[observed] if name == "candidate" else None)
    candidate, control = metrics["candidate"], metrics["incumbent_control"]
    candidate_critical = candidate["regimes"]["critical_low_fm"]
    control_critical = control["regimes"]["critical_low_fm"]
    checks = {
        "same_row_samples": candidate["samples"] == control["samples"] == len(indices),
        "mae_5pct_better": candidate["mae"] <= 0.95 * control["mae"],
        "absolute_bias_no_worse": abs(candidate["bias"]) <= abs(control["bias"]),
        "critical_support": candidate["critical_threshold"]["support"] >= MIN_CRITICAL_SAMPLES,
        "critical_mae_no_worse": bool(candidate_critical and control_critical and candidate_critical["mae"] <= control_critical["mae"]),
        "interval_coverage": 0.78 <= candidate["interval_coverage"] <= 0.82,
        "quantile_ordering": candidate["quantile_order_violation_rate"] == 0.0,
    }
    return {
        "status": "final_evaluation", "pass": all(checks.values()), "checks": checks,
        "dataset_sha256": manifest["dataset_sha256"], "split_version": manifest["version"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path), "row_key": ROW_KEY,
        "row_key_sha256": row_key_sha256, "samples": len(indices), "sequence_keys": len(keys),
        "metrics": metrics, "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "prospective_shadow_required": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate one selected checkpoint on the locked holdout")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=paths.ALIGNED_DIR / "station_leads.csv")
    parser.add_argument("--split-manifest", type=Path, default=paths.REPORTS_DIR / "station_split_manifest.json")
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "final_station_evaluation.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite locked evaluation report: {args.output}")
    report = evaluate(args.checkpoint, args.dataset, args.split_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    compact = {"pass": report["pass"], "checks": report["checks"],
               "candidate": report["metrics"]["candidate"],
               "incumbent_control": report["metrics"]["incumbent_control"]}
    print(json.dumps(compact, indent=2))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
