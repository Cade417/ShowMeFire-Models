"""Single locked historical-relock evaluation for hybrid station V3."""
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
from spatial.hybrid_contract import load_or_create_manifest
from spatial.hybrid_calibration import apply_calibration
from spatial.hybrid_features import FEATURES
from spatial.hybrid_training import load_frame, predict_checkpoint, prepare_with_base, sequence_arrays
from spatial.station_contract import MIN_CRITICAL_SAMPLES, ROW_KEY, detailed_metrics, sha256_file


def scored(frame, indices, prediction):
    columns = list(dict.fromkeys([*ROW_KEY, "target_fm", "initial_fm", "lead_hour", "valid_time"]))
    result = frame.loc[indices, columns].copy(); result["prediction"] = prediction
    result["month"] = pd.to_datetime(result.valid_time, utc=True).dt.month
    return result


def add_tail_metrics(report, actual, prediction):
    error = np.abs(np.asarray(actual) - np.asarray(prediction))
    report["tail_error"] = {"p90_absolute_error": float(np.quantile(error, 0.9)),
                            "p95_absolute_error": float(np.quantile(error, 0.95)),
                            "over_5_fraction": float(np.mean(error > 5))}


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid V3 exactly once on the historical relock")
    parser.add_argument("--dataset", type=Path, default=paths.ALIGNED_DIR / "station_leads.csv")
    parser.add_argument("--manifest", type=Path, default=paths.REPORTS_DIR / "hybrid_v3_split_manifest.json")
    parser.add_argument("--candidate-dir", type=Path, default=paths.MODELS_DIR / "hybrid_v3_candidate")
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "hybrid_v3_final_evaluation.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite locked V3 evaluation: {args.output}")
    frame = load_frame(args.dataset); manifest = load_or_create_manifest(args.manifest, frame, args.dataset, FEATURES, create=False)
    contract = json.loads((args.candidate_dir / "contract.json").read_text())
    calibration_path = args.candidate_dir / "calibration.json"; calibration = json.loads(calibration_path.read_text())
    base_path, residual_path = args.candidate_dir / "base_xgboost.json", args.candidate_dir / "residual_gru.pt"
    expected = {"base_model_sha256": sha256_file(base_path), "residual_model_sha256": sha256_file(residual_path),
                "manifest_sha256": manifest["manifest_sha256"], "dataset_sha256": manifest["dataset_sha256"]}
    mismatches = [key for key, value in expected.items() if contract.get(key) != value]
    mismatches += [f"calibration_{key}" for key, value in expected.items() if calibration.get(key) != value]
    if mismatches:
        raise SystemExit(f"V3 contract mismatch: {', '.join(mismatches)}")
    base = xgb.XGBRegressor(); base.load_model(base_path)
    prepared = prepare_with_base(frame, manifest["locked_test_runs"], base)
    arrays, row_indices, keys = sequence_arrays(prepared)
    checkpoint = torch.load(residual_path, map_location="cpu", weights_only=False)
    raw = predict_checkpoint(checkpoint, arrays, "cuda" if torch.cuda.is_available() else "cpu")
    observed = arrays[3].astype(bool); indices = row_indices[observed]; actual = arrays[2][observed]
    quantiles = apply_calibration(raw[observed], calibration)
    candidates = {"candidate": quantiles[:, 1], "incumbent_control": arrays[1][observed],
                  "physics": prepared.loc[indices, "physics_fm"].to_numpy(),
                  "persistence": prepared.loc[indices, "initial_fm"].to_numpy()}
    metrics = {}
    for name, prediction in candidates.items():
        table = scored(prepared, indices, prediction)
        metrics[name] = detailed_metrics(table, quantiles if name == "candidate" else None)
        add_tail_metrics(metrics[name], actual, prediction)
    candidate, control = metrics["candidate"], metrics["incumbent_control"]
    candidate_critical, control_critical = candidate["regimes"]["critical_low_fm"], control["regimes"]["critical_low_fm"]
    candidate_fnr = candidate["critical_threshold"]["false_negative_rate"]
    control_fnr = control["critical_threshold"]["false_negative_rate"]
    checks = {"same_row_samples": candidate["samples"] == control["samples"] == len(indices),
              "mae_5pct_better": candidate["mae"] <= 0.95 * control["mae"],
              "rmse_no_worse": candidate["rmse"] <= control["rmse"],
              "absolute_bias_no_worse": abs(candidate["bias"]) <= abs(control["bias"]),
              "critical_support": candidate["critical_threshold"]["support"] >= MIN_CRITICAL_SAMPLES,
              "critical_mae_no_worse": bool(candidate_critical and control_critical and candidate_critical["mae"] <= control_critical["mae"]),
              "critical_fnr_within_2pp": bool(candidate_fnr is not None and control_fnr is not None and candidate_fnr <= control_fnr + 0.02),
              "interval_coverage": 0.78 <= candidate["interval_coverage"] <= 0.82,
              "quantile_ordering": candidate["quantile_order_violation_rate"] == 0.0,
              "contract_complete": not mismatches}
    row_keys = prepared.loc[indices, ROW_KEY].astype(str).agg("|".join, axis=1)
    import hashlib
    report = {"status": "historical_relock_final", "pass": all(checks.values()), "checks": checks,
              "historical_relock": True, "prospective_shadow_required": True,
              "dataset_sha256": manifest["dataset_sha256"], "manifest_sha256": manifest["manifest_sha256"],
              "split_version": manifest["version"], "feature_schema_version": contract["feature_schema_version"],
              "base_model_sha256": expected["base_model_sha256"], "residual_model_sha256": expected["residual_model_sha256"],
              "calibration_sha256": sha256_file(calibration_path),
              "contract_sha256": sha256_file(args.candidate_dir / "contract.json"),
              "row_key": ROW_KEY, "row_key_sha256": hashlib.sha256("\n".join(row_keys).encode()).hexdigest(),
              "samples": len(indices), "sequences": len(keys), "metrics": metrics,
              "evaluated_at": datetime.now(timezone.utc).isoformat()}
    args.output.write_text(json.dumps(report, indent=2))
    compact = {"pass": report["pass"], "checks": checks,
               "candidate": {key: candidate[key] for key in ("mae", "rmse", "bias", "r2", "samples", "interval_coverage", "quantile_order_violation_rate")},
               "incumbent": {key: control[key] for key in ("mae", "rmse", "bias", "r2", "samples")}}
    print(json.dumps(compact, indent=2)); raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
