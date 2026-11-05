"""Fit bias and conformal interval calibration on the dedicated 80-90% partition."""
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
from spatial.hybrid_features import FEATURES
from spatial.hybrid_calibration import apply_calibration, fit_calibration
from spatial.hybrid_training import load_frame, predict_checkpoint, prepare_with_base, sequence_arrays
from spatial.station_contract import basic_metrics, sha256_file


def main():
    parser = argparse.ArgumentParser(description="Calibrate the fitted hybrid V3 bundle")
    parser.add_argument("--dataset", type=Path, default=paths.ALIGNED_DIR / "station_leads.csv")
    parser.add_argument("--manifest", type=Path, default=paths.REPORTS_DIR / "hybrid_v3_split_manifest.json")
    parser.add_argument("--candidate-dir", type=Path, default=paths.MODELS_DIR / "hybrid_v3_candidate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); output = args.output or args.candidate_dir / "calibration.json"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite calibration artifact: {output}")
    frame = load_frame(args.dataset); manifest = load_or_create_manifest(args.manifest, frame, args.dataset, FEATURES, create=False)
    contract = json.loads((args.candidate_dir / "contract.json").read_text())
    base_path, residual_path = args.candidate_dir / "base_xgboost.json", args.candidate_dir / "residual_gru.pt"
    if sha256_file(base_path) != contract["base_model_sha256"] or sha256_file(residual_path) != contract["residual_model_sha256"]:
        raise SystemExit("Candidate asset checksum mismatch")
    if contract["manifest_sha256"] != manifest["manifest_sha256"]:
        raise SystemExit("Candidate/split mismatch")
    base = xgb.XGBRegressor(); base.load_model(base_path)
    prepared = prepare_with_base(frame, manifest["calibration_runs"], base)
    arrays, indices, _ = sequence_arrays(prepared)
    checkpoint = torch.load(residual_path, map_location="cpu", weights_only=False)
    prediction = predict_checkpoint(checkpoint, arrays, "cuda" if torch.cuda.is_available() else "cpu")
    observed = arrays[3].astype(bool); actual = arrays[2][observed]; uncalibrated = prediction[observed]
    parameters = fit_calibration(actual, uncalibrated)
    bias_offset, conformal = parameters["bias_offset"], parameters["conformal_expansion"]
    calibrated = apply_calibration(uncalibrated, parameters)
    dates = pd.to_datetime(prepared.loc[indices[observed], "valid_time"], utc=True)
    artifact = {"status": "calibrated", "dataset_sha256": manifest["dataset_sha256"],
                "manifest_sha256": manifest["manifest_sha256"], "split_version": manifest["version"],
                "feature_schema_version": contract["feature_schema_version"],
                "base_model_sha256": contract["base_model_sha256"],
                "residual_model_sha256": contract["residual_model_sha256"],
                "calibration_window": {"start": dates.min().isoformat(), "end": dates.max().isoformat(),
                                       "runs": len(manifest["calibration_runs"]), "samples": int(len(actual))},
                **parameters,
                "uncalibrated_metrics": basic_metrics(actual, uncalibrated[:, 1]),
                "calibrated_metrics": basic_metrics(actual, calibrated[:, 1]),
                "calibrated_interval_coverage": float(np.mean((actual >= calibrated[:, 0]) & (actual <= calibrated[:, 2]))),
                "historical_relock": True, "prospective_shadow_required": True,
                "created_at": datetime.now(timezone.utc).isoformat()}
    output.write_text(json.dumps(artifact, indent=2)); print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
