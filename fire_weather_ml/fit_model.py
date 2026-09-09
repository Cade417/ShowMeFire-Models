"""
Fits the fire_weather_ml v1 candidate on the built station panel (see
panel.py / docs/fire_weather_ml_plan.md for how that panel gets built -
Phase 2 work, not yet run) and persists a candidate bundle to
paths.FIRE_WEATHER_ML_CANDIDATE_DIR - the artifact evaluate.py scores and
register_beta.py promotes. Mirrors risk_fusion/fit_risk_fusion.py's shape.

Usage:
    python -m fire_weather_ml.fit_model
    python -m fire_weather_ml.fit_model --panel some/other/panel.csv --output-dir some/other/path
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths
from fire_weather_ml import contract, model_bundle
from fire_weather_ml.features import MODEL_FEATURE_COLUMNS


def build_contract(train_panel: pd.DataFrame) -> Dict:
    """
    Records the split manifest actually used plus the sha256 of the frozen
    feature/label modules - a later evaluation/registration run that
    recomputes these and gets a different hash means something upstream
    changed without the contract being updated to match.
    """
    split_manifest = contract.create_manifest(contract.add_split_columns(train_panel))
    return {
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "feature_module_sha256": model_bundle.sha256_file(REPO_ROOT / "fire_weather_ml" / "features.py"),
        "label_module_sha256": model_bundle.sha256_file(REPO_ROOT / "fire_weather_ml" / "rothermel_labels.py"),
        "label_column": model_bundle.DEFAULT_LABEL_COLUMN,
        "split_manifest": split_manifest,
        "model_family": "xgboost_regressor",
        "advisory_only": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=paths.FIRE_WEATHER_ML_DIR / "station_panel.csv")
    parser.add_argument("--output-dir", type=Path, default=paths.FIRE_WEATHER_ML_CANDIDATE_DIR)
    args = parser.parse_args()

    print(f"Loading training panel from {args.panel}...")
    train_panel = pd.read_csv(args.panel)
    print(f"{len(train_panel)} rows, {train_panel['station_id'].nunique()} stations")

    print("Fitting the fire_weather_ml XGBoost regressor...")
    bundle = model_bundle.fit(train_panel)

    print("Calibrating the 0-100 ML Fire Weather Risk Score against this model's own predictions...")
    in_sample_predictions = model_bundle.score(train_panel.dropna(subset=[bundle["label_column"]]), bundle)
    bundle["risk_calibration"] = model_bundle.calibrate_risk_score(in_sample_predictions)

    print(f"Persisting bundle to {args.output_dir}...")
    model_bundle.save(bundle, args.output_dir)

    candidate_contract = build_contract(train_panel)
    (args.output_dir / "contract.json").write_text(json.dumps(candidate_contract, indent=2, default=str), encoding="utf-8")

    print(f"Bundle written: {sorted(p.name for p in args.output_dir.iterdir())}")


if __name__ == "__main__":
    main()
