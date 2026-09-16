"""
Fits the fire_risk_fusion GLM-only v1 model (monthly-baseline +
fast-weather residual, per risk_fusion/model_bundle.py) on the real
2014-2020 labeled panel and persists a candidate bundle to
paths.RISK_FUSION_CANDIDATE_DIR - the artifact evaluate_risk_fusion.py
scores and register_risk_fusion_beta.py promotes.

Usage:
    python -m risk_fusion.fit_risk_fusion
    python -m risk_fusion.fit_risk_fusion --output-dir some/other/path
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
from risk_fusion import features, model_bundle
from risk_fusion.risk_fusion_contract import add_split_columns, create_manifest

TRAINING_PANEL_PATH = paths.RISK_FUSION_DIR / "labeled_panel_2014_2020.csv"


def build_contract(train_panel: pd.DataFrame) -> Dict:
    """
    Records the split manifest actually used plus the sha256 of the
    frozen feature module and the offset definition (effort.py) - a
    later evaluation/registration run that recomputes these and gets a
    different hash means something upstream changed without the contract
    being updated to match, which is exactly what this is meant to catch.
    """
    split_manifest = create_manifest(add_split_columns(train_panel))
    return {
        "feature_module_schema": features.FEATURE_MODULE_SCHEMA,
        "feature_module_sha256": model_bundle.sha256_file(REPO_ROOT / "risk_fusion" / "features.py"),
        "offset_definition_sha256": model_bundle.sha256_file(REPO_ROOT / "risk_fusion" / "effort.py"),
        "split_manifest": split_manifest,
        "training_window": {"start": model_bundle.TRAINING_WINDOW_START, "end": model_bundle.TRAINING_WINDOW_END},
        "model_family": "glm",
        "count_family": "poisson",
        "advisory_only": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=paths.RISK_FUSION_CANDIDATE_DIR)
    args = parser.parse_args()

    print(f"Loading training panel from {TRAINING_PANEL_PATH}...")
    train_panel = model_bundle.load_training_panel(TRAINING_PANEL_PATH)
    print(f"{len(train_panel)} rows, {train_panel['county_fips'].nunique()} counties, "
          f"{int(train_panel['event_count'].sum())} total labeled events")

    print("Fitting the monthly-baseline + weather-residual model...")
    bundle = model_bundle.fit(train_panel)

    print(f"Persisting bundle to {args.output_dir}...")
    model_bundle.save(bundle, args.output_dir)

    contract = build_contract(train_panel)
    (args.output_dir / "contract.json").write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")

    print(f"Bundle written: {sorted(p.name for p in args.output_dir.iterdir())}")


if __name__ == "__main__":
    main()
