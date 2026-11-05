"""Fit one frozen, non-registering V5 bundle from development-only selection."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.evaluate_baselines import FEATURES as INCUMBENT_FEATURES
from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
from spatial.rule_contract import RULE_SPEC_SHA256
from spatial.station_contract import sha256_file
from spatial.v5_contract import FEATURE_SCHEMA_VERSION, load_or_create, reject_forbidden
from spatial.v5_features import BASE_FEATURES, FEATURES, SPECIALIST_FEATURES, regime_labels
from spatial.v5_guard import fit_guard, fit_uncertainty
from spatial.v5_training import (crossfit_base, crossfit_incumbent, fit_base, fit_incumbent,
                                 fit_specialist, load_frame, predict_base, predict_incumbent,
                                 load_or_prepare_static, predict_specialist, prepare, xgb_execution_device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=paths.V4_ALIGNED_DATASET)
    parser.add_argument("--v4-manifest", type=Path, default=paths.V4_SPLIT_MANIFEST)
    parser.add_argument("--manifest", type=Path, default=paths.V5_SPLIT_MANIFEST)
    parser.add_argument("--search", type=Path, default=paths.REPORTS_DIR / "v5_summer_guarded_search.json")
    parser.add_argument("--output-dir", type=Path, default=paths.V5_CANDIDATE_DIR)
    parser.add_argument("--feature-cache", type=Path, default=paths.REPORTS_DIR / "v5_static_features.pkl")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite immutable V5 candidate {args.output_dir}")
    frame = load_frame(args.dataset)
    manifest = load_or_create(args.manifest, frame, args.dataset, args.v4_manifest, FEATURES, create=False)
    reject_forbidden(manifest, manifest["development_runs"], "V5 fit")
    search = json.loads(args.search.read_text())
    if search.get("manifest_sha256") != manifest["manifest_sha256"] or search.get("forbidden_accessed"):
        raise SystemExit("V5 search/evidence contract mismatch")
    base_config = search["selected_base_config"]; specialist_config = search["selected_specialist_config"]
    base_family = search["selected_base_family"]
    physics_variant = search["physics_variant"]
    base_prepared = load_or_prepare_static(frame, args.feature_cache, manifest["manifest_sha256"], physics_variant)
    pooled = []
    for fold in manifest["folds"]:
        print(f"Fitting frozen V5 guard evidence {fold['name']}", flush=True)
        train_index = frame.index[frame.run_id.isin(set(fold["train_runs"]))]
        valid_index = frame.index[frame.run_id.isin(set(fold["validation_runs"]))]
        if base_family == "rain_aware":
            oof, _ = crossfit_base(frame, fold["train_runs"], base_config, physics_variant, prepared=base_prepared)
            base_model = fit_base(frame, fold["train_runs"], base_config, physics_variant, prepared=base_prepared)
            base_prediction = predict_base(base_model, frame.loc[valid_index], physics_variant, base_prepared)
        else:
            oof, _ = crossfit_incumbent(frame, fold["train_runs"])
            base_model = fit_incumbent(frame, fold["train_runs"])
            base_prediction = predict_incumbent(base_model, frame.loc[valid_index])
        training = prepare(frame.loc[train_index], oof.loc[train_index], physics_variant, base_prepared)
        validation = prepare(frame.loc[valid_index], base_prediction, physics_variant, base_prepared)
        specialist = fit_specialist(training, fold["train_runs"], specialist_config)
        scored = validation[validation.target_mask == 1].copy()
        scored["raw_correction"] = predict_specialist(specialist, scored)
        scored["regime"] = regime_labels(scored); scored["fold"] = fold["name"]
        scored["actual"] = scored.target_fm; scored["base"] = scored.incumbent_base_fm
        pooled.append(scored)
    scored = pd.concat(pooled, ignore_index=True)
    guard = fit_guard(scored)
    uncertainty = fit_uncertainty(scored, guard)

    print("Fitting full-development grouped OOF base", flush=True)
    if base_family == "rain_aware":
        base_oof, provenance = crossfit_base(frame, manifest["development_runs"], base_config, physics_variant, prepared=base_prepared)
    else:
        base_oof, provenance = crossfit_incumbent(frame, manifest["development_runs"])
    development_index = frame.index[frame.run_id.isin(set(manifest["development_runs"]))]
    specialist_training = prepare(frame.loc[development_index], base_oof.loc[development_index], physics_variant, base_prepared)
    specialist_model = fit_specialist(specialist_training, manifest["development_runs"], specialist_config)
    base_model = (fit_base(frame, manifest["development_runs"], base_config, physics_variant, prepared=base_prepared)
                  if base_family == "rain_aware" else fit_incumbent(frame, manifest["development_runs"]))

    args.output_dir.mkdir(parents=True)
    base_path = args.output_dir / "base_xgboost.json"; specialist_path = args.output_dir / "specialist_xgboost.json"
    guard_path = args.output_dir / "guard.json"; uncertainty_path = args.output_dir / "uncertainty.json"
    base_model.save_model(base_path); specialist_model.save_model(specialist_path)
    guard_path.write_text(json.dumps(guard, indent=2)); uncertainty_path.write_text(json.dumps(uncertainty, indent=2))
    contract = {
        "status": "experimental_unregistered", "model_family": f"{base_family}-xgboost-plus-guarded-shallow-xgboost",
        "split_version": manifest["version"], "manifest_sha256": manifest["manifest_sha256"],
        "development_partition_sha256": manifest["development_partition_sha256"],
        "forbidden_runs_sha256": manifest["forbidden_runs_sha256"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "features": FEATURES,
        "base_features": (BASE_FEATURES if base_family == "rain_aware" else INCUMBENT_FEATURES),
        "specialist_features": SPECIALIST_FEATURES,
        "rule_spec_sha256": RULE_SPEC_SHA256,
        "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
        "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
        "physics_variant": physics_variant, "base_family": base_family, "base_config": base_config, "specialist_config": specialist_config,
        "xgb_device": xgb_execution_device(), "base_model_sha256": sha256_file(base_path),
        "specialist_model_sha256": sha256_file(specialist_path), "guard_sha256": sha256_file(guard_path),
        "uncertainty_sha256": sha256_file(uncertainty_path), "crossfit_blocks": 5,
        "crossfit_scored_runs": sum(len(item["scored_runs"]) for item in provenance),
        "development_oof_metrics": search["development_metrics"],
        "development_oof_base_metrics": search["development_base_metrics"],
        "prospective_after": manifest["prospective_after"], "prospective_shadow_required": True,
        "beta_registration_allowed": False, "created_at": datetime.now(timezone.utc).isoformat(),
    }
    contract_path = args.output_dir / "contract.json"; contract_path.write_text(json.dumps(contract, indent=2))
    print(json.dumps({"candidate_dir": str(args.output_dir), "status": contract["status"],
                      "nonzero_guard_cells": sum(item["weight"] > 0 for item in guard.values()),
                      "prospective_shadow_required": True, "beta_registration_allowed": False}, indent=2))


if __name__ == "__main__": main()
