"""Create immutable, paired development evidence for promotion policy V2."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.station_contract import sha256_file
from spatial.v4_metrics import categories
from spatial.v5_contract import load_or_create, reject_forbidden
from spatial.v5_evidence import POLICY_VERSION, dataframe_sha256, evaluate_policy, policy_sha256
from spatial.v5_features import FEATURES, regime_labels
from spatial.v5_guard import apply_guard, intervals
from spatial.v5_training import (crossfit_base, crossfit_incumbent, fit_base, fit_incumbent,
                                 fit_specialist, load_frame, load_or_prepare_static, predict_base,
                                 predict_incumbent, predict_specialist, prepare)


def _bundle(candidate_dir: Path):
    contract = json.loads((candidate_dir / "contract.json").read_text())
    guard = json.loads((candidate_dir / "guard.json").read_text())
    uncertainty = json.loads((candidate_dir / "uncertainty.json").read_text())
    assets = {name: sha256_file(candidate_dir / name) for name in
              ("base_xgboost.json", "specialist_xgboost.json", "guard.json", "uncertainty.json", "contract.json")}
    return contract, guard, uncertainty, assets


def build_paired_rows(frame, manifest, search, candidate_dir, feature_cache):
    contract, guard, uncertainty, assets = _bundle(candidate_dir)
    if contract.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise RuntimeError("V5 bundle and split manifest do not match")
    physics = contract["physics_variant"]
    static = load_or_prepare_static(frame, feature_cache, manifest["manifest_sha256"], physics)
    pooled = []
    for fold in manifest["folds"]:
        reject_forbidden(manifest, fold["train_runs"] + fold["validation_runs"], f"offline evidence {fold['name']}")
        train_index = frame.index[frame.run_id.isin(set(fold["train_runs"]))]
        valid_index = frame.index[frame.run_id.isin(set(fold["validation_runs"]))]
        if contract["base_family"] == "rain_aware":
            base_oof, _ = crossfit_base(frame, fold["train_runs"], search["selected_base_config"], physics, prepared=static)
            base_model = fit_base(frame, fold["train_runs"], search["selected_base_config"], physics, prepared=static)
            base_valid = predict_base(base_model, frame.loc[valid_index], physics, static)
        else:
            base_oof, _ = crossfit_incumbent(frame, fold["train_runs"])
            base_model = fit_incumbent(frame, fold["train_runs"])
            base_valid = predict_incumbent(base_model, frame.loc[valid_index])
        training = prepare(frame.loc[train_index], base_oof.loc[train_index], physics, static)
        validation = prepare(frame.loc[valid_index], base_valid, physics, static)
        specialist = fit_specialist(training, fold["train_runs"], search["selected_specialist_config"])
        scored = validation[(validation.target_mask == 1) & validation.target_fm.notna()].copy()
        correction = predict_specialist(specialist, scored)
        regimes = regime_labels(scored)
        available = scored.precip_available.fillna(0).to_numpy() > 0
        candidate, weights, caps, reasons = apply_guard(
            scored.incumbent_base_fm.to_numpy(float), correction, scored.lead_hour, regimes, guard, available)
        quantiles = intervals(candidate, regimes, uncertainty)

        # The incumbent is independently fit on the exact same fold and scored rows.
        incumbent_model = fit_incumbent(frame, fold["train_runs"])
        incumbent = predict_incumbent(incumbent_model, scored)
        actual_category = categories(scored.target_fm, scored.target_rh, scored.target_wind_ms)
        candidate_category = categories(candidate, scored.hrrr_rh, scored.hrrr_wind_ms)
        incumbent_category = categories(incumbent, scored.hrrr_rh, scored.hrrr_wind_ms)
        valid_categories = np.asarray([(a is not None and c is not None and i is not None)
                                       for a, c, i in zip(actual_category, candidate_category, incumbent_category)])
        evidence = pd.DataFrame({
            "run_id": scored.run_id.astype(str), "station_id": scored.station_id.astype(str),
            "valid_time": scored.valid_time.astype(str), "fold": fold["name"],
            "bootstrap_block": scored.forecast_init_time.dt.strftime("%Y-%m-%d"),
            "actual_fm": scored.target_fm.to_numpy(float), "incumbent_fm": incumbent,
            "v5_base_fm": scored.incumbent_base_fm.to_numpy(float), "candidate_fm": candidate,
            "p10": quantiles[:, 0], "p50": quantiles[:, 1], "p90": quantiles[:, 2],
            "actual_category": actual_category, "candidate_category": candidate_category,
            "incumbent_category": incumbent_category,
            "summer": scored.valid_time.dt.month.isin([6, 7, 8]).to_numpy(),
            "critical": scored.target_fm.to_numpy(float) <= 6,
            "rain_event": ((scored.active_rain_indicator > 0) | (scored.post_rain_3h_indicator > 0)).to_numpy(),
            "target_rh": scored.target_rh.to_numpy(float), "target_wind_ms": scored.target_wind_ms.to_numpy(float),
            "forecast_rh": scored.hrrr_rh.to_numpy(float), "forecast_wind_ms": scored.hrrr_wind_ms.to_numpy(float),
            "guard_weight": weights, "guard_cap": caps, "guard_reason": reasons,
        })
        pooled.append(evidence.loc[valid_categories])
    return pd.concat(pooled, ignore_index=True), assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=paths.V4_ALIGNED_DATASET)
    parser.add_argument("--v4-manifest", type=Path, default=paths.V4_SPLIT_MANIFEST)
    parser.add_argument("--manifest", type=Path, default=paths.V5_SPLIT_MANIFEST)
    parser.add_argument("--search", type=Path, default=paths.REPORTS_DIR / "v5_summer_guarded_search.json")
    parser.add_argument("--candidate-dir", type=Path, default=paths.V5_CANDIDATE_DIR)
    parser.add_argument("--feature-cache", type=Path, default=paths.REPORTS_DIR / "v5_static_features.pkl")
    parser.add_argument("--rows", type=Path, default=paths.REPORTS_DIR / "v5_offline_v2_paired_rows.csv.gz")
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "v5_offline_v2_evaluation.json")
    args = parser.parse_args()
    if args.rows.exists() or args.output.exists():
        raise SystemExit("Refusing to overwrite immutable V5 policy-V2 evidence")
    frame = load_frame(args.dataset)
    manifest = load_or_create(args.manifest, frame, args.dataset, args.v4_manifest, FEATURES, create=False)
    search = json.loads(args.search.read_text())
    if search.get("manifest_sha256") != manifest["manifest_sha256"] or search.get("forbidden_accessed"):
        raise SystemExit("V5 search/evidence contract mismatch")
    paired, assets = build_paired_rows(frame, manifest, search, args.candidate_dir, args.feature_cache)
    evidence = evaluate_policy(paired)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.rows, index=False, compression="gzip")
    report = {
        "status": "offline_beta_evaluation", "policy_version": POLICY_VERSION,
        "policy_sha256": policy_sha256(), "pass": evidence["pass"],
        "beta_registration_allowed": evidence["pass"], "production_eligible": False,
        "prospective_shadow_required": True, "evidence": evidence,
        "paired_rows": str(args.rows), "paired_rows_sha256": sha256_file(args.rows),
        "paired_dataframe_sha256": dataframe_sha256(paired), "samples": len(paired),
        "dataset_sha256": sha256_file(args.dataset), "manifest_sha256": manifest["manifest_sha256"],
        "feature_schema_version": search.get("feature_schema_version", contract_value(args.candidate_dir, "feature_schema_version")),
        "rule_spec_sha256": contract_value(args.candidate_dir, "rule_spec_sha256"),
        "precipitation_contract_sha256": contract_value(args.candidate_dir, "precipitation_contract_sha256"),
        "bundle_assets": assets, "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": report["pass"], "samples": len(paired),
                      "mae": evidence["bootstrap"]["mae"],
                      "failed_checks": [k for k, v in evidence["checks"].items() if not v],
                      "report": str(args.output)}, indent=2))
    raise SystemExit(0 if report["pass"] else 2)


def contract_value(directory, key):
    return json.loads((Path(directory) / "contract.json").read_text()).get(key)


if __name__ == "__main__":
    main()
