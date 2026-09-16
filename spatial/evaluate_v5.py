"""Evaluate a frozen V5 candidate on immutable prospective rows only."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.evaluate_baselines import FEATURES as INCUMBENT_FEATURES
from spatial.station_contract import basic_metrics, detailed_metrics
from spatial.v4_metrics import category_metrics
from spatial.v5_contract import load_or_create, prospective_runs, reject_forbidden
from spatial.v5_features import FEATURES
from spatial.v5_runtime import load_bundle, score
from spatial.v5_training import load_frame


def subgroup(actual, prediction, mask, minimum=100):
    return basic_metrics(actual[mask], prediction[mask]) if int(mask.sum()) >= minimum else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=paths.V4_ALIGNED_DATASET)
    parser.add_argument("--v4-manifest", type=Path, default=paths.V4_SPLIT_MANIFEST)
    parser.add_argument("--manifest", type=Path, default=paths.V5_SPLIT_MANIFEST)
    parser.add_argument("--candidate-dir", type=Path, default=paths.V5_CANDIDATE_DIR)
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "v5_summer_guarded_prospective_evaluation.json")
    args = parser.parse_args()
    if args.output.exists(): raise SystemExit(f"Refusing to overwrite immutable prospective report {args.output}")
    frame = load_frame(args.dataset); manifest = load_or_create(args.manifest, frame, args.dataset, args.v4_manifest, FEATURES, create=False)
    runs = prospective_runs(manifest, frame); reject_forbidden(manifest, runs, "V5 prospective evaluation")
    if not runs: raise SystemExit("No eligible V5 prospective runs")
    selected = frame[frame.run_id.isin(set(runs)) & (frame.target_mask == 1)].copy()
    runtime = load_bundle(args.candidate_dir); prepared, scored = score(runtime, selected)
    actual = prepared.target_fm.to_numpy(float); candidate = scored["prediction"]; base = scored["base"]
    development = frame[frame.run_id.isin(set(manifest["development_runs"])) & (frame.target_mask == 1)].dropna(subset=["target_fm"])
    incumbent_model = xgb.XGBRegressor(n_estimators=300, learning_rate=.05, max_depth=5,
                                       objective="reg:squarederror", random_state=417, tree_method="hist")
    incumbent_model.fit(development[INCUMBENT_FEATURES], development.target_fm)
    incumbent = incumbent_model.predict(prepared[INCUMBENT_FEATURES])
    summer = prepared.valid_time.dt.month.isin([6, 7, 8]).to_numpy(); critical = actual <= 6
    rainy = prepared.active_rain_indicator.to_numpy() > 0; post_rain = prepared.post_rain_3h_indicator.to_numpy() > 0
    dates = set(prepared.valid_time.dt.strftime("%Y-%m-%d"))
    candidate_metrics, incumbent_metrics, base_metrics = basic_metrics(actual, candidate), basic_metrics(actual, incumbent), basic_metrics(actual, base)
    candidate_summer, incumbent_summer = subgroup(actual, candidate, summer), subgroup(actual, incumbent, summer)
    candidate_critical, incumbent_critical = subgroup(actual, candidate, critical), subgroup(actual, incumbent, critical)
    candidate_categories = category_metrics(actual, prepared.target_rh, prepared.target_wind_ms, candidate, prepared.hrrr_rh, prepared.hrrr_wind_ms)
    incumbent_categories = category_metrics(actual, prepared.target_rh, prepared.target_wind_ms, incumbent, prepared.hrrr_rh, prepared.hrrr_wind_ms)
    intervals = scored["intervals"]; coverage = float(np.mean((actual >= intervals[:, 0]) & (actual <= intervals[:, 2])))
    candidate_fnr = float(np.mean(candidate[critical] > 6)) if critical.any() else None
    incumbent_fnr = float(np.mean(incumbent[critical] > 6)) if critical.any() else None
    checks = {
        "minimum_30_days": len(dates) >= manifest["prospective_requirements"]["minimum_days"],
        "mae_5pct_better": candidate_metrics["mae"] <= .95 * incumbent_metrics["mae"],
        "rmse_no_worse": candidate_metrics["rmse"] <= incumbent_metrics["rmse"],
        "absolute_bias_no_worse": abs(candidate_metrics["bias"]) <= abs(incumbent_metrics["bias"]),
        "summer_support": candidate_summer is not None,
        "summer_mae_better": candidate_summer is not None and candidate_summer["mae"] < incumbent_summer["mae"],
        "summer_bias_no_worse": candidate_summer is not None and abs(candidate_summer["bias"]) <= abs(incumbent_summer["bias"]),
        "critical_support": candidate_critical is not None,
        "critical_mae_no_worse": candidate_critical is not None and candidate_critical["mae"] <= incumbent_critical["mae"],
        "critical_fnr": candidate_fnr is not None and candidate_fnr <= incumbent_fnr + .02,
        "elevated_period": candidate_categories.get("elevated_support", 0) >= 100,
        "elevated_recall": candidate_categories.get("elevated_recall") is not None and candidate_categories["elevated_recall"] >= incumbent_categories["elevated_recall"] - .02,
        "elevated_far": candidate_categories.get("elevated_false_alarm_ratio") is not None and candidate_categories["elevated_false_alarm_ratio"] <= incumbent_categories["elevated_false_alarm_ratio"] + .02,
        "macro_f1": candidate_categories.get("macro_f1", 0) >= incumbent_categories.get("macro_f1", 0),
        "over_one_category": candidate_categories.get("over_one_category_fraction", 1) <= incumbent_categories.get("over_one_category_fraction", 1),
        "post_rain_no_worse": subgroup(actual, candidate, post_rain) is not None and subgroup(actual, candidate, post_rain)["mae"] <= subgroup(actual, incumbent, post_rain)["mae"],
        "interval_coverage": .78 <= coverage <= .82,
        "interval_ordering": bool(np.all(intervals[:, 0] <= intervals[:, 1])) and bool(np.all(intervals[:, 1] <= intervals[:, 2])),
    }
    def detail(prediction, interval=None):
        table = prepared[["target_fm", "initial_fm", "lead_hour", "station_id", "valid_time"]].copy()
        table["prediction"] = prediction; table["month"] = table.valid_time.dt.month
        return detailed_metrics(table, interval)
    report = {"status": "prospective", "pass": all(checks.values()),
              "beta_registration_allowed": all(checks.values()), "checks": checks,
              "days": len(dates), "runs": len(runs), "samples": len(actual), "coverage": coverage,
              "candidate": detail(candidate, intervals), "rain_aware_base": detail(base), "incumbent": detail(incumbent),
              "candidate_summer": candidate_summer, "incumbent_summer": incumbent_summer,
              "candidate_critical": candidate_critical, "incumbent_critical": incumbent_critical,
              "rain": {"candidate": subgroup(actual, candidate, rainy), "incumbent": subgroup(actual, incumbent, rainy), "support": int(rainy.sum())},
              "post_rain": {"candidate": subgroup(actual, candidate, post_rain), "incumbent": subgroup(actual, incumbent, post_rain), "support": int(post_rain.sum())},
              "candidate_categories": candidate_categories, "incumbent_categories": incumbent_categories,
              "candidate_critical_fnr": candidate_fnr, "incumbent_critical_fnr": incumbent_fnr,
              "fallback_fraction": float(np.mean(scored["fallback"])),
              "manifest_sha256": manifest["manifest_sha256"], "evaluated_at": datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": report["pass"], "beta_registration_allowed": report["beta_registration_allowed"],
                      "days": report["days"], "candidate_mae": candidate_metrics["mae"],
                      "incumbent_mae": incumbent_metrics["mae"],
                      "failed_checks": [key for key, value in checks.items() if not value], "report": str(args.output)}, indent=2))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__": main()
