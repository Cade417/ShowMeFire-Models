"""Development-only search for the V5 rain-aware base and guarded specialist."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.station_contract import basic_metrics
from spatial.v4_metrics import category_metrics
from spatial.v5_contract import load_or_create, reject_forbidden
from spatial.v5_features import FEATURES, regime_labels
from spatial.v5_guard import apply_guard, fit_guard
from spatial.v5_training import (base_configs, best_iterations, crossfit_base, crossfit_incumbent,
                                 fit_base, fit_incumbent, fit_specialist,
                                 load_frame, load_or_prepare_static, predict_base, predict_incumbent, predict_specialist, prepare,
                                 specialist_configs, xgb_execution_device)


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def score_rows(frame, prediction):
    actual = frame.target_fm.to_numpy(float); prediction = np.asarray(prediction, float)
    overall = basic_metrics(actual, prediction)
    summer = frame.valid_time.dt.month.isin([6, 7, 8]).to_numpy()
    critical = actual <= 6
    result = {**overall,
              "summer_mae": basic_metrics(actual[summer], prediction[summer])["mae"] if summer.sum() >= 100 else None,
              "summer_rmse": basic_metrics(actual[summer], prediction[summer])["rmse"] if summer.sum() >= 100 else None,
              "critical_mae": float(np.mean(np.abs(actual[critical] - prediction[critical]))) if critical.sum() >= 100 else None}
    categories = category_metrics(actual, frame.target_rh, frame.target_wind_ms, prediction, frame.hrrr_rh, frame.hrrr_wind_ms)
    result["elevated_recall"] = categories.get("elevated_recall")
    result["elevated_false_alarm_ratio"] = categories.get("elevated_false_alarm_ratio")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=paths.V4_ALIGNED_DATASET)
    parser.add_argument("--v4-manifest", type=Path, default=paths.V4_SPLIT_MANIFEST)
    parser.add_argument("--manifest", type=Path, default=paths.V5_SPLIT_MANIFEST)
    parser.add_argument("--physics-variant", default="rain25_scale2")
    parser.add_argument("--state", type=Path, default=paths.REPORTS_DIR / "v5_summer_guarded_search.state.json")
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "v5_summer_guarded_search.json")
    parser.add_argument("--minimum-guard-rows", type=int, default=300)
    parser.add_argument("--feature-cache", type=Path, default=paths.REPORTS_DIR / "v5_static_features.pkl")
    args = parser.parse_args()
    frame = load_frame(args.dataset)
    manifest = load_or_create(args.manifest, frame, args.dataset, args.v4_manifest, FEATURES)
    execution_device = xgb_execution_device()
    base_prepared = load_or_prepare_static(frame, args.feature_cache, manifest["manifest_sha256"], args.physics_variant)
    state = {"manifest_sha256": manifest["manifest_sha256"], "xgb_device": execution_device,
             "base_trials": [], "specialist_trials": []}
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state.get("manifest_sha256") != manifest["manifest_sha256"] or state.get("xgb_device") != execution_device:
            raise RuntimeError("V5 search resume contract mismatch")
    state.setdefault("base_trials", state.get("base_trials", []))
    state.setdefault("specialist_trials", state.get("specialist_ranking", []))
    base_complete = {(json.dumps(x["config"], sort_keys=True), x["fold"]) for x in state["base_trials"]}
    configs = base_configs()
    for number, config in enumerate(configs, 1):
        for fold in manifest["folds"]:
            reject_forbidden(manifest, fold["train_runs"] + fold["validation_runs"], "V5 base search")
            key = (json.dumps(config, sort_keys=True), fold["name"])
            if key in base_complete: continue
            print(f"V5 base {number}/{len(configs)} {fold['name']} {config}", flush=True)
            model = fit_base(frame, fold["train_runs"], config, args.physics_variant, fold["validation_runs"], base_prepared)
            selected = frame[frame.run_id.isin(set(fold["validation_runs"])) & (frame.target_mask == 1)]
            prediction = predict_base(model, selected, args.physics_variant, base_prepared)
            state["base_trials"].append({"config": config, "fold": fold["name"],
                                          **score_rows(selected, prediction),
                                          "best_iteration": best_iterations(model, 800)})
            base_complete.add(key); atomic(args.state, state)
    base_ranking = []
    for config in configs:
        trials = [x for x in state["base_trials"] if x["config"] == config]
        base_ranking.append({"config": config, "folds": len(trials),
                             "mean_mae": float(np.mean([x["mae"] for x in trials])),
                             "mean_rmse": float(np.mean([x["rmse"] for x in trials])),
                             "mean_absolute_bias": float(np.mean([abs(x["bias"]) for x in trials])),
                             "mean_summer_mae": float(np.mean([x["summer_mae"] for x in trials if x["summer_mae"] is not None])),
                             "median_trees": int(np.median([x["best_iteration"] for x in trials]))})
    best_global = min(x["mean_mae"] for x in base_ranking)
    base_ranking.sort(key=lambda x: (x["mean_mae"] > best_global * 1.01, x["mean_summer_mae"], x["mean_mae"], x["mean_rmse"], x["mean_absolute_bias"]))
    selected_base = dict(base_ranking[0]["config"], n_estimators=base_ranking[0]["median_trees"])

    incumbent_trials = []
    for fold in manifest["folds"]:
        model = fit_incumbent(frame, fold["train_runs"])
        selected = frame[frame.run_id.isin(set(fold["validation_runs"])) & (frame.target_mask == 1)]
        incumbent_trials.append({"fold": fold["name"], **score_rows(selected, predict_incumbent(model, selected))})
    incumbent_summary = {"mean_mae": float(np.mean([x["mae"] for x in incumbent_trials])),
                         "mean_rmse": float(np.mean([x["rmse"] for x in incumbent_trials])),
                         "mean_summer_mae": float(np.mean([x["summer_mae"] for x in incumbent_trials if x["summer_mae"] is not None]))}
    rain_summary = base_ranking[0]
    base_family = "rain_aware" if (rain_summary["mean_mae"] < incumbent_summary["mean_mae"] and
                                    rain_summary["mean_summer_mae"] <= incumbent_summary["mean_summer_mae"] and
                                    rain_summary["mean_rmse"] <= 1.01 * incumbent_summary["mean_rmse"]) else "incumbent"
    if state.get("specialist_base_family") != base_family:
        state["specialist_trials"] = []
    state["specialist_base_family"] = base_family

    # Base predictions are independent of specialist configuration, so prepare each fold once.
    fold_data = {}
    for fold in manifest["folds"]:
        print(f"Preparing V5 grouped OOF base for {fold['name']}", flush=True)
        if base_family == "rain_aware":
            oof, _ = crossfit_base(frame, fold["train_runs"], selected_base, args.physics_variant, prepared=base_prepared)
            base_model = fit_base(frame, fold["train_runs"], selected_base, args.physics_variant, prepared=base_prepared)
            base_predict = lambda rows: predict_base(base_model, rows, args.physics_variant, base_prepared)
        else:
            oof, _ = crossfit_incumbent(frame, fold["train_runs"])
            base_model = fit_incumbent(frame, fold["train_runs"])
            base_predict = lambda rows: predict_incumbent(base_model, rows)
        train_index = frame.index[frame.run_id.isin(set(fold["train_runs"]))]
        valid_index = frame.index[frame.run_id.isin(set(fold["validation_runs"]))]
        training = prepare(frame.loc[train_index], oof.loc[train_index], args.physics_variant, base_prepared)
        valid_base = base_predict(frame.loc[valid_index])
        validation = prepare(frame.loc[valid_index], valid_base, args.physics_variant, base_prepared)
        fold_data[fold["name"]] = (training, validation, fold)

    specialist_ranking = []
    for number, config in enumerate(specialist_configs(), 1):
        existing = next((item for item in state["specialist_trials"] if item["config"] == config), None)
        if existing is not None:
            specialist_ranking.append(existing); continue
        pooled = []
        for fold_name, (training, validation, fold) in fold_data.items():
            print(f"V5 specialist {number}/{len(specialist_configs())} {fold_name} {config}", flush=True)
            specialist = fit_specialist(training, fold["train_runs"], config)
            scored = validation[validation.target_mask == 1].copy()
            scored["raw_correction"] = predict_specialist(specialist, scored)
            scored["regime"] = regime_labels(scored); scored["fold"] = fold_name
            scored["actual"] = scored.target_fm; scored["base"] = scored.incumbent_base_fm
            pooled.append(scored)
        scored = pd.concat(pooled, ignore_index=True)
        guard = fit_guard(scored, args.minimum_guard_rows)
        prediction, _, _, _ = apply_guard(scored.incumbent_base_fm, scored.raw_correction,
                                           scored.lead_hour, scored.regime, guard,
                                           scored.precip_available.fillna(0) > 0)
        metrics = score_rows(scored, prediction); base_metrics = score_rows(scored, scored.incumbent_base_fm)
        trial = {"config": config, "guard": guard, "metrics": metrics, "base_metrics": base_metrics,
                 "nonzero_guard_cells": int(sum(value["weight"] > 0 for value in guard.values()))}
        # Replace a prior matching trial to keep resume/state deterministic.
        state["specialist_trials"] = [x for x in state["specialist_trials"] if x["config"] != config] + [trial]
        atomic(args.state, state); specialist_ranking.append(trial)
    specialist_ranking.sort(key=lambda x: (
        x["metrics"]["mae"] > x["base_metrics"]["mae"],
        x["metrics"]["summer_mae"] > x["base_metrics"]["summer_mae"],
        x["metrics"]["summer_mae"], x["metrics"]["mae"], x["metrics"]["rmse"], abs(x["metrics"]["bias"]),
        x["config"]["max_depth"],
    ))
    selected = specialist_ranking[0]
    output = {"status": "complete", "manifest_sha256": manifest["manifest_sha256"],
              "physics_variant": args.physics_variant, "xgb_device": execution_device,
              "base_trials": state["base_trials"], "base_ranking": base_ranking,
              "selected_base_family": base_family, "selected_base_config": selected_base,
              "specialist_base_family": base_family,
              "incumbent_base_trials": incumbent_trials, "incumbent_base_summary": incumbent_summary,
              "rejected_rain_aware_base_summary": rain_summary if base_family == "incumbent" else None,
              "specialist_ranking": specialist_ranking,
              "selected_specialist_config": selected["config"], "selected_guard": selected["guard"],
              "development_metrics": selected["metrics"], "development_base_metrics": selected["base_metrics"],
              "selection_policy": "development OOF only; global/summer safe guarded correction; forbidden evidence excluded",
              "forbidden_accessed": False, "beta_registration_allowed": False}
    atomic(args.output, output); atomic(args.state, output)
    print(json.dumps({"selected_base_family": base_family, "selected_base": selected_base, "selected_specialist": selected["config"],
                      "development": selected["metrics"], "base": selected["base_metrics"],
                      "nonzero_guard_cells": selected["nonzero_guard_cells"], "output": str(args.output)}, indent=2))


if __name__ == "__main__": main()
