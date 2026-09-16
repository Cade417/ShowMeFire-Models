"""Atomic/resumable 16-configuration enhanced XGBoost search."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.station_contract import basic_metrics
from spatial.v4_contract import load_or_create, reject_forbidden
from spatial.v4_features import FEATURES
from spatial.v4_training import base_configs, fit_base, load_frame, predict_base, prebase, xgb_execution_device
from spatial.v4_metrics import category_metrics


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, default=paths.V4_ALIGNED_DATASET)
    parser.add_argument("--v3-manifest", type=Path, default=paths.REPORTS_DIR / "hybrid_v3_split_manifest.json")
    parser.add_argument("--manifest", type=Path, default=paths.V4_SPLIT_MANIFEST)
    parser.add_argument("--physics", type=Path, default=paths.REPORTS_DIR / "v4_precipitation-v1_physics_selection.json")
    parser.add_argument("--state", type=Path, default=paths.REPORTS_DIR / "v4_precipitation-v1_base_search.state.json")
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "v4_precipitation-v1_base_search.json")
    args = parser.parse_args(); frame = load_frame(args.dataset)
    manifest = load_or_create(args.manifest, frame, args.dataset, args.v3_manifest, FEATURES, create=False)
    physics = json.loads(args.physics.read_text()); variant = physics["selected_variant"]
    prepared = prebase(frame, variant)
    execution_device = xgb_execution_device()
    records = []
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state["manifest_sha256"] != manifest["manifest_sha256"]: raise RuntimeError("V4 base resume mismatch")
        if state.get("xgb_device") != execution_device: raise RuntimeError("V4 base resume device mismatch")
        records = state["trials"]
    complete = {(json.dumps(x["config"], sort_keys=True), x["fold"]) for x in records}
    for number, config in enumerate(base_configs(), 1):
        for fold in manifest["folds"]:
            reject_forbidden(manifest, fold["train_runs"] + fold["validation_runs"], "base search")
            key = (json.dumps(config, sort_keys=True), fold["name"])
            if key in complete: continue
            print(f"base config={number}/16 fold={fold['name']} {config}", flush=True)
            model = fit_base(frame, fold["train_runs"], config, variant, fold["validation_runs"], prepared=prepared)
            selected = frame[frame.run_id.isin(set(fold["validation_runs"])) & (frame.target_mask == 1)]
            prediction = predict_base(model, selected, variant, prepared=prepared); score = basic_metrics(selected.target_fm, prediction)
            summer = selected.valid_time.dt.month.isin([6, 7, 8]).to_numpy()
            summer_score = basic_metrics(selected.target_fm.to_numpy()[summer], prediction[summer]) if summer.sum() >= 100 else None
            critical=selected.target_fm.to_numpy()<=6
            critical_mae=float(np.mean(np.abs(prediction[critical]-selected.target_fm.to_numpy()[critical]))) if critical.sum()>=100 else None
            category_score=category_metrics(selected.target_fm,selected.target_rh,selected.target_wind_ms,prediction,selected.hrrr_rh,selected.hrrr_wind_ms)
            records.append({"config": config, "fold": fold["name"], **score,
                            "critical_mae":critical_mae,"elevated_recall":category_score.get("elevated_recall"),
                            "summer_mae":summer_score["mae"] if summer_score else None,
                            "summer_rmse":summer_score["rmse"] if summer_score else None,
                            "best_iteration": getattr(model, "best_iteration", None)})
            complete.add(key); atomic(args.state, {"status": "in_progress", "manifest_sha256": manifest["manifest_sha256"],
                                                  "xgb_device": execution_device, "trials": records})
    ranking = []
    for config in base_configs():
        values = [x for x in records if x["config"] == config]
        ranking.append({"config": config, "mean_mae": sum(x["mae"] for x in values)/len(values),
                        "mean_rmse": sum(x["rmse"] for x in values)/len(values),
                        "mean_absolute_bias": sum(abs(x["bias"]) for x in values)/len(values), "trials": len(values)})
        critical_values=[x["critical_mae"] for x in values if x["critical_mae"] is not None];recall_values=[x["elevated_recall"] for x in values if x["elevated_recall"] is not None]
        ranking[-1]["mean_critical_mae"]=sum(critical_values)/len(critical_values) if critical_values else 1e9
        ranking[-1]["mean_elevated_recall"]=sum(recall_values)/len(recall_values) if recall_values else -1.0
        summer_mae=[x["summer_mae"] for x in values if x["summer_mae"] is not None]
        summer_rmse=[x["summer_rmse"] for x in values if x["summer_rmse"] is not None]
        ranking[-1]["mean_summer_mae"]=sum(summer_mae)/len(summer_mae) if summer_mae else 1e9
        ranking[-1]["mean_summer_rmse"]=sum(summer_rmse)/len(summer_rmse) if summer_rmse else 1e9
    best_global = min(x["mean_mae"] for x in ranking)
    ranking.sort(key=lambda x: (x["mean_mae"] > best_global * 1.01, x["mean_summer_mae"], x["mean_mae"], x["mean_summer_rmse"], x["mean_rmse"], x["mean_absolute_bias"],x["mean_critical_mae"],-x["mean_elevated_recall"]))
    output = {"status": "complete", "physics_variant": variant, "trials": records, "ranking": ranking,
              "selected_config": ranking[0]["config"], "manifest_sha256": manifest["manifest_sha256"],
              "xgb_device": execution_device,
              "selection_policy": "lowest summer MAE among configurations within 1% of best global MAE",
              "forbidden_accessed": False}
    atomic(args.output, output); atomic(args.state, output); print(json.dumps({"selected_config": ranking[0], "output": str(args.output)}, indent=2))


if __name__ == "__main__": main()
