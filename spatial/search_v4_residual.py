"""Atomic/resumable 12-configuration guarded residual search."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.station_contract import basic_metrics
from spatial.v4_contract import load_or_create, reject_forbidden
from spatial.v4_features import FEATURES
from spatial.v4_training import (load_frame, prepare_fold, prebase, residual_configs,
                                 sequence_arrays, train_residual, predict, xgb_execution_device)
from spatial.search_v4_base import atomic
from spatial.v4_metrics import category_metrics
import numpy as np


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, default=paths.V4_ALIGNED_DATASET)
    parser.add_argument("--v3-manifest", type=Path, default=paths.REPORTS_DIR / "hybrid_v3_split_manifest.json")
    parser.add_argument("--manifest", type=Path, default=paths.V4_SPLIT_MANIFEST)
    parser.add_argument("--base-search", type=Path, default=paths.REPORTS_DIR / "v4_precipitation-v1_base_search.json")
    parser.add_argument("--state", type=Path, default=paths.REPORTS_DIR / "v4_precipitation-v1_residual_search.state.json")
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "v4_precipitation-v1_residual_search.json")
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args(); frame = load_frame(args.dataset)
    manifest = load_or_create(args.manifest, frame, args.dataset, args.v3_manifest, FEATURES, create=False)
    base_report = json.loads(args.base_search.read_text()); base_config = base_report["selected_config"]; variant = base_report["physics_variant"]
    execution_device = xgb_execution_device()
    if base_report.get("xgb_device") != execution_device:
        raise RuntimeError("V4 base and residual preparation device mismatch")
    prebase_frame = prebase(frame, variant)
    prepared = {}
    for fold in manifest["folds"]:
        reject_forbidden(manifest, fold["train_runs"] + fold["validation_runs"], "residual search")
        print(f"preparing {fold['name']} OOF enhanced-base features", flush=True)
        train, valid, _, _ = prepare_fold(
            frame, fold["train_runs"], fold["validation_runs"], base_config, variant,
            prepared=prebase_frame,
        )
        valid_arrays,valid_indices,_=sequence_arrays(valid)
        prepared[fold["name"]] = (sequence_arrays(train)[0], valid_arrays,valid_indices,valid)
    records = []
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state["manifest_sha256"] != manifest["manifest_sha256"]: raise RuntimeError("V4 residual resume mismatch")
        if state.get("xgb_device") != execution_device: raise RuntimeError("V4 residual resume device mismatch")
        records = state["trials"]
    complete = {(json.dumps(x["config"], sort_keys=True), x["fold"]) for x in records}
    for number, config in enumerate(residual_configs(), 1):
        for fold in manifest["folds"]:
            key = (json.dumps(config, sort_keys=True), fold["name"])
            if key in complete: continue
            print(f"residual config={number}/12 fold={fold['name']} {config}", flush=True)
            training,validation,valid_indices,valid_frame=prepared[fold["name"]]
            model, checkpoint, report = train_residual(training,validation, epochs=args.epochs, patience=args.patience, **config)
            prediction = predict(model, validation, checkpoint["mean"], checkpoint["std"], next(model.parameters()).device)
            observed = validation[3].astype(bool); score = basic_metrics(validation[2][observed], prediction[..., 3][observed])
            flat=valid_indices[observed];actual=validation[2][observed];p50=prediction[...,3][observed];critical=actual<=6
            summer=valid_frame.loc[flat,"valid_time"].dt.month.isin([6,7,8]).to_numpy()
            summer_score=basic_metrics(actual[summer],p50[summer]) if summer.sum()>=100 else None
            critical_mae=float(np.mean(np.abs(p50[critical]-actual[critical]))) if critical.sum()>=100 else None
            category_score=category_metrics(actual,valid_frame.loc[flat,"target_rh"],valid_frame.loc[flat,"target_wind_ms"],p50,
                                            valid_frame.loc[flat,"hrrr_rh"],valid_frame.loc[flat,"hrrr_wind_ms"])
            records.append({"config": config, "fold": fold["name"], **score,"critical_mae":critical_mae,
                            "elevated_recall":category_score.get("elevated_recall"), "best_epoch": report["best_epoch"]})
            records[-1].update({"summer_mae":summer_score["mae"] if summer_score else None,
                                "summer_rmse":summer_score["rmse"] if summer_score else None})
            complete.add(key); atomic(args.state, {"status": "in_progress", "manifest_sha256": manifest["manifest_sha256"],
                                                  "xgb_device": execution_device, "trials": records})
    ranking=[]
    for config in residual_configs():
        values=[x for x in records if x["config"] == config]
        ranking.append({"config":config,"mean_mae":sum(x["mae"] for x in values)/len(values),
                        "mean_rmse":sum(x["rmse"] for x in values)/len(values),
                        "mean_absolute_bias":sum(abs(x["bias"]) for x in values)/len(values),"trials":len(values),
                        "median_best_epoch":sorted(x["best_epoch"] for x in values)[len(values)//2]})
        critical_values=[x["critical_mae"] for x in values if x["critical_mae"] is not None];recall_values=[x["elevated_recall"] for x in values if x["elevated_recall"] is not None]
        ranking[-1]["mean_critical_mae"]=sum(critical_values)/len(critical_values) if critical_values else 1e9
        ranking[-1]["mean_elevated_recall"]=sum(recall_values)/len(recall_values) if recall_values else -1.0
        summer_mae=[x["summer_mae"] for x in values if x["summer_mae"] is not None];summer_rmse=[x["summer_rmse"] for x in values if x["summer_rmse"] is not None]
        ranking[-1]["mean_summer_mae"]=sum(summer_mae)/len(summer_mae) if summer_mae else 1e9
        ranking[-1]["mean_summer_rmse"]=sum(summer_rmse)/len(summer_rmse) if summer_rmse else 1e9
    best_global=min(x["mean_mae"] for x in ranking)
    ranking.sort(key=lambda x:(x["mean_mae"]>best_global*1.01,x["mean_summer_mae"],x["mean_mae"],x["mean_summer_rmse"],x["mean_rmse"],x["mean_absolute_bias"],x["mean_critical_mae"],-x["mean_elevated_recall"],x["config"]["hidden_size"]))
    output={"status":"complete","base_config":base_config,"physics_variant":variant,"trials":records,"ranking":ranking,
            "selected_config":ranking[0]["config"],"selected_fixed_epochs":ranking[0]["median_best_epoch"],
            "selection_policy":"lowest summer MAE among configurations within 1% of best global MAE",
            "manifest_sha256":manifest["manifest_sha256"],"xgb_device":execution_device,
            "forbidden_accessed":False}
    atomic(args.output,output);atomic(args.state,output);print(json.dumps({"selected":ranking[0],"output":str(args.output)},indent=2))


if __name__ == "__main__": main()
