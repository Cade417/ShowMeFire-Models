"""Fit the selected V4 base/residual bundle and OOF lead guard."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.rule_contract import RULE_SPEC_SHA256
from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
from spatial.station_contract import sha256_file
from spatial.v4_contract import FEATURE_SCHEMA_VERSION, load_or_create, reject_forbidden
from spatial.v4_features import FEATURES, add_v4_features
from spatial.v4_guard import fit_lead_guard
from spatial.v4_training import (crossfit_base, fit_base, load_frame, predict, prepare_fold,
                                 prebase, residual_configs, sequence_arrays, train_residual, xgb_execution_device)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",type=Path,default=paths.V4_ALIGNED_DATASET)
    parser.add_argument("--v3-manifest",type=Path,default=paths.REPORTS_DIR/"hybrid_v3_split_manifest.json")
    parser.add_argument("--manifest",type=Path,default=paths.V4_SPLIT_MANIFEST)
    parser.add_argument("--base-search",type=Path,default=paths.REPORTS_DIR/"v4_precipitation-v1_base_search.json")
    parser.add_argument("--residual-search",type=Path,default=paths.REPORTS_DIR/"v4_precipitation-v1_residual_search.json")
    parser.add_argument("--output-dir",type=Path,default=paths.V4_CANDIDATE_DIR)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    frame=load_frame(args.dataset);manifest=load_or_create(args.manifest,frame,args.dataset,args.v3_manifest,FEATURES,create=False)
    reject_forbidden(manifest,manifest["development_runs"],"V4 fit")
    base_report=json.loads(args.base_search.read_text());residual_report=json.loads(args.residual_search.read_text())
    for report in (base_report,residual_report):
        if report["manifest_sha256"] != manifest["manifest_sha256"] or report.get("forbidden_accessed"):
            raise SystemExit("V4 search contract mismatch")
    base_config=base_report["selected_config"].copy(); variant=base_report["physics_variant"]
    prebase_frame=prebase(frame,variant)
    iterations=[x.get("best_iteration") for x in base_report["trials"] if x["config"]==base_report["selected_config"] and x.get("best_iteration") is not None]
    if iterations: base_config["n_estimators"]=max(1,round(statistics.median(iterations))+1)
    residual_config=residual_report["selected_config"];fixed_epochs=int(residual_report["selected_fixed_epochs"])
    oof_rows=[]
    for fold in manifest["folds"]:
        print(f"fitting lead guard {fold['name']}",flush=True)
        train,valid,_,_=prepare_fold(frame,fold["train_runs"],fold["validation_runs"],base_config,variant,
                                    prepared=prebase_frame)
        train_arrays=sequence_arrays(train)[0];valid_arrays,row_indices,_=sequence_arrays(valid)
        model,checkpoint,_=train_residual(train_arrays,valid_arrays,epochs=fixed_epochs,fixed_epochs=fixed_epochs,**residual_config)
        raw=predict(model,valid_arrays,checkpoint["mean"],checkpoint["std"],next(model.parameters()).device)
        observed=valid_arrays[3].astype(bool); flat_indices=row_indices[observed]
        oof_rows.append(pd.DataFrame({"fold":fold["name"],"lead_hour":valid.loc[flat_indices,"lead_hour"].to_numpy(),
            "actual":valid_arrays[2][observed],"base":valid_arrays[1][observed],"residual_p50":raw[...,3][observed]}))
    guard=fit_lead_guard(pd.concat(oof_rows,ignore_index=True))
    print("preparing full-development OOF base",flush=True)
    base_oof,provenance=crossfit_base(frame,manifest["development_runs"],base_config,variant,prepared=prebase_frame)
    index=frame.index[frame.run_id.isin(set(manifest["development_runs"]))]
    prepared=add_v4_features(frame.loc[index],base_oof.loc[index],variant);arrays=sequence_arrays(prepared)[0]
    model,checkpoint,training_report=train_residual(arrays,None,fixed_epochs=fixed_epochs,epochs=fixed_epochs,**residual_config)
    base_model=fit_base(frame,manifest["development_runs"],base_config,variant,prepared=prebase_frame)
    args.output_dir.mkdir(parents=True);base_path=args.output_dir/"base_xgboost.json";residual_path=args.output_dir/"guarded_gru.pt"
    base_model.save_model(base_path);torch.save(checkpoint,residual_path)
    guard_path=args.output_dir/"lead_guard.json";guard_path.write_text(json.dumps(guard,indent=2))
    contract={"status":"fitted_uncalibrated","split_version":manifest["version"],"manifest_sha256":manifest["manifest_sha256"],
      "dataset_sha256_at_creation":manifest["dataset_sha256_at_creation"],"frozen_partition_sha256":manifest["frozen_partition_sha256"],"forbidden_runs_sha256":manifest["forbidden_runs_sha256"],
      "feature_schema_version":FEATURE_SCHEMA_VERSION,"features":FEATURES,"rule_spec_sha256":RULE_SPEC_SHA256,
      "precipitation_contract_version":PRECIPITATION_CONTRACT_VERSION,
      "precipitation_contract_sha256":PRECIPITATION_CONTRACT_SHA256,
      "physics_variant":variant,"base_config":base_config,"residual_config":residual_config,"fixed_epochs":fixed_epochs,
      "xgb_device":xgb_execution_device(),
      "base_model_sha256":sha256_file(base_path),"residual_model_sha256":sha256_file(residual_path),
      "lead_guard_sha256":sha256_file(guard_path),"quantiles":[5,10,25,50,75,90,95],
      "crossfit_blocks":5,"crossfit_scored_runs":sum(len(x["scored_runs"]) for x in provenance),
      "prospective_shadow_required":True,"beta_registration_allowed":False,"training_report":training_report,
      "created_at":datetime.now(timezone.utc).isoformat()}
    contract_path=args.output_dir/"contract.json";contract_path.write_text(json.dumps(contract,indent=2))
    print(json.dumps({"candidate_dir":str(args.output_dir),"lead_guard":guard,"beta_registration_allowed":False},indent=2))


if __name__=="__main__":main()
