"""Calibrate a fitted V4 bundle on the dedicated split halves."""
from __future__ import annotations

import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent));import paths
from spatial.rule_contract import category
from spatial.station_contract import sha256_file
from spatial.v4_calibration import (apply_bias,apply_conformal,apply_isotonic,category_probability,
                                    choose_bias,conformal,fit_isotonic)
from spatial.v4_contract import load_or_create,reject_forbidden
from spatial.v4_features import FEATURES
from spatial.v4_metrics import probability_metrics
from spatial.v4_runtime import load_bundle,score
from spatial.v4_training import load_frame


def flatten(runtime,frame,runs):
    selected=frame[frame.run_id.isin(set(runs))];prepared,arrays,indices,_,quantiles=score(runtime,selected)
    observed=arrays[3].astype(bool);flat=indices[observed]
    return {"actual":arrays[2][observed],"quantiles":quantiles[observed],"leads":arrays[4][observed],
            "rh":prepared.loc[flat,"hrrr_rh"].to_numpy(),"wind":prepared.loc[flat,"hrrr_wind_ms"].to_numpy(),
            "target_rh":prepared.loc[flat,"target_rh"].to_numpy(),"target_wind":prepared.loc[flat,"target_wind_ms"].to_numpy()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",type=Path,default=paths.V4_ALIGNED_DATASET)
    parser.add_argument("--v3-manifest",type=Path,default=paths.REPORTS_DIR/"hybrid_v3_split_manifest.json");parser.add_argument("--manifest",type=Path,default=paths.V4_SPLIT_MANIFEST)
    parser.add_argument("--candidate-dir",type=Path,default=paths.V4_CANDIDATE_DIR);parser.add_argument("--output",type=Path)
    args=parser.parse_args();output=args.output or args.candidate_dir/"calibration.json"
    if output.exists():raise SystemExit(f"Refusing to overwrite {output}")
    frame=load_frame(args.dataset);manifest=load_or_create(args.manifest,frame,args.dataset,args.v3_manifest,FEATURES,create=False)
    reject_forbidden(manifest,manifest["calibration_runs"],"V4 calibration");runtime=load_bundle(args.candidate_dir)
    if runtime["contract"]["manifest_sha256"]!=manifest["manifest_sha256"]:raise SystemExit("V4 calibration contract mismatch")
    fit=flatten(runtime,frame,manifest["calibration_fit_runs"]);valid=flatten(runtime,frame,manifest["calibration_validation_runs"])
    selection=choose_bias(fit["actual"],fit["quantiles"],fit["leads"],valid["actual"],valid["quantiles"],valid["leads"])
    all_actual=np.concatenate((fit["actual"],valid["actual"]));all_q=np.concatenate((fit["quantiles"],valid["quantiles"]));all_leads=np.concatenate((fit["leads"],valid["leads"]))
    if selection["type"]=="none":offsets={"global":0.0}
    elif selection["type"]=="global":offsets={"global":float(np.clip(np.mean(all_actual-all_q[:,3]),-1.5,1.5))}
    else:
        global_offset=float(np.clip(np.mean(all_actual-all_q[:,3]),-1.5,1.5));offsets={}
        for lead in np.unique(all_leads):
            selected=all_leads==lead;shrink=float(selected.sum()/(selected.sum()+200.0));raw=float(np.mean(all_actual[selected]-all_q[selected,3]))
            offsets[str(float(lead))]=float(np.clip(shrink*raw+(1-shrink)*global_offset,-1.5,1.5))
    biased=apply_bias(all_q,offsets,all_leads);expansion=conformal(all_actual,biased);calibrated=apply_conformal(biased,expansion)
    fit_biased=apply_bias(fit["quantiles"],offsets,fit["leads"])
    valid_biased=apply_bias(valid["quantiles"],offsets,valid["leads"])
    validation_expansion=conformal(fit["actual"],fit_biased)
    fit_q=apply_conformal(fit_biased,validation_expansion)
    valid_q=apply_conformal(valid_biased,validation_expansion)
    fit_probability=category_probability(fit_q,fit["rh"],fit["wind"])
    fit_category=np.asarray([category(f,r,w*1.9438444924406) for f,r,w in zip(fit["actual"],fit["target_rh"],fit["target_wind"])],object)
    fit_mask=np.asarray([x is not None for x in fit_category]);isotonic=fit_isotonic(fit_probability[fit_mask],np.asarray(fit_category[fit_mask],int)>=2)
    valid_probability=apply_isotonic(category_probability(valid_q,valid["rh"],valid["wind"]),isotonic)
    valid_category=np.asarray([category(f,r,w*1.9438444924406) for f,r,w in zip(valid["actual"],valid["target_rh"],valid["target_wind"])],object)
    valid_mask=np.asarray([x is not None for x in valid_category]);prob_metrics=probability_metrics(np.asarray(valid_category[valid_mask],int)>=2,valid_probability[valid_mask])
    artifact={"status":"calibrated","manifest_sha256":manifest["manifest_sha256"],"frozen_partition_sha256":manifest["frozen_partition_sha256"],
      "bias_selection":selection,"bias_offsets":offsets,"conformal_expansion":expansion,"target_coverage":.8,
      "validation_conformal_expansion":validation_expansion,
      "validation_interval_coverage":float(np.mean((valid["actual"]>=valid_q[:,1])&(valid["actual"]<=valid_q[:,5]))),
      "interval_coverage":float(np.mean((all_actual>=calibrated[:,1])&(all_actual<=calibrated[:,5]))),
      "probability_isotonic":isotonic,"probability_validation":prob_metrics,"calibration_runs":len(manifest["calibration_runs"]),
      "prospective_shadow_required":True,"created_at":datetime.now(timezone.utc).isoformat()}
    output.write_text(json.dumps(artifact,indent=2));print(json.dumps(artifact,indent=2))


if __name__=="__main__":main()
