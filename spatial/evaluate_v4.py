"""Evaluate V4 on calibration-validation or immutable prospective evidence."""
from __future__ import annotations

import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0,str(Path(__file__).resolve().parent.parent));import paths
from spatial.rule_contract import category
from spatial.v4_calibration import apply_bias,apply_conformal,apply_isotonic,category_probability
from spatial.v4_contract import load_or_create,prospective_runs,reject_forbidden
from spatial.v4_features import FEATURES
from spatial.v4_metrics import basic_and_tail,category_metrics,probability_metrics
from spatial.v4_runtime import load_bundle,score
from spatial.v4_training import load_frame
from spatial.evaluate_baselines import FEATURES as INCUMBENT_FEATURES
from spatial.station_contract import ROW_KEY,detailed_metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--partition",choices=("calibration-validation","prospective"),default="calibration-validation")
    parser.add_argument("--dataset",type=Path,default=paths.V4_ALIGNED_DATASET);parser.add_argument("--v3-manifest",type=Path,default=paths.REPORTS_DIR/"hybrid_v3_split_manifest.json")
    parser.add_argument("--manifest",type=Path,default=paths.V4_SPLIT_MANIFEST);parser.add_argument("--candidate-dir",type=Path,default=paths.V4_CANDIDATE_DIR)
    parser.add_argument("--output",type=Path);args=parser.parse_args()
    output=args.output or paths.REPORTS_DIR/f"v4_precipitation-v1_{args.partition}_evaluation.json"
    if args.partition=="prospective" and output.exists():raise SystemExit(f"Refusing to overwrite prospective report {output}")
    frame=load_frame(args.dataset);manifest=load_or_create(args.manifest,frame,args.dataset,args.v3_manifest,FEATURES,create=False)
    runs=manifest["calibration_validation_runs"] if args.partition=="calibration-validation" else prospective_runs(manifest,frame)
    reject_forbidden(manifest,runs,"V4 evaluation")
    if not runs:raise SystemExit("No eligible prospective runs")
    runtime=load_bundle(args.candidate_dir,True);prepared,arrays,indices,_,raw=score(runtime,frame[frame.run_id.isin(set(runs))])
    observed=arrays[3].astype(bool);flat=indices[observed];actual=arrays[2][observed];base=arrays[1][observed];leads=arrays[4][observed]
    calibration=runtime["calibration"]
    expansion=(calibration.get("validation_conformal_expansion",calibration["conformal_expansion"])
               if args.partition=="calibration-validation" else calibration["conformal_expansion"])
    quantiles=apply_conformal(apply_bias(raw[observed],calibration["bias_offsets"],leads),expansion);candidate=quantiles[:,3]
    target_rh=prepared.loc[flat,"target_rh"].to_numpy();target_wind=prepared.loc[flat,"target_wind_ms"].to_numpy();forecast_rh=prepared.loc[flat,"hrrr_rh"].to_numpy();forecast_wind=prepared.loc[flat,"hrrr_wind_ms"].to_numpy()
    candidate_category=category_metrics(actual,target_rh,target_wind,candidate,forecast_rh,forecast_wind);base_category=category_metrics(actual,target_rh,target_wind,base,forecast_rh,forecast_wind)
    incumbent_train=frame[frame.run_id.isin(set(manifest["development_runs"])) & (frame.target_mask==1)].dropna(subset=INCUMBENT_FEATURES+["target_fm"])
    incumbent_model=xgb.XGBRegressor(n_estimators=300,learning_rate=.05,max_depth=5,objective="reg:squarederror",random_state=417)
    incumbent_model.fit(incumbent_train[INCUMBENT_FEATURES],incumbent_train.target_fm)
    incumbent=incumbent_model.predict(prepared.loc[flat,INCUMBENT_FEATURES]);incumbent_category=category_metrics(actual,target_rh,target_wind,incumbent,forecast_rh,forecast_wind)
    probability=apply_isotonic(category_probability(quantiles,forecast_rh,forecast_wind),calibration["probability_isotonic"])
    observed_category=np.asarray([category(f,r,w*1.9438444924406) for f,r,w in zip(actual,target_rh,target_wind)],object);pmask=np.asarray([x is not None for x in observed_category])
    prob=probability_metrics(np.asarray(observed_category[pmask],int)>=2,probability[pmask])
    def detailed(prediction,q=None):
        columns=list(dict.fromkeys([*ROW_KEY,"target_fm","initial_fm","lead_hour","valid_time"]));table=prepared.loc[flat,columns].copy()
        table["prediction"]=prediction;table["month"]=table.valid_time.dt.month
        report=detailed_metrics(table,q);report["tail_error"]=basic_and_tail(actual,prediction)["tail_error"];return report
    candidate_metrics=detailed(candidate,quantiles[:,[1,3,5]]);base_metrics=detailed(base);incumbent_metrics=detailed(incumbent)
    physics=prepared.loc[flat,"physics_fm"].to_numpy();persistence=prepared.loc[flat,"initial_fm"].to_numpy()
    critical=actual<=6
    summer=prepared.loc[flat,"valid_time"].dt.month.isin([6,7,8]).to_numpy()
    interval_rain=prepared.loc[flat,"hrrr_precip_increment_mm"].fillna(0).to_numpy()
    hours_since_rain=prepared.loc[flat,"hours_since_forecast_rain"].fillna(999).to_numpy()
    rainy=interval_rain>0.1;heavy_rain=interval_rain>=10
    post_rain=(~rainy)&(hours_since_rain<=3)
    summer_low_fm=summer&(actual<=9)
    def subgroup(mask,prediction,minimum=100):
        return basic_and_tail(actual[mask],prediction[mask]) if int(mask.sum())>=minimum else None
    candidate_summer=basic_and_tail(actual[summer],candidate[summer]) if summer.sum()>=100 else None
    incumbent_summer=basic_and_tail(actual[summer],incumbent[summer]) if summer.sum()>=100 else None
    candidate_critical=basic_and_tail(actual[critical],candidate[critical]) if critical.sum()>=100 else None
    incumbent_critical=basic_and_tail(actual[critical],incumbent[critical]) if critical.sum()>=100 else None
    rain_events={
      "rain": {"support":int(rainy.sum()),"candidate":subgroup(rainy,candidate),"incumbent":subgroup(rainy,incumbent)},
      "post_rain_3h": {"support":int(post_rain.sum()),"candidate":subgroup(post_rain,candidate),"incumbent":subgroup(post_rain,incumbent)},
      "heavy_rain": {"support":int(heavy_rain.sum()),"candidate":subgroup(heavy_rain,candidate,30),"incumbent":subgroup(heavy_rain,incumbent,30)},
      "summer_low_fm": {"support":int(summer_low_fm.sum()),"candidate":subgroup(summer_low_fm,candidate),"incumbent":subgroup(summer_low_fm,incumbent)},
    }
    candidate_fnr=float(np.mean(candidate[critical]>6)) if critical.any() else None;incumbent_fnr=float(np.mean(incumbent[critical]>6)) if critical.any() else None
    coverage=float(np.mean((actual>=quantiles[:,1])&(actual<=quantiles[:,5])));ordering=float(np.mean(np.any(np.diff(quantiles,axis=1)<0,axis=1)))
    dates=set(prepared.loc[flat,"valid_time"].dt.strftime("%Y-%m-%d"));checks={
      "mae_5pct_better":candidate_metrics["mae"]<=.95*incumbent_metrics["mae"],"rmse_no_worse":candidate_metrics["rmse"]<=incumbent_metrics["rmse"],
      "absolute_bias_no_worse":abs(candidate_metrics["bias"])<=abs(incumbent_metrics["bias"]),"critical_support":candidate_critical is not None,
      "summer_support":candidate_summer is not None,
      "summer_mae_5pct_better":candidate_summer is not None and candidate_summer["mae"]<=.95*incumbent_summer["mae"],
      "summer_rmse_no_worse":candidate_summer is not None and candidate_summer["rmse"]<=incumbent_summer["rmse"],
      "critical_mae_no_worse":candidate_critical is not None and candidate_critical["mae"]<=incumbent_critical["mae"],
      "critical_fnr":candidate_fnr is not None and candidate_fnr<=incumbent_fnr+.02,
      "interval_coverage":.78<=coverage<=.82,"quantile_ordering":ordering==0,
      "category_support":candidate_category.get("supported",False),"elevated_recall":candidate_category.get("elevated_recall") is not None and candidate_category["elevated_recall"]>=incumbent_category["elevated_recall"]-.02,
      "elevated_far":candidate_category.get("elevated_false_alarm_ratio") is not None and candidate_category["elevated_false_alarm_ratio"]<=incumbent_category["elevated_false_alarm_ratio"]+.02,
      "macro_f1":candidate_category.get("macro_f1",0)>=incumbent_category.get("macro_f1",0),"over_one_category":candidate_category.get("over_one_category_fraction",1)<=incumbent_category.get("over_one_category_fraction",1),
      "probability_ece":prob.get("ece") is not None and prob["ece"]<=.05,"positive_brier_skill":prob.get("brier_skill") is not None and prob["brier_skill"]>0,
      "prospective_days":args.partition!="prospective" or len(dates)>=manifest["prospective_requirements"]["minimum_days"],
      "elevated_period":args.partition!="prospective" or candidate_category.get("elevated_support",0)>0}
    report={"status":args.partition,"pass":all(checks.values()),"beta_registration_allowed":args.partition=="prospective" and all(checks.values()),
      "checks":checks,"days":len(dates),"runs":len(runs),"samples":len(actual),"coverage":coverage,"ordering_violation_rate":ordering,
      "candidate":candidate_metrics,"enhanced_base":base_metrics,"incumbent":incumbent_metrics,
      "physics":detailed(physics),"persistence":detailed(persistence),
      "candidate_critical":candidate_critical,"incumbent_critical":incumbent_critical,
      "candidate_summer":candidate_summer,"incumbent_summer":incumbent_summer,
      "rain_event_metrics":rain_events,
      "candidate_critical_fnr":candidate_fnr,"incumbent_critical_fnr":incumbent_fnr,
      "candidate_categories":candidate_category,"enhanced_base_categories":base_category,"incumbent_categories":incumbent_category,
      "probability":prob,"manifest_sha256":manifest["manifest_sha256"],"evaluated_at":datetime.now(timezone.utc).isoformat()}
    output.write_text(json.dumps(report,indent=2))
    summary={"status":report["status"],"pass":report["pass"],"beta_registration_allowed":report["beta_registration_allowed"],
      "samples":report["samples"],"days":report["days"],"candidate_mae":candidate_metrics["mae"],
      "incumbent_mae":incumbent_metrics["mae"],"candidate_summer_mae":candidate_summer["mae"] if candidate_summer else None,
      "incumbent_summer_mae":incumbent_summer["mae"] if incumbent_summer else None,"interval_coverage":coverage,
      "failed_checks":[key for key,value in checks.items() if not value],"report":str(output)}
    print(json.dumps(summary,indent=2));raise SystemExit(0 if report["pass"] else 2)


if __name__=="__main__":main()
