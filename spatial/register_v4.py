"""Register V4 only after an immutable passing prospective report."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent));import paths
from models.versioning import register_trained_model
from spatial.station_contract import sha256_file
from spatial.rule_contract import RULE_SPEC_SHA256
from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256

def validate_registration(candidate_dir, report):
    if report.get("status")!="prospective" or not report.get("pass") or not report.get("beta_registration_allowed"):
        raise RuntimeError("V4 beta registration refused: prospective gate is incomplete or failed")
    shadow=json.loads((candidate_dir/"shadow_bundle_manifest.json").read_text())
    if shadow.get("rule_spec_sha256")!=RULE_SPEC_SHA256:raise RuntimeError("V4 registration rule mismatch")
    if shadow.get("precipitation_contract_sha256")!=PRECIPITATION_CONTRACT_SHA256:raise RuntimeError("V4 registration precipitation mismatch")
    mismatches=[name for name,digest in shadow.get("assets",{}).items() if not(candidate_dir/name).exists() or sha256_file(candidate_dir/name)!=digest]
    if mismatches:raise RuntimeError(f"V4 registration asset mismatch: {', '.join(mismatches)}")
    return shadow

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--candidate-dir",type=Path,default=paths.V4_CANDIDATE_DIR)
    parser.add_argument("--evaluation",type=Path,default=paths.REPORTS_DIR/"v4_precipitation-v1_prospective_evaluation.json");args=parser.parse_args()
    report=json.loads(args.evaluation.read_text())
    try:validate_registration(args.candidate_dir,report)
    except RuntimeError as error:raise SystemExit(str(error)) from error
    assets={"model":{"path":args.candidate_dir/"guarded_gru.pt"},"base_model":{"path":args.candidate_dir/"base_xgboost.json"},
            "lead_guard":{"path":args.candidate_dir/"lead_guard.json"},"contract":{"path":args.candidate_dir/"contract.json"},
            "calibration":{"path":args.candidate_dir/"calibration.json"}}
    version=register_trained_model("fuel_moisture_station_guarded",channel="beta",assets=assets,
      performance={"checks":report["checks"],"candidate":report["candidate"],"incumbent":report["incumbent"],
                   "manifest_sha256":report["manifest_sha256"],"prospective_days":report["days"]})
    # api/services/v4_shadow.py scores directly from this raw candidate_dir
    # (SMF_V4_SHADOW_BUNDLE), not the versioned copy register_trained_model
    # just made under models/versions/ - write the assigned version back
    # here too, same pattern fire_weather_ml/register_beta.py already uses.
    (args.candidate_dir/"registered_version.json").write_text(
        json.dumps({"model_type":"fuel_moisture_station_guarded","version":version},indent=2),encoding="utf-8")
    print(json.dumps({"registered_version":version,"production_changed":False},indent=2))
if __name__=="__main__":main()
