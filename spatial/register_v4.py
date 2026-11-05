"""Register V4 only after an immutable passing prospective report."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent));import paths
from models.register import ModelRegistrationSpec, register_beta
from spatial.station_contract import sha256_file
from spatial.rule_contract import RULE_SPEC_SHA256
from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256

MODEL_TYPE = "fuel_moisture_station_guarded"
ASSET_FILENAMES = {"model": "guarded_gru.pt", "base_model": "base_xgboost.json",
                    "lead_guard": "lead_guard.json", "contract": "contract.json", "calibration": "calibration.json"}

def validate_registration(candidate_dir, report):
    if report.get("status")!="prospective" or not report.get("pass") or not report.get("beta_registration_allowed"):
        raise RuntimeError("V4 beta registration refused: prospective gate is incomplete or failed")
    shadow=json.loads((candidate_dir/"shadow_bundle_manifest.json").read_text())
    if shadow.get("rule_spec_sha256")!=RULE_SPEC_SHA256:raise RuntimeError("V4 registration rule mismatch")
    if shadow.get("precipitation_contract_sha256")!=PRECIPITATION_CONTRACT_SHA256:raise RuntimeError("V4 registration precipitation mismatch")
    mismatches=[name for name,digest in shadow.get("assets",{}).items() if not(candidate_dir/name).exists() or sha256_file(candidate_dir/name)!=digest]
    if mismatches:raise RuntimeError(f"V4 registration asset mismatch: {', '.join(mismatches)}")
    return shadow

def _build_performance(report, context):
    return {"checks":report["checks"],"candidate":report["candidate"],"incumbent":report["incumbent"],
            "manifest_sha256":report["manifest_sha256"],"prospective_days":report["days"]}

SPEC = ModelRegistrationSpec(
    model_type=MODEL_TYPE,
    asset_filenames=ASSET_FILENAMES,
    validate_report=validate_registration,
    build_candidate=lambda candidate_dir: None,
    build_performance=_build_performance,
)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--candidate-dir",type=Path,default=paths.V4_CANDIDATE_DIR)
    parser.add_argument("--evaluation",type=Path,default=paths.REPORTS_DIR/"v4_precipitation-v1_prospective_evaluation.json");args=parser.parse_args()
    try:
        version = register_beta(SPEC, report_path=args.evaluation, candidate_dir=args.candidate_dir)
    except RuntimeError as error:raise SystemExit(str(error)) from error
    print(json.dumps({"registered_version":version,"production_changed":False},indent=2))
if __name__=="__main__":main()
