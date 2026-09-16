"""Finalize a calibrated V4 directory for explicit API shadow loading only."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent));import paths
from spatial.rule_contract import RULE_SPEC_SHA256,check_api_copy
from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
from spatial.station_contract import sha256_file

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--candidate-dir",type=Path,default=paths.V4_CANDIDATE_DIR);args=parser.parse_args()
    output=args.candidate_dir/"shadow_bundle_manifest.json"
    if output.exists():raise SystemExit(f"Refusing to overwrite {output}")
    check_api_copy();contract=json.loads((args.candidate_dir/"contract.json").read_text());calibration=args.candidate_dir/"calibration.json"
    if not calibration.exists():raise SystemExit("V4 calibration is required before shadow export")
    value={"status":"experimental_shadow_only","registry_channel":None,"beta_registration_allowed":False,
      "rule_spec_sha256":RULE_SPEC_SHA256,"manifest_sha256":contract["manifest_sha256"],
      "precipitation_contract_version":PRECIPITATION_CONTRACT_VERSION,
      "precipitation_contract_sha256":PRECIPITATION_CONTRACT_SHA256,
      "assets":{name:sha256_file(args.candidate_dir/name) for name in ("base_xgboost.json","guarded_gru.pt","lead_guard.json","contract.json","calibration.json")}}
    output.write_text(json.dumps(value,indent=2));print(json.dumps(value,indent=2))
if __name__=="__main__":main()
