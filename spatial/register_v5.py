"""Register V5 as beta only after a passing immutable prospective report."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from models.versioning import register_trained_model
from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256
from spatial.rule_contract import RULE_SPEC_SHA256
from spatial.station_contract import sha256_file


def validate_registration(candidate_dir, report):
    if report.get("status") != "prospective" or not report.get("pass") or not report.get("beta_registration_allowed"):
        raise RuntimeError("V5 beta registration refused: prospective gate is incomplete or failed")
    shadow = json.loads((candidate_dir / "shadow_bundle_manifest.json").read_text())
    if shadow.get("registry_channel") is not None or shadow.get("rule_spec_sha256") != RULE_SPEC_SHA256:
        raise RuntimeError("V5 shadow/registry contract mismatch")
    if shadow.get("precipitation_contract_sha256") != PRECIPITATION_CONTRACT_SHA256:
        raise RuntimeError("V5 precipitation contract mismatch")
    mismatches = [name for name, digest in shadow.get("assets", {}).items()
                  if not (candidate_dir / name).exists() or sha256_file(candidate_dir / name) != digest]
    if mismatches: raise RuntimeError(f"V5 registration asset mismatch: {', '.join(mismatches)}")
    return shadow


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--candidate-dir", type=Path, default=paths.V5_CANDIDATE_DIR)
    parser.add_argument("--evaluation", type=Path, default=paths.REPORTS_DIR / "v5_summer_guarded_prospective_evaluation.json")
    args = parser.parse_args(); report = json.loads(args.evaluation.read_text())
    try: validate_registration(args.candidate_dir, report)
    except RuntimeError as error: raise SystemExit(str(error)) from error
    assets = {name: {"path": args.candidate_dir / filename} for name, filename in {
        "model": "specialist_xgboost.json", "base_model": "base_xgboost.json", "guard": "guard.json",
        "uncertainty": "uncertainty.json", "contract": "contract.json"}.items()}
    version = register_trained_model("fuel_moisture_station_summer_guarded", channel="beta", assets=assets,
                                     performance={"checks": report["checks"], "candidate": report["candidate"],
                                                  "incumbent": report["incumbent"], "prospective_days": report["days"],
                                                  "manifest_sha256": report["manifest_sha256"]})
    print(json.dumps({"registered_version": version, "production_changed": False}, indent=2))


if __name__ == "__main__": main()
