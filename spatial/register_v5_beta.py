"""Register a V5 offline-qualified beta without changing production serving."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from models.versioning import register_trained_model
from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256
from spatial.rule_contract import RULE_SPEC_SHA256
from spatial.station_contract import sha256_file
from spatial.v5_evidence import POLICY_VERSION, policy_sha256


def validate_beta_registration(candidate_dir, report, rows_path=None):
    if report.get("status") != "offline_beta_evaluation" or report.get("policy_version") != POLICY_VERSION:
        raise RuntimeError("V5 beta registration refused: policy-V2 offline report required")
    if report.get("policy_sha256") != policy_sha256() or not report.get("pass") or not report.get("beta_registration_allowed"):
        raise RuntimeError("V5 beta registration refused: offline policy-V2 gate failed or changed")
    if report.get("production_eligible") is not False or not report.get("prospective_shadow_required"):
        raise RuntimeError("V5 beta report must remain non-production and require prospective shadow")
    contract = json.loads((candidate_dir / "contract.json").read_text())
    checks = {
        "manifest_sha256": contract.get("manifest_sha256"),
        "feature_schema_version": contract.get("feature_schema_version"),
        "rule_spec_sha256": RULE_SPEC_SHA256,
        "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
    }
    for key, expected in checks.items():
        if report.get(key) != expected or contract.get(key) != expected:
            raise RuntimeError(f"V5 beta registration refused: {key} mismatch")
    for filename, digest in report.get("bundle_assets", {}).items():
        path = candidate_dir / filename
        if not path.exists() or sha256_file(path) != digest:
            raise RuntimeError(f"V5 beta registration refused: asset mismatch {filename}")
    paired = Path(rows_path or report.get("paired_rows", ""))
    if not paired.exists() or sha256_file(paired) != report.get("paired_rows_sha256"):
        raise RuntimeError("V5 beta registration refused: paired evidence mismatch")
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=paths.V5_CANDIDATE_DIR)
    parser.add_argument("--evaluation", type=Path, default=paths.REPORTS_DIR / "v5_offline_v2_evaluation.json")
    parser.add_argument("--paired-rows", type=Path)
    args = parser.parse_args()
    report = json.loads(args.evaluation.read_text())
    try:
        validate_beta_registration(args.candidate_dir, report, args.paired_rows)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    assets = {name: {"path": args.candidate_dir / filename} for name, filename in {
        "model": "specialist_xgboost.json", "base_model": "base_xgboost.json", "guard": "guard.json",
        "uncertainty": "uncertainty.json", "contract": "contract.json"}.items()}
    evidence = report["evidence"]
    version = register_trained_model(
        "fuel_moisture_station_summer_guarded", channel="beta", assets=assets,
        performance={
            "evidence_status": "offline_beta", "production_eligible": False,
            "prospective_shadow_required": True, "promotion_policy": POLICY_VERSION,
            "policy_sha256": report["policy_sha256"], "paired_rows_sha256": report["paired_rows_sha256"],
            "bootstrap": evidence["bootstrap"], "support": evidence["support"],
            "checks": evidence["checks"], "manifest_sha256": report["manifest_sha256"],
        })
    # api/services/v5_shadow.py scores directly from this raw candidate_dir
    # (SMF_V5_SHADOW_BUNDLE), not the versioned copy register_trained_model
    # just made under models/versions/ - write the assigned version back
    # here too, same pattern fire_weather_ml/register_beta.py already uses.
    (args.candidate_dir / "registered_version.json").write_text(
        json.dumps({"model_type": "fuel_moisture_station_summer_guarded", "version": version}, indent=2),
        encoding="utf-8")

    print(json.dumps({"registered_version": version, "channel": "beta", "evidence_status": "offline_beta",
                      "production_eligible": False, "production_changed": False}, indent=2))


if __name__ == "__main__":
    main()
