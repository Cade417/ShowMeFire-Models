"""Register a V5 offline-qualified beta without changing production serving."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from models.register import ModelRegistrationSpec, register_beta
from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256
from spatial.rule_contract import RULE_SPEC_SHA256
from spatial.station_contract import sha256_file
from spatial.v5_evidence import POLICY_VERSION, policy_sha256

MODEL_TYPE = "fuel_moisture_station_summer_guarded"

ASSET_FILENAMES = {
    "model": "specialist_xgboost.json", "base_model": "base_xgboost.json", "guard": "guard.json",
    "uncertainty": "uncertainty.json", "contract": "contract.json",
}


def _make_validate_report(rows_path):
    def _validate_report(candidate_dir, report):
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
        return report
    return _validate_report


def _build_performance(report, context):
    evidence = report["evidence"]
    return {
        "evidence_status": "offline_beta", "production_eligible": False,
        "prospective_shadow_required": True, "promotion_policy": POLICY_VERSION,
        "policy_sha256": report["policy_sha256"], "paired_rows_sha256": report["paired_rows_sha256"],
        "bootstrap": evidence["bootstrap"], "support": evidence["support"],
        "checks": evidence["checks"], "manifest_sha256": report["manifest_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=paths.V5_CANDIDATE_DIR)
    parser.add_argument("--evaluation", type=Path, default=paths.REPORTS_DIR / "v5_offline_v2_evaluation.json")
    parser.add_argument("--paired-rows", type=Path)
    args = parser.parse_args()

    spec = ModelRegistrationSpec(
        model_type=MODEL_TYPE,
        asset_filenames=ASSET_FILENAMES,
        validate_report=_make_validate_report(args.paired_rows),
        build_candidate=lambda candidate_dir: None,  # nothing to rebuild - fit_v5.py already wrote the bundle
        build_performance=_build_performance,
    )

    try:
        version = register_beta(spec, report_path=args.evaluation, candidate_dir=args.candidate_dir)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    print(json.dumps({"registered_version": version, "channel": "beta", "evidence_status": "offline_beta",
                      "production_eligible": False, "production_changed": False}, indent=2))


if __name__ == "__main__":
    main()
