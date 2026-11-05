"""Register a complete, gate-approved V3 hybrid bundle without touching V2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from models.register import ModelRegistrationSpec, register_beta
from spatial.station_contract import sha256_file

MODEL_TYPE = "fuel_moisture_station_hybrid"
ASSET_FILENAMES = {"base_model": "base_xgboost.json", "model": "residual_gru.pt",
                    "calibration": "calibration.json", "contract": "contract.json"}


def validate_bundle(report, candidate_dir):
    """Return verified bundle paths or reject a partial/failed candidate."""
    if not report.get("pass") or not report.get("checks") or not all(report["checks"].values()):
        raise RuntimeError("Hybrid V3 failed the final gate; registration refused")
    candidate_dir = Path(candidate_dir)
    paths_by_role = {role: candidate_dir / filename for role, filename in ASSET_FILENAMES.items()}
    expected = {"base_model": report.get("base_model_sha256"),
                "model": report.get("residual_model_sha256"),
                "calibration": report.get("calibration_sha256"),
                "contract": report.get("contract_sha256")}
    mismatches = [role for role, path in paths_by_role.items()
                  if not expected[role] or not path.exists() or sha256_file(path) != expected[role]]
    if mismatches:
        raise RuntimeError(f"Hybrid bundle asset mismatch: {', '.join(mismatches)}")
    return paths_by_role


def _validate_report(candidate_dir, report):
    validate_bundle(report, candidate_dir)
    return report


def _build_performance(report, context):
    return {"checks": report["checks"], "candidate": report["metrics"]["candidate"],
            "incumbent_control": report["metrics"]["incumbent_control"],
            "dataset_sha256": report["dataset_sha256"], "manifest_sha256": report["manifest_sha256"],
            "split_version": report["split_version"], "feature_schema_version": report["feature_schema_version"],
            "historical_relock": True, "prospective_shadow_required": True}


SPEC = ModelRegistrationSpec(
    model_type=MODEL_TYPE,
    asset_filenames=ASSET_FILENAMES,
    validate_report=_validate_report,
    build_candidate=lambda candidate_dir: None,
    build_performance=_build_performance,
    write_back_registered_version=False,  # not read directly by any *_shadow.py service
)


def main():
    parser = argparse.ArgumentParser(description="Register a passing hybrid V3 bundle as a separate beta type")
    parser.add_argument("--candidate-dir", type=Path, default=paths.MODELS_DIR / "hybrid_v3_candidate")
    parser.add_argument("--evaluation", type=Path, default=paths.REPORTS_DIR / "hybrid_v3_final_evaluation.json")
    args = parser.parse_args()
    try:
        version = register_beta(SPEC, report_path=args.evaluation, candidate_dir=args.candidate_dir)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"registered_version": version, "model_type": MODEL_TYPE,
                      "channel": "beta", "production_changed": False}, indent=2))


if __name__ == "__main__":
    main()
