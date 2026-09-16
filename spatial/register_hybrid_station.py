"""Register a complete, gate-approved V3 hybrid bundle without touching V2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from models.versioning import register_trained_model
from spatial.station_contract import sha256_file


def validate_bundle(report, candidate_dir):
    """Return verified bundle paths or reject a partial/failed candidate."""
    if not report.get("pass") or not report.get("checks") or not all(report["checks"].values()):
        raise RuntimeError("Hybrid V3 failed the final gate; registration refused")
    candidate_dir = Path(candidate_dir)
    paths_by_role = {"base_model": candidate_dir / "base_xgboost.json",
                     "model": candidate_dir / "residual_gru.pt",
                     "calibration": candidate_dir / "calibration.json",
                     "contract": candidate_dir / "contract.json"}
    expected = {"base_model": report.get("base_model_sha256"),
                "model": report.get("residual_model_sha256"),
                "calibration": report.get("calibration_sha256"),
                "contract": report.get("contract_sha256")}
    mismatches = [role for role, path in paths_by_role.items()
                  if not expected[role] or not path.exists() or sha256_file(path) != expected[role]]
    if mismatches:
        raise RuntimeError(f"Hybrid bundle asset mismatch: {', '.join(mismatches)}")
    return paths_by_role


def main():
    parser = argparse.ArgumentParser(description="Register a passing hybrid V3 bundle as a separate beta type")
    parser.add_argument("--candidate-dir", type=Path, default=paths.MODELS_DIR / "hybrid_v3_candidate")
    parser.add_argument("--evaluation", type=Path, default=paths.REPORTS_DIR / "hybrid_v3_final_evaluation.json")
    args = parser.parse_args(); report = json.loads(args.evaluation.read_text())
    try:
        paths_by_role = validate_bundle(report, args.candidate_dir)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    performance = {"checks": report["checks"], "candidate": report["metrics"]["candidate"],
                   "incumbent_control": report["metrics"]["incumbent_control"],
                   "dataset_sha256": report["dataset_sha256"], "manifest_sha256": report["manifest_sha256"],
                   "split_version": report["split_version"], "feature_schema_version": report["feature_schema_version"],
                   "historical_relock": True, "prospective_shadow_required": True}
    version = register_trained_model("fuel_moisture_station_hybrid", performance=performance, channel="beta",
                                     assets={role: {"path": path} for role, path in paths_by_role.items()})
    print(json.dumps({"registered_version": version, "model_type": "fuel_moisture_station_hybrid",
                      "channel": "beta", "production_changed": False}, indent=2))


if __name__ == "__main__":
    main()
