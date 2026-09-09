"""
Registers the fire_weather_ml v1 candidate bundle (from fit_model.py) as a
beta model, after re-verifying the offline evaluation report (evaluate.py)
actually passed every checkable gate. Mirrors
risk_fusion/register_risk_fusion_beta.py's validate-then-register shape.

Registers a BETA candidate only, to THIS repo's own training-side registry
(models/versioning.py) - advisory_only, not production eligible, and not
promotable to anything public-facing until a future phase adds a real
prospective evaluation. Nothing here touches the api/ repo, its registry,
or any live serving path - importing this model_type into api/'s own
registry (api/models/versioning.py::validate_promotion_candidate) is
separate, later work, done only once there's something worth serving.

Usage:
    python -m fire_weather_ml.register_beta
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths
from models.versioning import register_trained_model
from fire_weather_ml import model_bundle

REQUIRED_METADATA_FIELDS = (
    "feature_module_sha256", "label_module_sha256", "label_column",
    "model_family", "training_row_count", "split_manifest_sha256", "advisory_only",
)


def validate_beta_registration(candidate_dir: Path, report: Dict) -> Dict:
    """
    Refuses to register unless: the report is advisory-only; every
    CHECKABLE gate (status == "fail") actually passed - "deferred"/
    "not_applicable" gates don't block v1 registration, they're exactly
    what later phases (real historical panel, occurrence cross-check,
    prospective shadow) exist to close; and every bundle asset the fit
    step should have written is present.
    """
    if report.get("advisory_only") is not True:
        raise RuntimeError("fire_weather_ml beta registration refused: report is not advisory_only")

    failing = [gate["name"] for gate in report.get("gates", []) if gate["status"] == "fail"]
    if failing:
        raise RuntimeError(f"fire_weather_ml beta registration refused: failing gates {failing}")

    contract_path = candidate_dir / "contract.json"
    if not contract_path.exists():
        raise RuntimeError(f"fire_weather_ml beta registration refused: missing {contract_path}")
    candidate_contract = json.loads(contract_path.read_text())
    if candidate_contract.get("advisory_only") is not True:
        raise RuntimeError("fire_weather_ml beta registration refused: contract is not advisory_only")

    for filename in model_bundle.BUNDLE_ASSET_FILENAMES.values():
        if not (candidate_dir / filename).exists():
            raise RuntimeError(f"fire_weather_ml beta registration refused: missing bundle asset {filename}")

    return candidate_contract


def build_metadata(report: Dict, candidate_contract: Dict) -> Dict:
    metadata = {
        "feature_module_sha256": candidate_contract.get("feature_module_sha256"),
        "label_module_sha256": candidate_contract.get("label_module_sha256"),
        "label_column": candidate_contract.get("label_column"),
        "model_family": candidate_contract.get("model_family"),
        "training_row_count": report.get("row_count"),
        "split_manifest_sha256": (candidate_contract.get("split_manifest") or {}).get("manifest_sha256"),
        "advisory_only": True,
    }
    missing = [field for field in REQUIRED_METADATA_FIELDS if metadata.get(field) is None]
    if missing:
        raise RuntimeError(f"fire_weather_ml beta registration refused: metadata missing {missing}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-dir", type=Path, default=paths.FIRE_WEATHER_ML_CANDIDATE_DIR)
    parser.add_argument("--evaluation", type=Path, default=paths.REPORTS_DIR / "fire_weather_ml_offline_evaluation.json")
    args = parser.parse_args()

    report = json.loads(args.evaluation.read_text())
    try:
        candidate_contract = validate_beta_registration(args.candidate_dir, report)
        metadata = build_metadata(report, candidate_contract)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    assets = {role: {"path": args.candidate_dir / filename}
              for role, filename in model_bundle.BUNDLE_ASSET_FILENAMES.items()}
    assets["contract"] = {"path": args.candidate_dir / "contract.json"}

    version = register_trained_model(
        "fire_weather_ml", channel="beta", assets=assets,
        performance={
            **metadata,
            "scores": report.get("scores"),
            "gates": report.get("gates"),
            "production_eligible": False,
            "prospective_shadow_required": True,
        },
    )
    print(json.dumps({"registered_version": version, "channel": "beta", "advisory_only": True,
                      "production_changed": False}, indent=2))


if __name__ == "__main__":
    main()
