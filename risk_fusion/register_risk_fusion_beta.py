"""
Registers the fire_risk_fusion GLM-only v1 candidate bundle (from
fit_risk_fusion.py) as a beta model, after re-verifying the offline
evaluation report (evaluate_risk_fusion.py) actually passed every
checkable gate. Mirrors spatial/register_v5_beta.py's
validate-then-register shape.

This registers a BETA candidate only - advisory_only, not production
eligible, and (per the project plan) not promotable to anything
public-facing until it accumulates a prospective shadow record. Nothing
here touches the api/ repo or any live serving path.

Usage:
    python -m risk_fusion.register_risk_fusion_beta
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
from models.register import ModelRegistrationSpec, register_beta
from risk_fusion import model_bundle
from risk_fusion.risk_fusion_evidence import POLICY_VERSION, policy_sha256

MODEL_TYPE = "fire_risk_fusion"
TRAINING_PANEL_PATH = paths.RISK_FUSION_DIR / "labeled_panel_2014_2020.csv"

# The training-side registry (models/versioning.py, this repo) does not
# yet enforce REQUIRED_RISK_FUSION_METADATA the way api/models/versioning.py
# does - populating every field anyway keeps this artifact forward-
# compatible once that check is added here too, rather than registering
# something that will fail a stricter check later without anyone noticing.
REQUIRED_RISK_FUSION_METADATA_FIELDS = (
    "label_manifest_sha256", "label_min_tier", "label_rows_by_tier", "cause_filter",
    "count_family", "model_family", "offset_definition_sha256", "feature_module_sha256",
    "policy_version", "policy_sha256", "guard_active_row_fraction", "advisory_only",
)

ASSET_FILENAMES = {**model_bundle.BUNDLE_ASSET_FILENAMES, "contract": "contract.json"}


def validate_beta_registration(candidate_dir: Path, report: Dict) -> Dict:
    """
    Refuses to register unless: the report is against the current policy
    version and is an advisory-only GLM evaluation; every CHECKABLE gate
    (status == "fail") actually passed - "deferred"/"not_applicable"/
    "assumed_from_label_pipeline" gates don't block v1 registration,
    they're exactly what the prospective shadow phase exists to close;
    and every bundle asset the fit step should have written is present.
    Returns the parsed contract for build_performance to derive metadata from.
    """
    if report.get("policy_version") != POLICY_VERSION:
        raise RuntimeError(
            f"fire_risk_fusion beta registration refused: report policy_version "
            f"{report.get('policy_version')!r} does not match current {POLICY_VERSION!r}"
        )
    if report.get("policy_sha256") != policy_sha256():
        raise RuntimeError(
            "fire_risk_fusion beta registration refused: report policy_sha256 does not match "
            "the current policy - a threshold or bootstrap parameter changed since this report was generated"
        )
    if report.get("advisory_only") is not True or report.get("model_family") != "glm":
        raise RuntimeError("fire_risk_fusion beta registration refused: report is not an advisory-only GLM evaluation")

    failing = [gate["name"] for gate in report.get("gates", []) if gate["status"] == "fail"]
    if failing:
        raise RuntimeError(f"fire_risk_fusion beta registration refused: failing gates {failing}")

    contract_path = candidate_dir / "contract.json"
    if not contract_path.exists():
        raise RuntimeError(f"fire_risk_fusion beta registration refused: missing {contract_path}")
    contract = json.loads(contract_path.read_text())
    if contract.get("advisory_only") is not True:
        raise RuntimeError("fire_risk_fusion beta registration refused: contract is not advisory_only")

    for filename in model_bundle.BUNDLE_ASSET_FILENAMES.values():
        if not (candidate_dir / filename).exists():
            raise RuntimeError(f"fire_risk_fusion beta registration refused: missing bundle asset {filename}")

    return contract


def build_metadata(report: Dict, contract: Dict) -> Dict:
    """
    REQUIRED_RISK_FUSION_METADATA, filled from what this increment
    actually has. label_manifest_sha256 is the sha256 of the labeled
    training panel CSV itself (labeled_panel_2014_2020.csv) - not the
    original FPA-FOD export manifest's own hash, which isn't threaded
    through this pipeline yet; noted explicitly rather than left looking
    like the same thing.
    """
    metadata = {
        "label_manifest_sha256": model_bundle.sha256_file(TRAINING_PANEL_PATH),
        "label_manifest_sha256_note": "sha256 of labeled_panel_2014_2020.csv, not the original export manifest hash",
        "label_min_tier": "official_source_confirmed",
        "label_rows_by_tier": "not_tracked_in_this_panel - enforced upstream in labels.py at panel-build time",
        "cause_filter": "prescribed,agricultural excluded - enforced upstream in labels.py",
        "count_family": contract.get("count_family"),
        "model_family": contract.get("model_family"),
        "offset_definition_sha256": contract.get("offset_definition_sha256"),
        "feature_module_sha256": contract.get("feature_module_sha256"),
        "policy_version": report.get("policy_version"),
        "policy_sha256": report.get("policy_sha256"),
        "guard_active_row_fraction": 1.0,
        "guard_active_row_fraction_note": "no guard/GBM residual in this GLM-only increment - 1.0 records the documented model_family=='glm' exception, not an actual guard result",
        "advisory_only": True,
    }
    missing = [field for field in REQUIRED_RISK_FUSION_METADATA_FIELDS if metadata.get(field) is None]
    if missing:
        raise RuntimeError(f"fire_risk_fusion beta registration refused: metadata missing {missing}")
    return metadata


def _build_performance(report: Dict, contract: Dict) -> Dict:
    return {
        **build_metadata(report, contract),
        "row_count": report.get("row_count"),
        "scores": report.get("scores"),
        "gates": report.get("gates"),
        "production_eligible": False,
        "prospective_shadow_required": True,
    }


SPEC = ModelRegistrationSpec(
    model_type=MODEL_TYPE,
    asset_filenames=ASSET_FILENAMES,
    validate_report=validate_beta_registration,
    build_candidate=lambda candidate_dir: None,  # nothing to rebuild - fit_risk_fusion.py already wrote the bundle
    build_performance=_build_performance,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-dir", type=Path, default=paths.RISK_FUSION_CANDIDATE_DIR)
    parser.add_argument("--evaluation", type=Path, default=paths.REPORTS_DIR / "risk_fusion_offline_evaluation.json")
    args = parser.parse_args()

    try:
        version = register_beta(SPEC, report_path=args.evaluation, candidate_dir=args.candidate_dir)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    print(json.dumps({"registered_version": version, "channel": "beta", "advisory_only": True,
                      "production_changed": False}, indent=2))


if __name__ == "__main__":
    main()
