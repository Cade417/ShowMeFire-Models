"""
Registers a fire_weather_index candidate bundle as a beta model, after
re-verifying the offline evaluation report actually passed every CHECKABLE
gate (deferred coverage/monotonicity gates don't block v1 registration -
that's exactly what accumulating more RRFS/FV3-HIRES history and a labeled
panel closes over time, same philosophy as risk_fusion's own v1 boundary).

Always registers to the `beta` channel, always advisory_only - promotion is
a separate, deliberate step this script never performs.

Usage:
    python -m fire_weather_index.register_beta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths
from fire_weather_index import calibrate, model_bundle
from models.register import ModelRegistrationSpec, register_beta

MODEL_TYPE = "fire_weather_index"


def _validate_report(candidate_dir: Path, report: Dict) -> Dict:
    if report.get("model_family") != "fire_weather_index":
        raise RuntimeError(f"refused: report model_family {report.get('model_family')!r} is not fire_weather_index")
    if not report.get("advisory_only"):
        raise RuntimeError("refused: report is not marked advisory_only")
    failed = [gate["name"] for gate in report.get("gates", []) if gate.get("status") == "fail"]
    if failed:
        raise RuntimeError(f"refused: gates failed: {failed}")
    return {}


def _build_candidate(candidate_dir: Path) -> Dict:
    """Runs calibration fresh and writes both bundle assets into candidate_dir."""
    calibration = calibrate.run()
    weights_asset = model_bundle.build_factor_weights_asset()
    thresholds_asset = model_bundle.build_category_thresholds_asset(
        calibration["thresholds"], calibration_report=calibration)
    model_bundle.save(candidate_dir, weights_asset, thresholds_asset)
    return calibration


SPEC = ModelRegistrationSpec(
    model_type=MODEL_TYPE,
    asset_filenames=model_bundle.BUNDLE_ASSET_FILENAMES,
    validate_report=_validate_report,
    build_candidate=_build_candidate,
    build_performance=lambda report, context: {"overall_pass": report["overall_pass"], "gates": report["gates"]},
    # api/models/versioning.py::REQUIRED_FIRE_WEATHER_INDEX_METADATA needs
    # model_family/advisory_only as an actual metadata= record, not just
    # inside performance - _validate_report above already confirmed both
    # fields on the report, so this just carries them through. Missing
    # metadata= entirely (the pre-existing gap this fixes) is exactly what
    # made an imported fire_weather_index beta fail the server's promotion
    # gate with "missing metadata: advisory_only, model_family".
    build_metadata=lambda report, context: {
        "model_family": report["model_family"], "advisory_only": report["advisory_only"],
    },
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=paths.REPORTS_DIR / "fire_weather_index_offline_evaluation.json",
                        help="Path to evaluate.py's report - registration is refused if this doesn't exist or fails a gate")
    parser.add_argument("--candidate-dir", type=Path, default=paths.FIRE_WEATHER_INDEX_CANDIDATE_DIR)
    args = parser.parse_args()

    version = register_beta(SPEC, report_path=args.report, candidate_dir=args.candidate_dir)
    print(f"Registered {MODEL_TYPE} beta version {version}")


if __name__ == "__main__":
    main()
