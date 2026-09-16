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
import json
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths
from fire_weather_index import calibrate, model_bundle
from models.versioning import register_trained_model

MODEL_TYPE = "fire_weather_index"


def validate_beta_registration(report: Dict) -> None:
    if report.get("model_family") != "fire_weather_index":
        raise RuntimeError(f"refused: report model_family {report.get('model_family')!r} is not fire_weather_index")
    if not report.get("advisory_only"):
        raise RuntimeError("refused: report is not marked advisory_only")
    failed = [gate["name"] for gate in report.get("gates", []) if gate.get("status") == "fail"]
    if failed:
        raise RuntimeError(f"refused: gates failed: {failed}")


def build_candidate(candidate_dir: Path) -> Dict:
    """Runs calibration fresh and writes both bundle assets into candidate_dir."""
    calibration = calibrate.run()
    weights_asset = model_bundle.build_factor_weights_asset()
    thresholds_asset = model_bundle.build_category_thresholds_asset(
        calibration["thresholds"], calibration_report=calibration)
    model_bundle.save(candidate_dir, weights_asset, thresholds_asset)
    return calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=paths.REPORTS_DIR / "fire_weather_index_offline_evaluation.json",
                        help="Path to evaluate.py's report - registration is refused if this doesn't exist or fails a gate")
    parser.add_argument("--candidate-dir", type=Path, default=paths.FIRE_WEATHER_INDEX_CANDIDATE_DIR)
    args = parser.parse_args()

    if not args.report.exists():
        raise SystemExit(f"No evaluation report at {args.report} - run `python -m fire_weather_index.evaluate` first")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_beta_registration(report)

    build_candidate(args.candidate_dir)
    assets = {role: str(args.candidate_dir / filename) for role, filename in model_bundle.BUNDLE_ASSET_FILENAMES.items()}
    version = register_trained_model(
        MODEL_TYPE, channel="beta",
        assets={role: {"path": path} for role, path in assets.items()},
        performance={"overall_pass": report["overall_pass"], "gates": report["gates"]},
    )

    # api/services/fire_weather_index_shadow.py scores directly from this raw
    # candidate_dir (SMF_FIRE_WEATHER_INDEX_BUNDLE), not the versioned copy
    # register_trained_model just made under models/versions/ - so the
    # assigned version string needs writing back here too, same reason
    # fire_weather_ml/register_beta.py already does this (this script was
    # previously missing it entirely, leaving model_version permanently
    # unknown to the shadow module and anything reading its bundle).
    (args.candidate_dir / "registered_version.json").write_text(
        json.dumps({"model_type": MODEL_TYPE, "version": version}, indent=2), encoding="utf-8")

    print(f"Registered {MODEL_TYPE} beta version {version}")


if __name__ == "__main__":
    main()
