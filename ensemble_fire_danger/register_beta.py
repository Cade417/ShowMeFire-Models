"""
Registers the ensemble_fire_danger candidate bundle (written by
retrain.py / fit + evaluate) as a beta in this repo's local registry,
after re-checking the evaluation report: no gate may be "fail" (deferred
gates - too few test days or events yet - do not block a v1 beta, same
philosophy as fire_weather_index). Always beta, always advisory_only;
promotion is a separate, deliberate server-side step.

Usage:
    python -m ensemble_fire_danger.register_beta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from models.register import ModelRegistrationSpec, register_beta  # noqa: E402

from . import model_bundle  # noqa: E402

MODEL_TYPE = "ensemble_fire_danger"


def _validate_report(candidate_dir: Path, report: Dict) -> Dict:
    if report.get("model_family") != MODEL_TYPE:
        raise RuntimeError(f"refused: report model_family {report.get('model_family')!r} is not {MODEL_TYPE}")
    if report.get("advisory_only") is not True:
        raise RuntimeError("refused: report is not marked advisory_only")
    failed = [gate["name"] for gate in report.get("gates", []) if gate.get("status") == "fail"]
    if failed:
        raise RuntimeError(f"refused: gates failed: {failed}")
    for name in model_bundle.BUNDLE_ASSET_FILENAMES.values():
        if not (candidate_dir / name).exists():
            raise RuntimeError(f"refused: candidate is missing {name} - run retrain/fit first")
    return {}


SPEC = ModelRegistrationSpec(
    model_type=MODEL_TYPE,
    asset_filenames=model_bundle.BUNDLE_ASSET_FILENAMES,
    validate_report=_validate_report,
    build_candidate=lambda candidate_dir: None,  # fit.py/retrain.py already wrote the bundle
    build_performance=lambda report, context: {
        "overall_pass": report["overall_pass"],
        "test_days": report.get("test_days"),
        "train_days": report.get("train_days"),
        "gates": report["gates"],
        "track_comparison": report.get("track_comparison"),
    },
    # api/models/versioning.py::REQUIRED_ENSEMBLE_FIRE_DANGER_METADATA
    build_metadata=lambda report, context: {
        "model_family": MODEL_TYPE,
        "advisory_only": True,
        "core_module_sha256": model_bundle.core_module_sha256(),
        "recommended_primary": (report.get("track_comparison") or {}).get("recommended_primary"),
    },
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=paths.ENSEMBLE_FD_REPORT)
    parser.add_argument("--candidate-dir", type=Path, default=paths.ENSEMBLE_FD_CANDIDATE_DIR)
    args = parser.parse_args(argv)
    version = register_beta(SPEC, report_path=args.report, candidate_dir=args.candidate_dir)
    model_bundle.set_version(args.candidate_dir, version)
    print(f"Registered {MODEL_TYPE} beta version {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
