"""Shared driver collapsing the repeated register_beta.py shape used by
fire_weather_index, fire_weather_ml, risk_fusion, and the spatial v4/v5/
station/hybrid candidates: validate an offline evaluation report, (re)build
bundle assets, register them as a beta candidate, and write the assigned
version back into the candidate directory for shadow-scoring services that
read the raw directory rather than the versioned copy under models/versions/.

Each model type keeps its own eval/calibration/bundle-shape code
(factors.py, calibrate.py, model_bundle.py's save()/load()) - only the
register/validate/write-back plumbing lives here. Model types differ in
whether validation needs to read anything besides the report (e.g.
fire_weather_ml re-reads its own contract.json out of candidate_dir) and
whether there's anything to rebuild before registering (fire_weather_index
recalibrates on every run; fire_weather_ml just registers whatever
fit_model.py already wrote) - `validate_report` returns a free-form
`context` dict threaded into `build_performance` so each model type can
carry forward whatever extra data it validated (e.g. a parsed contract)
without the driver needing to know its shape.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

from models.versioning import register_trained_model


@dataclass
class ModelRegistrationSpec:
    model_type: str
    asset_filenames: Dict[str, str]
    validate_report: Callable[[Path, dict], dict]
    build_candidate: Callable[[Path], None]
    build_performance: Callable[[dict, dict], dict]
    write_back_registered_version: bool = True


def register_beta(spec: ModelRegistrationSpec, report_path: Path, candidate_dir: Path) -> str:
    """Runs one model type's full register-as-beta sequence and returns the
    assigned version string. `report_path` must point at that model type's
    own offline evaluation report (produced by its evaluate.py)."""
    if not report_path.exists():
        raise SystemExit(f"No evaluation report at {report_path} - run this model type's evaluate.py first")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    context = spec.validate_report(candidate_dir, report)
    spec.build_candidate(candidate_dir)

    assets = {role: str(candidate_dir / filename) for role, filename in spec.asset_filenames.items()}
    version = register_trained_model(
        spec.model_type, channel="beta", assets=assets,
        performance=spec.build_performance(report, context),
    )

    if spec.write_back_registered_version:
        (candidate_dir / "registered_version.json").write_text(
            json.dumps({"model_type": spec.model_type, "version": version}, indent=2), encoding="utf-8")

    return version
