"""
The ensemble_fire_danger bundle: four small JSON files, read on the server
by api/services/ensemble_fire_danger/bundle.py.

  member_config.json         the member registry the calibration was fitted
                             with (runtime uses it instead of its vendored
                             default, so members/weights/radius always match
                             the calibration)
  calibration.json           isotonic knots per track/flavour/category
  categorical_thresholds.json  tau_k per track
  evaluation.json            the offline evaluation report (for provenance;
                             the runtime only reads version/checksums)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BUNDLE_ASSET_FILENAMES = {
    "member_config": "member_config.json",
    "calibration": "calibration.json",
    "categorical_thresholds": "categorical_thresholds.json",
    "evaluation": "evaluation.json",
}
CORE_PATH = Path(__file__).with_name("core.py")
CONFIG_PATH = Path(__file__).with_name("member_config.json")


def core_module_sha256() -> str:
    """LF-normalized, identical to api/services/ensemble_fire_danger/bundle.py::core_module_sha256."""
    return hashlib.sha256(CORE_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def save(candidate_dir: Path, fitted: dict, evaluation: Optional[dict] = None, version: Optional[str] = None) -> Path:
    candidate_dir = Path(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    calibration = dict(fitted["calibration"], version=version, fitted_at=stamp,
                       core_module_sha256=core_module_sha256())
    thresholds = dict(fitted["categorical_thresholds"], version=version, fitted_at=stamp)
    files = {
        "member_config": json.loads(CONFIG_PATH.read_text(encoding="utf-8")),
        "calibration": calibration,
        "categorical_thresholds": thresholds,
        "evaluation": evaluation or {"note": "not yet evaluated"},
    }
    for role, name in BUNDLE_ASSET_FILENAMES.items():
        (candidate_dir / name).write_text(json.dumps(files[role], indent=2), encoding="utf-8")
    (candidate_dir / "fit_report.json").write_text(json.dumps(fitted.get("fit_report", {}), indent=2),
                                                   encoding="utf-8")
    return candidate_dir


def load(candidate_dir: Path) -> dict:
    candidate_dir = Path(candidate_dir)
    read = lambda name: json.loads((candidate_dir / name).read_text(encoding="utf-8"))  # noqa: E731
    return {
        "calibration": read(BUNDLE_ASSET_FILENAMES["calibration"]),
        "categorical_thresholds": read(BUNDLE_ASSET_FILENAMES["categorical_thresholds"]),
        "member_config": read(BUNDLE_ASSET_FILENAMES["member_config"]),
    }


def set_version(candidate_dir: Path, version: str) -> None:
    for name in (BUNDLE_ASSET_FILENAMES["calibration"], BUNDLE_ASSET_FILENAMES["categorical_thresholds"]):
        path = Path(candidate_dir) / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = version
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
