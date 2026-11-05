"""
The fire_weather_index model bundle: factor weights/anchors and calibrated
category cutpoints, saved as one JSON file per role (same convention as
risk_fusion/model_bundle.py). There is no fitted GLM here - "fitting" this
model means calibrating category cutpoints (see calibrate.py), not
estimating factor weights from a label.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from fire_weather_index import factors

BUNDLE_ASSET_FILENAMES = {
    "factor_weights": "factor_weights.json",
    "category_thresholds": "category_thresholds.json",
}


def build_factor_weights_asset(weights: Optional[Dict[str, float]] = None) -> Dict:
    weights = weights or factors.FACTOR_WEIGHTS
    return {
        "schema": "fire-weather-index-factor-weights-v1",
        "weights": weights,
        "raw_score_ceiling": {
            "value": factors.RAW_SCORE_CEILING,
            "anchor": "peak raw county score, eastern Missouri (centroid_lon > -91.0), 2025-03-14",
            "source": "scripts/find_ceiling_anchor_score.py",
        },
        "ramp_anchors": {
            "rh": {"benign": factors.RH_BENIGN, "extreme": factors.RH_EXTREME},
            "wind": {"benign": factors.WIND_BENIGN, "extreme": factors.WIND_EXTREME},
            "vpd": {"benign": factors.VPD_BENIGN_KPA, "extreme": factors.VPD_EXTREME_KPA, "status": "first_pass"},
            "kbdi": {"benign": factors.KBDI_BENIGN_MM, "extreme": factors.KBDI_EXTREME_MM, "status": "first_pass"},
            "cure": {"benign": factors.CURE_GREEN_CEILING_C, "extreme": factors.CURE_CURED_FLOOR_C},
            "precip_relief": {"benign": 0.0, "extreme": factors.PRECIP_RELIEF_FULL_MM, "status": "first_pass"},
        },
    }


def build_category_thresholds_asset(thresholds: list, calibration_report: Optional[Dict] = None) -> Dict:
    """thresholds: 4 increasing score cutpoints splitting [0,1] into 5 bands
    (Low/Moderate/Elevated/Critical/Extreme), from calibrate.py."""
    return {
        "schema": "fire-weather-index-category-thresholds-v1",
        "thresholds": thresholds,
        "category_labels": ["Low", "Moderate", "Elevated", "Critical", "Extreme"],
        "calibration_report": calibration_report or {},
    }


def save(directory: Path, factor_weights_asset: Dict, category_thresholds_asset: Dict) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / BUNDLE_ASSET_FILENAMES["factor_weights"]).write_text(
        json.dumps(factor_weights_asset, indent=2), encoding="utf-8")
    (directory / BUNDLE_ASSET_FILENAMES["category_thresholds"]).write_text(
        json.dumps(category_thresholds_asset, indent=2), encoding="utf-8")


def load(directory: Path) -> Dict:
    directory = Path(directory)
    missing = [role for role, filename in BUNDLE_ASSET_FILENAMES.items() if not (directory / filename).exists()]
    if missing:
        raise FileNotFoundError(f"fire_weather_index bundle at {directory} is missing assets: {missing}")
    return {
        "factor_weights": json.loads((directory / BUNDLE_ASSET_FILENAMES["factor_weights"]).read_text(encoding="utf-8")),
        "category_thresholds": json.loads((directory / BUNDLE_ASSET_FILENAMES["category_thresholds"]).read_text(encoding="utf-8")),
    }


def score_to_category(score: float, thresholds: list) -> int:
    """0=Low .. 4=Extreme, via calibrated cutpoints - no if/elif branching
    over physical quantities, just a sorted-cutpoint lookup over the
    already-computed continuous score."""
    category = 0
    for cutpoint in thresholds:
        if score >= cutpoint:
            category += 1
        else:
            break
    return min(category, 4)
