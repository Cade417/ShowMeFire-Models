"""Canonical fire-danger contract and cross-repository drift check."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULE_PATH = ROOT / "rules" / "fire_danger_rules.json"
RULE_SPEC = json.loads(RULE_PATH.read_text(encoding="utf-8"))
RULE_SPEC_VERSION = RULE_SPEC["version"]
RULE_SPEC_SHA256 = hashlib.sha256(RULE_PATH.read_bytes()).hexdigest()
CATEGORY_LABELS = tuple(RULE_SPEC["categories"])


def category(fm, rh, wind_kts):
    try:
        import math
        if any(value is None or not math.isfinite(float(value)) for value in (fm, rh, wind_kts)):
            return None
    except (TypeError, ValueError):
        return None
    fm, rh, wind = map(float, (fm, rh, wind_kts)); t = RULE_SPEC["thresholds"]
    if fm >= t["low_fm"]: return 0
    if fm < t["extreme_fm"] and rh < t["extreme_rh"] and wind >= t["extreme_wind"]: return 4
    if fm < t["elevated_fm"] and rh < t["critical_rh"] and wind >= t["critical_wind"]: return 3
    if fm < t["elevated_fm"] and ((rh < t["elevated_rh"] and wind >= t["elevated_wind"]) or
                                  (rh < t["elevated_very_dry_rh"] and wind >= t["elevated_very_dry_wind"])): return 2
    if fm < t["low_fm"] and (rh < t["moderate_rh"] or wind >= t["moderate_wind"]): return 1
    return 0


def check_api_copy(api_path=None):
    api_path = Path(api_path or ROOT.parent / "api" / "core" / "fire_danger_rules.json")
    if not api_path.exists() or api_path.read_bytes() != RULE_PATH.read_bytes():
        raise RuntimeError(f"Canonical fire-danger rule drift: {api_path}")
    return RULE_SPEC_SHA256


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--api-path", type=Path); arguments = parser.parse_args()
    print(check_api_copy(arguments.api_path))
