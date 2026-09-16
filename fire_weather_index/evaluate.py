"""
Offline evidence report for fire_weather_index. Mirrors risk_fusion's
evaluate_risk_fusion.py shape (named gates, each with an explicit status -
pass/fail/deferred - nothing silently omitted), but the gates here fit
what this model actually is: a calibrated numeric score, not a fitted
count model. There is no held-out log-score/calibration-slope gate because
there is no fitted distribution to score against - the monotonicity check
in calibrate.py is the closest analogue, and IS one of the gates below.

Usage:
    python -m fire_weather_index.evaluate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths
from fire_weather_index import calibrate

MIN_COVERAGE_FOR_PASS = 0.5  # a source needs to cover at least half the panel before its coverage gate can "pass"


def _coverage_gate(name: str, fraction: float) -> dict:
    if fraction >= MIN_COVERAGE_FOR_PASS:
        return {"name": name, "status": "pass", "value": fraction}
    return {"name": name, "status": "deferred",
            "reason": f"{fraction:.1%} coverage - still building history, not yet enough to weight this source's "
                      f"contribution with confidence (threshold: {MIN_COVERAGE_FOR_PASS:.0%})",
            "value": fraction}


def build_report(panel_path: Path = None) -> dict:
    panel_path = panel_path or paths.FIRE_WEATHER_INDEX_PANEL
    calibration = calibrate.run(panel_path=panel_path)

    import pandas as pd
    panel = pd.read_csv(panel_path, dtype={"county_fips": str})

    gates = []
    scored_fraction = calibration["scored_rows"] / calibration["panel_rows"] if calibration["panel_rows"] else 0.0
    gates.append({"name": "factor_availability", "status": "pass" if scored_fraction >= 0.95 else "fail",
                  "value": scored_fraction,
                  "reason": "fraction of county-days with at least one usable factor (denominator > 0 in compute_score)"})

    rrfs_fraction = float(panel["rrfs_available"].mean()) if "rrfs_available" in panel else 0.0
    fv3hires_fraction = float(panel["fv3hires_available"].mean()) if "fv3hires_available" in panel else 0.0
    gates.append(_coverage_gate("rrfs_coverage_fraction", rrfs_fraction))
    gates.append(_coverage_gate("fv3hires_coverage_fraction", fv3hires_fraction))

    monotonicity = calibration["monotonicity"]
    if monotonicity.get("status") == "skipped":
        gates.append({"name": "monotonicity_vs_fire_occurrence", "status": "deferred", "reason": monotonicity["reason"]})
    else:
        gates.append({"name": "monotonicity_vs_fire_occurrence",
                      "status": "pass" if monotonicity["is_monotonic_non_decreasing"] else "fail",
                      "value": monotonicity["mean_event_rate_by_category"]})

    overall_pass = all(gate["status"] in ("pass", "deferred") for gate in gates)
    return {
        "model_family": "fire_weather_index",
        "advisory_only": True,
        "panel_rows": calibration["panel_rows"],
        "scored_rows": calibration["scored_rows"],
        "category_thresholds": calibration["thresholds"],
        "gates": gates,
        "overall_pass": overall_pass,
    }


def main():
    report = build_report()
    output_path = paths.REPORTS_DIR / "fire_weather_index_offline_evaluation.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    for gate in report["gates"]:
        print(f"  [{gate['status']:>8}] {gate['name']}")
    print(f"overall_pass={report['overall_pass']}")


if __name__ == "__main__":
    main()
