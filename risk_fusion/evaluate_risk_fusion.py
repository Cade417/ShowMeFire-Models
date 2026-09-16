"""
Offline evidence report for the fire_risk_fusion GLM-only v1 candidate
(monthly-baseline + fast-weather residual) - the out-of-fold comparisons
and promotion-policy gates from the project plan's Stage 3, run against
the real 2014-2020 labeled panel.

Not every gate in the full project plan is checkable from a GLM-only,
no-rule-Monte-Carlo, pre-prospective-shadow increment. Every gate below
is reported with an explicit status - "pass"/"fail" (checkable here),
"deferred" (needs a later phase, e.g. prospective shadow), or
"not_applicable" (needs machinery this increment doesn't build, e.g.
rule Monte Carlo). None are silently omitted.

Usage:
    python -m risk_fusion.evaluate_risk_fusion
    python -m risk_fusion.evaluate_risk_fusion --output some/other/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths
from risk_fusion import diagnostics, fit_glm, model_bundle, risk_fusion_evidence as evidence
from risk_fusion.risk_fusion_contract import add_split_columns

TRAINING_PANEL_PATH = paths.RISK_FUSION_DIR / "labeled_panel_2014_2020.csv"
DEFAULT_OUTPUT_PATH = paths.REPORTS_DIR / "risk_fusion_offline_evaluation.json"
MIN_EFFECTIVE_EPISODES = 24
MIN_EPISODES_WITH_EVENTS = 8
MIN_OFFICIAL_EVENT_SUPPORT = 300
BOOTSTRAP_PROBABILITY_THRESHOLD = 0.90

POWER_STATEMENT = (
    "The unit of independent information is the synoptic episode (a "
    "contiguous 14-day block), not the county-day row. This evaluation's "
    "episode and event counts (see effective_episodes/official_event_support "
    "below) set the resolution of what a paired bootstrap can detect here - "
    "consult those counts before treating any 'fails to beat baseline X' "
    "result as evidence of no effect, rather than of insufficient power."
)


def _offset_exponent_diagnostics(panel: pd.DataFrame) -> Dict:
    """
    Estimates how steeply log_effort actually enters the mean model
    (fit_glm.fit_effort_exponent) instead of assuming a fixed offset with
    coefficient exactly 1. The real finding on this panel: a fixed-offset
    assumption doesn't hold (coefficient ~1.05, 95% CI excludes 1.0) -
    which is why model_bundle.fit() now scales log_effort by this same
    estimate instead of assuming 1 (see "raw" below for that finding).

    "scaled_self_consistency" re-checks the ALREADY-SCALED column's
    coefficient, which lands on ~1.0 essentially by construction
    (rescaling a covariate by a constant exactly divides its fitted
    coefficient by that constant, for any GLM). This is a regression/
    implementation-consistency check, not an independent re-validation of
    the offset assumption - a failure here would mean a real bug (e.g.
    the wrong exponent applied), not a genuine misspecification finding.
    """
    panel = fit_glm.add_month_dummies(panel)
    raw = fit_glm.fit_effort_exponent(panel, offset_column="log_effort")

    scaled_panel = panel.copy()
    scaled_panel["log_effort_scaled"] = raw["coefficient"] * scaled_panel["log_effort"]
    scaled = fit_glm.fit_effort_exponent(scaled_panel, offset_column="log_effort_scaled")

    return {
        "raw": {**raw, "contains_one": raw["ci_low"] <= 1.0 <= raw["ci_high"]},
        "effort_exponent_applied": raw["coefficient"],
        "scaled_self_consistency": {**scaled, "contains_one": scaled["ci_low"] <= 1.0 <= scaled["ci_high"]},
    }


def build_report(panel: pd.DataFrame) -> Dict:
    panel = add_split_columns(panel)
    panel = fit_glm.add_month_dummies(panel)

    offset_check = _offset_exponent_diagnostics(panel)
    effort_exponent = offset_check["effort_exponent_applied"]
    panel["log_effort_scaled"] = effort_exponent * panel["log_effort"]

    result = fit_glm.crossfit_compare_residual(panel, offset_column="log_effort_scaled")

    y = result["event_count"].to_numpy()
    lam_offset_only = result["lam_offset_only"].to_numpy()
    lam_climatology = result["lam_climatological_doy"].to_numpy()
    lam_candidate = result["lam_climatology_plus_weather"].to_numpy()
    block_ids = result["episode_id"].to_numpy()

    scores = {
        "offset_only": evidence.log_score(y, lam_offset_only),
        "climatological_monthly": evidence.log_score(y, lam_climatology),
        "candidate": evidence.log_score(y, lam_candidate),
    }
    bootstrap_vs_offset_only = evidence.paired_block_bootstrap(y, lam_candidate, lam_offset_only, block_ids)
    bootstrap_vs_climatology = evidence.paired_block_bootstrap(y, lam_candidate, lam_climatology, block_ids)

    binary_consistency = diagnostics.poisson_binary_consistency(
        y, lam_candidate,
        count_model_params=len(fit_glm.MONTH_DUMMY_COLUMNS) + 1 + len(fit_glm.FAST_WEATHER_FEATURES) + 1,
    )
    calibration = diagnostics.calibration_diagnostics(y, lam_candidate)

    total_episodes = int(panel["episode_id"].nunique())
    episodes_with_events = int(panel.loc[panel["event_count"] > 0, "episode_id"].nunique())
    total_events = int(panel["event_count"].sum())

    gates = [
        {"name": "log_score_lower_than_offset_only", "status": "pass" if (
            scores["candidate"] < scores["offset_only"]
            and bootstrap_vs_offset_only["probability_candidate_better"] >= BOOTSTRAP_PROBABILITY_THRESHOLD
        ) else "fail", "point_estimate_delta": bootstrap_vs_offset_only["point_estimate_delta"],
         "probability_candidate_better": bootstrap_vs_offset_only["probability_candidate_better"]},
        {"name": "log_score_lower_than_climatology", "status": "pass" if (
            scores["candidate"] < scores["climatological_monthly"]
            and bootstrap_vs_climatology["probability_candidate_better"] >= BOOTSTRAP_PROBABILITY_THRESHOLD
        ) else "fail", "point_estimate_delta": bootstrap_vs_climatology["point_estimate_delta"],
         "probability_candidate_better": bootstrap_vs_climatology["probability_candidate_better"]},
        {"name": "log_score_lower_than_rule_only", "status": "not_applicable",
         "note": "No rule-Monte-Carlo features are wired into this GLM-only increment's panel."},
        {"name": "log_score_lower_than_redflag_only", "status": "not_applicable",
         "note": "No Red Flag Warning feature is wired into this GLM-only increment's panel."},
        {"name": "poisson_deviance_no_worse", "status": "pass" if (
            evidence.poisson_deviance(y, lam_candidate) <= evidence.poisson_deviance(y, lam_climatology)
        ) else "fail", "note": "Compared against climatology, not rule_only (unavailable this increment)."},
        {"name": "calibration_slope", "status": "pass" if 0.85 <= calibration["calibration_slope"] <= 1.15 else "fail",
         "value": calibration["calibration_slope"]},
        {"name": "reliability_max_bin_gap", "status": "pass" if calibration["reliability_max_bin_gap"] <= 0.05 else "fail",
         "value": calibration["reliability_max_bin_gap"]},
        {"name": "poisson_binary_consistency", "status": "pass" if binary_consistency["max_decile_gap"] <= 0.05 else "fail",
         "value": binary_consistency},
        {"name": "count_family_justified", "status": "pass" if binary_consistency["recommended_count_family"] == "poisson" else "fail",
         "note": f"binary_consistency recommends {binary_consistency['recommended_count_family']}; contract declares poisson."},
        {"name": "offset_identifiability", "status": "pass" if (
            offset_check["scaled_self_consistency"]["contains_one"]
            and 0.0 < offset_check["effort_exponent_applied"] < 2.0
        ) else "fail", "value": offset_check,
         "note": "log_effort's coefficient is now ESTIMATED (model_bundle.fit's effort_exponent), not assumed "
                 "to be 1 - see 'raw' for the finding that motivated this. This checks the applied scaling is "
                 "self-consistent and the estimated exponent is within a sane range, not that the raw "
                 "coefficient equals 1 (it doesn't have to anymore)."},
        {"name": "no_unverified_labels", "status": "assumed_from_label_pipeline",
         "note": "labels.py's build_labels() aggregates only official_source_confirmed-tier events into event_count; not re-verified from raw records here."},
        {"name": "tier_recorded", "status": "assumed_from_label_pipeline",
         "note": "label_min_tier=official_source_confirmed is enforced upstream in labels.py, not re-derivable from this panel alone."},
        {"name": "cause_filter_applied", "status": "assumed_from_label_pipeline",
         "note": "labels.py's apply_cause_filter excludes EXCLUDED_CAUSE_CATEGORIES={'prescribed','agricultural'} before aggregation."},
        {"name": "duplicate_collapse_rate", "status": "assumed_from_label_pipeline",
         "note": "Verified against real FPA-FOD duplicates during label ingestion (see labels.py MAX_DUPLICATE_COLLAPSE_RATE); not recomputed here."},
        {"name": "coverage_complete", "status": "not_applicable",
         "note": "effort.py does not yet track day-level source_coverage_fraction (documented gap in effort.py's module docstring)."},
        {"name": "guard_active_row_fraction", "status": "pass",
         "note": "model_family == 'glm' - the policy's documented exception, no guard/GBM residual in this increment."},
        {"name": "mirror_integrity", "status": "not_applicable", "note": "No contract-mirrored module changed in this increment."},
        {"name": "rule_mc_parity", "status": "not_applicable", "note": "No rule Monte Carlo in this GLM-only increment."},
        {"name": "rule_mc_determinism", "status": "not_applicable", "note": "No rule Monte Carlo in this GLM-only increment."},
        {"name": "public_path_unchanged", "status": "not_applicable", "note": "No shadow-serving change in this increment."},
        {"name": "feature_extrapolation_fraction", "status": "deferred",
         "note": "Needs an accumulated live-serving population to compare against training ranges - not yet accumulated."},
        {"name": "auxiliary_sign_agreement", "status": "not_applicable", "note": "No admin_reviewed auxiliary head fit in this increment."},
        {"name": "effective_episodes", "status": "pass" if (
            total_episodes >= MIN_EFFECTIVE_EPISODES and episodes_with_events >= MIN_EPISODES_WITH_EVENTS
        ) else "fail", "total_episodes": total_episodes, "episodes_with_events": episodes_with_events},
        {"name": "official_event_support", "status": "pass" if total_events >= MIN_OFFICIAL_EVENT_SUPPORT else "fail",
         "total_events": total_events, "note": "Prospective-window portion of this gate is not_applicable - no prospective window yet."},
        {"name": "minimum_prospective_days", "status": "deferred", "note": "Requires the (not yet started) prospective shadow phase."},
        {"name": "power_statement_present", "status": "pass"},
    ]

    return {
        "policy_version": evidence.POLICY_VERSION,
        "policy_sha256": evidence.policy_sha256(),
        "model_family": "glm",
        "count_family": "poisson",
        "advisory_only": True,
        "row_count": int(len(panel)),
        "effort_exponent": effort_exponent,
        "scores": scores,
        "gates": gates,
        "overall_pass": all(g["status"] in ("pass", "deferred", "not_applicable", "assumed_from_label_pipeline") for g in gates),
        "power_statement": POWER_STATEMENT,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    print(f"Loading training panel from {TRAINING_PANEL_PATH}...")
    panel = model_bundle.load_training_panel(TRAINING_PANEL_PATH)

    print("Running out-of-fold comparisons and gates...")
    report = build_report(panel)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    failing = [g["name"] for g in report["gates"] if g["status"] == "fail"]
    print(f"overall_pass={report['overall_pass']} failing_gates={failing}")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
