"""
Offline evidence report for the fire_weather_ml candidate, mirroring
risk_fusion/evaluate_risk_fusion.py's explicit-status-per-gate shape (no
gate silently omitted: "pass"/"fail" where checkable in this increment,
"deferred" for what needs a later phase, "not_applicable" for what needs
machinery this increment doesn't build).

Primary gates check the model's actual job: predicting the Rothermel-
computed physical label on held-out data, beating a naive baseline that
only sees instantaneous weather (no KBDI/GDD memory) - this is where the
model's ML value-add (vs. just running Rothermel directly) has to show up,
since a model with equal accuracy but no speed/scale advantage over the
physics calculation itself wouldn't be worth serving.

Secondary gate (`fire_occurrence_ranking_advisory`) is a real fire-
occurrence/observed-fire-behavior cross-check but is *always* reported as
"deferred" or "not_applicable" in its status (never "pass"/"fail") - it can
inform confidence but must never gate promotion, per this model family's
explicit design decision not to train or gate against occurrence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from fire_weather_ml import contract, model_bundle
from fire_weather_ml.features import FEATURE_COLUMNS

POLICY_VERSION = "fire-weather-ml-evaluation-policy-v1"
MIN_TEST_ROWS = 200
MIN_R2_VS_BASELINE_IMPROVEMENT = 0.05  # candidate must explain at least 5 more percentage points of variance than the naive baseline

# Features the naive baseline is allowed to see: current-instant weather and
# terrain only - no KBDI/GDD memory. If the candidate can't beat this, the
# extra memory features aren't earning their keep.
BASELINE_FEATURE_COLUMNS = tuple(c for c in FEATURE_COLUMNS if c not in ("kbdi", "gdd_accum"))


def _fit_and_score_fold(train: pd.DataFrame, test: pd.DataFrame, feature_columns) -> Dict:
    bundle = model_bundle.fit(train, feature_columns=feature_columns)
    predictions = model_bundle.score(test.dropna(subset=[model_bundle.DEFAULT_LABEL_COLUMN]), bundle)
    truth = test.loc[predictions.index, model_bundle.DEFAULT_LABEL_COLUMN]
    return {
        "mae": float(mean_absolute_error(truth, predictions)),
        "r2": float(r2_score(truth, predictions)),
        "test_rows": int(len(predictions)),
    }


def build_report(panel: pd.DataFrame) -> Dict:
    panel = contract.add_split_columns(panel)
    labeled = panel.dropna(subset=[model_bundle.DEFAULT_LABEL_COLUMN])

    candidate_folds, baseline_folds = [], []
    for train_index, test_index in contract.crossfit_indices(panel):
        train, test = panel.loc[train_index], panel.loc[test_index]
        if train[model_bundle.DEFAULT_LABEL_COLUMN].notna().sum() < 1 or test[model_bundle.DEFAULT_LABEL_COLUMN].notna().sum() < 1:
            continue
        candidate_folds.append(_fit_and_score_fold(train, test, FEATURE_COLUMNS))
        baseline_folds.append(_fit_and_score_fold(train, test, BASELINE_FEATURE_COLUMNS))

    candidate_mae = float(np.mean([f["mae"] for f in candidate_folds])) if candidate_folds else None
    candidate_r2 = float(np.mean([f["r2"] for f in candidate_folds])) if candidate_folds else None
    baseline_mae = float(np.mean([f["mae"] for f in baseline_folds])) if baseline_folds else None
    baseline_r2 = float(np.mean([f["r2"] for f in baseline_folds])) if baseline_folds else None

    total_rows = int(len(labeled))
    total_episodes = int(panel["episode_id"].nunique())

    gates = [
        {"name": "sufficient_test_rows", "status": "pass" if total_rows >= MIN_TEST_ROWS else "fail",
         "total_rows": total_rows, "minimum_required": MIN_TEST_ROWS},
        {"name": "beats_naive_weather_only_baseline", "status": (
            "pass" if (candidate_r2 is not None and baseline_r2 is not None
                       and candidate_r2 - baseline_r2 >= MIN_R2_VS_BASELINE_IMPROVEMENT) else "fail"
        ), "candidate_r2": candidate_r2, "baseline_r2": baseline_r2,
         "candidate_mae": candidate_mae, "baseline_mae": baseline_mae},
        {"name": "fire_occurrence_ranking_advisory", "status": "deferred",
         "note": "Real fire-occurrence/observed-fire-behavior cross-check is future work - see "
                 "docs/fire_weather_ml_plan.md. Explicitly never a pass/fail promotion gate for this "
                 "model family, even once implemented - it's a confidence signal, not the training target."},
        {"name": "emulation_cost_advantage_documented", "status": "deferred",
         "note": "Comparing inference cost/latency against running pyretechnics directly is Phase 3 work, "
                 "once a real historical panel and a trained candidate both exist."},
    ]

    return {
        "policy_version": POLICY_VERSION,
        "model_family": "xgboost_regressor",
        "label_column": model_bundle.DEFAULT_LABEL_COLUMN,
        "advisory_only": True,
        "row_count": total_rows,
        "total_episodes": total_episodes,
        "candidate_folds": candidate_folds,
        "baseline_folds": baseline_folds,
        "scores": {"candidate_mae": candidate_mae, "candidate_r2": candidate_r2,
                   "baseline_mae": baseline_mae, "baseline_r2": baseline_r2},
        "gates": gates,
        "overall_pass": all(g["status"] in ("pass", "deferred", "not_applicable") for g in gates),
    }


def main():
    import argparse
    import sys

    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT))
    import paths

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=paths.FIRE_WEATHER_ML_DIR / "station_panel.csv")
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "fire_weather_ml_offline_evaluation.json")
    args = parser.parse_args()

    panel = pd.read_csv(args.panel)
    report = build_report(panel)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    failing = [g["name"] for g in report["gates"] if g["status"] == "fail"]
    print(f"overall_pass={report['overall_pass']} failing_gates={failing}")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
