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
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from fire_weather_ml import contract, emulation_cost, model_bundle, occurrence_crosscheck
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
    labeled_test = test.dropna(subset=[model_bundle.DEFAULT_LABEL_COLUMN])
    predictions = model_bundle.score(labeled_test, bundle)
    truth = labeled_test.loc[predictions.index, model_bundle.DEFAULT_LABEL_COLUMN]
    return {
        "mae": float(mean_absolute_error(truth, predictions)),
        "r2": float(r2_score(truth, predictions)),
        "test_rows": int(len(predictions)),
        "predictions": predictions,
    }


def build_report(panel: pd.DataFrame, fire_labels_path: Path | None = None) -> Dict:
    panel = contract.add_split_columns(panel)
    labeled = panel.dropna(subset=[model_bundle.DEFAULT_LABEL_COLUMN])

    candidate_folds, baseline_folds = [], []
    oof_predictions = []
    for train_index, test_index in contract.crossfit_indices(panel):
        train, test = panel.loc[train_index], panel.loc[test_index]
        if train[model_bundle.DEFAULT_LABEL_COLUMN].notna().sum() < 1 or test[model_bundle.DEFAULT_LABEL_COLUMN].notna().sum() < 1:
            continue
        candidate_fold = _fit_and_score_fold(train, test, FEATURE_COLUMNS)
        baseline_folds.append(_fit_and_score_fold(train, test, BASELINE_FEATURE_COLUMNS))
        oof_predictions.append(candidate_fold.pop("predictions"))
        candidate_folds.append(candidate_fold)

    candidate_mae = float(np.mean([f["mae"] for f in candidate_folds])) if candidate_folds else None
    candidate_r2 = float(np.mean([f["r2"] for f in candidate_folds])) if candidate_folds else None
    baseline_mae = float(np.mean([f["mae"] for f in baseline_folds])) if baseline_folds else None
    baseline_r2 = float(np.mean([f["r2"] for f in baseline_folds])) if baseline_folds else None
    oof_predictions_series = pd.concat(oof_predictions) if oof_predictions else pd.Series(dtype=float)

    total_rows = int(len(labeled))
    total_episodes = int(panel["episode_id"].nunique())

    # A single candidate fit on ALL labeled rows, for the emulation-cost
    # timing and (if usable) the occurrence cross-check - not one of the
    # held-out cross-validation folds above, since those are deliberately
    # partial fits for unbiased accuracy measurement, not the best model
    # this data can produce.
    full_candidate_bundle = model_bundle.fit(labeled, feature_columns=FEATURE_COLUMNS) if total_rows else None

    if fire_labels_path is None:
        REPO_ROOT = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(REPO_ROOT))
        import paths
        candidates = sorted(paths.FIRE_LABELS_DIR.glob("*.csv"))
        fire_labels_path = candidates[-1] if candidates else None

    if fire_labels_path is None or not Path(fire_labels_path).is_file():
        occurrence_result = {"available": False, "reason": "no fire_labels CSV found under paths.FIRE_LABELS_DIR"}
    elif oof_predictions_series.empty:
        occurrence_result = {"available": False, "reason": "no out-of-fold candidate predictions to check (no complete folds)"}
    else:
        occurrence_result = occurrence_crosscheck.compute_occurrence_ranking(
            panel, oof_predictions_series, Path(fire_labels_path))

    cost_result = (
        emulation_cost.measure_emulation_speedup(panel, full_candidate_bundle)
        if full_candidate_bundle is not None
        else {"available": False, "reason": "no labeled rows to fit a candidate for timing"}
    )

    gates = [
        {"name": "sufficient_test_rows", "status": "pass" if total_rows >= MIN_TEST_ROWS else "fail",
         "total_rows": total_rows, "minimum_required": MIN_TEST_ROWS},
        {"name": "beats_naive_weather_only_baseline", "status": (
            "pass" if (candidate_r2 is not None and baseline_r2 is not None
                       and candidate_r2 - baseline_r2 >= MIN_R2_VS_BASELINE_IMPROVEMENT) else "fail"
        ), "candidate_r2": candidate_r2, "baseline_r2": baseline_r2,
         "candidate_mae": candidate_mae, "baseline_mae": baseline_mae},
        {"name": "fire_occurrence_ranking_advisory", "status": "deferred", "result": occurrence_result,
         "note": "Always deferred regardless of result - a confidence signal, never a pass/fail promotion "
                 "gate for this model family (it never trains or gates against occurrence). See "
                 "occurrence_crosscheck.py's module docstring for the current real data-availability gap "
                 "if result.available is False."},
        {"name": "emulation_cost_advantage_documented", "status": "pass" if cost_result.get("available") else "deferred",
         "result": cost_result,
         "note": "Compares the trained candidate's scoring time against running the real Rothermel physics "
                 "(rothermel_labels.py) row-by-row on the same sample - the actual point of an ML emulator."},
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

    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT))
    import paths

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, default=paths.FIRE_WEATHER_ML_DIR / "station_panel.csv")
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "fire_weather_ml_offline_evaluation.json")
    parser.add_argument("--fire-labels", type=Path, default=None,
                        help="Override the fire_labels CSV used for the advisory occurrence cross-check "
                             "(defaults to the newest CSV under paths.FIRE_LABELS_DIR).")
    args = parser.parse_args()

    panel = pd.read_csv(args.panel)
    report = build_report(panel, fire_labels_path=args.fire_labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    failing = [g["name"] for g in report["gates"] if g["status"] == "fail"]
    print(f"overall_pass={report['overall_pass']} failing_gates={failing}")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
