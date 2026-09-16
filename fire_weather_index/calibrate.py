"""
Calibrates fire_weather_index's category cutpoints and VALIDATES the score
against historical fire occurrence - it does not fit factor weights to any
label. See this package's __init__.py for why that distinction matters.

Two separate things happen here, and only here:
1. Category cutpoints: chosen as PERCENTILES of the score's own historical
   distribution (a standard technique for building a fire-weather index -
   see e.g. how NFDRS/ERC percentile classes are set), not derived from
   fire counts at all.
2. Monotonicity check: confirms that mean historical fire-occurrence rate
   increases from Low to Extreme under those percentile-based categories.
   This is what actually justifies the score - a percentile split that
   didn't track real outcomes would be useless even if statistically tidy.

Usage:
    python -m fire_weather_index.calibrate
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths
from fire_weather_index import factors, model_bundle
from risk_fusion.labels import build_labels

# Default category proportions: Low is by far the most common day, Extreme
# the rarest - matches the shape fire-weather categorical schemes generally
# take (see e.g. the live rule's own historical category distribution).
# FIRST_PASS: not derived from this project's own historical distribution
# yet, just a reasonable starting split.
DEFAULT_CATEGORY_PERCENTILES = (40, 70, 85, 95)


def score_panel(panel: pd.DataFrame, weights: Dict[str, float] = None) -> pd.DataFrame:
    """Adds a `score` column (None where every factor was missing)."""
    panel = panel.copy()
    scores = []
    for _, row in panel.iterrows():
        row_factors = factors.compute_factors(row.to_dict())
        scores.append(factors.compute_score(row_factors, weights))
    panel["score"] = scores
    return panel


def percentile_thresholds(scores: np.ndarray, percentiles=DEFAULT_CATEGORY_PERCENTILES) -> List[float]:
    valid = scores[np.isfinite(scores)]
    if valid.size == 0:
        raise ValueError("no valid scores to calibrate thresholds from")
    thresholds = [float(np.percentile(valid, p)) for p in percentiles]
    for i in range(1, len(thresholds)):
        if thresholds[i] <= thresholds[i - 1]:
            thresholds[i] = thresholds[i - 1] + 1e-6
    return thresholds


def check_monotonicity(scored_panel: pd.DataFrame, labeled_counts: pd.DataFrame, thresholds: List[float]) -> Dict:
    """Joins scored county-days against historical event counts (LEFT join -
    the weather panel is dense, so no match means a genuine zero, same
    principle as risk_fusion/join_panel.py) and reports mean event rate per
    category. Monotonic non-decreasing mean rate across Low->Extreme is the
    pass condition - it is NOT required to be strictly increasing (small
    samples in high categories are noisy), just non-decreasing."""
    merged = scored_panel.merge(labeled_counts, on=["county_fips", "valid_local_date"], how="left")
    merged["event_count"] = merged["event_count"].fillna(0)
    merged = merged.dropna(subset=["score"])
    merged["category"] = merged["score"].apply(lambda s: model_bundle.score_to_category(s, thresholds))

    by_category = merged.groupby("category")["event_count"].agg(["mean", "count"]).reindex(range(5))
    rates = by_category["mean"].to_numpy()
    valid_rates = rates[~np.isnan(rates)]
    is_monotonic = bool(np.all(np.diff(valid_rates) >= -1e-9))

    return {
        "category_labels": ["Low", "Moderate", "Elevated", "Critical", "Extreme"],
        "mean_event_rate_by_category": {int(cat): (float(rate) if not np.isnan(rate) else None) for cat, rate in enumerate(rates)},
        "row_count_by_category": {int(cat): int(count) for cat, count in by_category["count"].items()},
        "is_monotonic_non_decreasing": is_monotonic,
        "total_rows_joined": int(len(merged)),
        "rows_with_any_event": int((merged["event_count"] > 0).sum()),
    }


def run(panel_path: Path = None, labels_csv: Path = None, labels_manifest: Path = None) -> Dict:
    panel_path = panel_path or paths.FIRE_WEATHER_INDEX_PANEL
    labels_csv = labels_csv or (paths.FIRE_LABELS_DIR / "fire_labels.csv")
    labels_manifest = labels_manifest or (paths.FIRE_LABELS_DIR / "fire_labels_manifest.json")

    panel = pd.read_csv(panel_path, dtype={"county_fips": str})
    scored = score_panel(panel)
    valid_scores = scored["score"].dropna().to_numpy(dtype=float)
    thresholds = percentile_thresholds(valid_scores)

    report = {
        "panel_rows": int(len(panel)),
        "scored_rows": int(len(valid_scores)),
        "thresholds": thresholds,
        "score_percentiles_used": list(DEFAULT_CATEGORY_PERCENTILES),
    }

    if labels_csv.exists() and labels_manifest.exists():
        label_result = build_labels(labels_csv, labels_manifest)
        report["monotonicity"] = check_monotonicity(scored, label_result["primary_counts"], thresholds)
        report["label_manifest_sha256"] = label_result["manifest"].get("csv_sha256")
    else:
        report["monotonicity"] = {"status": "skipped", "reason": f"no fire labels export found at {labels_csv}"}

    return report


def main():
    report = run()
    print(f"panel_rows={report['panel_rows']} scored_rows={report['scored_rows']}")
    print(f"thresholds={report['thresholds']}")
    monotonicity = report["monotonicity"]
    if monotonicity.get("status") == "skipped":
        print(f"monotonicity check SKIPPED: {monotonicity['reason']}")
    else:
        print(f"mean_event_rate_by_category={monotonicity['mean_event_rate_by_category']}")
        print(f"is_monotonic_non_decreasing={monotonicity['is_monotonic_non_decreasing']}")


if __name__ == "__main__":
    main()
