"""
Joins the weather-only county-day panel (build_county_days.py) with the
deduped label counts (labels.py) into the first real labeled county-day
dataset for fire_risk_fusion.

LEFT JOIN weather -> labels, filling missing event_count with a genuine
zero: the weather panel is dense over every (county, day) that had a
cached HRRR run, so a day with no matching label row is a real "zero
fires observed", not a missing value - exactly the "no invented
negatives" property the project plan calls out as the reason for this
formulation. Label rows with no matching weather row (dates outside the
HRRR archive's coverage - see the module docstring in
scripts/backfill_hrrr.py for the 2014-09 start) are dropped, and how many
were dropped is always reported rather than silently discarded.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd


def build_labeled_panel(weather_panel: pd.DataFrame, primary_counts: pd.DataFrame) -> Dict:
    """
    weather_panel: one row per (county_fips, valid_local_date) with weather
        features, no target column (build_county_days.py's output).
    primary_counts: sparse (county_fips, valid_local_date, event_count, ...)
        from labels.to_county_day_counts().

    Returns {"panel": DataFrame, "labels_dropped_no_weather_match": int,
    "weather_days_with_fire": int, "weather_days_total": int}.
    """
    weather_panel = weather_panel.copy()
    weather_panel["county_fips"] = weather_panel["county_fips"].astype(str)
    weather_panel["valid_local_date"] = weather_panel["valid_local_date"].astype(str)

    if primary_counts.empty:
        panel = weather_panel.copy()
        panel["event_count"] = 0
        panel["acres_sum"] = 0.0
        panel["acres_max"] = 0.0
        return {
            "panel": panel,
            "labels_dropped_no_weather_match": 0,
            "weather_days_with_fire": 0,
            "weather_days_total": len(panel),
        }

    counts = primary_counts.copy()
    counts["county_fips"] = counts["county_fips"].astype(str)
    counts["valid_local_date"] = counts["valid_local_date"].astype(str)

    weather_keys = set(zip(weather_panel["county_fips"], weather_panel["valid_local_date"]))
    label_keys = set(zip(counts["county_fips"], counts["valid_local_date"]))
    dropped = len(label_keys - weather_keys)

    panel = weather_panel.merge(counts, on=["county_fips", "valid_local_date"], how="left")
    panel["event_count"] = panel["event_count"].fillna(0).astype(int)
    panel["acres_sum"] = panel["acres_sum"].fillna(0.0)
    panel["acres_max"] = panel["acres_max"].fillna(0.0)

    return {
        "panel": panel,
        "labels_dropped_no_weather_match": dropped,
        "weather_days_with_fire": int((panel["event_count"] > 0).sum()),
        "weather_days_total": len(panel),
    }
