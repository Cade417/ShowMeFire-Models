"""
Advisory-only cross-check: does the model's out-of-fold predicted spread
rate rank higher on real fire-occurrence dates/locations near a station
than on an average day? This is a confidence signal, never a training
target and never a promotion gate for this model family (see
docs/fire_weather_ml_plan.md's "why Rothermel-as-label, not occurrence-as-
label" section) - `evaluate.py` always reports this gate's `status` as
`deferred`, whatever this module returns.

Requires real date overlap between `paths.FIRE_LABELS_DIR`'s fire-
occurrence records and the panel being evaluated. As of this writing there
is NONE - `fire_labels_20260808.csv` covers 2011-01-01 through
2020-12-31, while the real historical panel (`historical_panel.py`) covers
2025-07-14 through 2026-08-02 (the fuel-moisture station archive doesn't
reach back to when fire_labels was last exported, and fire_labels hasn't
been re-exported since). This is a genuine, checked data-availability gap,
not a bug - `check_date_overlap` measures it explicitly rather than the
ranking logic silently returning nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Same spirit as risk_fusion/labels.py's cause filter (exclude causes that
# aren't a genuine weather-driven wildfire signal) - independently applied
# here, not imported. "unknown" is additionally excluded: unlike
# risk_fusion's occurrence-count model (where an undetermined cause still
# counts as *a* fire), this advisory check is specifically asking "did
# weather-driven fire behavior risk track real wildfire risk," which an
# unverified-cause record can't cleanly answer either way.
EXCLUDED_CAUSE_CATEGORIES = {"debris_burn", "unknown", "prescribed", "agricultural"}

MATCH_RADIUS_KM = 25.0
MATCH_WINDOW_HOURS = 6.0
EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def check_date_overlap(fire_labels_occurred_at: pd.Series, panel_valid_time: pd.Series) -> Dict:
    """Measures the actual date-range overlap (or lack of one) - never assumed."""
    labels_start, labels_end = fire_labels_occurred_at.min(), fire_labels_occurred_at.max()
    panel_start, panel_end = panel_valid_time.min(), panel_valid_time.max()
    overlap_start = max(labels_start, panel_start)
    overlap_end = min(labels_end, panel_end)
    has_overlap = overlap_start <= overlap_end
    return {
        "has_overlap": bool(has_overlap),
        "fire_labels_range": [str(labels_start), str(labels_end)],
        "panel_range": [str(panel_start), str(panel_end)],
        "overlap_range": [str(overlap_start), str(overlap_end)] if has_overlap else None,
    }


def compute_occurrence_ranking(
    panel: pd.DataFrame,
    predictions: pd.Series,
    fire_labels_path: Path,
) -> Dict:
    """
    `predictions` must be indexed like `panel` (out-of-fold candidate
    predictions, not the physical label itself - this checks the MODEL's
    output, not just the label it was trained on). Returns `{available:
    False, reason: ...}` if there's no real date overlap or no matched
    fire-station pairs; otherwise a percentile-rank statistic (mean
    percentile of the predicted spread rate among that station's own
    predictions, at the times/stations nearest a real fire) - a value near
    50% means no relationship, higher means the model tended to predict
    elevated risk near real fires.
    """
    fire_labels = pd.read_csv(fire_labels_path)
    fire_labels["occurred_at"] = pd.to_datetime(fire_labels["occurred_at"])
    fire_labels = fire_labels[~fire_labels["cause_category"].isin(EXCLUDED_CAUSE_CATEGORIES)]

    panel_valid_time = pd.to_datetime(panel["valid_time"])
    overlap = check_date_overlap(fire_labels["occurred_at"], panel_valid_time)
    if not overlap["has_overlap"]:
        return {"available": False, "reason": "no date overlap between fire_labels and the panel", **overlap}

    overlap_start, overlap_end = pd.to_datetime(overlap["overlap_range"][0]), pd.to_datetime(overlap["overlap_range"][1])
    fire_labels = fire_labels[fire_labels["occurred_at"].between(overlap_start, overlap_end)]
    if fire_labels.empty:
        return {"available": False, "reason": "no fire events (after cause filtering) fall in the overlap window", **overlap}

    if not {"lat", "lon"}.issubset(panel.columns):
        return {"available": False, "reason": "panel has no lat/lon columns to match stations to fire locations", **overlap}

    percentiles = []
    for station_id, station_rows in panel.groupby("station_id"):
        station_rows = station_rows.assign(_valid_time=panel_valid_time.loc[station_rows.index])
        station_predictions = predictions.reindex(station_rows.index).dropna()
        if len(station_predictions) < 10:
            continue
        rank_pct = station_predictions.rank(pct=True)

        station_lat, station_lon = station_rows["lat"].iloc[0], station_rows["lon"].iloc[0]
        distance_km = _haversine_km(
            fire_labels["latitude"].to_numpy(), fire_labels["longitude"].to_numpy(),
            np.full(len(fire_labels), station_lat), np.full(len(fire_labels), station_lon),
        )
        nearby_fires = fire_labels[distance_km <= MATCH_RADIUS_KM]
        if nearby_fires.empty:
            continue

        station_times = station_rows.loc[station_predictions.index, "_valid_time"]
        for fire_time in nearby_fires["occurred_at"]:
            time_delta_hours = (station_times - fire_time).abs().dt.total_seconds() / 3600.0
            nearest_index = time_delta_hours.idxmin()
            if time_delta_hours.loc[nearest_index] > MATCH_WINDOW_HOURS:
                continue
            percentiles.append(float(rank_pct.loc[nearest_index]))

    if not percentiles:
        return {"available": False, "reason": "no station was within match radius/window of any filtered fire event", **overlap}

    return {
        "available": True,
        "matched_fire_station_pairs": len(percentiles),
        "mean_prediction_percentile": float(np.mean(percentiles)),
        "null_expectation": 0.5,
        **overlap,
    }
