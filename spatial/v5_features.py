"""Causal rain-aware features and regime labels for V5."""
from __future__ import annotations

import numpy as np
import pandas as pd

from spatial.v4_features import FEATURES as V4_FEATURES, add_v4_features

EXTRA_FEATURES = [
    "precip_6h_mm", "precip_24h_mm", "precip_6h_partial", "precip_24h_partial",
    "precip_intensity_mmph", "rain_occurrence_interval", "active_rain_indicator",
    "post_rain_3h_indicator", "precip_missing_indicator", "precip_quality_issue",
    "forecast_drying_rate", "post_rain_drying_interaction",
]
FEATURES = V4_FEATURES + EXTRA_FEATURES
BASE_FEATURES = [name for name in FEATURES if name not in {
    "incumbent_base_fm", "physics_minus_incumbent", "rain_physics_minus_base"
}]
SPECIALIST_FEATURES = [
    "incumbent_base_fm", "rain_physics_minus_base", "lead_hour", "summer_indicator",
    "hrrr_precip_accum_mm", "hrrr_precip_increment_mm", "precip_3h_mm", "precip_6h_mm",
    "precip_24h_mm", "precip_intensity_mmph", "precip_duration_hours",
    "hours_since_forecast_rain", "active_rain_indicator", "post_rain_3h_indicator",
    "precip_partial_window_flag", "precip_quality_issue", "vpd_kpa", "forecast_emc",
    "hot_dry_interaction", "initial_emc_gap", "temp_lead_change_c", "rh_lead_change",
    "wind_lead_change_ms", "forecast_drying_rate", "post_rain_drying_interaction",
    "valid_hour_sin", "valid_hour_cos", "valid_doy_sin", "valid_doy_cos", "lat", "lon",
]


def _window_sum(times, values, position, hours):
    current = times.iloc[position]
    if pd.isna(current):
        start = max(0, position - int(hours) + 1)
        return float(values.iloc[start:position + 1].sum()), float(position - start + 1)
    cutoff = current - pd.Timedelta(hours=hours)
    selected = (times > cutoff) & (times <= current)
    duration = 0.0 if not selected.any() else (current - times[selected].min()).total_seconds() / 3600 + 1.0
    return float(values[selected].sum()), float(min(hours, duration))


def add_v5_features(frame, base_prediction, physics_variant="legacy"):
    result = add_v4_features(frame, base_prediction, physics_variant)
    ordered = result.sort_values(["run_id", "station_id", "lead_hour"]).copy()
    for name in EXTRA_FEATURES:
        ordered[name] = 0.0
    for _, indices in ordered.groupby(["run_id", "station_id"], sort=False).groups.items():
        indices = list(indices)
        group = ordered.loc[indices]
        times = pd.to_datetime(group.get("valid_time"), utc=True, errors="coerce").reset_index(drop=True)
        increment = group.hrrr_precip_increment_mm.fillna(0).clip(lower=0).reset_index(drop=True)
        interval = group.get("precip_interval_hours", pd.Series(1.0, index=group.index)).fillna(0).clip(lower=0).reset_index(drop=True)
        rh = group.hrrr_rh.reset_index(drop=True)
        for position, row_index in enumerate(indices):
            six, six_duration = _window_sum(times, increment, position, 6)
            day, day_duration = _window_sum(times, increment, position, 24)
            ordered.at[row_index, "precip_6h_mm"] = six
            ordered.at[row_index, "precip_24h_mm"] = day
            ordered.at[row_index, "precip_6h_partial"] = float(six_duration < 6)
            ordered.at[row_index, "precip_24h_partial"] = float(day_duration < 24)
            duration = float(interval.iloc[position])
            amount = float(increment.iloc[position])
            ordered.at[row_index, "precip_intensity_mmph"] = amount / duration if duration > 0 else 0.0
            ordered.at[row_index, "rain_occurrence_interval"] = float(amount > 0)
            ordered.at[row_index, "active_rain_indicator"] = float(amount > 0.1)
            age = float(group.hours_since_forecast_rain.iloc[position])
            ordered.at[row_index, "post_rain_3h_indicator"] = float(amount <= 0.1 and age <= 3)
            available = float(group.precip_available.iloc[position])
            ordered.at[row_index, "precip_missing_indicator"] = float(available <= 0)
            reset = float(group.precip_reset_flag.iloc[position])
            partial = float(group.precip_partial_window_flag.iloc[position])
            ordered.at[row_index, "precip_quality_issue"] = float(reset > 0 or available <= 0 or partial > 0)
            rh_change = 0.0 if position == 0 else float(rh.iloc[position] - rh.iloc[position - 1])
            drying_rate = max(0.0, -rh_change) / max(duration, 1.0)
            ordered.at[row_index, "forecast_drying_rate"] = drying_rate
            ordered.at[row_index, "post_rain_drying_interaction"] = drying_rate * float(amount <= 0.1 and age <= 3)
    for name in EXTRA_FEATURES:
        result.loc[ordered.index, name] = ordered[name].to_numpy()
    return result


def regime_labels(frame):
    valid = pd.to_datetime(frame.valid_time, utc=True, errors="coerce")
    summer = valid.dt.month.isin([6, 7, 8]).to_numpy()
    active = frame.active_rain_indicator.fillna(0).to_numpy() > 0
    post = frame.post_rain_3h_indicator.fillna(0).to_numpy() > 0
    labels = np.full(len(frame), "other", dtype=object)
    labels[summer] = "summer_dry"
    labels[post] = "post_rain"
    labels[active] = "active_rain"
    return labels
