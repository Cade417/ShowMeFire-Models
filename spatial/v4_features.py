"""Causal V4 features including precipitation increments and rain physics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from spatial.hybrid_features import FEATURES as V3_FEATURES, add_hybrid_features
from spatial.physics import evolve_fm, evolve_fm_with_rain

RAIN_VARIANTS = {
    "legacy": None,
    "rain25_scale2": (25.0, 2.0),
    "rain30_scale2": (30.0, 2.0),
    "rain30_scale4": (30.0, 4.0),
}
PRECIP_FEATURES = ["hrrr_precip_increment_mm", "precip_reset_flag", "precip_partial_window_flag",
                   "precip_available", "precip_duration_hours", "precip_3h_mm",
                   "hours_since_forecast_rain", "wet_dry_transition"]
EXTRA_FEATURES = [*PRECIP_FEATURES,
                  "rain_physics_fm", "rain_physics_minus_base", "summer_indicator",
                  "hot_dry_interaction", "initial_emc_gap"]
FEATURES = V3_FEATURES + EXTRA_FEATURES
TREE_FEATURES = [name for name in FEATURES if name not in
                 {"incumbent_base_fm", "physics_minus_incumbent", "rain_physics_minus_base"}]


def precipitation_features(frame):
    result = frame.copy(); ordered = result.sort_values(["run_id", "station_id", "lead_hour"])
    group = ordered.groupby(["run_id", "station_id"], sort=False)
    cumulative_name = "hrrr_precip_accum_mm" if "hrrr_precip_accum_mm" in ordered else "hrrr_precip_mm"
    cumulative = ordered[cumulative_name].clip(lower=0)
    if "hrrr_precip_increment_mm" in ordered:
        increment = ordered.hrrr_precip_increment_mm.clip(lower=0).fillna(0)
        reset = ordered.get("precip_reset_flag", pd.Series(0, index=ordered.index)).fillna(0).astype(bool)
    else:
        raw_increment = group[cumulative_name].diff().fillna(cumulative)
        reset = raw_increment < 0; increment = raw_increment.clip(lower=0).fillna(0)
    if "precip_interval_hours" in ordered:
        interval_hours = ordered.precip_interval_hours.clip(lower=0).fillna(0)
    else:
        interval_hours = group.lead_hour.diff().fillna(ordered.lead_hour).clip(lower=0)
    ordered["hrrr_precip_increment_mm"] = increment
    ordered["precip_reset_flag"] = reset.astype(float)
    ordered["precip_partial_window_flag"] = ordered.get("precip_partial_window_flag", interval_hours.ne(1)).fillna(1).astype(float)
    ordered["precip_available"] = ordered.get("precip_available", cumulative.notna()).fillna(0).astype(float)
    raining = increment > 0
    duration_values = pd.Series(0.0, index=ordered.index)
    rolling_values = pd.Series(0.0, index=ordered.index)
    since_values = pd.Series(999.0, index=ordered.index)
    for _, indices in group.groups.items():
        indices = list(indices); rain_age = 999.0; rain_duration = 0.0
        times = pd.to_datetime(ordered.loc[indices, "valid_time"], utc=True, errors="coerce") if "valid_time" in ordered else None
        for offset, row_index in enumerate(indices):
            hours = float(interval_hours.loc[row_index])
            if bool(raining.loc[row_index]):
                rain_age = 0.0; rain_duration += hours
            else:
                rain_age = min(999.0, rain_age + hours); rain_duration = 0.0
            duration_values.loc[row_index] = rain_duration
            since_values.loc[row_index] = rain_age
            if times is not None and pd.notna(times.iloc[offset]):
                cutoff = times.iloc[offset] - pd.Timedelta(hours=3)
                included = [idx for pos, idx in enumerate(indices[:offset + 1]) if times.iloc[pos] > cutoff]
            else:
                included = indices[max(0, offset - 2):offset + 1]
            rolling_values.loc[row_index] = increment.loc[included].sum()
    ordered["precip_duration_hours"] = duration_values
    ordered["precip_3h_mm"] = rolling_values
    ordered["hours_since_forecast_rain"] = since_values
    drying = ordered.hrrr_rh.diff().fillna(0) < 0
    ordered["wet_dry_transition"] = (raining.astype(int) - drying.astype(int)).astype(float)
    for name in PRECIP_FEATURES: result.loc[ordered.index, name] = ordered[name].to_numpy()
    return result


def add_v4_features(frame, base_prediction, physics_variant="legacy"):
    result = precipitation_features(frame)
    result = add_hybrid_features(result, base_prediction)
    values = RAIN_VARIANTS[physics_variant]; output = pd.Series(index=result.index, dtype=float)
    for _, group in result.sort_values("lead_hour").groupby(["run_id", "station_id"]):
        if values is None:
            prediction = evolve_fm(group.initial_fm.iloc[0], group.hrrr_temp_c, group.hrrr_rh)
        else:
            prediction = evolve_fm_with_rain(group.initial_fm.iloc[0], group.hrrr_temp_c, group.hrrr_rh,
                                             group.hrrr_precip_increment_mm, *values)
        output.loc[group.index] = prediction
    result["rain_physics_fm"] = output
    result["rain_physics_minus_base"] = result.rain_physics_fm - result.incumbent_base_fm
    valid = pd.to_datetime(result.valid_time, utc=True, errors="coerce")
    result["summer_indicator"] = valid.dt.month.isin([6, 7, 8]).astype(float)
    result["hot_dry_interaction"] = (result.hrrr_temp_c - 25.0).clip(lower=0) * result.vpd_kpa
    result["initial_emc_gap"] = result.initial_fm - result.forecast_emc
    return result
