"""Causal feature engineering for the V3 hybrid station model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from spatial.physics import nelson_emc
from spatial.station_features import FEATURES as V2_FEATURES, add_causal_features

BASE_FEATURE = "incumbent_base_fm"
EXTRA_FEATURES = [
    BASE_FEATURE, "physics_minus_incumbent", "vpd_kpa", "forecast_emc",
    "precip_occurrence", "log1p_precip", "temp_lead_change_c",
    "rh_lead_change", "wind_lead_change_ms", "precip_lead_change_mm",
]
FEATURES = V2_FEATURES + EXTRA_FEATURES


def add_hybrid_features(frame: pd.DataFrame, base_prediction) -> pd.DataFrame:
    result = add_causal_features(frame)
    result[BASE_FEATURE] = np.asarray(base_prediction, dtype=float)
    result["physics_minus_incumbent"] = result["physics_fm"] - result[BASE_FEATURE]
    saturation = 0.6108 * np.exp(17.27 * result.hrrr_temp_c / (result.hrrr_temp_c + 237.3))
    result["vpd_kpa"] = saturation * (1 - result.hrrr_rh.clip(0, 100) / 100)
    result["forecast_emc"] = nelson_emc(result.hrrr_temp_c, result.hrrr_rh)
    precipitation = result.hrrr_precip_mm.clip(lower=0).fillna(0)
    result["precip_occurrence"] = (precipitation > 0).astype(float)
    result["log1p_precip"] = np.log1p(precipitation)
    ordered = result.sort_values(["run_id", "station_id", "lead_hour"])
    grouped = ordered.groupby(["run_id", "station_id"], sort=False)
    changes = {
        "temp_lead_change_c": grouped.hrrr_temp_c.diff(),
        "rh_lead_change": grouped.hrrr_rh.diff(),
        "wind_lead_change_ms": grouped.hrrr_wind_ms.diff(),
        "precip_lead_change_mm": grouped.hrrr_precip_mm.diff(),
    }
    for name, values in changes.items():
        result.loc[ordered.index, name] = values.fillna(0).to_numpy()
    result["precip_lead_change_mm"] = result["precip_lead_change_mm"].clip(lower=0)
    return result
