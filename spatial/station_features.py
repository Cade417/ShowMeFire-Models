"""Causal feature contract shared by station-model training and evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_SCHEMA_VERSION = "station-causal-v2"

LEGACY_FEATURES = [
    "initial_fm", "initial_age_hours", "rtma_temp_c", "rtma_rh",
    "rtma_wind_ms", "hrrr_temp_c", "hrrr_rh", "hrrr_wind_ms",
    "hrrr_precip_mm", "lead_hour", "lat", "lon",
]

ENGINEERED_FEATURES = [
    "valid_hour_sin", "valid_hour_cos", "valid_doy_sin", "valid_doy_cos",
    "temp_change_c", "rh_change", "wind_change_ms",
]

FEATURES = LEGACY_FEATURES + ENGINEERED_FEATURES


def add_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with features available at forecast initialization.

    `valid_time` and HRRR values are forecast inputs. No target or realized
    post-initialization observation is used here.
    """
    result = frame.copy()
    valid = pd.to_datetime(result["valid_time"], utc=True, errors="coerce")
    hour_angle = 2 * np.pi * (valid.dt.hour + valid.dt.minute / 60.0) / 24.0
    doy_angle = 2 * np.pi * (valid.dt.dayofyear - 1) / 365.2425
    result["valid_hour_sin"] = np.sin(hour_angle)
    result["valid_hour_cos"] = np.cos(hour_angle)
    result["valid_doy_sin"] = np.sin(doy_angle)
    result["valid_doy_cos"] = np.cos(doy_angle)
    result["temp_change_c"] = result["hrrr_temp_c"] - result["rtma_temp_c"]
    result["rh_change"] = result["hrrr_rh"] - result["rtma_rh"]
    result["wind_change_ms"] = result["hrrr_wind_ms"] - result["rtma_wind_ms"]
    return result
