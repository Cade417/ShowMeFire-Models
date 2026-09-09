"""
fire_weather_ml's ML INPUT feature set - deliberately not the same column
set as rothermel_labels.py's LABEL_INPUT_COLUMNS, even though there's
overlap (both need current weather/fuel-moisture/terrain). The point of
training an ML model against a physics-computed label rather than just
re-running the physics is to let it learn from context the physics
calculation itself has no way to use - antecedent drought (KBDI) and
accumulated warmth (GDD) are exactly that: multi-day memory the
instantaneous Rothermel calculation never sees.

KBDI/GDD here are independent implementations (not imports from
risk_fusion, which has its own KBDI accrual tied to county-day geometry -
see model-training/docs/fire_weather_ml_plan.md for why this model family
doesn't share code with risk_fusion) using the standard published formulas.
Both are Phase 1 scaffolding: correct in shape/formula, not yet validated
against a real calibrated mean-annual-precipitation source per station
(risk_fusion/county_precip_normals.json exists as *data*, and could be
reused as a data input in Phase 2 without importing risk_fusion's code -
not wired up yet).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

KBDI_MAX = 800.0  # KBDI's own fixed ceiling (hundredths of an inch of soil moisture deficit)
KBDI_RAIN_INTERCEPTION_IN = 0.2  # standard KBDI convention: first 0.2in of a rain event is canopy/litter interception, not runoff into the soil-moisture deficit term
GDD_BASE_TEMP_C = 10.0  # standard base temperature for warm-season grass/brush green-up accumulation


def kbdi_step(previous_kbdi: float, daily_rain_in: float, max_temp_f: float, mean_annual_precip_in: float) -> float:
    """
    One day's Keetch-Byram Drought Index update (Keetch & Byram 1968).
    `previous_kbdi` and the return value are in KBDI's native hundredths-
    of-an-inch units (0-800). `daily_rain_in`/`max_temp_f`/
    `mean_annual_precip_in` are the standard formula's native units
    (inches, degrees F) - callers convert from metric on the way in.
    """
    net_rain = max(0.0, daily_rain_in - KBDI_RAIN_INTERCEPTION_IN)
    after_rain = max(0.0, previous_kbdi - net_rain * 100.0)
    drought_factor = (
        (800.0 - after_rain) * (0.968 * np.exp(0.0486 * max_temp_f) - 8.30)
        / (1.0 + 10.88 * np.exp(-0.0441 * mean_annual_precip_in))
        * 1e-3
    )
    return float(np.clip(after_rain + max(0.0, drought_factor), 0.0, KBDI_MAX))


def kbdi_series(daily_rain_in: pd.Series, max_temp_f: pd.Series, mean_annual_precip_in: float,
                initial_kbdi: float = 0.0) -> pd.Series:
    """Sequential KBDI accrual over a chronologically-ordered daily series (KBDI is stateful, not row-independent)."""
    if len(daily_rain_in) != len(max_temp_f):
        raise ValueError("daily_rain_in and max_temp_f must be the same length")
    values = []
    kbdi = initial_kbdi
    for rain, temp in zip(daily_rain_in.to_numpy(), max_temp_f.to_numpy()):
        kbdi = kbdi_step(kbdi, float(rain), float(temp), mean_annual_precip_in)
        values.append(kbdi)
    return pd.Series(values, index=daily_rain_in.index, name="kbdi")


def gdd_step(previous_accum: float, mean_temp_c: float, base_temp_c: float = GDD_BASE_TEMP_C) -> float:
    """One day's growing-degree-day accumulation, resetting each spring is the caller's responsibility (pass initial_accum=0 at the season start)."""
    return float(previous_accum + max(0.0, mean_temp_c - base_temp_c))


def gdd_series(mean_temp_c: pd.Series, base_temp_c: float = GDD_BASE_TEMP_C, initial_accum: float = 0.0) -> pd.Series:
    values = []
    accum = initial_accum
    for temp in mean_temp_c.to_numpy():
        accum = gdd_step(accum, float(temp), base_temp_c)
        values.append(accum)
    return pd.Series(values, index=mean_temp_c.index, name="gdd_accum")


FEATURE_COLUMNS = (
    "temp_c", "rh_pct", "wind_ms", "precip_mm",
    "fm1_pct", "fm10_pct", "fm100_pct",
    "kbdi", "gdd_accum",
    "slope_deg", "aspect_deg", "canopy_cover_pct", "canopy_height_m",
)


def assemble_features(
    weather: pd.DataFrame,
    mean_annual_precip_in: float,
    initial_kbdi: float = 0.0,
    initial_gdd: float = 0.0,
) -> pd.DataFrame:
    """
    Builds the FEATURE_COLUMNS set from a chronologically-sorted per-
    station daily/hourly weather+fuel-moisture+terrain frame. `weather`
    must already carry temp_c, rh_pct, wind_ms, precip_mm, fm1_pct,
    fm10_pct, fm100_pct, slope_deg, aspect_deg, canopy_cover_pct,
    canopy_height_m (see panel.py for how those get joined together);
    this function's job is only to derive the two memory features
    (kbdi, gdd_accum) on top of that and assemble the final column set.
    """
    required = ("temp_c", "rh_pct", "wind_ms", "precip_mm", "fm1_pct", "fm10_pct", "fm100_pct",
                "slope_deg", "aspect_deg", "canopy_cover_pct", "canopy_height_m")
    missing = [c for c in required if c not in weather.columns]
    if missing:
        raise ValueError(f"weather frame is missing required columns: {missing}")

    daily_rain_in = weather["precip_mm"] / 25.4
    max_temp_f = weather["temp_c"] * 9.0 / 5.0 + 32.0
    features = weather.copy()
    features["kbdi"] = kbdi_series(daily_rain_in, max_temp_f, mean_annual_precip_in, initial_kbdi)
    features["gdd_accum"] = gdd_series(weather["temp_c"], initial_accum=initial_gdd)
    return features[list(FEATURE_COLUMNS)]
