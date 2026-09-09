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
Per-station mean-annual-precipitation normals (KBDI's climate-normal term)
come from `precip_normals.py`, which reuses risk_fusion/county_precip_normals.json
as *data* (real 1991-2020 NOAA normals), not by importing risk_fusion's code.

Also home to `derive_fm1_fm10_fm100` and `live_moisture_percent` - Phase 2
additions for deriving inputs the real historical station data doesn't
directly observe (see their own docstrings for what's derived vs. observed
and why).
"""
from __future__ import annotations

import math
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


# --- Phase 2: deriving fm1/fm100 and live-fuel moisture the real station
# history doesn't directly observe ---------------------------------------

NELSON_EMC_MIN_PERCENT = 1.0
NELSON_EMC_MAX_PERCENT = 40.0
TAU_1_HR = 1.0
TAU_10_HR = 10.0
TAU_100_HR = 100.0

GDD_GREENUP_START = 200.0  # accumulated base-10C degree-days where green-up is assumed to begin
GDD_GREENUP_FULL = 1000.0  # ...and where it's assumed complete
LIVE_HERBACEOUS_CURED_PCT = 60.0   # Scott-Burgan L2 (cured) standard scenario value
LIVE_HERBACEOUS_GREEN_PCT = 120.0  # Scott-Burgan L4 (green) standard scenario value
LIVE_WOODY_CURED_PCT = 90.0
LIVE_WOODY_GREEN_PCT = 150.0


def nelson_emc(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    Nelson dead-fuel equilibrium moisture content (the standard published
    formula also independently used in api/services/spread_rate_moisture.py -
    same equations, separately implemented here per this project's
    model-training-never-imports-api boundary).
    """
    rh = np.asarray(rh, dtype=float)
    temp_c = np.asarray(temp_c, dtype=float)
    emc = np.where(
        rh <= 10,
        0.03 + 0.2626 * rh - 0.00104 * rh * temp_c,
        np.where(
            rh <= 50,
            2.22 - 0.160 * rh + 0.01660 * temp_c,
            21.06 - 0.4944 * rh + 0.005565 * rh ** 2 - 0.00063 * rh * temp_c,
        ),
    )
    return np.clip(emc, NELSON_EMC_MIN_PERCENT, NELSON_EMC_MAX_PERCENT)


def _advance_dead_fuel_state(state: dict, emc: float, precip_mm: float) -> dict:
    """One step of free-running (unanchored) 1/10/100-hr lag toward `emc`, with a rain bump - same shape as api/services/spread_rate_moisture.py::_advance_moisture, independently implemented, operating on scalars here rather than grids."""
    fm1 = state["fm1"] + (emc - state["fm1"]) * (1.0 - math.exp(-1.0 / TAU_1_HR))
    fm10 = state["fm10"] + (emc - state["fm10"]) * (1.0 - math.exp(-1.0 / TAU_10_HR))
    fm100 = state["fm100"] + (emc - state["fm100"]) * (1.0 - math.exp(-1.0 / TAU_100_HR))
    if precip_mm > 0.5:
        fm1 = min(fm1 + 0.5 * precip_mm, NELSON_EMC_MAX_PERCENT)
        fm10 = min(fm10 + 0.15 * precip_mm, NELSON_EMC_MAX_PERCENT)
        fm100 = min(fm100 + 0.05 * precip_mm, NELSON_EMC_MAX_PERCENT)
    return {"fm1": fm1, "fm10": fm10, "fm100": fm100}


def derive_fm1_fm10_fm100(
    temp_c: pd.Series, rh: pd.Series, precip_mm: pd.Series, observed_fm10_pct: pd.Series,
) -> pd.DataFrame:
    """
    The real station history (`training-data/aligned/station_leads*.csv`)
    only has a single fuel-moisture reading per station-hour (treated as
    the observed 10-hr class) - no separate 1-hr/100-hr observations. This
    derives them rather than observing them: a free-running Nelson-EMC
    1/10/100-hr lag propagation (seeded from EMC, same shape as
    api/services/spread_rate_moisture.py's live conditioning, independently
    implemented), then shifts ALL THREE classes at every timestep by the
    residual between this same propagation's own free-running 10-hr
    estimate and the REAL observed 10-hr reading. This is the same
    "anchor a modeled field to a real point observation via an additive
    residual" idea as that module's RAWS 10-hr correction, applied here
    temporally per-station (a real 10-hr reading exists at every historical
    timestep, unlike the live product's spatially-sparse RAWS coverage).

    This IS an approximation for the 1-hr/100-hr classes - flagged
    explicitly, not presented as observed. The 10-hr class in the returned
    frame is simply the real `observed_fm10_pct` passed through unchanged.

    All four input Series must be the same length, in chronological order,
    for a single station (fuel-moisture memory is sequential/stateful).
    """
    if not (len(temp_c) == len(rh) == len(precip_mm) == len(observed_fm10_pct)):
        raise ValueError("temp_c, rh, precip_mm, and observed_fm10_pct must be the same length")

    emc = nelson_emc(temp_c, rh)
    fm1_free, fm10_free, fm100_free = [], [], []
    state = None
    for e, precip in zip(np.asarray(emc), precip_mm.to_numpy()):
        if state is None:
            state = {"fm1": float(e), "fm10": float(e), "fm100": float(e)}
        else:
            state = _advance_dead_fuel_state(state, float(e), float(precip))
        fm1_free.append(state["fm1"])
        fm10_free.append(state["fm10"])
        fm100_free.append(state["fm100"])

    index = temp_c.index
    fm10_free_series = pd.Series(fm10_free, index=index)
    correction = observed_fm10_pct.to_numpy() - fm10_free_series.to_numpy()
    fm1 = np.clip(np.asarray(fm1_free) + correction, NELSON_EMC_MIN_PERCENT, NELSON_EMC_MAX_PERCENT)
    fm100 = np.clip(np.asarray(fm100_free) + correction, NELSON_EMC_MIN_PERCENT, NELSON_EMC_MAX_PERCENT)
    return pd.DataFrame({
        "fm1_pct": pd.Series(fm1, index=index),
        "fm10_pct": observed_fm10_pct,
        "fm100_pct": pd.Series(fm100, index=index),
    })


def green_factor(gdd_accum: float) -> float:
    """0 (fully cured) to 1 (fully green), ramping linearly over [GDD_GREENUP_START, GDD_GREENUP_FULL] - an approximation, not calibrated against real Missouri green-up observations."""
    return float(np.clip((gdd_accum - GDD_GREENUP_START) / (GDD_GREENUP_FULL - GDD_GREENUP_START), 0.0, 1.0))


def live_moisture_percent(gdd_accum: pd.Series) -> pd.DataFrame:
    """
    Maps accumulated growing-degree-days to Scott-Burgan L2 (cured) <-> L4
    (green) live-fuel-moisture scenario values - same shape as
    api/services/spread_rate_moisture.py::live_moisture_percent
    (statewide-GDD-proxy mapped to standard scenario endpoints),
    independently implemented here per this project's boundary.
    """
    green = gdd_accum.apply(green_factor)
    herbaceous = LIVE_HERBACEOUS_CURED_PCT + green * (LIVE_HERBACEOUS_GREEN_PCT - LIVE_HERBACEOUS_CURED_PCT)
    woody = LIVE_WOODY_CURED_PCT + green * (LIVE_WOODY_GREEN_PCT - LIVE_WOODY_CURED_PCT)
    return pd.DataFrame({"live_herbaceous_pct": herbaceous, "live_woody_pct": woody})
