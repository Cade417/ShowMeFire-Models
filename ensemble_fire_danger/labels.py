"""
"Observed" peak fire danger category per grid cell for one forecast day -
the verification target the ensemble probabilities are calibrated against.

observed = the SAME operational rule (rules/fire_danger_rules.json via
risk_fusion/rule_uncertainty.py's parity-checked category_vectorized),
applied hour by hour to the hourly RTMA analysis (spatial/rtma_capture.py)
over the same valid window as the forecast (16Z-03Z = 10:00-21:00 CT),
maxed over the window - exactly how the public map aggregates a forecast.

Documented choices / limitations (see docs/ensemble_fire_danger.md §6):
- There is no gridded OBSERVED 10-hr fuel moisture. The label's FM is the
  same XGBoost fuel-moisture model driven by RTMA temperature/RH/wind (the
  model DailyForecast runs on HRRR), with precipitation from the HRRR
  control member's own 1-h forecast precip (RTMA has no precip field).
  So the label is "what the rule says about the analysed weather", which
  is what the product forecasts - not fire occurrence.
- Wind uses the same 0.8 near-surface reduction (label_wind_factor) the
  rule's HRRR input gets: the category thresholds are defined on that
  reduced wind, and RTMA's 10 m analysis is the like-for-like quantity.
- Seasonal (green-up) dampening is NOT applied to labels or to the training
  members - historical per-day GDD state is not archived. It only ever
  demotes marginal cells in spring, identically for forecast and label.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from spatial.rtma_capture import fetch_rtma

from . import core
from .fm_features import predict_fm_hourly

LABEL_WIND_FACTOR = 0.8


def _fetch_rtma_retrying(when: datetime, attempts: int = 4):
    """fetch_rtma has no retry of its own; transient DNS/connection drops
    (seen live 2026-09-23 mid-panel-build) should not cost a whole day."""
    for attempt in range(attempts):
        try:
            return fetch_rtma(when)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def rtma_hours(valid_times: Sequence[pd.Timestamp]) -> List[xr.Dataset]:
    out = []
    for valid in valid_times:
        path = _fetch_rtma_retrying(datetime(valid.year, valid.month, valid.day, valid.hour))
        with xr.open_dataset(path) as ds:
            out.append(ds.load())
    return out


def observed_peak(valid_times: Sequence[pd.Timestamp], make_regridder: Callable, booster, feature_names,
                  categorize: Callable, precip_mm: Optional[np.ndarray], wind_factor: float = LABEL_WIND_FACTOR
                  ) -> Dict[str, np.ndarray]:
    """-> {"peak": (y,x) int8, "min_rh": (y,x), "max_wind_kts": (y,x), "min_fm": (y,x)} on the target grid."""
    hours = rtma_hours(valid_times)
    # cfgrib/Herbie RTMA longitudes are 0-360; the target grid (and every
    # member, see grib_idx._field_from_gid) is -180..180. Without this every
    # target cell falls outside the source hull and the label is all-missing.
    rtma_lon = hours[0]["longitude"].values
    rtma_lon = np.where(rtma_lon > 180.0, rtma_lon - 360.0, rtma_lon)
    regridder = make_regridder(hours[0]["latitude"].values, rtma_lon)
    temp = np.stack([regridder(h["t2m"].values) for h in hours]) - 273.15
    rh = np.clip(np.stack([regridder(h["r2"].values) for h in hours]), 0.0, 100.0)
    wind = np.stack([regridder(np.hypot(h["u10"].values, h["v10"].values)) for h in hours])
    precip = precip_mm if precip_mm is not None and precip_mm.shape == temp.shape else np.zeros_like(temp)
    fm = np.clip(predict_fm_hourly(booster, feature_names, temp, rh, wind, precip, valid_times), 1.0, 40.0)
    wind_kts = wind * wind_factor * core.MPS_TO_KNOTS
    return {
        "peak": core.peak_category(fm, rh, wind_kts, categorize),
        "min_rh": np.nanmin(rh, axis=0),
        "max_wind_kts": np.nanmax(wind_kts, axis=0),
        "min_fm": np.nanmin(fm, axis=0),
    }
