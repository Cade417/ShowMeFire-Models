"""
Fit the outlook's candidate corrections on TRAIN rows only.

1. Interval correction (per variable x lead group), for 10-hour fuel moisture and
   each weather variable. The ensemble's p10-p90 spread is only weather-scenario
   spread, so at long leads it under-covers. We fit
       centre' = centre + bias            bias = median(obs - centre)
       lo/hi'  = centre' -/+ scale * (spread below/above the raw centre)
   with the smallest scale (grid) whose training coverage reaches INTERVAL_NOMINAL.
   Groups with fewer than MIN_ROWS rows keep bias 0 / scale 1 and are marked
   "identity" - never extrapolated from another lead group.

2. Agreement calibration (per threshold x lead group): isotonic map from raw
   scenario agreement to the observed frequency of that threshold, fitted only
   when the group has MIN_EVENTS events AND non-events. Identity otherwise.
   apply_agreement() enforces Critical <= Elevated after calibration.

This does NOT extend any short-range learned fuel-moisture model: the fuel
correction is a plain bias/width adjustment of the worker's physics-based state,
and it is only a candidate until evaluate.py's gates and the shadow pilot pass.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import contract

MIN_ROWS = 200
MIN_EVENTS = 20
SCALE_GRID = np.round(np.arange(0.5, 6.001, 0.05), 2)
THRESHOLDS = {"elevated": ("agreement_elevated", "obs_elevated"),
              "critical": ("agreement_critical", "obs_critical")}


def variables(df: pd.DataFrame) -> list[str]:
    out = []
    for name in ("fm10", *contract.WEATHER_VARIABLES):
        cols = contract.variable_columns(name)
        if all(cols[k] in df.columns for k in ("center", "lo", "hi", "obs")):
            out.append(name)
    return out


def _identity(n: int, reason: str) -> dict:
    return {"method": "identity", "bias": 0.0, "scale": 1.0, "n": int(n), "reason": reason}


def fit_interval(rows: pd.DataFrame, cols: Dict[str, str]) -> dict:
    d = rows[[cols["center"], cols["lo"], cols["hi"], cols["obs"]]].dropna()
    if len(d) < MIN_ROWS:
        return _identity(len(d), f"fewer than {MIN_ROWS} rows")
    c, lo, hi, y = (d[cols[k]].to_numpy(float) for k in ("center", "lo", "hi", "obs"))
    bias = float(np.median(y - c))
    centre = c + bias
    below = np.maximum(c - lo, 0.0)
    above = np.maximum(hi - c, 0.0)
    scale, reached = float(SCALE_GRID[-1]), False
    for s in SCALE_GRID:
        cov = float(np.mean((y >= centre - s * below) & (y <= centre + s * above)))
        if cov >= contract.INTERVAL_NOMINAL:
            scale, reached = float(s), True
            break
    return {"method": "bias_scale", "bias": bias, "scale": scale, "n": int(len(d)),
            "nominal": contract.INTERVAL_NOMINAL, "reached_nominal": reached}


def apply_interval(params: Optional[dict], centre, lo, hi):
    """Corrected (lo, centre, hi). None / identity params return the inputs."""
    centre, lo, hi = (np.asarray(x, dtype=float) for x in (centre, lo, hi))
    if not params or params.get("method") == "identity":
        return lo, centre, hi
    c2 = centre + params["bias"]
    s = params["scale"]
    return c2 - s * np.maximum(centre - lo, 0.0), c2, c2 + s * np.maximum(hi - centre, 0.0)


def isotonic_knots(p: np.ndarray, y: np.ndarray) -> Optional[dict]:
    from sklearn.isotonic import IsotonicRegression
    if len(p) < MIN_ROWS:
        return None
    pos = int(np.sum(y > 0.5))
    if pos < MIN_EVENTS or len(y) - pos < MIN_EVENTS:
        return None
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y.astype(float))
    return {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}


def fit_agreement(rows: pd.DataFrame) -> dict:
    out: dict = {}
    for thr, (pcol, ycol) in THRESHOLDS.items():
        out[thr] = {}
        for group in contract.LEAD_GROUPS:
            d = rows.loc[rows["lead_group"] == group, [pcol, ycol]].dropna()
            knots = isotonic_knots(d[pcol].to_numpy(float), d[ycol].to_numpy(float))
            out[thr][group] = (
                {"method": "isotonic", "n": int(len(d)), **knots} if knots
                else {"method": "identity", "n": int(len(d)),
                      "reason": f"needs {MIN_ROWS} rows and {MIN_EVENTS} events and non-events"})
    return out


def apply_agreement(calibration: dict, group: str, elevated, critical):
    """Calibrated (elevated, critical); critical is clamped to never exceed elevated."""
    def one(thr, p):
        spec = calibration.get(thr, {}).get(group, {})
        p = np.asarray(p, dtype=float)
        if spec.get("method") != "isotonic":
            return p
        return np.interp(p, spec["x"], spec["y"])
    e = one("elevated", elevated)
    c = np.minimum(one("critical", critical), e)
    return e, c


def fit(df: pd.DataFrame, side: pd.Series) -> dict:
    train = df[(side == "train") & df["qc_ok"]]
    intervals: dict = {}
    for name in variables(df):
        cols = contract.variable_columns(name)
        intervals[name] = {g: fit_interval(train[train["lead_group"] == g], cols) for g in contract.LEAD_GROUPS}
    return {
        "family": "fire_outlook",
        "label_text": "scenario agreement",
        "train_rows": int(len(train)),
        "interval": intervals,
        "agreement": fit_agreement(train),
        "notes": ("Candidate only. Fitted on train rows (contiguous periods x held-in stations). "
                  "Agreement stays labelled 'scenario agreement' until review and the 30-day staff shadow pilot."),
    }
