"""
Fit the ensemble_fire_danger calibration on the TRAIN split.

Per track (members / ensprod), per probability flavour (point /
neighborhood), per category k (>= Moderate .. >= Extreme):

- isotonic regression of the observed exceedance (0/1) on the raw
  smoothed ensemble fraction - monotone by construction, no parametric
  shape assumed, which suits a ~10-40 member ensemble whose raw fraction is
  typically under-dispersive (too confident) at the extremes. Stored as at
  most MAX_KNOTS (x, y) knots; core.apply_calibration interpolates them.
- identity (no knots) whenever that (track, flavour, k) has fewer than
  MIN_EVENTS positive or negative training rows - a calibration fitted to
  a handful of events is worse than none. Extreme will sit here until real
  Extreme days accumulate; evaluate.py reports it as "deferred", not fail.

Categorical thresholds tau_k (point flavour only - the categorical map is
a point forecast): the calibrated-probability threshold maximizing CSI for
">= k" subject to frequency bias in [BIAS_LO, BIAS_HI], searched over
0.05..0.95. Falls back to config default_categorical_threshold (0.5 =
ensemble median) when there are too few events or no threshold meets the
bias constraint.

Output: in-memory dict {"calibration": ..., "categorical_thresholds": ...,
"fit_report": ...}; model_bundle.py writes it.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

MIN_EVENTS = 20
MAX_KNOTS = 60
BIAS_LO, BIAS_HI = 0.8, 1.25
FLAVOURS = {"point": ("p_point", "obs_point"), "neighborhood": ("p_nbhd", "obs_nbhd")}


def isotonic_knots(p: np.ndarray, y: np.ndarray) -> Optional[dict]:
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(p, y.astype("float64"))
    xs = np.linspace(0.0, 1.0, 201)
    ys = model.predict(xs)
    # keep only knots where the fitted curve changes slope/level
    keep = np.r_[True, np.abs(np.diff(ys, 2)) > 1e-9, True]
    xk, yk = xs[keep], ys[keep]
    if len(xk) > MAX_KNOTS:
        idx = np.unique(np.linspace(0, len(xk) - 1, MAX_KNOTS).round().astype(int))
        xk, yk = xk[idx], yk[idx]
    return {"x": [round(float(v), 4) for v in xk], "y": [round(float(v), 5) for v in np.maximum.accumulate(yk)]}


def best_threshold(p: np.ndarray, y: np.ndarray, default: float) -> Dict[str, float]:
    best = {"threshold": default, "csi": None, "bias": None, "source": "default"}
    positives = int(y.sum())
    if positives < MIN_EVENTS:
        best["source"] = "default (too few events)"
        return best
    for tau in np.round(np.arange(0.05, 0.96, 0.05), 2):
        forecast = p >= tau
        hits = int(np.sum(forecast & y))
        false_alarms = int(np.sum(forecast & ~y))
        misses = int(np.sum(~forecast & y))
        denom = hits + false_alarms + misses
        if denom == 0:
            continue
        csi = hits / denom
        bias = (hits + false_alarms) / positives
        if BIAS_LO <= bias <= BIAS_HI and (best["csi"] is None or csi > best["csi"]):
            best = {"threshold": float(tau), "csi": round(csi, 4), "bias": round(bias, 4), "source": "fitted"}
    if best["source"] == "default":
        best["source"] = "default (no threshold within bias bounds)"
    return best


def fit(panel: pd.DataFrame, split: Dict[str, str], default_threshold: float = 0.5) -> dict:
    train = panel[panel["day"].map(split).eq("train")]
    calibration = {"method": "isotonic", "min_events": MIN_EVENTS, "tracks": {}}
    thresholds = {"method": "max_csi_bias_bounded", "bias_bounds": [BIAS_LO, BIAS_HI], "tracks": {}}
    report = {"train_days": int(train["day"].nunique()), "tracks": {}}
    for track, rows in train.groupby("track"):
        calibration["tracks"][track] = {"point": {}, "neighborhood": {}}
        thresholds["tracks"][track] = {}
        report["tracks"][track] = {}
        for k, krows in rows.groupby("k"):
            entry = {}
            for flavour, (p_col, y_col) in FLAVOURS.items():
                p = krows[p_col].to_numpy("float64")
                y = krows[y_col].to_numpy(bool)
                positives, negatives = int(y.sum()), int((~y).sum())
                if positives >= MIN_EVENTS and negatives >= MIN_EVENTS:
                    calibration["tracks"][track][flavour][str(k)] = isotonic_knots(p, y)
                    entry[flavour] = {"status": "fitted", "events": positives, "rows": len(y)}
                else:
                    entry[flavour] = {"status": "identity (too few events)", "events": positives, "rows": len(y)}
            knots = calibration["tracks"][track]["point"].get(str(k))
            p_cal = np.interp(krows["p_point"].to_numpy("float64"), knots["x"], knots["y"]) if knots \
                else krows["p_point"].to_numpy("float64")
            tau = best_threshold(p_cal, krows["obs_point"].to_numpy(bool), default_threshold)
            thresholds["tracks"][track][str(k)] = tau["threshold"]
            entry["categorical_threshold"] = tau
            report["tracks"][track][str(k)] = entry
    return {"calibration": calibration, "categorical_thresholds": thresholds, "fit_report": report}
