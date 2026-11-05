"""Scoring helpers for forecast-vs-obs and pairwise series comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd

VARIABLE_MAP = {
    "Temperature (C)": ("pred_temp", "obs_temp"),
    "Relative Humidity (%)": ("pred_rh", "obs_rh"),
    "Wind Speed (m/s)": ("pred_wind", "obs_wind"),
    "Fuel Moisture (%)": ("pred_fm", "obs_fm"),
}

FD_LABELS = {0: "Low", 1: "Moderate", 2: "Elevated", 3: "Critical", 4: "Extreme"}


def _fire_danger(fm, rh, wind_kts):
    """Mirror ShowMeFire public FD rules (same thresholds as end-of-day reports)."""
    if fm is None or (isinstance(fm, float) and np.isnan(fm)):
        return None
    if rh is None or (isinstance(rh, float) and np.isnan(rh)):
        return None
    if wind_kts is None or (isinstance(wind_kts, float) and np.isnan(wind_kts)):
        return None
    fm, rh, wind_kts = float(fm), float(rh), float(wind_kts)
    if fm < 7 and rh < 20 and wind_kts >= 25:
        return 4
    if fm < 9 and rh < 25 and wind_kts >= 15:
        return 3
    if fm < 9 and ((rh < 25 and wind_kts >= 10) or (rh < 35 and wind_kts >= 15)):
        return 2
    if fm < 15 and (rh < 45 or wind_kts >= 10):
        return 1
    if fm >= 15:
        return 0
    return 0


def attach_obs_fire_danger(obs_df: pd.DataFrame) -> pd.DataFrame:
    if obs_df.empty:
        return obs_df
    out = obs_df.copy()
    out["obs_fire_danger"] = [
        _fire_danger(fm, rh, (wind * 1.94384) if wind is not None and not pd.isna(wind) else None)
        for fm, rh, wind in zip(out.get("obs_fm", []), out.get("obs_rh", []), out.get("obs_wind", []))
    ]
    return out


def calculate_metrics(merged_df: pd.DataFrame, variable_map=None) -> dict:
    variable_map = variable_map or VARIABLE_MAP
    metrics = {}
    for name, (pred_col, obs_col) in variable_map.items():
        if pred_col not in merged_df.columns or obs_col not in merged_df.columns:
            metrics[name] = {"mae": None, "rmse": None, "bias": None, "count": 0, "correlation": None}
            continue
        valid = merged_df[[pred_col, obs_col]].dropna().copy()
        if valid.empty:
            metrics[name] = {"mae": None, "rmse": None, "bias": None, "count": 0, "correlation": None}
            continue
        y_true = pd.to_numeric(valid[obs_col], errors="coerce")
        y_pred = pd.to_numeric(valid[pred_col], errors="coerce")
        mask = y_true.notna() & y_pred.notna()
        y_true, y_pred = y_true[mask], y_pred[mask]
        if y_true.empty:
            metrics[name] = {"mae": None, "rmse": None, "bias": None, "count": 0, "correlation": None}
            continue
        err = y_pred - y_true
        corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else None
        metrics[name] = {
            "mae": round(float(np.mean(np.abs(err))), 4),
            "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "bias": round(float(np.mean(err)), 4),
            "count": int(len(y_true)),
            "correlation": None if corr is None or np.isnan(corr) else round(corr, 4),
        }
    return metrics


def metrics_by_station(merged_df: pd.DataFrame, pred_col="pred_fm", obs_col="obs_fm") -> pd.DataFrame:
    rows = []
    if merged_df.empty or pred_col not in merged_df.columns:
        return pd.DataFrame()
    for stid, group in merged_df.groupby("stid"):
        valid = group[[pred_col, obs_col]].dropna()
        if valid.empty:
            continue
        err = valid[pred_col] - valid[obs_col]
        rows.append({
            "stid": stid,
            "count": len(valid),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
        })
    return pd.DataFrame(rows).sort_values("mae") if rows else pd.DataFrame()


def metrics_by_hour(merged_df: pd.DataFrame, pred_col="pred_fm", obs_col="obs_fm") -> pd.DataFrame:
    if merged_df.empty or "timestamp" not in merged_df.columns:
        return pd.DataFrame()
    work = merged_df.dropna(subset=[pred_col, obs_col]).copy()
    if work.empty:
        return pd.DataFrame()
    work["hour"] = pd.to_datetime(work["timestamp"], utc=True).dt.hour
    rows = []
    for hour, group in work.groupby("hour"):
        err = group[pred_col] - group[obs_col]
        rows.append({
            "hour": int(hour),
            "count": len(group),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
        })
    return pd.DataFrame(rows).sort_values("hour")


def pairwise_series_delta(left: pd.DataFrame, right: pd.DataFrame, value_col="pred_fm") -> pd.DataFrame:
    """Align two forecast series on stid+timestamp and compute value deltas."""
    if left.empty or right.empty:
        return pd.DataFrame()
    a = left[["stid", "timestamp", value_col]].rename(columns={value_col: "left"})
    b = right[["stid", "timestamp", value_col]].rename(columns={value_col: "right"})
    merged = a.merge(b, on=["stid", "timestamp"], how="inner")
    merged["delta"] = merged["right"] - merged["left"]
    merged["abs_delta"] = merged["delta"].abs()
    return merged


def flatten_metric_block(block: dict | None, prefix: str = "") -> dict:
    """Pull top-level scalar metrics out of nested eval JSON blocks."""
    if not isinstance(block, dict):
        return {}
    out = {}
    for key in ("mae", "rmse", "bias", "r2", "samples", "interval_coverage",
                "persistence_mae", "physics_mae", "improvement_over_persistence"):
        if key in block and not isinstance(block[key], dict):
            out[f"{prefix}{key}" if prefix else key] = block[key]
    return out


def fd_confusion(merged_df: pd.DataFrame, pred_col="pred_fire_danger", obs_col="obs_fire_danger") -> pd.DataFrame:
    if merged_df.empty or pred_col not in merged_df.columns or obs_col not in merged_df.columns:
        return pd.DataFrame()
    valid = merged_df[[pred_col, obs_col]].dropna()
    if valid.empty:
        return pd.DataFrame()
    cats = sorted(set(valid[pred_col].astype(int)).union(set(valid[obs_col].astype(int))))
    matrix = pd.crosstab(
        valid[obs_col].astype(int).map(lambda c: FD_LABELS.get(c, str(c))),
        valid[pred_col].astype(int).map(lambda c: FD_LABELS.get(c, str(c))),
        dropna=False,
    )
    order = [FD_LABELS[c] for c in cats if c in FD_LABELS]
    matrix = matrix.reindex(index=order, columns=order, fill_value=0)
    return matrix
