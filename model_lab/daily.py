"""Cached product-aware daily archive scoring for the unified dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from paths import DATA_ROOT
from model_lab.data import discover_forecast_dates, discover_forecast_series, load_scored_series
from model_lab.metrics import attach_obs_fire_danger, calculate_metrics

CACHE_PATH = DATA_ROOT / "model_lab" / "daily_scores.json"


def _fd_metrics(frame: pd.DataFrame) -> dict:
    frame = attach_obs_fire_danger(frame)
    valid = frame.dropna(subset=["pred_fire_danger", "obs_fire_danger"])
    if valid.empty:
        return {"macro_f1": None, "accuracy": None, "count": 0}
    actual = valid["obs_fire_danger"].astype(int)
    predicted = valid["pred_fire_danger"].astype(int)
    return {
        "macro_f1": float(f1_score(actual, predicted, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0)),
        "accuracy": float((actual == predicted).mean()),
        "count": int(len(valid)),
    }


def score_daily_series(dates: list[str] | None = None, series: list[str] | None = None) -> pd.DataFrame:
    dates = dates or list(discover_forecast_dates())
    series = series or list(discover_forecast_series())
    rows = []
    for date_token in dates:
        for series_key in series:
            merged, meta = load_scored_series(date_token, series_key)
            if merged.empty:
                continue
            fm = calculate_metrics(merged).get("Fuel Moisture (%)", {})
            fd = _fd_metrics(merged)
            rows.append({
                "date": date_token,
                "series": series_key,
                "fm_mae": fm.get("mae"),
                "fm_rmse": fm.get("rmse"),
                "fm_bias": fm.get("bias"),
                "fd_macro_f1": fd.get("macro_f1"),
                "fd_accuracy": fd.get("accuracy"),
                "count": fd.get("count") or fm.get("count") or 0,
                "forecast_path": meta.get("forecast_path"),
            })
    return pd.DataFrame(rows)


def cache_daily_scores(days: int = 14, force: bool = False) -> pd.DataFrame:
    all_dates = list(discover_forecast_dates())
    dates = all_dates[-days:]
    if CACHE_PATH.exists() and not force:
        try:
            cached = pd.DataFrame(json.loads(CACHE_PATH.read_text(encoding="utf-8")))
            if not cached.empty and set(dates).issubset(set(cached["date"].astype(str))):
                return cached[cached["date"].isin(dates)].copy()
        except (OSError, ValueError, TypeError):
            pass
    result = score_daily_series(dates=dates)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(result.to_json(orient="records", indent=2), encoding="utf-8")
    return result


def daily_summary(product: str = "fuel_moisture", days: int = 14, force: bool = False) -> pd.DataFrame:
    scores = cache_daily_scores(days=days, force=force)
    if scores.empty:
        return scores
    metric = "fm_mae" if product == "fuel_moisture" else "fd_macro_f1"
    grouped = scores.groupby("series", as_index=False).agg(
        days=("date", "count"),
        primary=(metric, "mean"),
        count=("count", "sum"),
    )
    grouped["product"] = product
    return grouped.sort_values("primary", ascending=(product == "fuel_moisture"))
