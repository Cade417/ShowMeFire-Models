"""Feature engineering for the detection dataset.

Returns (features DataFrame, {column: group}). Nothing derived from NERIS
dispatches matched to *this* detection or from labels is allowed in here
(leakage) - NERIS appears only as prior-period county/month call density.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

EARTH_KM = 6371.0088
LAND_COVER_CLASSES = ["Cropland", "Developed", "Trees", "Grass", "Water", "Shrub", "Bare", "Wetland", "Snow"]
SOURCES = ["viirs", "modis", "ngfs", "goes"]
GRID_DEG = 0.25  # ~25 km spatial CV blocks


def _load_base_features():
    """Reuse the live model's feature code (api/detection-confidence-model/features.py)."""
    api_root = Path(os.getenv("SMF_API_REPO_ROOT", paths.REPO_ROOT.parent / "api"))
    path = api_root / "detection-confidence-model" / "features.py"
    spec = importlib.util.spec_from_file_location("_smf_detection_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _count_within(points_rad, radius_km, times=None, window_h=None):
    """Neighbour counts (excluding self) within radius, optionally within +-window_h."""
    from sklearn.neighbors import BallTree
    tree = BallTree(points_rad, metric="haversine")
    idx = tree.query_radius(points_rad, r=radius_km / EARTH_KM)
    if times is None:
        return np.array([len(i) - 1 for i in idx])
    return np.array([int((np.abs(times[i] - times[k]) <= window_h).sum()) - 1 for k, i in enumerate(idx)])


def build_features(det: pd.DataFrame, neris: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    base = _load_base_features()
    det = det.reset_index(drop=True)
    X = pd.DataFrame(index=det.index)
    groups: dict = {}

    def add(name, values, group):
        X[name] = values
        groups[name] = group

    # --- base block: exactly the live scorer's features
    vec = np.array([base.build_feature_vector(r) for r in det.to_dict("records")], dtype=float)
    for i, name in enumerate(base.FEATURE_NAMES):
        add(name, vec[:, i], "base")

    # --- sensor
    for c in ("frp", "bright_t7", "bright_t13", "pixel_area", "solar_zenith_angle", "satellite_zenith_angle", "weather_danger_prob"):
        add(c, pd.to_numeric(det.get(c), errors="coerce"), "sensor")
    add("bt_ratio", X["bright_t13"] / X["bright_t7"], "sensor")
    add("frp_per_area", X["frp"] / X["pixel_area"].replace(0, np.nan), "sensor")
    add("detection_confidence_pct", pd.to_numeric(det.get("detection_confidence_pct"), errors="coerce"), "sensor")
    for s in SOURCES:
        add(f"src_{s}", (det["source"].str.lower() == s).astype(float), "sensor")
    add("satellite_code", det["satellite"].astype("category").cat.codes.astype(float), "sensor")
    foot = det.get("footprint_geojson")
    add("has_footprint", foot.notna().astype(float) if foot is not None else 0.0, "sensor")

    # --- land / fuel
    lc = det["land_cover"].fillna("")
    for cls in LAND_COVER_CLASSES:
        add(f"lc_{cls.lower()}", [base.land_cover_fraction(v, cls) for v in lc], "land")
    add("lc_n_classes", [sum(1 for p in v.split(",") if p.strip()) for v in lc], "land")
    add("lc_dominant_frac", [max([float(p.partition(":")[2] or 0) for p in v.split(",") if ":" in p] or [0.0]) / 100 for v in lc], "land")
    add("weather_danger_cat", det["weather_danger_category"].astype("category").cat.codes.astype(float), "land")

    # --- temporal
    t = pd.to_datetime(det["occurred_at"], utc=True, errors="coerce")
    local = t - pd.Timedelta(hours=5)
    add("hour_local", local.dt.hour.astype(float), "temporal")
    add("hour_sin", np.sin(2 * np.pi * local.dt.hour / 24), "temporal")
    add("hour_cos", np.cos(2 * np.pi * local.dt.hour / 24), "temporal")
    doy = local.dt.dayofyear
    add("doy_sin", np.sin(2 * np.pi * doy / 365.25), "temporal")
    add("doy_cos", np.cos(2 * np.pi * doy / 365.25), "temporal")
    add("month", local.dt.month.astype(float), "temporal")
    add("dow", local.dt.dayofweek.astype(float), "temporal")
    add("is_weekend", (local.dt.dayofweek >= 5).astype(float), "temporal")
    add("burn_season", local.dt.month.isin([2, 3, 4, 10, 11]).astype(float), "temporal")  # MO spring/fall burn windows
    add("ingest_lag_h", (pd.to_datetime(det.get("first_seen_at"), utc=True, errors="coerce") - t).dt.total_seconds() / 3600, "temporal")

    # --- spatial + repeat history + cross-sensor
    valid = t.notna().to_numpy()
    pts = np.radians(det[["latitude", "longitude"]].to_numpy(dtype=float))
    hours = (t.astype("int64") // 3_600_000_000_000).to_numpy(dtype=float)
    hours[~valid] = np.nan
    if len(det) > 1:
        for r in (1, 5, 10):
            add(f"n_within_{r}km", _count_within(pts, r), "spatial")
        add("n_within_2km_3h", _count_within(pts, 2, np.nan_to_num(hours), 3), "cross_sensor")
        add("n_within_2km_24h", _count_within(pts, 2, np.nan_to_num(hours), 24), "cross_sensor")
        add("n_within_5km_7d", _count_within(pts, 5, np.nan_to_num(hours), 24 * 7), "temporal")
        add("n_within_1km_365d", _count_within(pts, 1, np.nan_to_num(hours), 24 * 365), "temporal")
        _cross_source(X, groups, det, pts, hours)
        from sklearn.neighbors import BallTree
        d, _ = BallTree(pts, metric="haversine").query(pts, k=2)
        add("nn_distance_km", d[:, 1] * EARTH_KM, "spatial")
    add("lat", det["latitude"].astype(float), "spatial")
    add("lon", det["longitude"].astype(float), "spatial")
    add("in_recurring_source", det["recurring_source_id"].notna().astype(float), "spatial")
    add("in_exclusion_zone", det["exclusion_zone_id"].notna().astype(float), "spatial")

    # --- incident context
    inc = det["incident_id"]
    add("incident_size", inc.map(inc.value_counts()).fillna(1).astype(float), "incident")
    span = det.assign(_t=t).groupby("incident_id")["_t"].agg(lambda s: (s.max() - s.min()).total_seconds() / 3600)
    add("incident_span_h", inc.map(span).fillna(0.0), "incident")
    n_src = det.groupby("incident_id")["source"].nunique()
    add("incident_n_sources", inc.map(n_src).fillna(1).astype(float), "incident")
    add("incident_max_frp", inc.map(det.groupby("incident_id")["frp"].max()), "incident")

    # --- prior-period NERIS county/month call density (no per-detection leakage)
    if neris is not None and not neris.empty:
        _prior_neris_density(X, groups, det, neris, local)

    X = X.replace([np.inf, -np.inf], np.nan)
    return X, groups


def _cross_source(X, groups, det, pts, hours):
    from sklearn.neighbors import BallTree
    idx = BallTree(pts, metric="haversine").query_radius(pts, r=2 / EARTH_KM)
    src = det["source"].str.lower().to_numpy()
    frp = pd.to_numeric(det["frp"], errors="coerce").to_numpy()
    n_other, max_other = np.zeros(len(det)), np.full(len(det), np.nan)
    for k, nb in enumerate(idx):
        ok = nb[(np.abs(np.nan_to_num(hours)[nb] - np.nan_to_num(hours)[k]) <= 3) & (src[nb] != src[k])]
        n_other[k] = len(set(src[ok]))
        if len(ok):
            max_other[k] = np.nanmax(frp[ok]) if not np.isnan(frp[ok]).all() else np.nan
    X["n_other_sources_2km_3h"], groups["n_other_sources_2km_3h"] = n_other, "cross_sensor"
    X["max_other_frp_2km_3h"], groups["max_other_frp_2km_3h"] = max_other, "cross_sensor"


def _prior_neris_density(X, groups, det, neris, local):
    """Statewide NERIS call count in the *previous* calendar month: a coarse reporting-intensity
    proxy that says nothing about this detection's own label."""
    per_month = neris.call_date.dt.to_period("M").value_counts()
    prev = local.dt.tz_localize(None).dt.to_period("M") - 1
    X["neris_prior_month_calls"] = prev.map(per_month).astype(float)
    groups["neris_prior_month_calls"] = "context"


def spatial_group(det: pd.DataFrame) -> np.ndarray:
    """Grid-cell ids used as CV groups so neighbouring detections never straddle train/test."""
    gx = np.floor(det["longitude"].to_numpy(dtype=float) / GRID_DEG).astype(int)
    gy = np.floor(det["latitude"].to_numpy(dtype=float) / GRID_DEG).astype(int)
    return gx * 100000 + gy
