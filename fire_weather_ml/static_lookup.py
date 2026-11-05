"""
Station -> static-terrain nearest-neighbor lookup against the already-built
fire-behavior static bundle (`data/static/bundles/fire_behavior_static_*.nc`
- same file api/services/fire_behavior_static.py::load_static_fields loads
in production). Independent implementation (no import from api/), reusing
the same cKDTree nearest-neighbor pattern already established in
api/services/spread_rate_moisture.py::_interpolate_field.

The bundle only covers the Missouri domain (bbox roughly
(-96.8, -88.1, 34.8, 41.8)) - a station outside that domain, or one whose
nearest valid cell is implausibly far away, has no real terrain to attach
and is excluded rather than guessed.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# A grid cell roughly this far from a station (in degrees) is not a
# meaningful nearest-neighbor match - the bundle's ~3.1km cell spacing
# means anything beyond a couple of cells' width is likely off-grid
# entirely rather than just imprecise.
MAX_MATCH_DISTANCE_DEG = 0.1


def load_static_bundle(bundle_path: Path) -> dict:
    """Loads the subset of bundle fields static_lookup needs, as plain numpy arrays."""
    with xr.open_dataset(bundle_path) as ds:
        return {
            "lat": np.asarray(ds.latitude.values, dtype=float),
            "lon": np.asarray(ds.longitude.values, dtype=float),
            "slope_deg": np.asarray(ds.slope_degrees.values, dtype=float),
            "aspect_sin": np.asarray(ds.aspect_sin.values, dtype=float),
            "aspect_cos": np.asarray(ds.aspect_cos.values, dtype=float),
            "canopy_cover_pct": np.asarray(ds.canopy_cover_pct.values, dtype=float),
            "canopy_height_m": np.asarray(ds.canopy_height_m.values, dtype=float),
            "fuel_model_code": np.rint(np.asarray(ds.fuel_model_fbfm40.values, dtype=float)).astype(np.int32),
            "valid_mask": np.asarray(ds.static_valid_mask.values, dtype=float) > 0.5,
        }


def _aspect_degrees(sin_component: np.ndarray, cos_component: np.ndarray) -> np.ndarray:
    return (np.degrees(np.arctan2(sin_component, cos_component)) + 360.0) % 360.0


def build_station_lookup(bundle: dict, stations: pd.DataFrame,
                         max_distance_deg: float = MAX_MATCH_DISTANCE_DEG) -> pd.DataFrame:
    """
    `stations` must have station_id/lat/lon columns. Returns a DataFrame
    indexed by station_id with fuel_model_code/slope_deg/aspect_deg/
    canopy_cover_pct/canopy_height_m/match_distance_deg - one row per
    station whose nearest VALID grid cell is within max_distance_deg;
    stations outside the bundle's domain or too far from any valid cell
    are silently excluded from the result (the caller decides what to do
    with a station that disappears here), not defaulted.
    """
    valid = bundle["valid_mask"]
    valid_lat = bundle["lat"][valid]
    valid_lon = bundle["lon"][valid]
    valid_rows, valid_cols = np.where(valid)

    tree = cKDTree(np.column_stack((valid_lat, valid_lon)))
    query_points = stations[["lat", "lon"]].to_numpy(dtype=float)
    distance, index = tree.query(query_points)

    rows, cols = valid_rows[index], valid_cols[index]
    result = pd.DataFrame({
        "station_id": stations["station_id"].to_numpy(),
        "match_distance_deg": distance,
        "fuel_model_code": bundle["fuel_model_code"][rows, cols],
        "slope_deg": bundle["slope_deg"][rows, cols],
        "aspect_deg": _aspect_degrees(bundle["aspect_sin"][rows, cols], bundle["aspect_cos"][rows, cols]),
        "canopy_cover_pct": bundle["canopy_cover_pct"][rows, cols],
        "canopy_height_m": bundle["canopy_height_m"][rows, cols],
    })
    return result[result["match_distance_deg"] <= max_distance_deg].set_index("station_id")
