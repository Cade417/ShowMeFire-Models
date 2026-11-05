"""
Real historical wind-FROM direction per station-hour, read directly from
the RTMA cache (`training-data/cache/rtma/rtma_{YYYYMMDD}_{HH}z.nc`) rather
than assumed or omitted. `training-data/aligned/station_leads*.csv` only
has wind SPEED (`rtma_wind_ms`) - direction was lost in that CSV's
per-lead-hour aggregation - but Rothermel needs a real wind-relative-to-
slope-aspect direction to compute spread direction/rate correctly.

Groups by unique hour rather than looking up per station-row: the same
cache file covers every station observed in that hour, so each file is
opened once regardless of how many stations/rows need it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


def _cache_path(cache_dir: Path, hour: pd.Timestamp) -> Path:
    return cache_dir / f"rtma_{hour:%Y%m%d}_{hour:%H}z.nc"


def wind_from_degrees(u_ms: np.ndarray, v_ms: np.ndarray) -> np.ndarray:
    """Meteorological wind-FROM direction in degrees clockwise from north."""
    return (np.degrees(np.arctan2(-u_ms, -v_ms)) + 360.0) % 360.0


def lookup_hour_wind(cache_dir: Path, hour: pd.Timestamp, stations: pd.DataFrame) -> pd.Series | None:
    """
    wind_from_deg per station_id for one hour, nearest RTMA grid cell to
    each station's lat/lon. None if that hour's cache file is missing -
    the caller must treat that as "no data for this hour", not a default.
    """
    path = _cache_path(cache_dir, hour)
    if not path.is_file():
        return None
    with xr.open_dataset(path) as ds:
        lat = np.asarray(ds.latitude.values, dtype=float)
        lon = np.asarray(ds.longitude.values, dtype=float)
        u = np.asarray(ds.u10.values, dtype=float)
        v = np.asarray(ds.v10.values, dtype=float)

    tree = cKDTree(np.column_stack((lat.ravel(), lon.ravel())))
    _, index = tree.query(stations[["lat", "lon"]].to_numpy(dtype=float))
    u_at_station = u.ravel()[index]
    v_at_station = v.ravel()[index]
    return pd.Series(
        wind_from_degrees(u_at_station, v_at_station), index=stations["station_id"].to_numpy(),
    )


def build_wind_direction_lookup(cache_dir: Path, hours: pd.Series, stations: pd.DataFrame) -> pd.DataFrame:
    """
    `hours` need not be unique - each distinct hour (floored) is looked up
    once. Returns a long (valid_time_hour, station_id, wind_from_deg)
    frame; hours whose cache file is missing are simply absent from the
    result, so a subsequent merge leaves those rows' wind_from_deg null
    rather than silently defaulted.
    """
    unique_hours = pd.to_datetime(hours).dt.floor("h").unique()
    frames = []
    for hour in unique_hours:
        winds = lookup_hour_wind(cache_dir, pd.Timestamp(hour), stations)
        if winds is None:
            continue
        frames.append(pd.DataFrame({
            "valid_time_hour": hour, "station_id": winds.index, "wind_from_deg": winds.to_numpy(),
        }))
    if not frames:
        return pd.DataFrame(columns=["valid_time_hour", "station_id", "wind_from_deg"])
    return pd.concat(frames, ignore_index=True)
