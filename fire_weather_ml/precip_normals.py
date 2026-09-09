"""
Station -> mean-annual-precipitation-normal lookup, for KBDI's climate-
normal term (features.py::kbdi_step's mean_annual_precip_in). Reuses
risk_fusion's already-computed DATA files - `risk_fusion/county_boundaries.geojson`
(county polygons) and `risk_fusion/county_precip_normals.json` (real 1991-2020
NOAA NCEI normals per county) - via a nearest-county spatial join, not by
importing any risk_fusion Python module. Data reuse is fine under this
project's model-family boundary; code reuse is not (see
docs/fire_weather_ml_plan.md).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

DEFAULT_COUNTY_BOUNDARIES_PATH = Path(__file__).resolve().parent.parent / "risk_fusion" / "county_boundaries.geojson"
DEFAULT_PRECIP_NORMALS_PATH = Path(__file__).resolve().parent.parent / "risk_fusion" / "county_precip_normals.json"

MM_PER_IN = 25.4


def load_county_precip_normals_in(path: Path = DEFAULT_PRECIP_NORMALS_PATH) -> Dict[str, float]:
    """fips -> mean_annual_precip_in (converted from the file's native mm)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {fips: entry["mean_annual_precip_mm"] / MM_PER_IN for fips, entry in payload["counties"].items()}


def station_to_county_fips(stations: pd.DataFrame, boundaries_path: Path = DEFAULT_COUNTY_BOUNDARIES_PATH) -> pd.Series:
    """
    `stations` must have station_id/lat/lon columns. Returns a Series
    indexed by station_id -> county fips, for stations whose point falls
    inside a real county polygon (a station outside every county - e.g.
    just across a state line - gets no fips, not a guessed nearest one).
    """
    counties = gpd.read_file(boundaries_path)
    points = gpd.GeoDataFrame(
        {"station_id": stations["station_id"].to_numpy()},
        geometry=[Point(lon, lat) for lon, lat in zip(stations["lon"], stations["lat"])],
        crs=counties.crs,
    )
    joined = gpd.sjoin(points, counties[["fips", "geometry"]], how="inner", predicate="within")
    return joined.set_index("station_id")["fips"]


def build_station_precip_normals(
    stations: pd.DataFrame,
    boundaries_path: Path = DEFAULT_COUNTY_BOUNDARIES_PATH,
    normals_path: Path = DEFAULT_PRECIP_NORMALS_PATH,
) -> Dict[str, float]:
    """station_id -> mean_annual_precip_in, for stations that matched a county with a recorded normal."""
    fips_by_station = station_to_county_fips(stations, boundaries_path)
    normals_by_fips = load_county_precip_normals_in(normals_path)
    return {
        station_id: normals_by_fips[fips]
        for station_id, fips in fips_by_station.items()
        if fips in normals_by_fips
    }
