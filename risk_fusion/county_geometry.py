"""
County geometry, area, and spatial-region assignment for the risk-fusion
county-day panel.

Loads the vendored WGS84 GeoJSON produced by
scripts/export_county_geometry.py (never reads api/'s filesystem at
runtime - see that script's docstring for why).

Two things this module deliberately does NOT do yet, both documented so
they can't be mistaken for something more authoritative:

1. Burnable area: computing county area x non-water/non-developed NLCD
   fraction requires the NLCD raster, and training-data/static/source/ is
   empty in this workspace - nothing has been downloaded. Until that
   bundle exists, burnable_area_km2 falls back to the county's full
   geometric area with burnable_fraction_source="geometric_area_only",
   which the manifest and any consumer must check before treating the
   effort offset as anything more than a rough area proxy.

2. Climate-division regions: real NOAA/NCEI Missouri climate divisions
   are an external reference table this module does not have a verified
   copy of, and getting county-to-division assignment wrong would
   silently corrupt the spatial blocking scheme the split contract
   depends on for leakage control. Rather than guess, REGION_METHOD
   below is an explicit, reproducible, geometry-only substitute (a
   longitude/latitude grid over county centroids) that serves the same
   purpose - independent spatial strata - without claiming to be an
   official division. Swap in real NCEI divisions later by replacing
   assign_regions() and bumping REGION_METHOD; nothing downstream should
   care which method produced the region id, only that folds respect it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from pyproj import Transformer
from shapely.geometry import shape as shapely_shape
from shapely.ops import transform as shapely_transform

REPO_ROOT = Path(__file__).resolve().parent.parent
GEOJSON_PATH = Path(__file__).resolve().parent / "county_boundaries.geojson"

REGION_METHOD = "lon-lat-grid-v1"
N_LON_BINS = 3
N_LAT_BINS = 2

_TO_ALBERS = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True).transform


def load_counties() -> List[Dict]:
    """Returns [{fips, name, geometry_wgs84, centroid_lon, centroid_lat, area_km2}, ...]."""
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(
            f"{GEOJSON_PATH} is missing - run scripts/export_county_geometry.py "
            "from a workspace checkout with the api repo alongside this one."
        )
    data = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    counties = []
    for feature in data["features"]:
        geom_wgs84 = shapely_shape(feature["geometry"])
        geom_albers = shapely_transform(_TO_ALBERS, geom_wgs84)
        centroid = geom_wgs84.centroid
        counties.append({
            "fips": feature["properties"]["fips"],
            "name": feature["properties"]["name"],
            "geometry_wgs84": geom_wgs84,
            "centroid_lon": centroid.x,
            "centroid_lat": centroid.y,
            "area_km2": geom_albers.area / 1_000_000.0,
        })
    return counties


def burnable_area_km2(county: Dict) -> tuple:
    """
    Returns (burnable_area_km2, source). Falls back to full geometric area
    until an NLCD-derived fraction is available - see module docstring.
    """
    return county["area_km2"], "geometric_area_only"


def assign_regions(counties: List[Dict]) -> Dict[str, int]:
    """
    Deterministic geometry-only regional stratification - see module
    docstring for why this is not the real NCEI climate divisions.
    Returns {fips: region_id}, region_id in [0, N_LON_BINS * N_LAT_BINS).
    """
    lons = sorted(c["centroid_lon"] for c in counties)
    lats = sorted(c["centroid_lat"] for c in counties)

    def _bin_edges(sorted_values, n_bins):
        edges = []
        for i in range(1, n_bins):
            idx = min(len(sorted_values) - 1, round(i * len(sorted_values) / n_bins))
            edges.append(sorted_values[idx])
        return edges

    lon_edges = _bin_edges(lons, N_LON_BINS)
    lat_edges = _bin_edges(lats, N_LAT_BINS)

    def _bin_index(value, edges):
        index = 0
        for edge in edges:
            if value > edge:
                index += 1
        return index

    assignments = {}
    for county in counties:
        lon_bin = _bin_index(county["centroid_lon"], lon_edges)
        lat_bin = _bin_index(county["centroid_lat"], lat_edges)
        region_id = lat_bin * N_LON_BINS + lon_bin
        assignments[county["fips"]] = region_id
    return assignments


def build_county_table() -> List[Dict]:
    """The full per-county reference row: fips, name, area, burnable area/source, region."""
    counties = load_counties()
    regions = assign_regions(counties)
    table = []
    for county in counties:
        area, source = burnable_area_km2(county)
        table.append({
            "fips": county["fips"],
            "name": county["name"],
            "area_km2": round(county["area_km2"], 3),
            "burnable_area_km2": round(area, 3),
            "burnable_fraction_source": source,
            "region_id": regions[county["fips"]],
            "region_method": REGION_METHOD,
            "centroid_lon": round(county["centroid_lon"], 5),
            "centroid_lat": round(county["centroid_lat"], 5),
        })
    return table
