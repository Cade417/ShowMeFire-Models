"""
Build the HRRR grid-cell -> Missouri county index used by
risk_fusion/build_county_days.py to aggregate cell-level weather up to
county-day rows.

Needs one real cached HRRR file for its grid shape/coordinates (any file
under CACHE_HRRR_DIR works - the grid is the same for every run of the
same product). Point-in-polygon against the vendored county boundaries
(risk_fusion/county_boundaries.geojson) with a per-county bounding-box
prefilter, same approach as api/services/county_lookup.py.

Usage:
    python scripts/build_county_cells.py
    python scripts/build_county_cells.py --hrrr-sample path/to/some_hrrr.nc
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths
from risk_fusion.county_geometry import load_counties

OUTPUT_PATH = REPO_ROOT / "risk_fusion" / "county_cells.json"

# api/forecast/DailyForecast.py crops HRRR in TWO steps before handing
# hourly grids to its shadow hooks: first core.domain.crop()'s buffered
# MO_BUFFERED_BBOX (what the plain "hrrr" source below reproduces, (273,
# 267)), then this second, tighter box (DailyForecast.py's own
# `mo_bounds`, ~line 1572) right before building the per-hour grids the
# hooks actually receive, (196, 205). A county_cells.json meant to index
# THOSE grids (api/core/risk_fusion_reference/county_cells.json) must be
# built against this narrower crop, not the buffered one - see
# api/core/risk_fusion_county_reference.py's docstring. Keep this in sync
# with DailyForecast.py's mo_bounds if that ever changes.
API_MO_BOUNDS = {"lat": (35.8, 40.8), "lon": (-95.8, -89.1)}

# fire_weather_index blends HRRR (backbone) with RRFS/FV3-HIRES, each on its
# own grid/projection/resolution - each needs its own cell->county index,
# built the same way as HRRR's but pointed at that source's own cache dir.
SOURCE_CONFIG = {
    "hrrr": {"cache_dir": paths.CACHE_HRRR_DIR, "glob": "hrrr_*.nc",
             "output": REPO_ROOT / "risk_fusion" / "county_cells.json"},
    "rrfs": {"cache_dir": paths.CACHE_RRFS_DIR, "glob": "rrfs_*.nc",
             "output": REPO_ROOT / "fire_weather_index" / "county_cells_rrfs.json"},
    "fv3hires": {"cache_dir": paths.CACHE_FV3HIRES_DIR, "glob": "fv3hires_*.nc",
                 "output": REPO_ROOT / "fire_weather_index" / "county_cells_fv3hires.json"},
    # Same HRRR cache as "hrrr", but additionally cropped to
    # API_MO_BOUNDS - use this to regenerate
    # api/core/risk_fusion_reference/county_cells.json (copy the output
    # over by hand, same as the plain "hrrr" output always has been).
    "hrrr_api": {"cache_dir": paths.CACHE_HRRR_DIR, "glob": "hrrr_*.nc",
                 "output": REPO_ROOT / "risk_fusion" / "county_cells_api.json",
                 "api_crop": True},
}


def _find_sample(explicit: Path | None, source: str = "hrrr") -> Path:
    if explicit:
        return explicit
    config = SOURCE_CONFIG[source]
    candidates = sorted(config["cache_dir"].glob(config["glob"]))
    if not candidates:
        raise SystemExit(
            f"No cached {source} file found under {config['cache_dir']}. "
            f"Pass --sample explicitly, or fetch one {source} run first."
        )
    return candidates[0]


def _crop_to_api_bounds(lat, lon):
    """
    Verbatim port of DailyForecast.py's second mo_bounds crop (the mask +
    bounding-box-of-mask logic around its own `mo_bounds` variable) - see
    API_MO_BOUNDS above for why this exists as a second step.
    """
    import numpy as np

    lat_mask = (lat >= API_MO_BOUNDS["lat"][0]) & (lat <= API_MO_BOUNDS["lat"][1])
    lon_mask = (lon >= API_MO_BOUNDS["lon"][0]) & (lon <= API_MO_BOUNDS["lon"][1])
    combined_mask = lat_mask & lon_mask

    rows_with_data = np.any(combined_mask, axis=1)
    cols_with_data = np.any(combined_mask, axis=0)
    row_indices = np.where(rows_with_data)[0]
    col_indices = np.where(cols_with_data)[0]
    if not len(row_indices) or not len(col_indices):
        raise SystemExit("API_MO_BOUNDS crop does not intersect the sampled HRRR grid")
    row_min, row_max = row_indices[0], row_indices[-1] + 1
    col_min, col_max = col_indices[0], col_indices[-1] + 1
    return lat[row_min:row_max, col_min:col_max], lon[row_min:row_max, col_min:col_max]


def build(sample: Path, api_crop: bool = False) -> dict:
    import numpy as np
    import xarray as xr
    from shapely.geometry import Point
    from shapely.prepared import prep

    with xr.open_dataset(sample, decode_cf=False) as ds:
        lat = np.asarray(ds["latitude"].values)
        lon = np.asarray(ds["longitude"].values)

    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)

    if api_crop:
        lat, lon = _crop_to_api_bounds(lat, lon)
    grid_shape = list(lat.shape)

    counties = load_counties()
    prepared = [(prep(c["geometry_wgs84"]), c["geometry_wgs84"].bounds, c["fips"]) for c in counties]

    assignments = {}
    y_size, x_size = lat.shape
    for y in range(y_size):
        for x in range(x_size):
            cell_lat = float(lat[y, x])
            cell_lon = float(lon[y, x])
            for geom, (minx, miny, maxx, maxy), fips in prepared:
                if not (minx <= cell_lon <= maxx and miny <= cell_lat <= maxy):
                    continue
                if geom.contains(Point(cell_lon, cell_lat)):
                    assignments[f"{y},{x}"] = fips
                    break

    counties_hit = len(set(assignments.values()))
    return {
        "schema": "county-cells-v1",
        "grid_shape": grid_shape,
        "source_file": f"{sample.name} (+ API_MO_BOUNDS crop)" if api_crop else sample.name,
        "cell_count": len(assignments),
        "counties_covered": counties_hit,
        "cell_to_fips": assignments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCE_CONFIG), default="hrrr")
    parser.add_argument("--sample", type=Path, default=None, help="Explicit sample file (overrides --source's auto-discovery)")
    parser.add_argument("--hrrr-sample", type=Path, default=None, help="Deprecated alias for --sample")
    parser.add_argument("--output", type=Path, default=None, help="Overrides the default output path for --source")
    args = parser.parse_args()

    explicit_sample = args.sample or args.hrrr_sample
    sample = _find_sample(explicit_sample, args.source)
    output_path = args.output or SOURCE_CONFIG[args.source]["output"]
    print(f"Building {args.source} county-cell index from {sample}...")
    index = build(sample, api_crop=SOURCE_CONFIG[args.source].get("api_crop", False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index), encoding="utf-8")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"Wrote {output_path}: {index['cell_count']} cells across {index['counties_covered']} counties "
          f"(grid {index['grid_shape']}, sha256={digest[:12]}...)")

    if index["counties_covered"] < 115:
        print(f"WARNING: only {index['counties_covered']}/115 counties have at least one grid cell centroid "
              "inside them - small counties may need nearest-cell fallback, not just centroid containment.")


if __name__ == "__main__":
    main()
