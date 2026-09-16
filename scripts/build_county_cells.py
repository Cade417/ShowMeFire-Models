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


def build(sample: Path) -> dict:
    import numpy as np
    import xarray as xr
    from shapely.geometry import Point
    from shapely.prepared import prep

    with xr.open_dataset(sample, decode_cf=False) as ds:
        lat = np.asarray(ds["latitude"].values)
        lon = np.asarray(ds["longitude"].values)
        grid_shape = list(lat.shape)

    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)

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
        "source_file": sample.name,
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
    index = build(sample)

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
