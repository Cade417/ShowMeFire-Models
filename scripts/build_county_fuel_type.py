#!/usr/bin/env python3
"""
Aggregates the existing static fire-behavior bundle (LANDFIRE Scott-Burgan
FBFM40 fuel model + canopy cover, already built by
static_features/build_fire_behavior_bundle.py - see data/static/bundles/) to
one row per Missouri county, for fire_weather_index's fuel-aware precip
relief threshold (fine grass fuels reach full relief with much less rain
than heavy timber litter/duff).

Reuses scripts/build_county_cells.py's own cell->county point-in-polygon
logic (the static bundle carries its own latitude/longitude coordinates,
same shape `build()` already expects from any source file) rather than
reimplementing that join, and risk_fusion/features.py's
reduce_cells_to_county for the actual aggregation.

Usage:
    python scripts/build_county_fuel_type.py
    python scripts/build_county_fuel_type.py --bundle path/to/specific_bundle.nc
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import xarray as xr

import paths
from build_county_cells import build as build_cell_index  # scripts/build_county_cells.py
from risk_fusion.features import reduce_cells_to_county

OUTPUT_PATH = REPO_ROOT / "fire_weather_index" / "county_fuel_type.json"
BUNDLE_GLOB = "fire_behavior_static_*.nc"
BUNDLE_VERSION_RE = re.compile(r"fire_behavior_static_(\d+)\.(\d+)\.nc$")

# Standard Scott-Burgan 40 fuel-model numeric groupings (LANDFIRE convention):
#   91/93/98/99  non-burnable (urban/water/barren/snow-ice)
#   101-109      grass (GR)
#   121-124      grass-shrub (GS)
#   141-149      shrub (SH)
#   161-165      timber understory (TU)
#   181-189      timber litter (TL)
#   201-204      slash-blowdown (SB)
# Legacy 1-13 (original 13 fuel models, pre-FBFM40) grouped the same way in
# case an older bundle ever carries them: 1-3 grass, 4-7 shrub, 8-10 timber
# litter, 11-13 slash.
def classify_fuel_group(fbfm40_code: Optional[float]) -> str:
    if fbfm40_code is None or not np.isfinite(fbfm40_code):
        return "unknown"
    code = int(round(fbfm40_code))
    if code in (91, 93, 98, 99) or code == 0:
        return "nonburnable"
    if 101 <= code <= 124 or 1 <= code <= 3:
        return "grass"
    if 141 <= code <= 149 or 4 <= code <= 7:
        return "shrub"
    if 161 <= code <= 204 or 8 <= code <= 13:
        return "timber"
    return "unknown"


def _find_latest_bundle(explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit
    candidates = sorted(
        paths.STATIC_BUNDLE_DIR.glob(BUNDLE_GLOB),
        key=lambda p: tuple(map(int, BUNDLE_VERSION_RE.search(p.name).groups())) if BUNDLE_VERSION_RE.search(p.name) else (0, 0),
    )
    if not candidates:
        raise SystemExit(f"No static fire-behavior bundle found under {paths.STATIC_BUNDLE_DIR} - "
                          "run static_features/build_fire_behavior_bundle.py first.")
    return candidates[-1]


def _dominant_code(values: np.ndarray) -> float:
    """Most common rounded fuel code among a county's cells - a mean of
    category codes is meaningless, this needs a mode."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    counts = Counter(int(round(v)) for v in finite)
    return float(counts.most_common(1)[0][0])


def build(bundle_path: Path) -> dict:
    cell_index = build_cell_index(bundle_path)
    cell_to_fips = cell_index["cell_to_fips"]

    with xr.open_dataset(bundle_path) as ds:
        fuel_codes = np.asarray(ds["fuel_model_fbfm40"].values, dtype="float64")
        canopy = np.asarray(ds["canopy_cover_pct"].values, dtype="float64")
        valid = np.asarray(ds["static_valid_mask"].values) == 1

    # Invalid/off-domain cells are coded 0 (confirmed live: every code==0
    # cell is also static_valid_mask==0) - excluded here rather than left
    # in, or they'd win the per-county dominant-class vote in any county
    # with enough edge/nodata cells, misclassifying it as "nonburnable"
    # for a reason that has nothing to do with its actual land cover.
    fuel_codes = np.where(valid, fuel_codes, np.nan)
    canopy = np.where(valid, canopy, np.nan)

    dominant_by_fips = reduce_cells_to_county(fuel_codes, cell_to_fips, _dominant_code)
    canopy_by_fips = reduce_cells_to_county(canopy, cell_to_fips, np.nanmean)

    counties = {}
    for fips in sorted(dominant_by_fips):
        dominant_code = dominant_by_fips[fips]
        counties[fips] = {
            "dominant_fbfm40": dominant_code,
            "fuel_group": classify_fuel_group(dominant_code),
            "canopy_cover_pct_mean": canopy_by_fips.get(fips),
        }
    return {
        "schema": "county-fuel-type-v1",
        "source_bundle": bundle_path.name,
        "grid_shape": cell_index["grid_shape"],
        "counties": counties,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None, help="Explicit static bundle .nc (default: latest under data/static/bundles)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    bundle_path = _find_latest_bundle(args.bundle)
    print(f"Aggregating fuel type from {bundle_path}...")
    result = build(bundle_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    group_counts = Counter(row["fuel_group"] for row in result["counties"].values())
    print(f"Wrote {args.output}: {len(result['counties'])} counties, fuel group counts: {dict(group_counts)}")


if __name__ == "__main__":
    main()
