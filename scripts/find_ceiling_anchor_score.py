"""
One-off: compute fire_weather_index's raw (pre-rescale) peak score for
eastern Missouri on 2025-03-14 - a real historic extreme fire-weather day
- to serve as the score-ceiling anchor (see factors.py::RAW_SCORE_CEILING).

Deliberately isolated from the production 91-day cache/panel: runs
add_stateful_features/add_antecedent_precip on a standalone 14-day frame
(2025-03-01 through 2025-03-14, HRRR only) rather than merging this into
the real panel, since that panel spans a completely different, much later
date range and mixing non-contiguous multi-year dates into the sequential
per-county GDD/KBDI accumulation would corrupt both.

Usage:
    python scripts/find_ceiling_anchor_score.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from fire_weather_index import factors
from fire_weather_index.build_county_days import add_antecedent_precip, process_one_run
from risk_fusion.build_county_days import add_stateful_features
from risk_fusion.county_geometry import load_counties

ANCHOR_DATE = "2025-03-14"
EASTERN_LON_CUTOFF = -91.0
ANCHOR_CACHE_DIR = REPO_ROOT / "data" / "cache" / "hrrr_ceiling_anchor"
CELL_CELLS_PATH = REPO_ROOT / "risk_fusion" / "county_cells.json"
FILENAME_RE = re.compile(r"hrrr_(\d{8})_(\d{2})z_f\d{2}-\d{2}\.nc$")


def _run_time_from_filename(path: Path):
    match = FILENAME_RE.search(path.name)
    if not match:
        return None
    return pd.Timestamp(f"{match.group(1)}T{match.group(2)}:00:00Z")


def main():
    counties = load_counties()
    name_by_fips = {c["fips"]: c["name"] for c in counties}
    eastern_fips = {c["fips"] for c in counties if c["centroid_lon"] > EASTERN_LON_CUTOFF}
    print(f"Eastern MO counties (centroid_lon > {EASTERN_LON_CUTOFF}), n={len(eastern_fips)}:")
    for fips in sorted(eastern_fips, key=lambda f: name_by_fips[f]):
        print(f"  {fips} {name_by_fips[fips]}")

    import json
    cell_index = json.loads(CELL_CELLS_PATH.read_text(encoding="utf-8"))
    cell_to_fips = cell_index["cell_to_fips"]
    expected_grid_shape = tuple(cell_index["grid_shape"])

    run_files = sorted(ANCHOR_CACHE_DIR.glob("hrrr_*.nc"))
    if not run_files:
        raise SystemExit(f"No cached HRRR files found in {ANCHOR_CACHE_DIR} - run the backfill first.")
    print(f"\nProcessing {len(run_files)} cached HRRR runs from {ANCHOR_CACHE_DIR}...")

    frames = []
    for path in run_files:
        run_time = _run_time_from_filename(path)
        if run_time is None:
            continue
        frame = process_one_run(path, run_time, cell_to_fips, expected_grid_shape=expected_grid_shape)
        if frame is not None:
            frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values("run_id").drop_duplicates(["county_fips", "valid_local_date"], keep="last")
    panel = add_stateful_features(panel)
    panel = add_antecedent_precip(panel)

    anchor_day = panel[panel["valid_local_date"] == ANCHOR_DATE]
    if anchor_day.empty:
        raise SystemExit(f"No rows for {ANCHOR_DATE} in the isolated panel - check the backfill covered this date.")

    results = []
    for _, row in anchor_day.iterrows():
        row_dict = row.to_dict()
        row_factors = factors.compute_factors(row_dict)
        score = factors.compute_score(row_factors)
        results.append((row["county_fips"], name_by_fips.get(row["county_fips"], "?"), score))

    eastern_results = sorted(
        ((fips, name, score) for fips, name, score in results if fips in eastern_fips and score is not None),
        key=lambda r: r[2], reverse=True,
    )

    print(f"\nEastern MO county scores for {ANCHOR_DATE} (raw, pre-rescale), descending:")
    for fips, name, score in eastern_results:
        print(f"  {score:.4f}  {name} ({fips})")

    if eastern_results:
        top_fips, top_name, top_score = eastern_results[0]
        print(f"\nRAW_SCORE_CEILING candidate = {top_score!r} ({top_name}, {top_fips})")
    else:
        print("\nNo eastern-MO scores computed - nothing to anchor on.")


if __name__ == "__main__":
    main()
