"""
Build the blended HRRR + RRFS + FV3-HIRES county-day panel for fire_weather_index.

HRRR is the required backbone (full history, proven extraction - the same
approach risk_fusion/build_county_days.py already uses). RRFS and FV3-HIRES
contribute additional, independently-captured versions of the same
variables (see spatial/rrfs_capture.py and spatial/fv3hires_capture.py -
both write the exact same variable names as HRRR: t2m, r2, u10, v10, tp,
optionally gust) wherever a cached run exists for a given county-day. The
per-source extraction logic below (_process_one_run) is therefore a single
generic function, not three copies - each source just brings its own
cell_to_fips map and cached file glob.

Blending: for each (county, day), whichever sources produced a row
contribute to a simple mean of that variable - HRRR is never overridden,
only averaged WITH RRFS/FV3-HIRES when they're also present. rrfs_available
/ fv3hires_available flags record which rows actually got a contribution
from each source, so a future retrain can tell blended-input rows apart
from HRRR-only ones without re-deriving it.

FV3-HIRES's temp_max_c is built from its (TMAX+TMIN)/2 proxy (see
fv3hires_capture.py's module docstring) - averaged in like everything
else, but this is a real caveat on precision, not just a naming detail.

Usage:
    python -m fire_weather_index.build_county_days --limit 60
    python -m fire_weather_index.build_county_days
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths
from risk_fusion import features as rff
from risk_fusion.build_county_days import add_stateful_features, _load_county_precip_normals
from spatial.precipitation import decode_forecast_precipitation

PACKAGE_DIR = Path(__file__).resolve().parent
COUNTY_REFERENCE_PATH = REPO_ROOT / "risk_fusion" / "county_reference.json"

SOURCES = {
    "hrrr": {"cache_dir": paths.CACHE_HRRR_DIR, "glob": "hrrr_*.nc",
             "filename_re": re.compile(r"hrrr_(\d{8})_(\d{2})z_f\d{2}-\d{2}\.nc$"),
             "cell_cells": REPO_ROOT / "risk_fusion" / "county_cells.json"},
    "rrfs": {"cache_dir": paths.CACHE_RRFS_DIR, "glob": "rrfs_*.nc",
             "filename_re": re.compile(r"rrfs_(\d{8})_(\d{2})z_f\d{2}-\d{2}\.nc$"),
             "cell_cells": PACKAGE_DIR / "county_cells_rrfs.json"},
    "fv3hires": {"cache_dir": paths.CACHE_FV3HIRES_DIR, "glob": "fv3hires_*.nc",
                 "filename_re": re.compile(r"fv3hires_(\d{8})_(\d{2})z_f\d{2}-\d{2}\.nc$"),
                 "cell_cells": PACKAGE_DIR / "county_cells_fv3hires.json"},
}

BLEND_COLUMNS = ["temp_max_c", "rh_mean", "rh_min_afternoon", "wind_kts_max", "wind_kts_p90",
                  "vpd_kpa_max", "precip_24h_mm"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_county_reference() -> dict:
    return {row["fips"]: row for row in _load_json(COUNTY_REFERENCE_PATH)["counties"]}


def _run_time_from_filename(path: Path, filename_re) -> Optional[pd.Timestamp]:
    match = filename_re.search(path.name)
    if not match:
        return None
    date_str, hour_str = match.groups()
    return pd.Timestamp(f"{date_str}T{hour_str}:00:00Z")


def _lead_hours(ds: xr.Dataset) -> np.ndarray:
    leads = np.asarray(ds.step.values)
    if np.issubdtype(leads.dtype, np.timedelta64):
        return (leads / np.timedelta64(1, "h")).astype(int)
    return leads.astype(int)


def process_one_run(path: Path, run_time: pd.Timestamp, cell_to_fips: dict,
                     expected_grid_shape: Optional[tuple] = None) -> Optional[pd.DataFrame]:
    """One row per county for the day this run's afternoon/evening leads fall
    on. Generic across HRRR/RRFS/FV3-HIRES - all three capture modules write
    the same variable names (see this module's docstring)."""
    raw = xr.open_dataset(path, decode_cf=False)
    for name in raw.variables:
        raw[name].attrs.pop("dtype", None)
    with xr.decode_cf(raw, decode_timedelta=True) as ds:
        leads = _lead_hours(ds)
        afternoon_mask = np.isin(leads, list(rff.AFTERNOON_LEAD_HOURS))
        full_mask = np.isin(leads, list(rff.FULL_LEAD_HOURS))
        if not full_mask.any():
            return None

        temp_c = np.asarray(ds["t2m"].values) - 273.15
        rh = np.asarray(ds["r2"].values)
        if expected_grid_shape is not None and tuple(rh.shape[-2:]) != tuple(expected_grid_shape):
            raise ValueError(
                f"{path.name}: grid shape {tuple(rh.shape[-2:])} does not match this source's "
                f"county_cells.json grid_shape {tuple(expected_grid_shape)} - re-fetch instead of trusting it."
            )
        wind_ms = np.hypot(np.asarray(ds["u10"].values), np.asarray(ds["v10"].values))
        wind_kts_reduced = wind_ms * rff.MPS_TO_KNOTS * rff.FIRE_WIND_REDUCTION_FACTOR
        vpd = rff.vapor_pressure_deficit_kpa(temp_c, rh)
        precipitation = decode_forecast_precipitation(ds)
        interval_mm = np.asarray(precipitation.interval_mm.values)

    valid_date = (run_time + pd.Timedelta(hours=int(leads[full_mask][0]))).tz_convert("America/Chicago").normalize()

    rows = []
    for fips in {v for v in cell_to_fips.values()}:
        cells = {k: v for k, v in cell_to_fips.items() if v == fips}
        temp_max = max(rff.reduce_cells_to_county(temp_c[i], cells, np.nanmax).get(fips, np.nan) for i in np.flatnonzero(full_mask))
        rh_full_vals = [rff.reduce_cells_to_county(rh[i], cells, np.nanmean).get(fips, np.nan) for i in np.flatnonzero(full_mask)]
        rh_afternoon_vals = [rff.reduce_cells_to_county(rh[i], cells, np.nanmin).get(fips, np.nan) for i in np.flatnonzero(afternoon_mask)]
        wind_full_vals = [rff.reduce_cells_to_county(wind_kts_reduced[i], cells, np.nanmax).get(fips, np.nan) for i in np.flatnonzero(full_mask)]
        vpd_full_vals = [rff.reduce_cells_to_county(vpd[i], cells, np.nanmax).get(fips, np.nan) for i in np.flatnonzero(full_mask)]
        precip_vals = [rff.reduce_cells_to_county(interval_mm[i], cells, np.nanmean).get(fips, np.nan) for i in np.flatnonzero(full_mask)]

        rows.append({
            "county_fips": fips,
            "valid_local_date": valid_date.strftime("%Y-%m-%d"),
            "run_id": run_time.strftime("%Y%m%d_%H"),
            "temp_max_c": temp_max,
            "rh_mean": float(np.nanmean(rh_full_vals)) if rh_full_vals else np.nan,
            "rh_min_afternoon": float(np.nanmin(rh_afternoon_vals)) if rh_afternoon_vals else np.nan,
            "wind_kts_max": float(np.nanmax(wind_full_vals)) if wind_full_vals else np.nan,
            "wind_kts_p90": float(np.nanpercentile(wind_full_vals, 90)) if wind_full_vals else np.nan,
            "vpd_kpa_max": float(np.nanmax(vpd_full_vals)) if vpd_full_vals else np.nan,
            "precip_24h_mm": float(np.nansum(precip_vals)) if precip_vals else np.nan,
        })
    return pd.DataFrame(rows)


def build_source_panel(source: str, limit: Optional[int] = None, since: Optional[date] = None) -> pd.DataFrame:
    """One county-day panel for a single source (no blending, no stateful features yet)."""
    config = SOURCES[source]
    cell_index = _load_json(config["cell_cells"])
    cell_to_fips = cell_index["cell_to_fips"]
    expected_grid_shape = tuple(cell_index["grid_shape"])

    run_files = sorted(config["cache_dir"].glob(config["glob"]))
    if since is not None:
        run_files = [path for path in run_files
                     if (run_time := _run_time_from_filename(path, config["filename_re"])) is not None
                     and run_time.date() >= since]
    if limit:
        run_files = run_files[:limit]

    frames = []
    for path in run_files:
        run_time = _run_time_from_filename(path, config["filename_re"])
        if run_time is None:
            continue
        try:
            frame = process_one_run(path, run_time, cell_to_fips, expected_grid_shape=expected_grid_shape)
            if frame is not None:
                frames.append(frame)
        except Exception as exc:
            print(f"  [{source}] skipped {path.name}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["county_fips", "valid_local_date", "run_id"] + BLEND_COLUMNS)

    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values("run_id").drop_duplicates(["county_fips", "valid_local_date"], keep="last")


def blend_sources(hrrr: pd.DataFrame, rrfs: pd.DataFrame, fv3hires: pd.DataFrame) -> pd.DataFrame:
    """Outer-joins the three per-source panels on (county_fips,
    valid_local_date) and averages BLEND_COLUMNS across whichever sources
    have a value for that row - HRRR is never overridden, only ever
    averaged together with RRFS/FV3-HIRES when they're also present.
    rrfs_available/fv3hires_available flags which rows got a contribution
    from that source."""
    key = ["county_fips", "valid_local_date"]
    hrrr = hrrr.rename(columns={c: f"{c}__hrrr" for c in BLEND_COLUMNS})
    rrfs = rrfs.rename(columns={c: f"{c}__rrfs" for c in BLEND_COLUMNS})[key + [f"{c}__rrfs" for c in BLEND_COLUMNS]]
    fv3hires = fv3hires.rename(columns={c: f"{c}__fv3hires" for c in BLEND_COLUMNS})[key + [f"{c}__fv3hires" for c in BLEND_COLUMNS]]

    merged = hrrr.merge(rrfs, on=key, how="left").merge(fv3hires, on=key, how="left")
    merged["rrfs_available"] = merged[f"{BLEND_COLUMNS[0]}__rrfs"].notna()
    merged["fv3hires_available"] = merged[f"{BLEND_COLUMNS[0]}__fv3hires"].notna()

    for column in BLEND_COLUMNS:
        source_columns = [f"{column}__hrrr", f"{column}__rrfs", f"{column}__fv3hires"]
        merged[column] = merged[source_columns].mean(axis=1, skipna=True)
        merged = merged.drop(columns=source_columns)

    keep = key + ["run_id"] + BLEND_COLUMNS + ["rrfs_available", "fv3hires_available",
                  "burnable_area_km2", "burnable_fraction_source", "region_id", "region_method"]
    return merged[[c for c in keep if c in merged.columns]]


ANTECEDENT_PRECIP_LOOKBACK_DAYS = 3  # FIRST_PASS - see add_antecedent_precip


def add_antecedent_precip(panel: pd.DataFrame, lookback_days: int = ANTECEDENT_PRECIP_LOOKBACK_DAYS) -> pd.DataFrame:
    """Adds antecedent_precip_mm/antecedent_precip_days per county-day: real
    rain from the PRECEDING `lookback_days` days, not the current day's own
    forecast-window precip_24h_mm.

    This is not a new live data source - it reuses the same amortization
    principle risk_fusion's own KBDI already depends on (see
    risk_fusion/features.py::keetch_byram_drought_index and this package's
    __init__.py docstring): once a day has passed, that day's own
    forecast-window precip_24h_mm IS a reasonable proxy for what actually
    fell, so summing the last few already-computed days' own precip_24h_mm
    values (sequentially per county, same shape as add_stateful_features's
    KBDI/GDD pass) gives a real "it rained recently" signal without a new
    fetch. antecedent_precip_days records how many of the lookback days
    actually had a prior row (0-`lookback_days`) - a fresh county history
    or a gap in the cache means fewer days counted, never a silently
    assumed zero-rain history."""
    panel = panel.sort_values(["county_fips", "valid_local_date"]).reset_index(drop=True)
    antecedent_mm = np.full(len(panel), np.nan)
    antecedent_days = np.zeros(len(panel), dtype=int)

    for _, group in panel.groupby("county_fips"):
        idx = group.index.to_numpy()
        precip = group["precip_24h_mm"].to_numpy()
        for position in range(len(idx)):
            window_start = max(0, position - lookback_days)
            window = precip[window_start:position]  # excludes the current day itself
            valid_window = window[np.isfinite(window)]
            antecedent_mm[idx[position]] = float(np.sum(valid_window)) if valid_window.size else 0.0
            antecedent_days[idx[position]] = int(valid_window.size)

    panel["antecedent_precip_mm"] = antecedent_mm
    panel["antecedent_precip_days"] = antecedent_days
    return panel


def build(limit: Optional[int] = None, since: Optional[date] = None) -> pd.DataFrame:
    print("Building HRRR panel (backbone)...")
    hrrr = build_source_panel("hrrr", limit=limit, since=since)
    if hrrr.empty:
        return pd.DataFrame()

    counties = _load_county_reference()
    hrrr["burnable_area_km2"] = hrrr["county_fips"].map(lambda f: counties.get(f, {}).get("burnable_area_km2"))
    hrrr["burnable_fraction_source"] = hrrr["county_fips"].map(lambda f: counties.get(f, {}).get("burnable_fraction_source"))
    hrrr["region_id"] = hrrr["county_fips"].map(lambda f: counties.get(f, {}).get("region_id"))
    hrrr["region_method"] = hrrr["county_fips"].map(lambda f: counties.get(f, {}).get("region_method"))

    print("Building RRFS panel (blend input)...")
    rrfs = build_source_panel("rrfs")
    print("Building FV3-HIRES panel (blend input)...")
    fv3hires = build_source_panel("fv3hires")

    print(f"  hrrr rows={len(hrrr)} rrfs rows={len(rrfs)} fv3hires rows={len(fv3hires)}")
    blended = blend_sources(hrrr, rrfs, fv3hires)
    panel = add_stateful_features(blended, precip_normals=_load_county_precip_normals())
    return add_antecedent_precip(panel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N cached HRRR runs (dev/testing).")
    parser.add_argument("--output", type=Path, default=paths.FIRE_WEATHER_INDEX_PANEL)
    args = parser.parse_args()

    panel = build(limit=args.limit)
    if panel.empty:
        print("No rows produced.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(panel)} rows, {panel['county_fips'].nunique()} counties, "
          f"{panel['valid_local_date'].nunique()} distinct days, "
          f"rrfs_available={int(panel['rrfs_available'].sum())}, fv3hires_available={int(panel['fv3hires_available'].sum())}")


if __name__ == "__main__":
    main()
