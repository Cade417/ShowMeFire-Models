"""
Build the county x day feature panel from the cached HRRR archive.

Stage 0i of the fire_risk_fusion plan: prove the county-day aggregation
pipeline end-to-end over real data and produce the exposure denominators
(burnable_area_km2 x days) the effort offset needs - WITHOUT a target
column, since no fire-event labels are wired in yet. Two-pass design:

Pass 1 (per HRRR run, independent): reduce each run's grid to one row
per county with same-day weather aggregates (temp/rh/wind/vpd/precip/gust).
This is what parallelizes; each run only needs its own file.

Pass 2 (per county, sequential over its own days only): KBDI and GDD are
stateful accumulators, so they must be computed per county across that
county's sorted daily series - never mixed across counties.

fm_* columns are intentionally absent. Fuel moisture requires running the
trained FM model (XGBoost/V5 inference), not just the raw archived HRRR
grid - that is a separate, heavier step out of scope for this backfill.
fm_features_available=False on every row makes the gap explicit rather
than silently shipping a zero or a guess.

mean_annual_precip_mm (needed by KBDI) now comes from real per-county NOAA
1991-2020 climate normals (risk_fusion/county_precip_normals.json, built by
risk_fusion/build_precip_normals.py from NOAA NCEI's public Climate at a
Glance County Time Series data service - see that module's docstring for
the verified source/endpoint). Falls back to the old archive-derived proxy
(mean_daily_precip x 365, explicitly NOT a climate normal) only for a
county missing from that file, so a stale/regenerated environment degrades
rather than breaks.

Usage:
    python -m risk_fusion.build_county_days --limit 60
    python -m risk_fusion.build_county_days
"""
from __future__ import annotations

import argparse
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
from risk_fusion.county_geometry import load_counties
from spatial.precipitation import decode_forecast_precipitation

COUNTY_REFERENCE_PATH = REPO_ROOT / "risk_fusion" / "county_reference.json"
COUNTY_CELLS_PATH = REPO_ROOT / "risk_fusion" / "county_cells.json"
COUNTY_PRECIP_NORMALS_PATH = REPO_ROOT / "risk_fusion" / "county_precip_normals.json"
RUN_FILENAME_RE = re.compile(r"hrrr_(\d{8})_(\d{2})z_f\d{2}-\d{2}\.nc$")


def _load_county_cells() -> dict:
    import json
    return json.loads(COUNTY_CELLS_PATH.read_text(encoding="utf-8"))


def _load_county_reference() -> dict:
    import json
    return {row["fips"]: row for row in json.loads(COUNTY_REFERENCE_PATH.read_text(encoding="utf-8"))["counties"]}


def _load_county_precip_normals() -> dict:
    """{fips: mean_annual_precip_mm} from real NOAA normals - empty dict (not an
    error) if build_precip_normals.py hasn't been run yet in this environment,
    so add_stateful_features() falls back to the archive-derived proxy per
    county rather than failing outright."""
    import json
    if not COUNTY_PRECIP_NORMALS_PATH.exists():
        return {}
    data = json.loads(COUNTY_PRECIP_NORMALS_PATH.read_text(encoding="utf-8"))
    return {fips: row["mean_annual_precip_mm"] for fips, row in data.get("counties", {}).items()}


def _run_date_from_filename(path: Path) -> Optional[pd.Timestamp]:
    match = RUN_FILENAME_RE.search(path.name)
    if not match:
        return None
    date_str, hour_str = match.groups()
    return pd.Timestamp(f"{date_str}T{hour_str}:00:00Z")


def _lead_hours(ds: xr.Dataset) -> np.ndarray:
    leads = np.asarray(ds.step.values)
    if np.issubdtype(leads.dtype, np.timedelta64):
        return (leads / np.timedelta64(1, "h")).astype(int)
    return leads.astype(int)


def process_one_run(path: Path, cell_to_fips: dict, expected_grid_shape: Optional[tuple] = None) -> Optional[pd.DataFrame]:
    """One row per county for the day this HRRR run's afternoon/evening leads fall on.

    expected_grid_shape, if given, MUST match this file's actual (y, x)
    grid shape - see county_cells.json's own recorded grid_shape. This
    guards against a real, observed failure mode: a full-CONUS,
    uncropped HRRR file (no domain_bbox attribute - this repo's own
    fetch_hrrr() always sets one) landing in the cache directory under a
    filename that collides with this repo's own Missouri-cropped naming
    convention. Without this check, county_cells.json's cell indices
    (built for the small Missouri crop) get silently applied to a
    completely different, much larger grid, reading arbitrary/meaningless
    cells instead of Missouri's - confirmed live against real cached
    files from outside this repo's own fetch path. Raising here (rather
    than silently producing wrong numbers) is deliberate; build()'s
    existing per-file try/except turns this into a visible "skipped"
    line instead of a silent corruption.
    """
    run_time = _run_date_from_filename(path)
    if run_time is None:
        return None

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
                f"{path.name}: grid shape {tuple(rh.shape[-2:])} does not match county_cells.json's "
                f"recorded grid_shape {tuple(expected_grid_shape)} (domain_bbox attr: "
                f"{raw.attrs.get('domain_bbox')!r}) - this file was not produced by this repo's own "
                f"Missouri-cropped fetch_hrrr(). Using county_cells.json's cell indices against it "
                f"would silently read the wrong geographic cells. Re-fetch this run instead of trusting it."
            )
        wind_ms = np.hypot(np.asarray(ds["u10"].values), np.asarray(ds["v10"].values))
        wind_kts_reduced = wind_ms * rff.MPS_TO_KNOTS * rff.FIRE_WIND_REDUCTION_FACTOR
        vpd = rff.vapor_pressure_deficit_kpa(temp_c, rh)

        gust_available = "gust" in ds.data_vars
        gust_kts = (np.asarray(ds["gust"].values) * rff.MPS_TO_KNOTS) if gust_available else None

        precipitation = decode_forecast_precipitation(ds)
        interval_mm = np.asarray(precipitation.interval_mm.values)

    valid_date = (run_time + pd.Timedelta(hours=int(leads[full_mask][0]))).tz_convert("America/Chicago").normalize()

    rows = []
    counties = _load_county_reference()
    for fips in {v for v in cell_to_fips.values()}:
        cells = {k: v for k, v in cell_to_fips.items() if v == fips}
        temp_max = max(rff.reduce_cells_to_county(temp_c[i], cells, np.nanmax).get(fips, np.nan) for i in np.flatnonzero(full_mask))
        rh_full_vals = [rff.reduce_cells_to_county(rh[i], cells, np.nanmean).get(fips, np.nan) for i in np.flatnonzero(full_mask)]
        rh_afternoon_vals = [rff.reduce_cells_to_county(rh[i], cells, np.nanmin).get(fips, np.nan) for i in np.flatnonzero(afternoon_mask)]
        wind_full_vals = [rff.reduce_cells_to_county(wind_kts_reduced[i], cells, np.nanmax).get(fips, np.nan) for i in np.flatnonzero(full_mask)]
        vpd_full_vals = [rff.reduce_cells_to_county(vpd[i], cells, np.nanmax).get(fips, np.nan) for i in np.flatnonzero(full_mask)]
        # Spatial reducer is nanmean (average precip depth across the
        # county's cells for this lead) - nansum across cells would add up
        # rainfall at every pixel and inflate the total by the county's
        # cell count. Summing ACROSS STEPS afterward is correct: that's
        # accumulating the interval amounts over the day's time window.
        precip_vals = [rff.reduce_cells_to_county(interval_mm[i], cells, np.nanmean).get(fips, np.nan) for i in np.flatnonzero(full_mask)]
        gust_max = (max(rff.reduce_cells_to_county(gust_kts[i], cells, np.nanmax).get(fips, np.nan)
                        for i in np.flatnonzero(full_mask)) if gust_available else None)

        county_row = counties.get(fips, {})
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
            "gust_available": gust_available,
            "gust_kts_max": gust_max,
            "burnable_area_km2": county_row.get("burnable_area_km2"),
            "burnable_fraction_source": county_row.get("burnable_fraction_source"),
            "region_id": county_row.get("region_id"),
            "region_method": county_row.get("region_method"),
            "fm_features_available": False,
        })
    return pd.DataFrame(rows)


def add_stateful_features(panel: pd.DataFrame, precip_normals: Optional[dict] = None) -> pd.DataFrame:
    """Adds KBDI and GDD per county, sequentially over each county's own sorted days.

    precip_normals: {fips: mean_annual_precip_mm}, real NOAA 1991-2020 normals
    (see _load_county_precip_normals/build_precip_normals.py). Defaults to
    loading COUNTY_PRECIP_NORMALS_PATH if not passed explicitly. A county
    missing from it falls back to the archive-derived proxy, tagged as such.
    """
    if precip_normals is None:
        precip_normals = _load_county_precip_normals()

    panel = panel.sort_values(["county_fips", "valid_local_date"]).reset_index(drop=True)
    kbdi_values = np.full(len(panel), np.nan)
    kbdi_valid = np.zeros(len(panel), dtype=bool)
    gdd_values = np.full(len(panel), np.nan)
    precip_normal_source = np.empty(len(panel), dtype=object)

    for fips, group in panel.groupby("county_fips"):
        idx = group.index.to_numpy()
        dates = pd.DatetimeIndex(group["valid_local_date"])
        daily_precip = group["precip_24h_mm"].to_numpy()
        daily_max_temp = group["temp_max_c"].to_numpy()

        real_normal = precip_normals.get(fips)
        if real_normal is not None:
            mean_annual_precip_mm = real_normal
            source = "noaa_cag_1991_2020_normal"
        else:
            # Not a real climate normal - archive-derived proxy, used only
            # when this county is missing from county_precip_normals.json.
            mean_annual_precip_mm = max(1.0, float(np.nanmean(daily_precip)) * 365.0)
            source = "archive_derived_proxy_not_a_climate_normal"

        kbdi_result = rff.keetch_byram_drought_index(daily_max_temp, np.nan_to_num(daily_precip), mean_annual_precip_mm)
        kbdi_values[idx] = kbdi_result["kbdi"]
        kbdi_valid[idx] = kbdi_result["valid"]
        precip_normal_source[idx] = source

        gdd_values[idx] = rff.growing_degree_days((daily_max_temp + daily_max_temp) / 2.0, dates)

    panel["kbdi"] = kbdi_values
    panel["kbdi_valid"] = kbdi_valid
    panel["gdd_accum_since_mar1"] = gdd_values
    panel["mean_annual_precip_mm_source"] = precip_normal_source

    calendar = rff.calendar_features(pd.DatetimeIndex(panel["valid_local_date"]))
    for column in calendar.columns:
        panel[column] = calendar[column].to_numpy()
    return panel


def build(limit: Optional[int] = None, since: Optional[date] = None,
          run_files: Optional[list] = None) -> pd.DataFrame:
    """
    run_files, if given, is used exactly as passed (e.g. a single target
    run for live scoring - see score_live.py) and skips the glob/since
    filtering entirely - this is the one to reach for when you want a
    SPECIFIC day, since `since` is an open-ended "on or after" filter and
    would otherwise pull in every cached run between that date and
    whatever's most recently cached (2,500+ files as of this writing) into
    one undifferentiated multi-date panel.

    since restricts the glob to runs on/after that calendar date - for
    trimming a stale-but-still-bounded prefix off the archive, not for
    isolating a single day. None (the default for both) preserves the
    original full-archive behavior.
    """
    county_cells = _load_county_cells()
    cell_to_fips = county_cells["cell_to_fips"]
    expected_grid_shape = tuple(county_cells["grid_shape"])
    if run_files is None:
        run_files = sorted(paths.CACHE_HRRR_DIR.glob("hrrr_*.nc"))
        if since is not None:
            run_files = [path for path in run_files
                         if (run_time := _run_date_from_filename(path)) is not None and run_time.date() >= since]
    if limit:
        run_files = run_files[:limit]

    frames = []
    for i, path in enumerate(run_files, 1):
        try:
            frame = process_one_run(path, cell_to_fips, expected_grid_shape=expected_grid_shape)
            if frame is not None:
                frames.append(frame)
        except Exception as exc:
            print(f"  skipped {path.name}: {exc}")
        if i % 50 == 0:
            print(f"  processed {i}/{len(run_files)} runs...")

    if not frames:
        return pd.DataFrame()

    daily_panel = pd.concat(frames, ignore_index=True)
    # One row per (county, day): if multiple runs cover the same day, keep the latest run.
    daily_panel = daily_panel.sort_values("run_id").drop_duplicates(["county_fips", "valid_local_date"], keep="last")
    return add_stateful_features(daily_panel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N cached HRRR runs (dev/testing).")
    parser.add_argument("--output", type=Path, default=paths.RISK_FUSION_PANEL)
    args = parser.parse_args()

    print(f"Building county-day panel from {paths.CACHE_HRRR_DIR}...")
    panel = build(limit=args.limit)
    if panel.empty:
        print("No rows produced.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(panel)} rows, {panel['county_fips'].nunique()} counties, "
          f"{panel['valid_local_date'].nunique()} distinct days")


if __name__ == "__main__":
    main()
