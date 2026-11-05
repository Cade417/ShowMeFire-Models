"""
Blends the fire_risk_fusion county-level "today vs seasonal norm" ratio
onto a REAL sub-county forecast - the production system's own per-station
hourly archive (training-data/archive/forecasts/station_forecasts_beta_*.json,
19 Missouri stations with real hourly fuel_moisture/rh/wind/fire_danger).

This is NOT the full ~20,000-cell production grid (that only exists
transiently inside the live api/ forecast run and needs a fine-grained
fuel-moisture model this project doesn't have yet - see the project plan).
It's a real, already-on-disk, genuinely sub-county spatial unit, used here
to validate the blending logic before any production wiring.

The blended value is a THIRD, clearly-labeled experimental figure sitting
alongside the station's own already-reported fire_danger category - never
overwriting or relabeling it, per the project's standing honesty
invariant of keeping rule-derived and history-informed outputs separate.

Usage:
    python -m risk_fusion.blend_station_forecast
    python -m risk_fusion.blend_station_forecast --archive path/to/station_forecasts_beta_20260711_12.json
    python -m risk_fusion.blend_station_forecast --since 2026-07-05 --until 2026-07-11
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths
from risk_fusion import features, model_bundle, rule_uncertainty as ru
from risk_fusion.build_county_days import build as build_county_days
from risk_fusion.county_geometry import load_counties
from spatial.rule_contract import RULE_SPEC

TRAINING_PANEL_PATH = paths.RISK_FUSION_DIR / "labeled_panel_2014_2020.csv"

# V5/plan-documented fallback uncertainty constants (see rule_uncertainty.py
# and the project plan's Monte Carlo section) - this archive carries no
# per-point uncertainty of its own, so these stand in for it.
FM_SIGMA_FALLBACK = 3.41 / 1.2816  # V5 global regime half-width -> sigma
RH_SIGMA_FALLBACK = 7.0
WIND_SIGMA_LOG_FALLBACK = 0.30


def load_station_archive(path: Path) -> pd.DataFrame:
    """
    Flattens {"run_date": ..., "stations": {sid: {lat, lon, forecasts: [...]}}}
    into one row per (station_id, hour).
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for station_id, station in payload["stations"].items():
        for forecast in station["forecasts"]:
            rows.append({
                "station_id": station_id,
                "lat": station["lat"],
                "lon": station["lon"],
                "time": forecast["time"],
                "fuel_moisture": forecast["fuel_moisture"],
                "rh": forecast["rh"],
                "wind_kts": forecast["wind_speed_ms"] * features.MPS_TO_KNOTS,
                "fire_danger": forecast["fire_danger"],
            })
    frame = pd.DataFrame(rows)
    frame.attrs["run_date"] = payload["run_date"]
    return frame


def station_to_county(lat: float, lon: float, counties: List[Dict]) -> Optional[str]:
    """Point-in-polygon against county_geometry.load_counties()'s WGS84 polygons. None if outside every county."""
    from shapely.geometry import Point

    point = Point(lon, lat)
    for county in counties:
        if county["geometry_wgs84"].contains(point):
            return county["fips"]
    return None


def station_peak_probability(station_hours: pd.DataFrame, thresholds: Dict) -> Dict:
    """
    Runs sample_category_probabilities per hour with the documented
    fallback sigmas, takes probability_at_or_above_elevated, then the MAX
    across hours - mirrors the real system's peak-over-the-day risk
    product. Also carries the archive's own already-reported peak
    fire_danger category, unmodified, for reference/parity.
    """
    fm = station_hours["fuel_moisture"].to_numpy()
    rh = station_hours["rh"].to_numpy()
    wind_kts = station_hours["wind_kts"].to_numpy()

    result = ru.sample_category_probabilities(
        fm=fm, rh=rh, wind_kts=wind_kts,
        fm_sigma=np.full_like(fm, FM_SIGMA_FALLBACK),
        rh_sigma=np.full_like(rh, RH_SIGMA_FALLBACK),
        wind_sigma_log=np.full_like(wind_kts, WIND_SIGMA_LOG_FALLBACK),
        thresholds=thresholds,
    )
    return {
        "peak_probability_at_or_above_elevated": float(np.max(result["probability_at_or_above_elevated"])),
        "peak_reported_category": int(station_hours["fire_danger"].max()),
    }


def county_weather_multiplier(bundle: Dict, run_date: date, county_fips: set) -> Dict[str, float]:
    """
    lam_full / lam_seasonal_only per county for run_date, using an
    ALREADY-FITTED model_bundle.fit() result - the same seasonal-baseline-
    vs-weather-adjusted decomposition validated ad hoc earlier this
    session, restricted to just the counties that actually contain a
    station (not all 115). Fitting is the caller's job (once, reused
    across every day) since it doesn't depend on run_date at all.
    """
    run_dt = datetime(run_date.year, run_date.month, run_date.day, 12)
    target_file = paths.CACHE_HRRR_DIR / f"hrrr_{run_dt:%Y%m%d}_12z_f04-15.nc"
    if not target_file.exists():
        raise FileNotFoundError(f"No cached HRRR run for {run_date} at {target_file} - fetch it first")

    target_panel = build_county_days(run_files=[target_file])
    target_panel = target_panel[target_panel["county_fips"].isin(county_fips)].copy()

    from risk_fusion import effort, fit_glm
    target_panel["county_fips"] = target_panel["county_fips"].astype(str).str.zfill(5)
    target_panel = fit_glm.add_month_dummies(target_panel)
    target_panel["log_effort"] = effort.log_effort_offset_batch(
        target_panel["county_fips"], bundle["rate_table"], bundle["county_reference"])
    target_panel["log_effort_scaled"] = bundle["effort_exponent"] * target_panel["log_effort"]

    lam_seasonal = fit_glm.predict(bundle["climatology_fit"], target_panel)
    lam_full = fit_glm.predict_residual(bundle["residual_fit"], bundle["climatology_fit"], target_panel)
    ratio = lam_full / lam_seasonal
    return dict(zip(target_panel["county_fips"], ratio))


def blend(station_table: pd.DataFrame, county_ratios: Dict[str, float]) -> pd.DataFrame:
    """
    blended = clip(peak_probability * county_ratio, 0, 1) - a labeled
    approximation, not a rigorously-derived joint probability
    (probabilities aren't strictly closed under multiplication).
    """
    station_table = station_table.copy()
    station_table["county_weather_multiplier"] = station_table["county_fips"].map(county_ratios)
    station_table["blended_probability"] = (
        station_table["peak_probability_at_or_above_elevated"] * station_table["county_weather_multiplier"]
    ).clip(0.0, 1.0)
    return station_table


OUTPUT_COLUMNS = ["run_date", "station_id", "county_fips", "county_name", "lat", "lon",
                  "peak_probability_at_or_above_elevated", "county_weather_multiplier",
                  "blended_probability", "peak_reported_category"]

# Matches both naming conventions seen in the real archive - an early
# "station_forecasts_YYYYMMDD_HHz.json" series (from 2026-03-27) and a
# later "station_forecasts_beta_YYYYMMDD_HHz.json" one (from 2026-03-28),
# same schema either way. Their date coverage has small, non-identical
# gaps (e.g. only one of the two has 2026-03-30), so treating both as one
# pool is what actually maximizes coverage.
ARCHIVE_FILENAME_RE = re.compile(r"station_forecasts_(?:beta_)?(\d{8})_\d{2}\.json$")


def archive_date(path: Path) -> Optional[date]:
    """Parses the run date out of a station_forecasts[_beta]_YYYYMMDD_HHz.json filename, or None if unrecognized."""
    match = ARCHIVE_FILENAME_RE.search(path.name)
    return datetime.strptime(match.group(1), "%Y%m%d").date() if match else None


def process_archive(archive_path: Path, counties: List[Dict], bundle: Dict) -> pd.DataFrame:
    """One archived day, start to finish: load -> per-station peak probability -> station->county
    -> county weather multiplier (against the already-fitted bundle) -> blend. Adds a run_date column."""
    print(f"Loading station archive {archive_path}...")
    hours = load_station_archive(archive_path)
    run_date_value = pd.Timestamp(hours.attrs["run_date"]).date()
    print(f"run_date={run_date_value}, {hours['station_id'].nunique()} stations, {len(hours)} station-hours")

    thresholds = RULE_SPEC["thresholds"]
    rows = []
    for station_id, group in hours.groupby("station_id"):
        lat, lon = group["lat"].iloc[0], group["lon"].iloc[0]
        fips = station_to_county(lat, lon, counties)
        peak = station_peak_probability(group, thresholds)
        rows.append({"station_id": station_id, "lat": lat, "lon": lon, "county_fips": fips, **peak})
    station_table = pd.DataFrame(rows)

    unresolved = station_table[station_table["county_fips"].isna()]
    if not unresolved.empty:
        print(f"WARNING: {len(unresolved)} station(s) did not resolve to a Missouri county: "
              f"{unresolved['station_id'].tolist()}")
    station_table = station_table.dropna(subset=["county_fips"])

    county_fips = set(station_table["county_fips"])
    ratios = county_weather_multiplier(bundle, run_date_value, county_fips)

    blended = blend(station_table, ratios)
    blended["run_date"] = run_date_value
    return blended


def _select_archives(archive: Optional[Path], since: Optional[str], until: Optional[str]) -> List[Path]:
    if archive is not None:
        if since or until:
            raise SystemExit("--archive cannot be combined with --since/--until")
        return [archive]

    all_candidates = sorted(paths.ARCHIVE_FORECASTS_DIR.glob("station_forecasts_*.json"))
    if not all_candidates:
        raise SystemExit(f"No station_forecasts*.json files found in {paths.ARCHIVE_FORECASTS_DIR}")

    # One file per date: sorted-name order puts the plain "station_forecasts_"
    # name before the "station_forecasts_beta_" one for the same date when
    # both exist (identical schema either way - see ARCHIVE_FILENAME_RE).
    by_date: Dict[date, Path] = {}
    for path in all_candidates:
        parsed = archive_date(path)
        if parsed is not None:
            by_date.setdefault(parsed, path)
    candidates = [by_date[d] for d in sorted(by_date)]

    if not since and not until:
        return [candidates[-1]]

    since_date = pd.Timestamp(since).date() if since else date.min
    until_date = pd.Timestamp(until).date() if until else date.max
    selected = [path for path in candidates if since_date <= archive_date(path) <= until_date]
    if not selected:
        raise SystemExit(f"No archived station files found between {since_date} and {until_date}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=None,
                        help="Path to a single station_forecasts_beta_*.json file (default: most recent available)")
    parser.add_argument("--since", type=str, default=None, help="YYYY-MM-DD - process every archived day on/after this date")
    parser.add_argument("--until", type=str, default=None, help="YYYY-MM-DD - process every archived day on/before this date")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    archive_paths = _select_archives(args.archive, args.since, args.until)
    print(f"Processing {len(archive_paths)} archived day(s): {[p.name for p in archive_paths]}")

    print("Fitting the seasonal-baseline model once, reused across every day...")
    train_panel = model_bundle.load_training_panel(TRAINING_PANEL_PATH)
    bundle = model_bundle.fit(train_panel)
    counties = load_counties()

    blended_days = [process_archive(path, counties, bundle) for path in archive_paths]
    combined = pd.concat(blended_days, ignore_index=True)

    names = {row["fips"]: row["name"] for row in json.loads((REPO_ROOT / "risk_fusion" / "county_reference.json").read_text())["counties"]}
    combined["county_name"] = combined["county_fips"].map(names)
    combined = combined.sort_values(["run_date", "blended_probability"], ascending=[True, False])

    if args.output:
        output_path = args.output
    elif len(archive_paths) == 1:
        output_path = paths.RISK_FUSION_DIR / f"station_blend_{combined['run_date'].iloc[0]}.csv"
    else:
        output_path = paths.RISK_FUSION_DIR / f"station_blend_{combined['run_date'].min()}_to_{combined['run_date'].max()}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined[OUTPUT_COLUMNS].to_csv(output_path, index=False)

    print(f"\nAll days, ranked by blended probability:")
    print(combined[OUTPUT_COLUMNS].to_string(index=False))
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
