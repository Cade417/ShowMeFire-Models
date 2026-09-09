"""
Phase 2's real historical panel builder - the actual orchestrator that
turns already-on-disk data into a training panel with Rothermel labels
attached, per docs/fire_weather_ml_plan.md's Phase 2 design. Ties together:

1. `paths.PRECIP_ALIGNED_DATASET` (station weather + observed 10-hr fuel
   moisture), filtered to the static bundle's Missouri domain.
2. `static_lookup.py` - per-station terrain from the already-built
   `data/static/bundles/fire_behavior_static_*.nc`.
3. `wind_direction.py` - real wind-FROM direction from the historical RTMA
   cache (the aligned CSV only has wind speed).
4. `precip_normals.py` - per-station mean-annual-precipitation (KBDI's
   climate-normal term), via nearest county.
5. `features.derive_fm1_fm10_fm100`/`live_moisture_percent` - deriving
   inputs the source data doesn't directly observe.
6. `panel.build_station_panel` - the Phase 1 join/feature/label assembly,
   unchanged.

Every station-exclusion step is counted and reported, not silently applied
- `build_historical_panel` returns both the panel and a coverage dict so a
caller (or `main()`) can see exactly how much was dropped and why.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths
from fire_weather_ml import features, precip_normals, static_lookup, wind_direction
from fire_weather_ml.panel import build_station_panel

# Same domain the static bundle was built over (west, east, south, north) -
# see data/static/bundles/fire_behavior_static_2026.4.json's source_manifest.bbox.
MO_BBOX = (-96.8, -88.1, 34.8, 41.8)

DEFAULT_STATIC_BUNDLE_PATH = REPO_ROOT.parent / "data" / "static" / "bundles" / "fire_behavior_static_2026.4.nc"
DEFAULT_RTMA_CACHE_DIR = paths.CACHE_RTMA_DIR

ALIGNED_WEATHER_COLUMNS = (
    "station_id", "lat", "lon", "valid_time", "rtma_temp_c", "rtma_rh",
    "rtma_wind_ms", "hrrr_precip_increment_mm", "initial_fm",
)


def load_aligned_weather(path: Path, bbox: Tuple[float, float, float, float] = MO_BBOX) -> pd.DataFrame:
    """Loads and renames the real station weather+fuel-moisture history, filtered to `bbox`."""
    frame = pd.read_csv(path, usecols=list(ALIGNED_WEATHER_COLUMNS))
    west, east, south, north = bbox
    in_domain = frame["lat"].between(south, north) & frame["lon"].between(west, east)
    frame = frame[in_domain].copy()
    frame["valid_time"] = pd.to_datetime(frame["valid_time"])
    frame = frame.rename(columns={
        "rtma_temp_c": "temp_c", "rtma_rh": "rh_pct", "rtma_wind_ms": "wind_ms",
        "hrrr_precip_increment_mm": "precip_mm", "initial_fm": "fm10_observed_pct",
    })
    # A handful of rows can have a null precip increment where the aligned
    # dataset's own precip_available flag is false - treating that as "no
    # rain this hour" (0.0) is the same assumption build_county_days.py
    # makes for missing precip elsewhere in this project, not a new one.
    frame["precip_mm"] = frame["precip_mm"].fillna(0.0)
    return frame.sort_values(["station_id", "valid_time"]).reset_index(drop=True)


def _derive_per_station_fields(weather: pd.DataFrame) -> pd.DataFrame:
    """Adds fm1_pct/fm10_pct/fm100_pct and live_herbaceous_pct/live_woody_pct, one station's chronological series at a time (both are sequential/stateful)."""
    derived_frames = []
    for _station_id, group in weather.groupby("station_id", sort=False):
        group = group.sort_values("valid_time").reset_index(drop=True)
        fm = features.derive_fm1_fm10_fm100(
            group["temp_c"], group["rh_pct"], group["precip_mm"], group["fm10_observed_pct"])
        live = features.live_moisture_percent(features.gdd_series(group["temp_c"]))
        group[["fm1_pct", "fm10_pct", "fm100_pct"]] = fm.reset_index(drop=True)
        group[["live_herbaceous_pct", "live_woody_pct"]] = live.reset_index(drop=True)
        derived_frames.append(group)
    return pd.concat(derived_frames, ignore_index=True)


def build_historical_panel(
    aligned_weather_path: Path = None,
    static_bundle_path: Path = DEFAULT_STATIC_BUNDLE_PATH,
    rtma_cache_dir: Path = DEFAULT_RTMA_CACHE_DIR,
) -> Tuple[pd.DataFrame, Dict]:
    aligned_weather_path = aligned_weather_path or paths.PRECIP_ALIGNED_DATASET
    weather = load_aligned_weather(aligned_weather_path)
    stations = weather.drop_duplicates("station_id")[["station_id", "lat", "lon"]].reset_index(drop=True)

    bundle = static_lookup.load_static_bundle(static_bundle_path)
    static_by_station = static_lookup.build_station_lookup(bundle, stations)
    stations_with_static = set(static_by_station.index)

    matched_stations = stations[stations["station_id"].isin(stations_with_static)].reset_index(drop=True)
    precip_by_station = precip_normals.build_station_precip_normals(matched_stations)
    stations_with_precip = set(precip_by_station)

    usable_station_ids = stations_with_static & stations_with_precip
    dropped_no_static = sorted(set(stations["station_id"]) - stations_with_static)
    dropped_no_precip_normal = sorted(stations_with_static - stations_with_precip)

    weather = weather[weather["station_id"].isin(usable_station_ids)].copy()
    usable_stations = matched_stations[matched_stations["station_id"].isin(usable_station_ids)]

    weather["valid_time_hour"] = weather["valid_time"].dt.floor("h")
    wind_lookup = wind_direction.build_wind_direction_lookup(rtma_cache_dir, weather["valid_time"], usable_stations)
    weather = weather.merge(wind_lookup, on=["valid_time_hour", "station_id"], how="left")
    rows_missing_wind = int(weather["wind_from_deg"].isna().sum())
    weather = weather.dropna(subset=["wind_from_deg"]).drop(columns=["valid_time_hour"])

    weather = _derive_per_station_fields(weather)

    panel = build_station_panel(weather, static_by_station, precip_by_station)

    labeled_rows = int(panel["ros_ch_per_h"].notna().sum())
    coverage = {
        "input_stations_in_domain": int(len(stations)),
        "stations_dropped_no_static_match": dropped_no_static,
        "stations_dropped_no_precip_normal": dropped_no_precip_normal,
        "usable_stations": int(len(usable_stations)),
        "rows_dropped_missing_wind_direction": rows_missing_wind,
        "total_rows": int(len(panel)),
        "labeled_rows": labeled_rows,
        "null_label_rate": (1.0 - labeled_rows / len(panel)) if len(panel) else None,
        "date_range": [str(panel["valid_time"].min()), str(panel["valid_time"].max())] if len(panel) else None,
    }
    return panel, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=paths.FIRE_WEATHER_ML_PANEL)
    parser.add_argument("--static-bundle", type=Path, default=DEFAULT_STATIC_BUNDLE_PATH)
    args = parser.parse_args()

    panel, coverage = build_historical_panel(static_bundle_path=args.static_bundle)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)

    print(json.dumps(coverage, indent=2, default=str))
    print(f"Panel written to {args.output}")


if __name__ == "__main__":
    main()
