"""
Joins per-station weather+fuel-moisture history to per-station static
terrain (fuel model, slope, aspect, canopy) and attaches both the ML
FEATURE_COLUMNS (features.py) and the Rothermel-computed training label
(rothermel_labels.py) - the training-side analog of
api/services/spread_rate.py's per-cell loop, but building a flat station-
hour panel instead of refreshing a live grid.

Phase 1 scope: this module assembles a panel from already-loaded
DataFrames. It does not itself pull historical RTMA/station data - that's
Phase 2 (see docs/fire_weather_ml_plan.md), which wires this against the
existing pullers (ingest_obs.py, backfill_synoptic.py, extract_hrrr.py) and
the already-built static bundle (data/static/bundles/fire_behavior_static_*).
"""
from __future__ import annotations

import pandas as pd

from fire_weather_ml import features, rothermel_labels

STATION_ID_COLUMN = "station_id"
TIMESTAMP_COLUMN = "valid_time"

STATIC_TERRAIN_COLUMNS = (
    "fuel_model_code", "slope_deg", "aspect_deg", "canopy_cover_pct", "canopy_height_m",
)


def join_static_terrain(weather: pd.DataFrame, static_by_station: pd.DataFrame) -> pd.DataFrame:
    """
    Left-joins per-station static terrain onto a weather frame by
    `station_id`. `static_by_station` must be indexed/keyed one row per
    station - terrain doesn't vary over time, unlike everything else in
    the panel. Raises on stations present in `weather` but missing from
    `static_by_station` rather than silently dropping or NaN-filling them,
    since a missing fuel model makes the Rothermel label meaningless, not
    just imprecise.
    """
    missing_columns = [c for c in STATIC_TERRAIN_COLUMNS if c not in static_by_station.columns]
    if missing_columns:
        raise ValueError(f"static_by_station is missing required columns: {missing_columns}")

    joined = weather.merge(
        static_by_station[list(STATIC_TERRAIN_COLUMNS)],
        left_on=STATION_ID_COLUMN, right_index=True, how="left", validate="many_to_one",
    )
    unmatched = joined[joined["fuel_model_code"].isna()][STATION_ID_COLUMN].unique()
    if len(unmatched):
        raise ValueError(f"stations missing static terrain: {sorted(unmatched.tolist())}")
    return joined


def build_station_panel(
    weather: pd.DataFrame,
    static_by_station: pd.DataFrame,
    mean_annual_precip_in_by_station: dict[str, float],
) -> pd.DataFrame:
    """
    Full Phase-1 assembly for one or more stations: join terrain, derive
    KBDI/GDD per station (each station's drought memory is independent -
    grouped, not accumulated across the whole panel at once), attach
    features, then attach the Rothermel-computed label. `weather` must
    already be sorted by (station_id, valid_time) - KBDI/GDD accrual is
    sequential and order-dependent.
    """
    required = (STATION_ID_COLUMN, TIMESTAMP_COLUMN, "wind_from_deg", "live_herbaceous_pct", "live_woody_pct")
    missing = [c for c in required if c not in weather.columns]
    if missing:
        raise ValueError(f"weather frame is missing required columns: {missing}")

    joined = join_static_terrain(weather, static_by_station)
    joined = joined.sort_values([STATION_ID_COLUMN, TIMESTAMP_COLUMN]).reset_index(drop=True)

    feature_frames = []
    for station_id, group in joined.groupby(STATION_ID_COLUMN, sort=False):
        precip_normal = mean_annual_precip_in_by_station.get(station_id)
        if precip_normal is None:
            raise ValueError(f"no mean-annual-precipitation normal provided for station {station_id!r}")
        station_features = features.assemble_features(group, mean_annual_precip_in=precip_normal)
        station_features[STATION_ID_COLUMN] = station_id
        station_features[TIMESTAMP_COLUMN] = group[TIMESTAMP_COLUMN].to_numpy()
        # rothermel_labels needs the raw wind-direction/live-fuel/terrain
        # inputs too, not just the derived FEATURE_COLUMNS - carry them
        # through unchanged rather than re-deriving them.
        for column in ("wind_from_deg", "live_herbaceous_pct", "live_woody_pct", *rothermel_labels.LABEL_INPUT_COLUMNS):
            if column in group.columns:
                station_features[column] = group[column].to_numpy()
        feature_frames.append(station_features)

    panel = pd.concat(feature_frames, ignore_index=True)
    return rothermel_labels.compute_labels_for_panel(panel)
