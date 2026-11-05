from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

VARIABLES = {
    "fuel_moisture": ("fuel_moisture_set_1", "fuel_moisture_value_1", "fuel_moisture"),
    "obs_temp": ("air_temp_set_1", "air_temp_value_1", "air_temp"),
    "obs_rh": ("relative_humidity_set_1", "relative_humidity_value_1", "relative_humidity"),
    "obs_wind_ms": ("wind_speed_set_1", "wind_speed_value_1", "wind_speed"),
    # Requested from Synoptic (services/synoptic.py's vars= list) but never
    # extracted until now. Same raw-key convention and units as obs_wind_ms -
    # no unit conversion is applied here for either, by existing precedent.
    "obs_wind_gust_ms": ("wind_gust_set_1", "wind_gust_value_1", "wind_gust"),
}


def _series(obs, candidates, count):
    for name in candidates:
        value = obs.get(name)
        if isinstance(value, list):
            return value[:count] + [None] * max(0, count - len(value))
    return [None] * count


def load_observations(directory: Path, progress=None) -> pd.DataFrame:
    records = []
    paths = sorted(Path(directory).glob("raw_data_*.json"))
    for number, path in enumerate(paths, 1):
        if progress is not None:
            progress(number, len(paths), path)
        try:
            body = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for station in body.get("STATION", []):
            obs = station.get("OBSERVATIONS", {})
            times = obs.get("date_time", [])
            values = {name: _series(obs, keys, len(times)) for name, keys in VARIABLES.items()}
            for index, stamp in enumerate(times):
                records.append({
                    "station_id": station.get("STID") or station.get("ID"),
                    "time": stamp,
                    "lat": station.get("LATITUDE"), "lon": station.get("LONGITUDE"),
                    **{name: series[index] for name, series in values.items()},
                })
    if not records:
        return pd.DataFrame(columns=["station_id", "time", "lat", "lon", *VARIABLES])
    frame = pd.DataFrame(records)
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    for col in ("lat", "lon", *VARIABLES):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["station_id", "time"]).drop_duplicates(["station_id", "time"], keep="last").sort_values(["station_id", "time"])


def causal_initial_and_targets(station_obs, init_time, valid_times, max_age_hours=3, tolerance_minutes=30):
    init_time = pd.Timestamp(init_time)
    if init_time.tzinfo is None:
        init_time = init_time.tz_localize("UTC")
    prior = station_obs[(station_obs.time <= init_time) & station_obs.fuel_moisture.notna()]
    initial = prior.iloc[-1] if not prior.empty else None
    initial_fm = float(initial.fuel_moisture) if initial is not None else float("nan")
    age = (init_time - initial.time).total_seconds() / 3600 if initial is not None else float("nan")
    if initial is None or age > max_age_hours:
        initial_fm = float("nan")
    targets = []
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    fm_obs = station_obs[station_obs.fuel_moisture.notna()]
    for valid in valid_times:
        valid = pd.Timestamp(valid)
        if valid.tzinfo is None:
            valid = valid.tz_localize("UTC")
        candidates = fm_obs[(fm_obs.time >= valid - tolerance) & (fm_obs.time <= valid + tolerance)]
        if candidates.empty:
            targets.append((float("nan"), None))
        else:
            nearest = candidates.iloc[(candidates.time - valid).abs().argmin()]
            targets.append((float(nearest.fuel_moisture), nearest.time))
    return initial_fm, age, targets


def causal_initial_and_weather_targets(station_obs, init_time, valid_times,
                                       max_age_hours=3, tolerance_minutes=30):
    """Align FM/RH/wind labels to one nearest station observation per valid time."""
    initial_fm, age, _ = causal_initial_and_targets(
        station_obs, init_time, [], max_age_hours=max_age_hours,
        tolerance_minutes=tolerance_minutes)
    tolerance = pd.Timedelta(minutes=tolerance_minutes); targets = []
    for valid in map(pd.Timestamp, valid_times):
        valid = valid.tz_localize("UTC") if valid.tzinfo is None else valid.tz_convert("UTC")
        candidates = station_obs[(station_obs.time >= valid - tolerance) &
                                 (station_obs.time <= valid + tolerance)]
        candidates = candidates[candidates.fuel_moisture.notna()]
        if candidates.empty:
            targets.append({"target_fm": float("nan"), "target_rh": float("nan"),
                            "target_wind_ms": float("nan"), "target_time": None,
                            "target_match_age_minutes": float("nan")})
            continue
        nearest = candidates.iloc[(candidates.time - valid).abs().argmin()]
        targets.append({"target_fm": float(nearest.fuel_moisture),
                        "target_rh": float(nearest.obs_rh) if pd.notna(nearest.obs_rh) else float("nan"),
                        "target_wind_ms": float(nearest.obs_wind_ms) if pd.notna(nearest.obs_wind_ms) else float("nan"),
                        "target_time": nearest.time,
                        "target_match_age_minutes": abs((nearest.time - valid).total_seconds()) / 60})
    return initial_fm, age, targets
