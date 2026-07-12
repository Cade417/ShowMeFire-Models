from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

VARIABLES = {
    "fuel_moisture": ("fuel_moisture_set_1", "fuel_moisture_value_1", "fuel_moisture"),
    "obs_temp": ("air_temp_set_1", "air_temp_value_1", "air_temp"),
    "obs_rh": ("relative_humidity_set_1", "relative_humidity_value_1", "relative_humidity"),
}


def _series(obs, candidates, count):
    for name in candidates:
        value = obs.get(name)
        if isinstance(value, list):
            return value[:count] + [None] * max(0, count - len(value))
    return [None] * count


def load_observations(directory: Path) -> pd.DataFrame:
    records = []
    for path in sorted(Path(directory).glob("raw_data_*.json")):
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
