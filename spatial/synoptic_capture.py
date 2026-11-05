"""Historical Synoptic station-observation fetcher for the training data store.

Mirrors the production api's services/synoptic.py defaults (9-state region,
network 2 - empirically confirmed to cover ~89 MO-region fuel-moisture
stations, vs. 1 station on network 1 despite network 1 being labeled "RAWS").
This is pure HTTP + JSON, no native-library concurrency hazards, so unlike
the HRRR/RTMA backfills, thread-based concurrency is safe here.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

DEFAULT_STATES = ["MO", "OK", "AR", "TN", "KY", "IL", "IA", "NE", "KS"]
DEFAULT_NETWORKS = [2]
TIMESERIES_URL = "https://api.synopticdata.com/v2/stations/timeseries"
REQUIRED_VARIABLES = {"fuel_moisture_set_1", "air_temp_set_1", "relative_humidity_set_1"}


def fetch_timeseries(start: datetime, end: datetime, token: str, states=None, networks=None, timeout=60) -> dict:
    """Fetch raw station timeseries for [start, end) (both UTC). Returns the
    raw Synoptic API JSON response (same shape production archives as-is)."""
    states = states or DEFAULT_STATES
    networks = networks or DEFAULT_NETWORKS
    params = {
        "token": token,
        "state": ",".join(states),
        "network": ",".join(map(str, networks)),
        "start": start.strftime("%Y%m%d%H%M"),
        "end": end.strftime("%Y%m%d%H%M"),
        "units": "english",
        "status": "active",
        "obtimezone": "UTC",
        "vars": "fuel_moisture,relative_humidity,air_temp,wind_speed,wind_gust,solar_radiation,precip_accum",
    }
    response = requests.get(TIMESERIES_URL, params=params, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if body.get("SUMMARY", {}).get("RESPONSE_CODE") != 1:
        raise RuntimeError(f"Synoptic API error: {body.get('SUMMARY', {}).get('RESPONSE_MESSAGE')}")
    return body


def split_by_day(response: dict, ignored_stations: set | None = None) -> dict[str, dict]:
    """Split one multi-day API response into {YYYY-MM-DD: response-shaped-dict},
    filtering each station's OBSERVATIONS arrays down to that day's entries."""
    ignored_stations = ignored_stations or set()
    by_day: dict[str, dict] = {}
    for station in response.get("STATION", []):
        if (station.get("STID") or station.get("ID")) in ignored_stations:
            continue
        obs = station.get("OBSERVATIONS", {})
        times = obs.get("date_time", [])
        day_indices: dict[str, list[int]] = {}
        for index, stamp in enumerate(times):
            day = stamp[:10]  # ISO date_time is already UTC, e.g. "2026-04-01T15:53:00Z"
            day_indices.setdefault(day, []).append(index)

        for day, indices in day_indices.items():
            day_response = by_day.setdefault(day, {"STATION": []})
            day_obs = {"date_time": [times[i] for i in indices]}
            for key, values in obs.items():
                if key == "date_time":
                    continue
                if isinstance(values, list) and len(values) == len(times):
                    day_obs[key] = [values[i] for i in indices]
            day_station = {k: v for k, v in station.items() if k != "OBSERVATIONS"}
            day_station["OBSERVATIONS"] = day_obs
            day_response["STATION"].append(day_station)
    return by_day


def validate_raw_data(path) -> tuple[bool, str]:
    import json
    try:
        with open(path) as f:
            body = json.load(f)
        stations = body.get("STATION")
        if stations is None:
            return False, "no STATION key"
        # An empty list is a legitimate outcome (e.g. every station for that
        # day was ignore-filtered) - only a missing STATION key means broken.
        if stations and not any(REQUIRED_VARIABLES & set(s.get("OBSERVATIONS", {})) for s in stations):
            return False, "no required variables present"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def daily_chunks(start: datetime, end: datetime, chunk_days: int):
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        yield cursor, chunk_end
        cursor = chunk_end
