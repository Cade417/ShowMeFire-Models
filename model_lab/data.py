"""Discover registry entries, shadow candidates, archives, and offline eval reports."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from paths import ARCHIVE_FORECASTS_DIR, ARCHIVE_RAW_DATA_DIR, DATA_ROOT, MODELS_DIR, REPORTS_DIR

DATE_TOKEN_RE = re.compile(r"(20\d{6})")
# station_forecasts_20260807_12.json              -> production
# station_forecasts_beta_20260807_12.json         -> beta
# station_forecasts_beta_mytag_20260807_12.json   -> beta_mytag
FORECAST_NAME_RE = re.compile(
    r"^station_forecasts(?:_beta(?:_([A-Za-z][A-Za-z0-9\-]*))?)?_(20\d{6})(?:_(\d{2}))?\.json$"
)

SERIES_COLORS = {
    "production": "#1F4E79",
    "beta": "#C45C26",
    "obs": "#2F6F4E",
}


def series_key_from_filename(name: str) -> tuple[str, str] | None:
    """Return (series_key, date_token) for a forecast archive filename."""
    match = FORECAST_NAME_RE.match(Path(name).name)
    if not match:
        return None
    tag, date_token, _hour = match.groups()
    if "beta" not in Path(name).name:
        return "production", date_token
    if tag:
        return f"beta_{tag}", date_token
    return "beta", date_token


def series_file_prefix(series_key: str) -> str:
    if series_key == "production":
        return "station_forecasts"
    if series_key == "beta":
        return "station_forecasts_beta"
    if series_key.startswith("beta_"):
        return f"station_forecasts_beta_{series_key[len('beta_'):]}"
    return f"station_forecasts_{series_key}"


@lru_cache(maxsize=1)
def discover_forecast_series() -> tuple[str, ...]:
    """All forecast series keys present under archive/forecasts."""
    if not ARCHIVE_FORECASTS_DIR.exists():
        return ("production", "beta")
    keys = set()
    for path in ARCHIVE_FORECASTS_DIR.glob("station_forecasts*.json"):
        parsed = series_key_from_filename(path.name)
        if parsed:
            keys.add(parsed[0])
    if not keys:
        keys.update({"production", "beta"})
    # Stable order: production, bare beta, then tagged betas, then anything else.
    def _sort_key(name: str):
        if name == "production":
            return (0, name)
        if name == "beta":
            return (1, name)
        if name.startswith("beta_"):
            return (2, name)
        return (3, name)
    return tuple(sorted(keys, key=_sort_key))


@dataclass(frozen=True)
class ModelRecord:
    model_type: str
    channel: str
    version: str | None
    file: str | None
    performance: dict
    trained_at: str | None
    path: Path | None
    source: str  # registry | shadow | history


def _resolve_under_data(relative: str | None) -> Path | None:
    if not relative:
        return None
    raw = Path(relative)
    if raw.is_absolute():
        return raw if raw.exists() else None
    # Registry paths are usually relative to DATA_ROOT (models/versions/...)
    # or accidentally relative to the old API root (models/...).
    candidates = [
        DATA_ROOT / relative,
        MODELS_DIR.parent / relative,  # training-data/...
        DATA_ROOT / "models" / raw.name,
        MODELS_DIR / "versions" / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return DATA_ROOT / relative


def load_registry_config() -> dict:
    config_path = MODELS_DIR / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def list_registry_models() -> list[ModelRecord]:
    config = load_registry_config()
    records: list[ModelRecord] = []
    for model_type, entry in config.items():
        if not isinstance(entry, dict):
            continue
        for channel in ("stable", "beta"):
            item = entry.get(channel)
            if not item:
                records.append(ModelRecord(
                    model_type=model_type, channel=channel, version=None, file=None,
                    performance={}, trained_at=None, path=None, source="registry",
                ))
                continue
            records.append(ModelRecord(
                model_type=model_type,
                channel=channel,
                version=item.get("version"),
                file=item.get("file"),
                performance=item.get("performance") or {},
                trained_at=item.get("trained_at") or item.get("promoted_at"),
                path=_resolve_under_data(item.get("file")),
                source="registry",
            ))
        for hist in entry.get("history") or []:
            records.append(ModelRecord(
                model_type=model_type,
                channel=hist.get("channel", "history"),
                version=hist.get("version"),
                file=hist.get("file"),
                performance=hist.get("performance") or {},
                trained_at=hist.get("trained_at") or hist.get("promoted_at") or hist.get("recorded_at"),
                path=_resolve_under_data(hist.get("file")),
                source="history",
            ))
    return records


def list_shadow_candidates() -> list[ModelRecord]:
    if not MODELS_DIR.exists():
        return []
    records = []
    for path in sorted(MODELS_DIR.iterdir()):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if "shadow" not in name and "candidate" not in name:
            continue
        manifest = None
        for candidate in ("manifest.json", "bundle_manifest.json", "metadata.json"):
            if (path / candidate).exists():
                try:
                    manifest = json.loads((path / candidate).read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    manifest = {"_raw": candidate}
                break
        performance = {}
        if isinstance(manifest, dict):
            performance = manifest.get("performance") or manifest.get("metrics") or {}
            if not performance and "mae" in manifest:
                performance = {k: manifest[k] for k in ("mae", "rmse", "bias", "r2") if k in manifest}
        records.append(ModelRecord(
            model_type=path.name,
            channel="shadow",
            version=None if not isinstance(manifest, dict) else manifest.get("version"),
            file=str(path),
            performance=performance if isinstance(performance, dict) else {},
            trained_at=None,
            path=path,
            source="shadow",
        ))
    return records


def list_eval_reports() -> list[Path]:
    if not REPORTS_DIR.exists():
        return []
    files = []
    for path in REPORTS_DIR.rglob("*.json"):
        name = path.name.lower()
        if any(token in name for token in ("evaluation", "baseline_metrics", "sequence_metrics", "search.state")):
            files.append(path)
        elif name.endswith("_evaluation.json") or "evaluation" in name:
            files.append(path)
    return sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)


def load_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def extract_date_token(name: str) -> str | None:
    match = DATE_TOKEN_RE.search(name)
    return match.group(1) if match else None


@lru_cache(maxsize=1)
def discover_forecast_dates() -> tuple[str, ...]:
    if not ARCHIVE_FORECASTS_DIR.exists():
        return ()
    dates = set()
    for path in ARCHIVE_FORECASTS_DIR.glob("station_forecasts*.json"):
        token = extract_date_token(path.name)
        if token:
            dates.add(token)
    return tuple(sorted(dates))


def list_series_for_date(date_token: str, series_keys: list[str] | None = None) -> dict[str, Path | None]:
    """Resolve forecast JSON paths for one date across one or more series keys."""
    keys = list(series_keys) if series_keys is not None else list(discover_forecast_series())
    found = {key: None for key in keys}
    if not ARCHIVE_FORECASTS_DIR.exists():
        return found
    for path in sorted(ARCHIVE_FORECASTS_DIR.glob(f"station_forecasts*{date_token}*.json")):
        parsed = series_key_from_filename(path.name)
        if not parsed:
            continue
        series_key, token = parsed
        if token != date_token or series_key not in found:
            continue
        # Keep the latest hour file for that series/date.
        found[series_key] = path
    return found


def resolve_series_path(date_token: str, series_key: str) -> Path | None:
    return list_series_for_date(date_token, [series_key]).get(series_key)


def find_raw_paths_for_forecast_date(date_token: str) -> list[Path]:
    """Return raw_data files that can validate a forecast initialized on date_token.

    Forecast valid times usually span the init day and the following morning, so
    we load both D and D+1 when present.
    """
    if not ARCHIVE_RAW_DATA_DIR.exists():
        return []
    from datetime import timedelta

    tokens = [date_token]
    try:
        dt = datetime.strptime(date_token, "%Y%m%d")
        tokens.append((dt + timedelta(days=1)).strftime("%Y%m%d"))
    except ValueError:
        pass
    paths: list[Path] = []
    seen = set()
    for token in tokens:
        for match in sorted(ARCHIVE_RAW_DATA_DIR.glob(f"raw_data_{token}*.json")):
            key = str(match.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(match)
    return paths


def find_raw_for_date(date_token: str) -> Path | None:
    paths = find_raw_paths_for_forecast_date(date_token)
    return paths[0] if paths else None


def forecast_to_dataframe(forecast_data: dict) -> pd.DataFrame:
    records = []
    for stid, data in (forecast_data.get("stations") or {}).items():
        for fc in data.get("forecasts") or []:
            ts = pd.Timestamp(fc["time"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            records.append({
                "stid": stid,
                "timestamp": ts.round("h"),
                "pred_temp": fc.get("temp_c"),
                "pred_rh": fc.get("rh"),
                "pred_wind": fc.get("wind_speed_ms"),
                "pred_fm": fc.get("fuel_moisture"),
                "pred_fire_danger": fc.get("fire_danger"),
                "lat": data.get("lat"),
                "lon": data.get("lon"),
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _fahrenheit_to_celsius(value):
    return (value - 32.0) * 5.0 / 9.0


def _mph_to_ms(value):
    return value * 0.44704


def observations_to_dataframe(raw_data) -> pd.DataFrame:
    records = []
    if isinstance(raw_data, list):
        stations = raw_data
    elif isinstance(raw_data, dict):
        stations = raw_data.get("STATION") or raw_data.get("stations") or []
        if isinstance(stations, dict):
            stations = list(stations.values()) if stations and isinstance(next(iter(stations.values()), None), dict) else [stations]
        if not stations and any(isinstance(v, dict) for v in raw_data.values()):
            # {STID: {...}} shape
            stations = [{"STID": key, **value} if isinstance(value, dict) else {"STID": key}
                        for key, value in raw_data.items()]
    else:
        stations = []

    for station in stations:
        stid = station.get("STID") or station.get("stid")
        if not stid:
            continue
        obs = station.get("OBSERVATIONS") or station
        times = obs.get("date_time") or obs.get("TIME") or []
        temps = obs.get("air_temp_set_1") or obs.get("air_temp") or []
        rhs = obs.get("relative_humidity_set_1") or obs.get("relative_humidity") or []
        winds = obs.get("wind_speed_set_1") or obs.get("wind_speed") or []
        fms = obs.get("fuel_moisture_set_1") or obs.get("fuel_moisture") or []
        for i, time_str in enumerate(times):
            try:
                ts = pd.Timestamp(time_str)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                ts = ts.round("h")
                temp_val = temps[i] if i < len(temps) else None
                wind_val = winds[i] if i < len(winds) else None
                # Synoptic archive in this project stores temp F / wind mph (endOfDayReport converts).
                if temp_val is not None:
                    temp_val = _fahrenheit_to_celsius(float(temp_val))
                if wind_val is not None:
                    wind_val = _mph_to_ms(float(wind_val))
                records.append({
                    "stid": stid,
                    "timestamp": ts,
                    "obs_temp": temp_val,
                    "obs_rh": rhs[i] if i < len(rhs) else None,
                    "obs_wind": wind_val,
                    "obs_fm": fms[i] if i < len(fms) else None,
                })
            except (ValueError, TypeError, IndexError):
                continue
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.groupby(["stid", "timestamp"], as_index=False).mean(numeric_only=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def merge_forecast_obs(forecast_df: pd.DataFrame, obs_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df.empty or obs_df.empty:
        return pd.DataFrame()
    return forecast_df.merge(obs_df, on=["stid", "timestamp"], how="inner")


def load_scored_series(date_token: str, series: str) -> tuple[pd.DataFrame, dict]:
    """Return (merged forecast+obs dataframe, metadata)."""
    forecast_path = resolve_series_path(date_token, series)
    raw_paths = find_raw_paths_for_forecast_date(date_token)
    meta = {
        "date": date_token,
        "series": series,
        "forecast_path": str(forecast_path) if forecast_path else None,
        "raw_paths": [str(p) for p in raw_paths],
        "raw_path": str(raw_paths[0]) if raw_paths else None,
    }
    if not forecast_path or not raw_paths:
        return pd.DataFrame(), meta
    forecast_df = forecast_to_dataframe(load_json(forecast_path))
    obs_frames = [observations_to_dataframe(load_json(path)) for path in raw_paths]
    obs_frames = [frame for frame in obs_frames if not frame.empty]
    if not obs_frames:
        return pd.DataFrame(), meta
    obs_df = pd.concat(obs_frames, ignore_index=True)
    obs_df = obs_df.groupby(["stid", "timestamp"], as_index=False).mean(numeric_only=True)
    merged = merge_forecast_obs(forecast_df, obs_df)
    meta["forecast_rows"] = len(forecast_df)
    meta["obs_rows"] = len(obs_df)
    meta["matched_rows"] = len(merged)
    return merged, meta


def score_date_range(dates: list[str], series: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score many dates; return (all matched rows with series label, per-date metrics table)."""
    from model_lab.metrics import attach_obs_fire_danger, calculate_metrics

    frames = []
    metric_rows = []
    for date_token in dates:
        merged, meta = load_scored_series(date_token, series)
        if merged.empty:
            metric_rows.append({"date": date_token, "series": series, "count": 0, "fm_mae": None, "fm_bias": None})
            continue
        merged = attach_obs_fire_danger(merged)
        merged = merged.copy()
        merged["series"] = series
        merged["date"] = date_token
        frames.append(merged)
        metrics = calculate_metrics(merged)
        fm = metrics.get("Fuel Moisture (%)", {})
        metric_rows.append({
            "date": date_token,
            "series": series,
            "count": fm.get("count") or 0,
            "fm_mae": fm.get("mae"),
            "fm_rmse": fm.get("rmse"),
            "fm_bias": fm.get("bias"),
            "fm_corr": fm.get("correlation"),
            "temp_mae": metrics.get("Temperature (C)", {}).get("mae"),
            "rh_mae": metrics.get("Relative Humidity (%)", {}).get("mae"),
            "wind_mae": metrics.get("Wind Speed (m/s)", {}).get("mae"),
            "forecast_path": meta.get("forecast_path"),
        })
    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_rows, pd.DataFrame(metric_rows)


def list_xgb_artifacts() -> list[ModelRecord]:
    """Registry + versions/*.json that look like XGBoost fuel moisture boosters."""
    records = []
    seen = set()
    for record in list_registry_models():
        if record.path and record.path.suffix == ".json" and record.path.exists():
            key = str(record.path.resolve())
            if key not in seen:
                seen.add(key)
                records.append(record)
    versions_dir = MODELS_DIR / "versions"
    if versions_dir.exists():
        for path in sorted(versions_dir.glob("fuel_moisture_*.json")):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            records.append(ModelRecord(
                model_type="fuel_moisture",
                channel="file",
                version=path.stem.replace("fuel_moisture_", ""),
                file=str(path),
                performance={},
                trained_at=None,
                path=path,
                source="versions",
            ))
    return records
