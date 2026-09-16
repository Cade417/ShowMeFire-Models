"""Build and validate a Fire Danger dataset from archived Synoptic observations."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import paths

FD_DATASET_PATH = paths.DATA_ROOT / "data" / "fire_danger_training_data.csv"
FD_REPORT_PATH = paths.DATA_ROOT / "reports" / "model_lab" / "fire_danger_dataset.json"


def _category(fuel_moisture: float, rh: float, wind_ms: float) -> int:
    wind_kts = wind_ms * 1.94384
    if fuel_moisture >= 15:
        return 0
    if fuel_moisture < 7 and rh < 20 and wind_kts >= 25:
        return 4
    if fuel_moisture < 9 and rh < 25 and wind_kts >= 15:
        return 3
    if fuel_moisture < 9 and ((rh < 35 and wind_kts >= 12) or (rh < 25 and wind_kts >= 5)):
        return 2
    if fuel_moisture < 15 and (rh < 45 or wind_kts >= 10):
        return 1
    return 0


def _as_list(observations: dict, key: str, length: int) -> list:
    values = observations.get(key) or []
    return list(values)[:length] + [None] * max(0, length - len(values))


def _load_raw_file(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stations = payload.get("STATION", []) if isinstance(payload, dict) else payload
    rows = []
    for station in stations or []:
        if station.get("STATE") != "MO":
            continue
        observations = station.get("OBSERVATIONS") or {}
        times = observations.get("date_time") or []
        length = len(times)
        fields = {
            "air_temp_f": _as_list(observations, "air_temp_set_1", length),
            "rel_humidity": _as_list(observations, "relative_humidity_set_1", length),
            "wind_speed_ms": _as_list(observations, "wind_speed_set_1", length),
            "target_fm": _as_list(observations, "fuel_moisture_set_1", length),
            "precip_accum_mm": _as_list(observations, "precip_accum_set_1", length),
        }
        for index, timestamp in enumerate(times):
            rows.append({
                "station_id": station.get("STID"),
                "station_name": station.get("NAME"),
                "lat": station.get("LATITUDE"),
                "lon": station.get("LONGITUDE"),
                "obs_time": timestamp,
                **{key: values[index] for key, values in fields.items()},
            })
    return rows


def build_fire_danger_dataset(
    *,
    raw_dir: Path = paths.ARCHIVE_RAW_DATA_DIR,
    output_path: Path = FD_DATASET_PATH,
    report_path: Path = FD_REPORT_PATH,
) -> dict:
    files = sorted(Path(raw_dir).glob("raw_data_*.json"))
    rows = []
    errors = []
    for path in files:
        try:
            rows.extend(_load_raw_file(path))
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": path.name, "error": str(exc)})

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No usable Synoptic observations found under {raw_dir}")

    numeric = ["air_temp_f", "rel_humidity", "wind_speed_ms", "target_fm", "precip_accum_mm"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["obs_time"] = pd.to_datetime(frame["obs_time"], errors="coerce", utc=True)
    frame["temp_c"] = (frame["air_temp_f"] - 32.0) * (5.0 / 9.0)
    before_dedup = len(frame)
    frame = frame.drop_duplicates(["station_id", "obs_time"]).sort_values(["station_id", "obs_time"])

    valid = frame.dropna(subset=["station_id", "obs_time", "temp_c", "rel_humidity", "wind_speed_ms", "target_fm"]).copy()
    valid = valid[
        valid["rel_humidity"].between(0, 100)
        & valid["wind_speed_ms"].between(0, 100)
        & valid["target_fm"].between(0, 100)
    ].copy()
    valid["hour"] = valid["obs_time"].dt.hour
    valid["month"] = valid["obs_time"].dt.month
    valid["emc_baseline"] = valid["rel_humidity"] / 5.0
    grouped = valid.groupby("station_id", group_keys=False)
    valid["temp_mean_3h"] = grouped["temp_c"].transform(lambda series: series.rolling(3, min_periods=1).mean())
    valid["rh_mean_3h"] = grouped["rel_humidity"].transform(lambda series: series.rolling(3, min_periods=1).mean())
    valid["temp_mean_6h"] = grouped["temp_c"].transform(lambda series: series.rolling(6, min_periods=1).mean())
    valid["rh_mean_6h"] = grouped["rel_humidity"].transform(lambda series: series.rolling(6, min_periods=1).mean())
    valid["fire_danger_category"] = [
        _category(fm, rh, wind)
        for fm, rh, wind in valid[["target_fm", "rel_humidity", "wind_speed_ms"]].itertuples(index=False, name=None)
    ]

    output_columns = [
        "station_id", "station_name", "lat", "lon", "obs_time", "target_fm",
        "temp_c", "rel_humidity", "wind_speed_ms", "hour", "month",
        "emc_baseline", "temp_mean_3h", "rh_mean_3h", "temp_mean_6h",
        "rh_mean_6h", "precip_accum_mm", "fire_danger_category",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid[output_columns].to_csv(output_path, index=False)

    support = {str(category): int(count) for category, count in valid["fire_danger_category"].value_counts().sort_index().items()}
    support = {str(category): support.get(str(category), 0) for category in range(5)}
    report = {
        "dataset_path": str(output_path),
        "source_directory": str(raw_dir),
        "source_files": len(files),
        "source_errors": errors,
        "rows_parsed": len(frame),
        "rows_valid": len(valid),
        "stations": int(valid["station_id"].nunique()),
        "first_observation": valid["obs_time"].min().isoformat() if not valid.empty else None,
        "last_observation": valid["obs_time"].max().isoformat() if not valid.empty else None,
        "class_support": support,
        "duplicate_rows_removed": int(before_dedup - len(frame)),
        "missing_rates": {
            column: float(frame[column].isna().mean())
            for column in ("temp_c", "rel_humidity", "wind_speed_ms", "target_fm")
        },
        "ready_for_basic_training": bool(
            len(valid) >= 1000
            and all(support[str(category)] >= 25 for category in (0, 1, 2))
        ),
        "ready_for_training": bool(
            len(valid) >= 1000
            and all(support[str(category)] >= 25 for category in range(5))
        ),
        "warnings": [
            f"class {category} has only {support[str(category)]} rows"
            for category in range(5)
            if support[str(category)] < 25
        ],
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
