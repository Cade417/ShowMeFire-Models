from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.observations import causal_initial_and_weather_targets, load_observations
from spatial.domain import MO_BUFFERED_BBOX
from spatial.precipitation import (
    PRECIPITATION_CONTRACT_SHA256,
    PRECIPITATION_CONTRACT_VERSION,
    decode_forecast_precipitation,
)

HRRR_RE = re.compile(r"hrrr_(\d{8})_(\d{2})z")
BUILDER_SCHEMA_VERSION = "aligned-precipitation-v1.2"


def _duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _report_progress(index, total, summary, total_rows, started_at, run_started_at):
    elapsed = time.perf_counter() - started_at
    average = elapsed / index if index else 0
    eta = average * (total - index)
    print(
        f"[{index}/{total}] {summary['hrrr']} status={summary['status']} "
        f"run_rows={summary.get('rows', 0)} total_rows={total_rows} "
        f"run_time={_duration(time.perf_counter() - run_started_at)} "
        f"elapsed={_duration(elapsed)} eta={_duration(eta)}",
        flush=True,
    )


def _open_hrrr(path):
    """Older cached HRRR files carry a stray 'dtype' attr on 'step' left over
    from GRIB->netCDF conversion, which collides with xarray's own encoding
    and fails CF-decoding outright. Strip it and decode manually instead."""
    raw = xr.open_dataset(path, decode_cf=False)
    for name in raw.variables:
        raw[name].attrs.pop("dtype", None)
    return xr.decode_cf(raw, decode_timedelta=True)


def _nearest(ds, lat, lon):
    grid_lon = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
    distance = (ds.latitude - lat) ** 2 + (grid_lon - lon) ** 2
    dims = list(distance.dims)
    flat = int(np.nanargmin(distance.values))
    indices = np.unravel_index(flat, distance.shape)
    return {dim: index for dim, index in zip(dims, indices)}


def _grid_signature(ds):
    latitude, longitude = ds.latitude, ds.longitude
    dims = tuple(latitude.dims)
    sizes = tuple((dim, int(latitude.sizes[dim])) for dim in dims)
    samples = []
    for fraction in (0.0, 0.5, 1.0):
        selector = {dim: min(latitude.sizes[dim] - 1, int((latitude.sizes[dim] - 1) * fraction)) for dim in dims}
        samples.extend((float(latitude.isel(**selector).values), float(longitude.isel(**selector).values)))
    return sizes, tuple(round(value, 6) for value in samples)


def _nearest_cached(cache, source, signature, ds, station_id, lat, lon):
    key = (source, signature, str(station_id), round(float(lat), 6), round(float(lon), 6))
    if key not in cache:
        cache[key] = _nearest(ds, lat, lon)
    return cache[key]


def _inside_precipitation_context(lat, lon):
    west, east, south, north = MO_BUFFERED_BBOX
    return west <= lon <= east and south <= lat <= north


def _value(point, name, default=np.nan):
    if name not in point:
        return default
    value = np.asarray(point[name].values).squeeze()
    return float(value) if value.size == 1 else default


def _scalar(array, selector, default=np.nan):
    try:
        value = np.asarray(array.isel(**{key: val for key, val in selector.items() if key in array.dims}).values).squeeze()
        return float(value) if value.size == 1 else default
    except (KeyError, IndexError, TypeError, ValueError):
        return default


def _temp_c(value):
    return value - 273.15 if np.isfinite(value) and value > 150 else value


def _valid_times(ds, init_time):
    if "valid_time" in ds:
        values = np.atleast_1d(ds.valid_time.values)
        if len(values) > 1:
            return pd.to_datetime(values, utc=True)
    if "step" in ds.coords:
        return pd.DatetimeIndex([init_time + pd.to_timedelta(step) for step in ds.step.values])
    return pd.DatetimeIndex([init_time])


def _step_hours(array):
    values = np.asarray(array.step.values)
    return values / np.timedelta64(1, "h") if np.issubdtype(values.dtype, np.timedelta64) else values.astype(float)


def build(max_age_hours=3, progress_every=1, output=None, run_limit=None, resume=True,
          start_run=None, end_run=None):
    observations = load_observations(paths.ARCHIVE_RAW_DATA_DIR)
    if observations.empty:
        raise RuntimeError(f"No station observations found in {paths.ARCHIVE_RAW_DATA_DIR}")
    rows = []
    run_summaries = []
    hrrr_paths = sorted(path for path in paths.CACHE_HRRR_DIR.glob("hrrr_*.nc") if HRRR_RE.search(path.name))
    if start_run:
        hrrr_paths = [path for path in hrrr_paths if "".join(HRRR_RE.search(path.name).groups()) >= str(start_run)]
    if end_run:
        hrrr_paths = [path for path in hrrr_paths if "".join(HRRR_RE.search(path.name).groups()) <= str(end_run)]
    if run_limit:
        hrrr_paths = hrrr_paths[:run_limit]
    total_runs = len(hrrr_paths)
    fragment_dir = paths.ALIGNED_DIR / f".{PRECIPITATION_CONTRACT_VERSION}-fragments"
    if resume: fragment_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    nearest_cache = {}
    print(
        f"Building aligned dataset from {total_runs} HRRR runs and "
        f"{observations['station_id'].nunique()} stations...",
        flush=True,
    )
    for index, hrrr_path in enumerate(hrrr_paths, start=1):
        run_started_at = time.perf_counter()
        match = HRRR_RE.search(hrrr_path.name)
        if not match:
            continue
        init_time = pd.Timestamp(f"{match.group(1)} {match.group(2)}:00", tz="UTC")
        rtma_path = paths.CACHE_RTMA_DIR / f"rtma_{init_time:%Y%m%d_%H}z.nc"
        context_path = paths.CACHE_HRRR_DIR / f"hrrr_precip_context_{init_time:%Y%m%d_%H}z_f00-03.nc"
        summary = {"hrrr": hrrr_path.name, "init_time": init_time.isoformat(), "rtma": rtma_path.name, "status": "pending"}
        print(f"[{index}/{total_runs}] Processing {hrrr_path.name}...", flush=True)
        if not rtma_path.exists():
            summary["status"] = "missing_rtma"
            run_summaries.append(summary)
            if index % progress_every == 0 or index == total_runs:
                _report_progress(index, total_runs, summary, len(rows), started_at, run_started_at)
            continue
        fragment = fragment_dir / f"{hrrr_path.stem}.csv"
        fragment_meta = fragment.with_suffix(".json")
        source_fingerprint = {
            "hrrr_mtime_ns": hrrr_path.stat().st_mtime_ns,
            "rtma_mtime_ns": rtma_path.stat().st_mtime_ns,
            "context_mtime_ns": context_path.stat().st_mtime_ns if context_path.exists() else None,
            "max_initial_age_hours": max_age_hours,
            "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
            "builder_schema_version": BUILDER_SCHEMA_VERSION,
        }
        if resume and fragment.exists() and fragment_meta.exists():
            try:
                if json.loads(fragment_meta.read_text()) == source_fingerprint:
                    cached = pd.read_csv(fragment)
                    rows.extend(cached.to_dict("records"))
                    summary.update({"status": "resumed", "rows": len(cached),
                                    "precip_context_f00_f03": context_path.exists()})
                    run_summaries.append(summary)
                    _report_progress(index, total_runs, summary, len(rows), started_at, run_started_at)
                    continue
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        try:
            run_row_start = len(rows)
            with _open_hrrr(hrrr_path) as hrrr, xr.open_dataset(rtma_path) as rtma:
                valid_times = _valid_times(hrrr, init_time)
                if "step" in hrrr.dims and len(valid_times) != hrrr.sizes["step"]:
                    valid_times = pd.DatetimeIndex([init_time + pd.to_timedelta(step) for step in hrrr.step.values])
                precipitation = decode_forecast_precipitation(hrrr)
                precipitation_leads = _step_hours(precipitation.cumulative_mm)
                hrrr_grid_signature = _grid_signature(hrrr)
                rtma_grid_signature = _grid_signature(rtma)
                context_precipitation = None
                context_leads = np.array([], dtype=float)
                context_grid_signature = None
                precipitation_context_available = False
                if context_path.exists():
                    with _open_hrrr(context_path) as context:
                        context_precipitation = decode_forecast_precipitation(context.load())
                        context_leads = _step_hours(context_precipitation.cumulative_mm)
                        context_grid_signature = _grid_signature(context_precipitation.cumulative_mm)
                        precipitation_context_available = True
                lead_indices = [
                    position for position, valid_time in enumerate(valid_times)
                    if 4 <= (pd.Timestamp(valid_time) - init_time).total_seconds() / 3600 <= 15
                ]
                run_rows = 0
                for station_id, station_obs in observations.groupby("station_id"):
                    location = station_obs.dropna(subset=["lat", "lon"])
                    if location.empty:
                        continue
                    lat, lon = float(location.iloc[-1].lat), float(location.iloc[-1].lon)
                    target_times = valid_times[lead_indices]
                    initial_fm, initial_age, targets = causal_initial_and_weather_targets(station_obs, init_time, target_times, max_age_hours=max_age_hours)
                    if not np.isfinite(initial_fm):
                        continue
                    hi = _nearest_cached(nearest_cache, "hrrr", hrrr_grid_signature, hrrr, station_id, lat, lon)
                    context_index = None
                    if context_precipitation is not None and _inside_precipitation_context(lat, lon):
                        context_index = _nearest_cached(
                            nearest_cache, "hrrr_context", context_grid_signature,
                            context_precipitation.cumulative_mm, station_id, lat, lon,
                        )
                    ri = _nearest_cached(nearest_cache, "rtma", rtma_grid_signature, rtma, station_id, lat, lon)
                    rtma_point = rtma.isel(**ri)
                    rtma_temp = _temp_c(_value(rtma_point, "t2m"))
                    rtma_rh = _value(rtma_point, "r2")
                    rtma_wind = np.hypot(_value(rtma_point, "u10"), _value(rtma_point, "v10"))
                    for lead_index, valid_time, target in zip(lead_indices, target_times, targets):
                        selector = {**hi}
                        if "step" in hrrr.dims:
                            selector["step"] = lead_index
                        point = hrrr.isel(**selector)
                        lead_hour = (pd.Timestamp(valid_time) - init_time).total_seconds() / 3600
                        matching_precip = np.flatnonzero(np.isclose(precipitation_leads, lead_hour))
                        if not len(matching_precip):
                            raise ValueError(f"precipitation lead {lead_hour} is unavailable")
                        precip_selector = {**hi, "step": int(matching_precip[0])}
                        u10, v10 = _value(point, "u10"), _value(point, "v10")
                        precip_cumulative = _scalar(precipitation.cumulative_mm, precip_selector)
                        precip_interval = _scalar(precipitation.interval_mm, precip_selector)
                        precip_interval_hours = _scalar(precipitation.interval_hours, precip_selector)
                        precip_reset = _scalar(precipitation.reset_flag, precip_selector, 0.0)
                        precip_partial = _scalar(precipitation.partial_window_flag, precip_selector, 1.0)
                        if context_index is not None and np.isclose(lead_hour, 4.0):
                            prior_positions = np.flatnonzero(context_leads < lead_hour)
                            if len(prior_positions):
                                prior_position = int(prior_positions[-1])
                                prior_cumulative = _scalar(
                                    context_precipitation.cumulative_mm,
                                    {**context_index, "step": prior_position},
                                )
                                if np.isfinite(prior_cumulative):
                                    raw_interval = precip_cumulative - prior_cumulative
                                    precip_reset = raw_interval < -1e-6
                                    precip_interval = max(0.0, raw_interval)
                                    precip_interval_hours = lead_hour - float(context_leads[prior_position])
                                    precip_partial = not np.isclose(precip_interval_hours, 1.0)
                        target_fm, target_time = target["target_fm"], target["target_time"]
                        rows.append({
                            "run_id": f"{init_time:%Y%m%d%H}", "forecast_init_time": init_time.isoformat(),
                            "valid_time": pd.Timestamp(valid_time).isoformat(), "lead_hour": (pd.Timestamp(valid_time) - init_time).total_seconds() / 3600,
                            "station_id": station_id, "lat": lat, "lon": lon,
                            "initial_fm": initial_fm, "initial_age_hours": initial_age, "initial_mask": 1,
                            "target_fm": target_fm, "target_time": target_time.isoformat() if target_time is not None else None,
                            "target_mask": int(np.isfinite(target_fm)),
                            "target_rh": target["target_rh"], "target_wind_ms": target["target_wind_ms"],
                            "target_rh_mask": int(np.isfinite(target["target_rh"])),
                            "target_wind_mask": int(np.isfinite(target["target_wind_ms"])),
                            "target_match_age_minutes": target["target_match_age_minutes"],
                            "rtma_analysis_time": init_time.isoformat(), "rtma_temp_c": rtma_temp, "rtma_rh": rtma_rh, "rtma_wind_ms": rtma_wind,
                            "hrrr_temp_c": _temp_c(_value(point, "t2m")), "hrrr_rh": _value(point, "r2"),
                            "hrrr_wind_ms": np.hypot(u10, v10),
                            "hrrr_precip_mm": precip_cumulative,
                            "hrrr_precip_accum_mm": precip_cumulative,
                            "hrrr_precip_increment_mm": precip_interval,
                            "precip_interval_hours": precip_interval_hours,
                            "precip_reset_flag": int(bool(precip_reset)),
                            "precip_partial_window_flag": int(bool(precip_partial)),
                            "precip_available": int(np.isfinite(precip_cumulative)),
                        })
                        run_rows += 1
                summary.update({"status": "complete", "rows": run_rows, "valid_times": len(target_times),
                                "precip_variable": precipitation.variable_name,
                                "precip_units": precipitation.source_units,
                                "precip_accumulation_kind": precipitation.accumulation_kind})
                summary["precip_context_f00_f03"] = precipitation_context_available
                if resume:
                    run_frame = pd.DataFrame(rows[run_row_start:])
                    temporary = fragment.with_suffix(".csv.tmp")
                    run_frame.to_csv(temporary, index=False); temporary.replace(fragment)
                    metadata_tmp = fragment_meta.with_suffix(".json.tmp")
                    metadata_tmp.write_text(json.dumps(source_fingerprint)); metadata_tmp.replace(fragment_meta)
        except Exception as exc:
            summary.update({"status": "failed", "error": str(exc)})
        run_summaries.append(summary)
        if index % progress_every == 0 or index == total_runs:
            _report_progress(index, total_runs, summary, len(rows), started_at, run_started_at)
    frame = pd.DataFrame(rows)
    run_failures = [item for item in run_summaries if item["status"] == "failed"]
    if run_failures:
        examples = "; ".join(f"{item['hrrr']}: {item.get('error', 'unknown error')}" for item in run_failures[:3])
        raise RuntimeError(f"Aligned build failed for {len(run_failures)} HRRR runs; no dataset written ({examples})")
    if frame.empty or not frame["hrrr_precip_accum_mm"].notna().any():
        raise RuntimeError("No usable precipitation values were decoded; no dataset written")
    if len(frame) >= 1000 and not (frame["hrrr_precip_accum_mm"].fillna(0) > 0).any():
        raise RuntimeError("Precipitation is constant zero across the archive; no dataset written")
    output = Path(output) if output else paths.ALIGNED_DIR / f"station_leads_{PRECIPITATION_CONTRACT_VERSION}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary_output, index=False)
    temporary_output.replace(output)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "max_initial_age_hours": max_age_hours,
        "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
        "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
        "builder_schema_version": BUILDER_SCHEMA_VERSION,
        "output": str(output),
        "rows": len(frame),
        "runs": run_summaries,
    }, indent=2))
    print(f"Wrote {len(frame)} aligned station/lead rows to {output}")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-initial-age-hours", type=float, default=3)
    parser.add_argument("--output", type=Path,
                        help="Versioned output CSV (default: station_leads_precipitation-v1.csv)")
    parser.add_argument("--run-limit", type=int, help="Process only the first N discovered runs (for smoke tests)")
    parser.add_argument("--start-run", help="First initialization as YYYYMMDDHH")
    parser.add_argument("--end-run", help="Last initialization as YYYYMMDDHH")
    parser.add_argument("--no-resume", action="store_true", help="Ignore versioned per-run resume fragments")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print a completion summary every N HRRR runs (default: 1)",
    )
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1")
    build(args.max_initial_age_hours, args.progress_every, args.output, args.run_limit, not args.no_resume,
          args.start_run, args.end_run)


if __name__ == "__main__":
    main()
