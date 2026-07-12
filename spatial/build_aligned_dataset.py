from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.observations import causal_initial_and_targets, load_observations

HRRR_RE = re.compile(r"hrrr_(\d{8})_(\d{2})z")


def _nearest(ds, lat, lon):
    grid_lon = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
    distance = (ds.latitude - lat) ** 2 + (grid_lon - lon) ** 2
    dims = list(distance.dims)
    flat = int(np.nanargmin(distance.values))
    indices = np.unravel_index(flat, distance.shape)
    return {dim: index for dim, index in zip(dims, indices)}


def _value(point, name, default=np.nan):
    if name not in point:
        return default
    value = np.asarray(point[name].values).squeeze()
    return float(value) if value.size == 1 else default


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


def build(max_age_hours=3):
    observations = load_observations(paths.ARCHIVE_RAW_DATA_DIR)
    if observations.empty:
        raise RuntimeError(f"No station observations found in {paths.ARCHIVE_RAW_DATA_DIR}")
    rows = []
    run_summaries = []
    for hrrr_path in sorted(paths.CACHE_HRRR_DIR.glob("hrrr_*.nc")):
        match = HRRR_RE.search(hrrr_path.name)
        if not match:
            continue
        init_time = pd.Timestamp(f"{match.group(1)} {match.group(2)}:00", tz="UTC")
        rtma_path = paths.CACHE_RTMA_DIR / f"rtma_{init_time:%Y%m%d_%H}z.nc"
        summary = {"hrrr": hrrr_path.name, "init_time": init_time.isoformat(), "rtma": rtma_path.name, "status": "pending"}
        if not rtma_path.exists():
            summary["status"] = "missing_rtma"
            run_summaries.append(summary)
            continue
        try:
            with xr.open_dataset(hrrr_path) as hrrr, xr.open_dataset(rtma_path) as rtma:
                valid_times = _valid_times(hrrr, init_time)
                if "step" in hrrr.dims and len(valid_times) != hrrr.sizes["step"]:
                    valid_times = pd.DatetimeIndex([init_time + pd.to_timedelta(step) for step in hrrr.step.values])
                run_rows = 0
                for station_id, station_obs in observations.groupby("station_id"):
                    location = station_obs.dropna(subset=["lat", "lon"])
                    if location.empty:
                        continue
                    lat, lon = float(location.iloc[-1].lat), float(location.iloc[-1].lon)
                    initial_fm, initial_age, targets = causal_initial_and_targets(station_obs, init_time, valid_times, max_age_hours=max_age_hours)
                    if not np.isfinite(initial_fm):
                        continue
                    hi = _nearest(hrrr, lat, lon)
                    ri = _nearest(rtma, lat, lon)
                    rtma_point = rtma.isel(**ri)
                    rtma_temp = _temp_c(_value(rtma_point, "t2m"))
                    rtma_rh = _value(rtma_point, "r2")
                    rtma_wind = np.hypot(_value(rtma_point, "u10"), _value(rtma_point, "v10"))
                    for lead_index, (valid_time, target) in enumerate(zip(valid_times, targets)):
                        selector = {**hi}
                        if "step" in hrrr.dims:
                            selector["step"] = lead_index
                        point = hrrr.isel(**selector)
                        u10, v10 = _value(point, "u10"), _value(point, "v10")
                        target_fm, target_time = target
                        rows.append({
                            "run_id": f"{init_time:%Y%m%d%H}", "forecast_init_time": init_time.isoformat(),
                            "valid_time": pd.Timestamp(valid_time).isoformat(), "lead_hour": (pd.Timestamp(valid_time) - init_time).total_seconds() / 3600,
                            "station_id": station_id, "lat": lat, "lon": lon,
                            "initial_fm": initial_fm, "initial_age_hours": initial_age, "initial_mask": 1,
                            "target_fm": target_fm, "target_time": target_time.isoformat() if target_time is not None else None,
                            "target_mask": int(np.isfinite(target_fm)),
                            "rtma_analysis_time": init_time.isoformat(), "rtma_temp_c": rtma_temp, "rtma_rh": rtma_rh, "rtma_wind_ms": rtma_wind,
                            "hrrr_temp_c": _temp_c(_value(point, "t2m")), "hrrr_rh": _value(point, "r2"),
                            "hrrr_wind_ms": np.hypot(u10, v10), "hrrr_precip_mm": _value(point, "apcp", 0.0),
                        })
                        run_rows += 1
                summary.update({"status": "complete", "rows": run_rows, "valid_times": len(valid_times)})
        except Exception as exc:
            summary.update({"status": "failed", "error": str(exc)})
        run_summaries.append(summary)
    frame = pd.DataFrame(rows)
    output = paths.ALIGNED_DIR / "station_leads.csv"
    frame.to_csv(output, index=False)
    (paths.ALIGNED_DIR / "build_manifest.json").write_text(json.dumps({"max_initial_age_hours": max_age_hours, "runs": run_summaries}, indent=2))
    print(f"Wrote {len(frame)} aligned station/lead rows to {output}")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-initial-age-hours", type=float, default=3)
    args = parser.parse_args()
    build(args.max_initial_age_hours)


if __name__ == "__main__":
    main()
