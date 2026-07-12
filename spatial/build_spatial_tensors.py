"""Build dynamic run tensors referencing one immutable static bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.build_aligned_dataset import HRRR_RE, _valid_times
from spatial.grid import apply_mapping, idw_initial_analysis, nearest_mapping
from spatial.observations import causal_initial_and_targets, load_observations
from spatial.physics import evolve_fm
from static_features.schema import validate_bundle


def _lon180(values):
    values = np.asarray(values)
    return np.where(values > 180, values - 360, values)


def _step_field(ds, name, target_x, target_y, steps, default=0.0):
    if name not in ds:
        return np.full((steps, len(target_y), len(target_x)), default, dtype="float32")
    field = ds[name]
    spatial_dims = ds.latitude.dims
    ydim, xdim = spatial_dims
    interpolated = field.interp({xdim: target_x, ydim: target_y}, method="linear")
    values = np.asarray(interpolated.values).squeeze()
    if values.ndim == 2: values = np.repeat(values[None], steps, axis=0)
    return values.astype("float32")


def build(static_bundle: Path, limit=None):
    static_manifest = validate_bundle(static_bundle)
    with xr.open_dataset(static_bundle) as static_ds:
        target_x, target_y = static_ds.x.values, static_ds.y.values
        grid_lat, grid_lon = static_ds.latitude.values, static_ds.longitude.values
    observations = load_observations(paths.ARCHIVE_RAW_DATA_DIR)
    output_dir = paths.ALIGNED_DIR / "spatial_tensors"; output_dir.mkdir(parents=True, exist_ok=True)
    built = 0
    for hrrr_path in sorted(paths.CACHE_HRRR_DIR.glob("hrrr_*.nc")):
        match = HRRR_RE.search(hrrr_path.name)
        if not match: continue
        run_id = f"{match.group(1)}{match.group(2)}"; output = output_dir / f"spatial_{run_id}.npz"
        if output.exists(): continue
        init = pd.Timestamp(f"{match.group(1)} {match.group(2)}:00", tz="UTC")
        rtma_path = paths.CACHE_RTMA_DIR / f"rtma_{init:%Y%m%d_%H}z.nc"
        if not rtma_path.exists(): continue
        with xr.open_dataset(hrrr_path) as hrrr, xr.open_dataset(rtma_path) as rtma:
            valid = _valid_times(hrrr, init); steps = len(valid)
            mapping, _ = nearest_mapping(rtma.latitude.values, rtma.longitude.values, grid_lat, grid_lon)
            rtma_temp = apply_mapping(rtma.t2m.values, mapping); rtma_temp = np.where(rtma_temp > 150, rtma_temp - 273.15, rtma_temp)
            rtma_rh = apply_mapping(rtma.r2.values, mapping)
            rtma_wind = np.hypot(apply_mapping(rtma.u10.values, mapping), apply_mapping(rtma.v10.values, mapping))

            station_ids, station_lat, station_lon, station_fm, station_age, station_targets = [], [], [], [], [], []
            for station_id, group in observations.groupby("station_id"):
                location = group.dropna(subset=["lat", "lon"])
                if location.empty: continue
                initial_fm, age, targets = causal_initial_and_targets(group, init, valid)
                if np.isfinite(initial_fm):
                    station_ids.append(station_id)
                    station_lat.append(float(location.iloc[-1].lat)); station_lon.append(float(location.iloc[-1].lon))
                    station_fm.append(initial_fm); station_age.append(age); station_targets.append(targets)
            if not station_fm: continue
            initial, distance, effective = idw_initial_analysis(station_lat, station_lon, station_fm, grid_lat, grid_lon)
            mask0 = np.zeros(grid_lat.shape, dtype="float32"); age_grid = np.full(grid_lat.shape, 3.0, dtype="float32")
            target = np.zeros((steps, 1, *grid_lat.shape), dtype="float32"); target_mask = np.zeros_like(target)
            train_mask = np.zeros_like(target); station_holdout_mask = np.zeros_like(target); region_holdout_mask = np.zeros_like(target)
            pixel_tree = cKDTree(np.column_stack((grid_lat.ravel(), grid_lon.ravel())))
            for station_id, lat, lon, age, targets in zip(station_ids, station_lat, station_lon, station_age, station_targets):
                pixel = int(pixel_tree.query([lat, lon])[1]); y, x = np.unravel_index(pixel, grid_lat.shape)
                mask0[y, x] = 1; age_grid[y, x] = age
                for step, (value, _) in enumerate(targets):
                    if np.isfinite(value):
                        target[step, 0, y, x] = value; target_mask[step, 0, y, x] = 1
                        held_station = int(hashlib.sha256(str(station_id).encode()).hexdigest(), 16) % 5 == 0
                        if held_station: station_holdout_mask[step, 0, y, x] = 1
                        elif lon >= -92.45: region_holdout_mask[step, 0, y, x] = 1
                        else: train_mask[step, 0, y, x] = 1
            temp = _step_field(hrrr, "t2m", target_x, target_y, steps); temp = np.where(temp > 150, temp - 273.15, temp)
            rh = _step_field(hrrr, "r2", target_x, target_y, steps)
            wind = np.hypot(_step_field(hrrr, "u10", target_x, target_y, steps), _step_field(hrrr, "v10", target_x, target_y, steps))
            precip = _step_field(hrrr, "apcp", target_x, target_y, steps)
            physics = evolve_fm(initial, temp, rh)[:, None].astype("float32")
            current = [initial, mask0, age_grid, distance, effective, rtma_temp, rtma_rh, rtma_wind]
            sequence = []
            for step in range(steps):
                hour = pd.Timestamp(valid[step]).hour * 2 * np.pi / 24
                sequence.append(np.stack([temp[step], rh[step], wind[step], precip[step], *current,
                                          np.full_like(initial, np.sin(hour)), np.full_like(initial, np.cos(hour))]))
            metadata = {
                "run_id": run_id, "forecast_init_time": init.isoformat(), "valid_times": [pd.Timestamp(v).isoformat() for v in valid],
                "dynamic_channels": ["hrrr_temp_c", "hrrr_rh", "hrrr_wind_speed", "hrrr_precip", "initial_fm_analysis",
                                     "initial_obs_mask", "initial_age_hours", "nearest_station_distance_deg", "effective_station_count",
                                     "rtma_temp_c", "rtma_rh", "rtma_wind_speed", "sin_hour", "cos_hour"],
                "static_bundle": static_bundle.name, "static_bundle_sha256": static_manifest["sha256"],
                "grid_fingerprint": static_manifest["grid_fingerprint"], "station_count": len(station_fm),
                "labels_are_observations_only": True,
            }
            np.savez_compressed(output, dynamic_sequence=np.asarray(sequence, dtype="float32"), physics=physics,
                                target=target, target_mask=target_mask, train_mask=train_mask, station_holdout_mask=station_holdout_mask,
                                region_holdout_mask=region_holdout_mask, metadata=json.dumps(metadata))
        built += 1
        if limit and built >= limit: break
    print(f"Built {built} dynamic tensor file(s) in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--static-bundle", required=True, type=Path); parser.add_argument("--limit", type=int)
    args = parser.parse_args(); build(args.static_bundle, args.limit)
