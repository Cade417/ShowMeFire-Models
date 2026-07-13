"""Build leakage-safe teacher/student tensors on the immutable static grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import timedelta
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

TARGET_LEADS = tuple(range(4, 16))
ANTECEDENT_HOURS = 12


def _step_field(ds, name, target_x, target_y, indices, default=0.0):
    if name not in ds:
        return np.full((len(indices), len(target_y), len(target_x)), default, dtype="float32")
    field = ds[name]; ydim, xdim = ds.latitude.dims
    values = np.asarray(field.interp({xdim: target_x, ydim: target_y}, method="linear").values).squeeze()
    if values.ndim == 2:
        values = values[None]
    return values[np.asarray(indices)].astype("float32")


def _load_rtma(path, mapping):
    with xr.open_dataset(path) as ds:
        missing = {"t2m", "r2", "u10", "v10"} - set(ds.variables)
        if missing:
            raise ValueError(f"{path.name} missing {sorted(missing)}")
        temp = apply_mapping(np.asarray(ds.t2m.values).squeeze(), mapping)
        temp = np.where(temp > 150, temp - 273.15, temp)
        rh = apply_mapping(np.asarray(ds.r2.values).squeeze(), mapping)
        # Speed is invariant under grid-to-earth vector rotation. Preserve this
        # explicit earth-relative scalar contract until vector channels are added.
        wind = np.hypot(apply_mapping(np.asarray(ds.u10.values).squeeze(), mapping),
                        apply_mapping(np.asarray(ds.v10.values).squeeze(), mapping))
    available = np.ones_like(temp, dtype="float32")
    return np.stack((temp, rh, wind, available)).astype("float32")


def _rtma_sequence(times, grid_lat, grid_lon, mapping_cache):
    frames = []
    for value in times:
        path = paths.CACHE_RTMA_DIR / f"rtma_{value:%Y%m%d_%H}z.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as ds:
            signature = (ds.latitude.shape, float(np.nanmin(ds.latitude)), float(np.nanmax(ds.latitude)),
                         float(np.nanmin(ds.longitude)), float(np.nanmax(ds.longitude)))
            if signature not in mapping_cache:
                mapping_cache[signature] = nearest_mapping(ds.latitude.values, ds.longitude.values, grid_lat, grid_lon)[0]
        frames.append(_load_rtma(path, mapping_cache[signature]))
    return np.stack(frames).astype("float32")


def build(static_bundle: Path, limit=None):
    static_manifest = validate_bundle(static_bundle)
    with xr.open_dataset(static_bundle) as static_ds:
        target_x, target_y = static_ds.x.values, static_ds.y.values
        grid_lat, grid_lon = static_ds.latitude.values, static_ds.longitude.values
    observations = load_observations(paths.ARCHIVE_RAW_DATA_DIR)
    output_dir = paths.ALIGNED_DIR / "spatial_tensors"; output_dir.mkdir(parents=True, exist_ok=True)
    built = 0; rejected = []; mapping_cache = {}
    for hrrr_path in sorted(paths.CACHE_HRRR_DIR.glob("hrrr_*.nc")):
        match = HRRR_RE.search(hrrr_path.name)
        if not match:
            continue
        run_id = f"{match.group(1)}{match.group(2)}"; output = output_dir / f"spatial_{run_id}.npz"
        if output.exists():
            continue
        init = pd.Timestamp(f"{match.group(1)} {match.group(2)}:00", tz="UTC")
        try:
            with xr.open_dataset(hrrr_path) as hrrr:
                all_valid = [pd.Timestamp(value) for value in _valid_times(hrrr, init)]
                lead_by_time = {int((value - init).total_seconds() // 3600): index for index, value in enumerate(all_valid)}
                if any(lead not in lead_by_time for lead in TARGET_LEADS):
                    raise ValueError(f"HRRR must contain exact leads {TARGET_LEADS}")
                indices = [lead_by_time[lead] for lead in TARGET_LEADS]
                valid = [init + pd.Timedelta(hours=lead) for lead in TARGET_LEADS]
                temp = _step_field(hrrr, "t2m", target_x, target_y, indices); temp = np.where(temp > 150, temp - 273.15, temp)
                rh = _step_field(hrrr, "r2", target_x, target_y, indices)
                wind = np.hypot(_step_field(hrrr, "u10", target_x, target_y, indices), _step_field(hrrr, "v10", target_x, target_y, indices))
                precip = _step_field(hrrr, "apcp", target_x, target_y, indices)
            antecedent_times = [init - pd.Timedelta(hours=hour) for hour in range(ANTECEDENT_HOURS, -1, -1)]
            realized_times = [init + pd.Timedelta(hours=hour) for hour in range(1, max(TARGET_LEADS) + 1)]
            antecedent = _rtma_sequence(antecedent_times, grid_lat, grid_lon, mapping_cache)
            realized = _rtma_sequence(realized_times, grid_lat, grid_lon, mapping_cache)

            station_ids, station_lat, station_lon, station_fm, station_age, station_targets = [], [], [], [], [], []
            for station_id, group in observations.groupby("station_id"):
                location = group.dropna(subset=["lat", "lon"])
                if location.empty:
                    continue
                initial_fm, age, targets = causal_initial_and_targets(group, init, valid)
                if np.isfinite(initial_fm):
                    station_ids.append(station_id); station_lat.append(float(location.iloc[-1].lat)); station_lon.append(float(location.iloc[-1].lon))
                    station_fm.append(initial_fm); station_age.append(age); station_targets.append(targets)
            if not station_fm:
                raise ValueError("no causal fuel-moisture state")
            initial, distance, effective = idw_initial_analysis(station_lat, station_lon, station_fm, grid_lat, grid_lon)
            mask0 = np.zeros(grid_lat.shape, dtype="float32"); age_grid = np.full(grid_lat.shape, 3.0, dtype="float32")
            shape = (len(TARGET_LEADS), 1, *grid_lat.shape)
            target = np.zeros(shape, dtype="float32"); target_mask = np.zeros(shape, dtype="float32")
            train_mask = np.zeros(shape, dtype="float32"); station_holdout_mask = np.zeros(shape, dtype="float32"); region_holdout_mask = np.zeros(shape, dtype="float32")
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
            current = np.stack((initial, mask0, age_grid, distance, effective)).astype("float32")
            hrrr_forecast = np.stack((temp, rh, wind, precip), axis=1).astype("float32")
            physics = evolve_fm(initial, temp, rh)[:, None].astype("float32")
            metadata = {"tensor_schema_version": 2, "run_id": run_id, "forecast_init_time": init.isoformat(),
                "antecedent_times": [value.isoformat() for value in antecedent_times],
                "realized_times": [value.isoformat() for value in realized_times], "valid_times": [value.isoformat() for value in valid],
                "hrrr_leads": list(TARGET_LEADS), "antecedent_channels": ["temp_c", "rh", "wind_speed", "availability"],
                "realized_channels": ["temp_c", "rh", "wind_speed", "availability"],
                "hrrr_channels": ["temp_c", "rh", "wind_speed", "precip"],
                "current_fm_channels": ["fm", "obs_mask", "age_hours", "nearest_station_distance", "effective_station_count"],
                "realized_rtma_future": {"training_only": True, "exported": False},
                "static_bundle": static_bundle.name, "static_bundle_sha256": static_manifest["sha256"],
                "grid_fingerprint": static_manifest["grid_fingerprint"], "station_count": len(station_fm),
                "labels_are_observations_only": True, "wind_contract": "earth_relative_speed"}
            np.savez_compressed(output, antecedent_rtma=antecedent, realized_rtma_future=realized,
                hrrr_forecast=hrrr_forecast, current_fm_state=current, physics_trajectory=physics,
                target=target, target_mask=target_mask, train_mask=train_mask,
                station_holdout_mask=station_holdout_mask, region_holdout_mask=region_holdout_mask,
                metadata=json.dumps(metadata))
            built += 1
        except Exception as exc:
            rejected.append({"run_id": run_id, "file": hrrr_path.name, "reason": str(exc)})
        if limit and built >= limit:
            break
    report = {"built": built, "rejected": rejected, "tensor_schema_version": 2}
    (paths.REPORTS_DIR / "spatial_tensor_build.json").write_text(json.dumps(report, indent=2))
    print(f"Built {built} teacher/student tensor file(s); rejected={len(rejected)}; output={output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--static-bundle", required=True, type=Path); parser.add_argument("--limit", type=int)
    args = parser.parse_args(); build(args.static_bundle, args.limit)
