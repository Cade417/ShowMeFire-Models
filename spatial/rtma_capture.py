"""Single-analysis RTMA fetcher for the training data store."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from herbie import Herbie

import paths

MO_BUFFERED_BBOX = (-96.8, -88.1, 34.8, 41.8)
REQUIRED_VARIABLES = {"t2m", "r2", "u10", "v10"}


def sanitize_dataset(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy(deep=False)
    if "step" in ds.dims and ds.sizes["step"] == 1:
        ds = ds.isel(step=0, drop=True)
    elif "step" in ds.coords and "step" not in ds.dims:
        ds = ds.drop_vars("step")
    for name in ds.variables:
        ds[name].attrs.pop("dtype", None)
        ds[name].attrs.pop("source", None)
        ds[name].encoding.pop("dtype", None)
    return ds


def as_dataset(value):
    if isinstance(value, list):
        if not value:
            raise RuntimeError("Herbie returned no RTMA datasets")
        value = xr.merge([sanitize_dataset(item) for item in value], compat="override")
    return sanitize_dataset(value)


def crop(ds: xr.Dataset) -> xr.Dataset:
    lon = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
    west, east, south, north = MO_BUFFERED_BBOX
    mask = (lon >= west) & (lon <= east) & (ds.latitude >= south) & (ds.latitude <= north)
    rows, cols = np.where(mask.values)
    if not len(rows):
        raise ValueError("RTMA grid does not intersect the configured domain")
    ydim, xdim = mask.dims
    return ds.isel({ydim: slice(rows.min(), rows.max() + 1), xdim: slice(cols.min(), cols.max() + 1)})


def relative_humidity(temp_k, dewpoint_k):
    temp_c, dewpoint_c = temp_k - 273.15, dewpoint_k - 273.15
    return (100 * np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c) - (17.625 * temp_c) / (243.04 + temp_c))).clip(0, 100)


def validate_rtma(path: Path) -> tuple[bool, str]:
    try:
        with xr.open_dataset(path) as ds:
            missing = REQUIRED_VARIABLES - set(ds.data_vars)
            if missing:
                return False, f"missing variables: {sorted(missing)}"
            if not all(np.isfinite(ds[name].values).any() for name in REQUIRED_VARIABLES):
                return False, "one or more required variables contain no finite values"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def fetch_rtma(run_dt: datetime, cache_dir: Path | None = None) -> Path:
    if run_dt.tzinfo is not None:
        run_dt = run_dt.astimezone(timezone.utc).replace(tzinfo=None)
    run_dt = run_dt.replace(minute=0, second=0, microsecond=0)
    cache_dir = Path(cache_dir or paths.CACHE_RTMA_DIR); cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"rtma_{run_dt:%Y%m%d_%H}z.nc"
    if target.exists():
        valid, _ = validate_rtma(target)
        if valid:
            return target
        target.unlink()

    herbie = Herbie(run_dt, fxx=0, model="rtma", product="anl")
    try:
        therm = as_dataset(herbie.xarray(":(?:TMP|DPT):2 m above ground:"))
    except Exception:
        therm = as_dataset(herbie.xarray(":(?:TMP|DPT):2 m"))
    try:
        wind = as_dataset(herbie.xarray(":(?:UGRD|VGRD):10 m above ground:"))
    except Exception:
        wind = as_dataset(herbie.xarray(":(?:UGRD|VGRD):10 m"))
    ds = sanitize_dataset(xr.merge([therm, wind], compat="override"))
    if "r2" not in ds:
        if "t2m" not in ds or "d2m" not in ds:
            raise KeyError(f"RTMA variables cannot derive RH: {list(ds.data_vars)}")
        ds["r2"] = relative_humidity(ds.t2m, ds.d2m)
        ds.r2.attrs.update({"long_name": "relative humidity", "units": "%", "derived_from": "t2m,d2m Magnus formula"})
    ds = sanitize_dataset(crop(ds).load())
    ds.attrs.update({"requested_analysis_time_utc": run_dt.isoformat() + "Z", "domain_bbox": str(MO_BUFFERED_BBOX)})
    temporary = target.with_suffix(".nc.tmp")
    try:
        ds.to_netcdf(temporary, engine="netcdf4")
        valid, reason = validate_rtma(temporary)
        if not valid:
            raise RuntimeError(f"RTMA cache verification failed: {reason}")
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target
