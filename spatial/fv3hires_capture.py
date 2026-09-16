"""Forward-only FV3-HIRES (HiResW FV3) fetcher for the training data store, cropped to Missouri.

Mirrors spatial/hrrr_capture.py's fetch/validate/sanitize pattern, with two
structural differences forced by what this product actually is:

1. NOMADS-only, no backfill. Unlike HRRR/RRFS (both mirrored to long-term
   AWS/GCS archives), FV3-HIRES is served only from NOMADS's rolling
   production directory (confirmed live 2026-09-14 - Herbie's `hiresw`
   template has no AWS/GCS source at all). NOMADS keeps only a short window
   of recent cycles, so there is no historical range to backfill - capture
   is forward-only from whenever this pipeline starts running, and a run
   not captured before it rolls off NOMADS is lost permanently.
2. No native 2m instantaneous temperature. Confirmed via live inventory
   (2026-09-14): this product carries TMP only at 80m above ground, not 2m -
   no DPT anywhere either. It does carry TMAX/TMIN at 2m (hourly max/min),
   so `t2m` here is a (tmax+tmin)/2 PROXY, not the same quantity HRRR/RRFS
   report - never treat it as directly comparable at the county-day join
   step; downstream code must use fv3hires_t2m_is_proxy to know this.
   GUST is not present in this product at all (no gust field to fetch).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from herbie import Herbie

import paths
from spatial.domain import MO_BUFFERED_BBOX, crop

REQUIRED_VARIABLES = {"t2m", "r2", "u10", "v10", "tp"}
DEFAULT_LEAD_HOURS = range(4, 16)  # matches hrrr_capture.py's afternoon-window convention


def sanitize_dataset(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy(deep=False)
    for name in ds.variables:
        ds[name].attrs.pop("dtype", None)
        ds[name].attrs.pop("source", None)
        ds[name].encoding.pop("dtype", None)
    return ds


def as_dataset(value):
    if isinstance(value, list):
        if not value:
            raise RuntimeError("Herbie returned no HiResW FV3 datasets")
        value = xr.merge([sanitize_dataset(item) for item in value], compat="override")
    return sanitize_dataset(value)


def classify_fetch_failure(exc: Exception) -> str:
    """"unavailable" (this cycle has already rolled off NOMADS's short
    retention window, or hasn't posted yet - step to a different run rather
    than retrying) or "transient" (worth retrying)."""
    message = str(exc).lower()
    if isinstance(exc, (KeyError, ValueError)):
        return "deterministic"
    if any(token in message for token in ("404", "not found", "no grib", "unavailable")):
        return "unavailable"
    return "transient"


def _fetch_lead(run_dt: datetime, fxx: int) -> xr.Dataset:
    herbie = Herbie(run_dt, fxx=fxx, model="hiresw", product="fv3_2p5km", domain="conus")
    therm = as_dataset(herbie.xarray(":(?:TMAX|TMIN):2 m above ground:"))
    if "tmax" not in therm or "tmin" not in therm:
        raise KeyError(f"HiResW FV3 2m TMAX/TMIN not found: {list(therm.data_vars)}")
    therm["t2m"] = (therm.tmax + therm.tmin) / 2.0
    therm.t2m.attrs.update({"long_name": "2m temperature proxy", "units": "K",
                             "derived_from": "(TMAX+TMIN)/2 - this product has no instantaneous 2m TMP"})
    rh = as_dataset(herbie.xarray(":RH:2 m above ground:"))
    wind = as_dataset(herbie.xarray(":(?:UGRD|VGRD):10 m above ground:"))
    precip = as_dataset(herbie.xarray(":APCP:surface:"))
    return sanitize_dataset(xr.merge([therm, rh, wind, precip], compat="override"))


def validate_fv3hires(path: Path) -> tuple[bool, str]:
    try:
        with xr.open_dataset(path) as ds:
            missing = REQUIRED_VARIABLES - set(ds.data_vars)
            if missing:
                return False, f"missing variables: {sorted(missing)}"
            if "step" not in ds.dims:
                return False, "missing step dimension"
            if not all(np.isfinite(ds[name].values).any() for name in REQUIRED_VARIABLES):
                return False, "one or more required variables contain no finite values"
            if np.any(np.asarray(ds["tp"].values) < 0):
                return False, "negative precipitation value"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def fetch_fv3hires(run_dt: datetime, cache_dir: Path | None = None, lead_hours=DEFAULT_LEAD_HOURS) -> Path:
    if run_dt.tzinfo is not None:
        run_dt = run_dt.astimezone(timezone.utc).replace(tzinfo=None)
    run_dt = run_dt.replace(minute=0, second=0, microsecond=0)
    cache_dir = Path(cache_dir or paths.CACHE_FV3HIRES_DIR); cache_dir.mkdir(parents=True, exist_ok=True)
    lead_hours = list(lead_hours)
    target = cache_dir / f"fv3hires_{run_dt:%Y%m%d_%H}z_f{lead_hours[0]:02d}-{lead_hours[-1]:02d}.nc"
    if target.exists():
        valid, _ = validate_fv3hires(target)
        if valid:
            return target
        target.unlink()

    steps = [_fetch_lead(run_dt, fxx) for fxx in lead_hours]
    ds = xr.concat(steps, dim="step")
    ds = sanitize_dataset(crop(ds).load())
    ds.attrs.update({"model": "hiresw", "product": "fv3_2p5km", "t2m_is_tmax_tmin_proxy": "true",
                      "requested_init_time_utc": run_dt.isoformat() + "Z", "domain_bbox": str(MO_BUFFERED_BBOX)})
    temporary = target.with_suffix(".nc.tmp")
    try:
        ds.to_netcdf(temporary, engine="netcdf4")
        valid, reason = validate_fv3hires(temporary)
        if not valid:
            raise RuntimeError(f"fetched HiResW FV3 failed validation: {reason}")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
