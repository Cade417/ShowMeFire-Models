"""Historical HRRR fetcher for the training data store, cropped to Missouri.

Mirrors spatial/rtma_capture.py's fetch/validate/sanitize pattern. Unlike a
single-analysis RTMA fetch, one HRRR "run" is a set of separate GRIB files -
one per forecast lead hour - concatenated along a step dimension.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from herbie import Herbie

import paths
from spatial.domain import MO_BUFFERED_BBOX, crop
from spatial.precipitation import decode_forecast_precipitation
from spatial.rtma_capture import relative_humidity

REQUIRED_VARIABLES = {"t2m", "r2", "u10", "v10", "tp"}
# gust is deliberately NOT required: it's a forward-only addition (the
# historical archive predates it), and HRRR occasionally omits the GUST
# field for a given cycle/lead. Fetches best-effort in _fetch_lead; its
# absence on a given row must be checked via risk_fusion_features's
# gust_available flag, never assumed present.
OPTIONAL_VARIABLES = {"gust"}
DEFAULT_LEAD_HOURS = range(4, 16)
PRECIP_CONTEXT_LEAD_HOURS = range(0, 4)

# Both are official NOAA Big Data Program mirrors with equivalent HRRR
# coverage - this is a reliability preference, not a data-completeness
# one. Measured live from this deployment environment: 15/15 successful
# requests against Google Cloud Storage vs. 1/15 against the AWS S3
# mirror (noaa-hrrr-bdp-pds), for the identical object. Herbie falls back
# through the full priority list on its own if the first source 404s for
# a specific object, so listing "aws" second costs nothing when google
# has the file, which is effectively always.
HERBIE_SOURCE_PRIORITY = ["google", "aws"]


def clear_local_cache(run_dt: datetime) -> None:
    """
    Herbie's own local download cache (~/data/{model}/{YYYYMMDD}/, distinct
    from paths.CACHE_HRRR_DIR - that's OUR output cache) can be left in a
    poisoned state after a failed subset download: the .idx file gets
    cached as present, but the actual .grib2 subset never gets written
    (observed live - a transient DNS resolution hiccup mid-download leaves
    exactly this state). Herbie then treats the .idx as proof the file is
    already fetched on every subsequent retry, so it fails deterministically
    forever afterward instead of retrying the actual download. Clearing
    this directory before each attempt guarantees a clean download path
    regardless of what a prior attempt left behind.
    """
    cache_dir = Path.home() / "data" / "hrrr" / f"{run_dt:%Y%m%d}"
    shutil.rmtree(cache_dir, ignore_errors=True)


def classify_fetch_failure(exc: Exception) -> str:
    """
    "deterministic" (retrying won't help, don't retry), "unavailable" (the
    requested run genuinely doesn't exist - e.g. a date before HRRR
    existed, or too recent for a mirror to have published it yet - step
    back to a different run instead of retrying), or "transient" (worth
    retrying, ideally after clear_local_cache()).
    """
    message = str(exc).lower()
    if isinstance(exc, (KeyError, ValueError)) or "decode variable" in message or "serialization" in message:
        return "deterministic"
    if any(token in message for token in ("404", "not found", "no grib", "unavailable")):
        return "unavailable"
    if isinstance(exc, FileNotFoundError):
        # Observed live: a transient network failure mid-download can leave
        # Herbie's local cache poisoned (.idx cached, .grib2 subset never
        # written), and every subsequent attempt against that same stale
        # cache raises a bare FileNotFoundError ("[Errno 2] No such file or
        # directory: ...", no "404"/"not found" wording) - NOT a signal
        # that the HRRR data itself is unavailable (a genuinely missing
        # date raises ValueError instead, verified against 2011-06-01,
        # before HRRR existed - that case is still caught by the token
        # check above whenever the message says so explicitly). Retryable:
        # clear_local_cache() before every attempt gives a retry a clean
        # download instead of repeating the same poisoned-cache failure.
        return "transient"
    return "transient"


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
            raise RuntimeError("Herbie returned no HRRR datasets")
        value = xr.merge([sanitize_dataset(item) for item in value], compat="override")
    return sanitize_dataset(value)


def validate_hrrr(path: Path) -> tuple[bool, str]:
    try:
        raw = xr.open_dataset(path, decode_cf=False)
        for name in raw.variables:
            raw[name].attrs.pop("dtype", None)
        with xr.decode_cf(raw, decode_timedelta=True) as ds:
            missing = REQUIRED_VARIABLES - set(ds.data_vars)
            if missing:
                return False, f"missing variables: {sorted(missing)}"
            if "step" not in ds.dims:
                return False, "missing step dimension"
            if not all(np.isfinite(ds[name].values).any() for name in REQUIRED_VARIABLES):
                return False, "one or more required variables contain no finite values"
            precipitation = decode_forecast_precipitation(ds)
            if ds.sizes["step"] >= 16:
                leads = np.asarray(ds.step.values)
                if np.issubdtype(leads.dtype, np.timedelta64):
                    leads = leads / np.timedelta64(1, "h")
                if float(np.nanmin(leads)) > 0 or float(np.nanmax(leads)) < 15:
                    return False, "missing f00-f15 precipitation context"
            if np.any(np.asarray(precipitation.interval_mm.values) < 0):
                return False, "negative decoded precipitation interval"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _fetch_gust(herbie: Herbie) -> xr.Dataset | None:
    """Best-effort GUST fetch. Returns None on any failure - never blocks the run."""
    try:
        return as_dataset(herbie.xarray(":GUST:surface:"))
    except Exception:
        return None


def _fetch_lead(run_dt: datetime, fxx: int) -> xr.Dataset:
    herbie = Herbie(run_dt, fxx=fxx, model="hrrr", product="sfc", priority=HERBIE_SOURCE_PRIORITY)
    try:
        therm = as_dataset(herbie.xarray(":(?:TMP|RH|DPT):2 m above ground:"))
    except Exception:
        therm = as_dataset(herbie.xarray(":(?:TMP|RH|DPT):2 m:"))
    if "r2" not in therm:
        # Early HRRR archives (verified live: 2014-09) do not carry a
        # native 2m RH field, only TMP and DPT - same gap rtma_capture.py
        # already handles for RTMA, so reuse its exact Magnus-formula
        # derivation rather than a second implementation of it.
        if "t2m" not in therm or "d2m" not in therm:
            raise KeyError(f"HRRR 2m thermal fields cannot derive RH: {list(therm.data_vars)}")
        therm["r2"] = relative_humidity(therm.t2m, therm.d2m)
        therm.r2.attrs.update({"long_name": "relative humidity", "units": "%", "derived_from": "t2m,d2m Magnus formula"})
    try:
        wind = as_dataset(herbie.xarray(":(?:UGRD|VGRD):10 m above ground:"))
    except Exception:
        wind = as_dataset(herbie.xarray(":(?:UGRD|VGRD):10 m:"))
    precip = as_dataset(herbie.xarray(":APCP:surface:"))
    parts = [therm, wind, precip]
    gust = _fetch_gust(herbie)
    if gust is not None:
        parts.append(gust)
    return sanitize_dataset(xr.merge(parts, compat="override"))


def _fetch_precip_lead(run_dt: datetime, fxx: int) -> xr.Dataset:
    herbie = Herbie(run_dt, fxx=fxx, model="hrrr", product="sfc", priority=HERBIE_SOURCE_PRIORITY)
    return as_dataset(herbie.xarray(":APCP:surface:"))


def fetch_hrrr(run_dt: datetime, cache_dir: Path | None = None, lead_hours=DEFAULT_LEAD_HOURS) -> Path:
    if run_dt.tzinfo is not None:
        run_dt = run_dt.astimezone(timezone.utc).replace(tzinfo=None)
    run_dt = run_dt.replace(minute=0, second=0, microsecond=0)
    cache_dir = Path(cache_dir or paths.CACHE_HRRR_DIR); cache_dir.mkdir(parents=True, exist_ok=True)
    lead_hours = list(lead_hours)
    target = cache_dir / f"hrrr_{run_dt:%Y%m%d_%H}z_f{lead_hours[0]:02d}-{lead_hours[-1]:02d}.nc"
    if target.exists():
        valid, _ = validate_hrrr(target)
        if valid:
            return target
        target.unlink()

    steps = [_fetch_lead(run_dt, fxx) for fxx in lead_hours]
    ds = xr.concat(steps, dim="step")
    ds = sanitize_dataset(crop(ds).load())
    ds.attrs.update({"model": "hrrr", "product": "sfc",
                      "requested_init_time_utc": run_dt.isoformat() + "Z", "domain_bbox": str(MO_BUFFERED_BBOX)})
    temporary = target.with_suffix(".nc.tmp")
    try:
        ds.to_netcdf(temporary, engine="netcdf4")
        valid, reason = validate_hrrr(temporary)
        if not valid:
            raise RuntimeError(f"fetched HRRR failed validation: {reason}")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def validate_precip_context(path: Path) -> tuple[bool, str]:
    try:
        raw = xr.open_dataset(path, decode_cf=False)
        for name in raw.variables:
            raw[name].attrs.pop("dtype", None)
        with xr.decode_cf(raw, decode_timedelta=True) as ds:
            precipitation = decode_forecast_precipitation(ds)
            if "step" not in precipitation.cumulative_mm.dims or precipitation.cumulative_mm.sizes["step"] != 4:
                return False, "precipitation context must contain f00-f03"
            leads = np.asarray(ds.step.values)
            if np.issubdtype(leads.dtype, np.timedelta64):
                leads = leads / np.timedelta64(1, "h")
            if list(map(float, leads)) != [0.0, 1.0, 2.0, 3.0]:
                return False, f"unexpected precipitation context leads: {leads.tolist()}"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def fetch_precip_context(run_dt: datetime, cache_dir: Path | None = None,
                         lead_hours=PRECIP_CONTEXT_LEAD_HOURS) -> Path:
    if run_dt.tzinfo is not None:
        run_dt = run_dt.astimezone(timezone.utc).replace(tzinfo=None)
    run_dt = run_dt.replace(minute=0, second=0, microsecond=0)
    cache_dir = Path(cache_dir or paths.CACHE_HRRR_DIR); cache_dir.mkdir(parents=True, exist_ok=True)
    lead_hours = list(lead_hours)
    target = cache_dir / f"hrrr_precip_context_{run_dt:%Y%m%d_%H}z_f{lead_hours[0]:02d}-{lead_hours[-1]:02d}.nc"
    if target.exists():
        valid, _ = validate_precip_context(target)
        if valid: return target
        target.unlink()
    steps = [_fetch_precip_lead(run_dt, fxx) for fxx in lead_hours]
    ds = sanitize_dataset(crop(xr.concat(steps, dim="step")).load())
    ds.attrs.update({"model": "hrrr", "product": "sfc", "content": "precipitation_context",
                     "requested_init_time_utc": run_dt.isoformat() + "Z", "domain_bbox": str(MO_BUFFERED_BBOX)})
    temporary = target.with_suffix(".nc.tmp")
    try:
        ds.to_netcdf(temporary, engine="netcdf4")
        valid, reason = validate_precip_context(temporary)
        if not valid: raise RuntimeError(f"fetched precipitation context failed validation: {reason}")
        temporary.replace(target)
    finally:
        if temporary.exists(): temporary.unlink()
    return target
