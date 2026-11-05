"""Historical RRFS fetcher for the training data store, cropped to Missouri.

Mirrors spatial/hrrr_capture.py's fetch/validate/sanitize pattern, but does
NOT use Herbie's built-in `model="rrfs"` template: that template targets the
old prototype bucket (`noaa-rrfs-pds`), which stopped updating 2026-08-12
when NOAA began the RRFS/REFS pre-operational parallel phase (operational
cutover targeted 2026-10-06, per NWS Service Change Notice 26-48). The real,
currently-growing feed lives in a different bucket - `noaa-rrfs-ops-pds` -
under a `rrfs.<YYYYMMDD>/<HH>/rrfs.t<HH>z.2dfld.13km.f<FFF>.na.grib2` key
layout that Herbie has no knowledge of at all. Confirmed live via direct S3
listing (2026-09-14): full North-America domain (so it includes Missouri -
the bucket's other candidate products don't: `firewx` covers a fixed
western-US regional nest only, and `refs` ensemble products carry only
precipitation-probability fields, no temp/RH/wind), history starting
2026-08-12, growing daily.

CORRECTION (2026-09-15, found via real 404s on non-synoptic hours): this
"13km.na" product is NOT published every hour - only at the 4 synoptic
cycles, 00/06/12/18 UTC. Every other hour instead publishes a completely
different product (`2dfld.2p5km.subh` - sub-hourly, different variable/
level layout, "hi"/"pr" file splits, unverified domain coverage) that this
module does not know how to read. NA_PRODUCT_CYCLE_HOURS enforces this -
fetch_rrfs raises immediately (a ValueError, classified "deterministic" by
classify_fetch_failure - retrying won't help, this hour will never publish
this product) for any other hour, rather than making a network round-trip
against a URL that deterministically 404s.

Fetches the needed GRIB messages directly via HTTP Range requests against
each cycle's `.idx` sidecar (the same byte-subsetting technique Herbie uses
internally) rather than downloading the full ~150MB per-lead file.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import xarray as xr

import paths
from spatial.domain import MO_BUFFERED_BBOX, crop
from spatial.rtma_capture import relative_humidity

RRFS_OPS_BASE = "https://noaa-rrfs-ops-pds.s3.amazonaws.com"
RRFS_OPS_ARCHIVE_START = datetime(2026, 8, 12)  # bucket has nothing before this date
# The "13km.na" product this module reads only exists at these 4 synoptic
# cycles - confirmed live (2026-09-15): every other hour 404s because a
# different, unsupported product (2dfld.2p5km.subh) is published instead.
NA_PRODUCT_CYCLE_HOURS = (0, 6, 12, 18)

REQUIRED_VARIABLES = {"t2m", "r2", "u10", "v10", "tp"}
# gust: same best-effort, never-assumed-present treatment as hrrr_capture.py's
# OPTIONAL_VARIABLES - this is a brand-new product, not guaranteed stable yet.
OPTIONAL_VARIABLES = {"gust"}
DEFAULT_LEAD_HOURS = range(4, 16)  # matches hrrr_capture.py's afternoon-window convention

# (GRIB shortName-equivalent search key, level) pairs, grouped so each HTTP
# byte-range fetch only pulls messages that share the same vertical level -
# merging TMP/RH/DPT (2m) with UGRD/VGRD (10m) in one cfgrib open raises
# cfgrib.dataset.DatasetBuildError ("key present and new value is different:
# key='heightAboveGround'") because cfgrib can't build one hypercube across
# two different heightAboveGround values. Verified live against a real cycle.
_THERMAL_MESSAGES = {("TMP", "2 m above ground"), ("DPT", "2 m above ground"), ("RH", "2 m above ground")}
_WIND_MESSAGES = {("UGRD", "10 m above ground"), ("VGRD", "10 m above ground")}
_GUST_MESSAGES = {("GUST", "surface")}
_PRECIP_MESSAGES = {("APCP", "surface")}


def _key(run_dt: datetime, fxx: int) -> str:
    return f"rrfs.{run_dt:%Y%m%d}/{run_dt:%H}/rrfs.t{run_dt:%H}z.2dfld.13km.f{fxx:03d}.na.grib2"


def _parse_idx(idx_text: str) -> list[tuple[int, int, str, str]]:
    entries = []
    for line in idx_text.strip().splitlines():
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 5:
            continue
        entries.append((int(parts[0]), int(parts[1]), parts[3], parts[4]))
    return entries


def _byte_ranges(entries: list[tuple[int, int, str, str]]) -> dict[int, tuple[int, int | None]]:
    """Maps each message's start offset to (start, end) - end is None for the last message (fetch to EOF)."""
    ordered = sorted(entries, key=lambda e: e[1])
    ranges = {}
    for i, (_, offset, _, _) in enumerate(ordered):
        end = ordered[i + 1][1] - 1 if i + 1 < len(ordered) else None
        ranges[offset] = (offset, end)
    return ranges


def _fetch_group(session: requests.Session, run_dt: datetime, fxx: int,
                  entries: list[tuple[int, int, str, str]], ranges: dict, wanted: set[tuple[str, str]]) -> bytes | None:
    matches = [e for e in entries if (e[2], e[3]) in wanted]
    if not matches:
        return None
    grib_url = f"{RRFS_OPS_BASE}/{_key(run_dt, fxx)}"
    out = bytearray()
    for _, offset, _, _ in matches:
        start, end = ranges[offset]
        range_header = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
        response = session.get(grib_url, headers={"Range": range_header}, timeout=60)
        response.raise_for_status()
        out += response.content
    return bytes(out)


def _open_group(data: bytes, tmp_dir: Path, name: str) -> xr.Dataset:
    """Writes to a genuinely unique temp filename, not a fixed one -
    concurrent worker processes (see scripts/backfill_rrfs.py's
    ProcessPoolExecutor) fetching different leads/cycles at the same time
    otherwise collide on the same path and, on Windows, fail with
    WinError 32 ("used by another process") instead of just overwriting
    (confirmed live under --workers > 1)."""
    fd, temp_name = tempfile.mkstemp(prefix=f"rrfs_{name}.", suffix=".grib2", dir=tmp_dir)
    path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}).load()
    finally:
        path.unlink(missing_ok=True)
    ds = ds.copy(deep=False)
    for var in ds.variables:
        ds[var].attrs.pop("dtype", None)
        ds[var].encoding.pop("dtype", None)
    return ds


def classify_fetch_failure(exc: Exception) -> str:
    """"deterministic" (bad request shape, retrying won't help), "unavailable"
    (this run genuinely isn't published - before RRFS_OPS_ARCHIVE_START, or
    too recent for the pipeline to have posted it yet), or "transient"."""
    message = str(exc).lower()
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            return "unavailable"
        if status is not None and 400 <= status < 500:
            return "deterministic"
        return "transient"
    if isinstance(exc, (KeyError, ValueError)) or "decode variable" in message:
        return "deterministic"
    return "transient"


def _fetch_lead(session: requests.Session, run_dt: datetime, fxx: int, tmp_dir: Path) -> xr.Dataset:
    idx_url = f"{RRFS_OPS_BASE}/{_key(run_dt, fxx)}.idx"
    response = session.get(idx_url, timeout=30)
    response.raise_for_status()
    entries = _parse_idx(response.text)
    if not entries:
        raise ValueError(f"empty or unparsable RRFS index: {idx_url}")
    ranges = _byte_ranges(entries)

    groups = []
    therm_bytes = _fetch_group(session, run_dt, fxx, entries, ranges, _THERMAL_MESSAGES)
    if therm_bytes is None:
        raise KeyError(f"RRFS thermal messages (TMP/DPT/RH 2m) not found in index: {idx_url}")
    therm = _open_group(therm_bytes, tmp_dir, "therm")
    if "r2" not in therm:
        if "t2m" not in therm or "d2m" not in therm:
            raise KeyError(f"RRFS 2m thermal fields cannot derive RH: {list(therm.data_vars)}")
        therm["r2"] = relative_humidity(therm.t2m, therm.d2m)
        therm.r2.attrs.update({"long_name": "relative humidity", "units": "%", "derived_from": "t2m,d2m Magnus formula"})
    groups.append(therm)

    wind_bytes = _fetch_group(session, run_dt, fxx, entries, ranges, _WIND_MESSAGES)
    if wind_bytes is None:
        raise KeyError(f"RRFS wind messages (UGRD/VGRD 10m) not found in index: {idx_url}")
    groups.append(_open_group(wind_bytes, tmp_dir, "wind"))

    precip_bytes = _fetch_group(session, run_dt, fxx, entries, ranges, _PRECIP_MESSAGES)
    if precip_bytes is not None:
        groups.append(_open_group(precip_bytes, tmp_dir, "precip"))
    # else: APCP genuinely absent at this lead (e.g. f000 analysis has no
    # accumulation window yet) - validate_rrfs will catch a run that never
    # gets a precip message across any of its fetched leads.

    gust_bytes = _fetch_group(session, run_dt, fxx, entries, ranges, _GUST_MESSAGES)
    if gust_bytes is not None:
        try:
            groups.append(_open_group(gust_bytes, tmp_dir, "gust"))
        except Exception:
            pass  # best-effort, same as hrrr_capture.py's _fetch_gust

    return xr.merge(groups, compat="override")


def validate_rrfs(path: Path) -> tuple[bool, str]:
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


def fetch_rrfs(run_dt: datetime, cache_dir: Path | None = None, lead_hours=DEFAULT_LEAD_HOURS) -> Path:
    if run_dt.tzinfo is not None:
        run_dt = run_dt.astimezone(timezone.utc).replace(tzinfo=None)
    run_dt = run_dt.replace(minute=0, second=0, microsecond=0)
    if run_dt < RRFS_OPS_ARCHIVE_START:
        raise ValueError(f"requested RRFS run {run_dt} predates the operational archive start {RRFS_OPS_ARCHIVE_START}")
    if run_dt.hour not in NA_PRODUCT_CYCLE_HOURS:
        raise ValueError(
            f"RRFS 'na' product is only published at {NA_PRODUCT_CYCLE_HOURS} UTC - "
            f"{run_dt} ({run_dt.hour:02d}z) does not have it (a different, unsupported product is published instead)"
        )

    cache_dir = Path(cache_dir or paths.CACHE_RRFS_DIR); cache_dir.mkdir(parents=True, exist_ok=True)
    lead_hours = list(lead_hours)
    target = cache_dir / f"rrfs_{run_dt:%Y%m%d_%H}z_f{lead_hours[0]:02d}-{lead_hours[-1]:02d}.nc"
    if target.exists():
        valid, _ = validate_rrfs(target)
        if valid:
            return target
        target.unlink()

    with requests.Session() as session:
        steps = [_fetch_lead(session, run_dt, fxx, cache_dir) for fxx in lead_hours]
    ds = xr.concat(steps, dim="step")
    ds = ds.copy(deep=False)
    ds = crop(ds).load()
    ds.attrs.update({"model": "rrfs", "product": "2dfld.13km.na",
                      "requested_init_time_utc": run_dt.isoformat() + "Z", "domain_bbox": str(MO_BUFFERED_BBOX)})
    temporary = target.with_suffix(".nc.tmp")
    try:
        ds.to_netcdf(temporary, engine="netcdf4")
        valid, reason = validate_rrfs(temporary)
        if not valid:
            raise RuntimeError(f"fetched RRFS failed validation: {reason}")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
