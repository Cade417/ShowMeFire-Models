"""
Grid-cell (pixel) level fire_weather_index scoring - the same ramp/weighted-
average model as factors.py, applied directly to a weather grid instead of
already-aggregated county-day rows. Produces a smooth, per-cell score/
category surface (matching the production Peak Fire Danger Forecast map's
style - continuous blobs that cut across county lines, not discrete county
polygons) rather than a county choropleth.

Only uses rh/wind/vpd/precip_relief, same as
api/services/fire_weather_index_shadow.py's live-scoring path - KBDI and
the seasonal cure/green-up factor need multi-day accumulated state a single
cached weather file doesn't carry, so they're treated as unavailable here
too (renormalized away by compute_score_grid, not a special case).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

from fire_weather_index import factors
from risk_fusion import features as rff
from spatial.precipitation import decode_forecast_precipitation


def _ramp_grid(values: np.ndarray, benign: float, extreme: float) -> np.ndarray:
    """Same linear ramp as factors._ramp, vectorized - NaN propagates
    naturally through arithmetic and np.clip, no special-casing needed."""
    fraction = (np.asarray(values, dtype="float64") - benign) / (extreme - benign)
    return np.clip(fraction, 0.0, 1.0)


def compute_factor_grids(path: Path) -> Dict[str, np.ndarray]:
    """Reads one cached HRRR/RRFS/FV3-HIRES county-day source file and
    returns per-cell rh_min_afternoon/wind_kts_max/vpd_kpa_max/
    precip_24h_mm grids - same reduction windows and formulas as
    fire_weather_index/build_county_days.py::process_one_run, just not
    collapsed to county aggregates."""
    raw = xr.open_dataset(path, decode_cf=False)
    for name in raw.variables:
        raw[name].attrs.pop("dtype", None)
    with xr.decode_cf(raw, decode_timedelta=True) as ds:
        leads = np.asarray(ds.step.values)
        if np.issubdtype(leads.dtype, np.timedelta64):
            leads = (leads / np.timedelta64(1, "h")).astype(int)
        afternoon_mask = np.isin(leads, list(rff.AFTERNOON_LEAD_HOURS))
        full_mask = np.isin(leads, list(rff.FULL_LEAD_HOURS))
        if not full_mask.any():
            raise ValueError(f"{path.name}: no leads in the day-1 aggregation window")

        temp_c = np.asarray(ds["t2m"].values) - 273.15
        rh = np.asarray(ds["r2"].values)
        wind_ms = np.hypot(np.asarray(ds["u10"].values), np.asarray(ds["v10"].values))
        wind_kts = wind_ms * rff.MPS_TO_KNOTS * rff.FIRE_WIND_REDUCTION_FACTOR
        vpd = rff.vapor_pressure_deficit_kpa(temp_c, rh)
        precipitation = decode_forecast_precipitation(ds)
        interval_mm = np.asarray(precipitation.interval_mm.values)
        lat = np.asarray(ds["latitude"].values)
        lon = np.asarray(ds["longitude"].values)

    afternoon_idx = np.flatnonzero(afternoon_mask) if afternoon_mask.any() else np.flatnonzero(full_mask)
    full_idx = np.flatnonzero(full_mask)

    return {
        "rh_min_afternoon": np.nanmin(rh[afternoon_idx], axis=0),
        "wind_kts_max": np.nanmax(wind_kts[full_idx], axis=0),
        "vpd_kpa_max": np.nanmax(vpd[full_idx], axis=0),
        "precip_24h_mm": np.nansum(interval_mm[full_idx], axis=0),
        "lat": lat,
        "lon": lon,
    }


def _to_lon180(lon: np.ndarray) -> np.ndarray:
    """Normalizes to the -180..180 convention - HRRR/FV3-HIRES store 0..360,
    RRFS already stores -180..180. griddata doesn't care about the
    convention, but every source has to agree on ONE before interpolating,
    or points that are actually close together (e.g. -96 and 264) look
    360 degrees apart."""
    lon = np.asarray(lon, dtype="float64")
    return np.where(lon > 180, lon - 360, lon)


def regrid_to_grid(source_lat: np.ndarray, source_lon: np.ndarray, source_value: np.ndarray,
                    target_lat: np.ndarray, target_lon: np.ndarray, method: str = "linear") -> np.ndarray:
    """Interpolates one source field (any curvilinear grid) onto a
    DIFFERENT target grid (any curvilinear grid) via scattered-point
    interpolation - HRRR/RRFS/FV3-HIRES share no common grid (confirmed
    live: different shapes, different native lon conventions, different
    domains), so `xarray.interp` (built for shared/regular axes) doesn't
    apply; `scipy.interpolate.griddata` treats every source cell as an
    independent (lon, lat, value) point and triangulates.

    Returns NaN wherever the target point falls outside the source's
    convex hull - griddata's own default behavior for methods other than
    "nearest", kept deliberately rather than worked around: a target pixel
    the source grid never actually covered has no real value to report,
    same "never assume present" principle as gust_available/rrfs_available
    elsewhere in this project."""
    from scipy.interpolate import griddata

    source_lon = _to_lon180(source_lon)
    target_lon = _to_lon180(target_lon)
    source_value = np.asarray(source_value, dtype="float64")

    valid = np.isfinite(source_value) & np.isfinite(source_lat) & np.isfinite(source_lon)
    if not valid.any():
        return np.full(np.asarray(target_lat).shape, np.nan)

    points = np.column_stack([source_lon[valid], source_lat[valid]])
    values = source_value[valid]
    target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
    interpolated = griddata(points, values, target_points, method=method)
    return interpolated.reshape(np.asarray(target_lat).shape)


def compute_antecedent_precip_grid(hrrr_path: Path, lookback_days: int = 3) -> Dict[str, object]:
    """Real rain from the preceding `lookback_days` HRRR runs (by calendar
    day, strictly before hrrr_path's own day) - same amortization principle
    as build_county_days.py::add_antecedent_precip (each prior day's own
    forecast-window precip_24h_mm stands in for that day's real rain, once
    the day has passed), just reading each prior day's cached HRRR file
    directly instead of an already-built panel column.

    Returns antecedent_precip_mm (same shape as hrrr_path's own grid, 0.0 -
    never NaN - where no prior day contributed, since "no antecedent data
    found" honestly means "0 additional rain known", not "unknown") and
    antecedent_precip_days (how many of the lookback days actually had a
    cached run, 0-lookback_days - never silently assumed full history)."""
    import re
    from datetime import datetime as _dt, timedelta as _td

    target_grids = compute_factor_grids(hrrr_path)
    target_shape = target_grids["precip_24h_mm"].shape

    match = re.search(r"hrrr_(\d{8})_(\d{2})z_f\d{2}-\d{2}\.nc$", hrrr_path.name)
    if not match:
        return {"antecedent_precip_mm": np.zeros(target_shape), "antecedent_precip_days": 0}
    current_run = _dt.strptime(match.group(1) + match.group(2), "%Y%m%d%H")

    total = np.zeros(target_shape)
    days_found = 0
    for offset in range(1, lookback_days + 1):
        prior_day = (current_run - _td(days=offset)).strftime("%Y%m%d")
        candidates = sorted(hrrr_path.parent.glob(f"hrrr_{prior_day}_*z_f*.nc"))
        if not candidates:
            continue
        try:
            prior_grids = compute_factor_grids(candidates[-1])
        except Exception:
            continue
        prior_precip = prior_grids["precip_24h_mm"]
        if prior_precip.shape == target_shape:
            total += np.nan_to_num(prior_precip)
        else:
            regridded = regrid_to_grid(prior_grids["lat"], prior_grids["lon"], prior_precip,
                                        target_grids["lat"], target_grids["lon"])
            total += np.nan_to_num(regridded)
        days_found += 1
    return {"antecedent_precip_mm": total, "antecedent_precip_days": days_found}


def compute_fuel_threshold_grid(target_lat: np.ndarray, target_lon: np.ndarray) -> Optional[np.ndarray]:
    """Per-pixel precip-relief-full threshold (mm), from the same static
    fire-behavior bundle scripts/build_county_fuel_type.py aggregates to
    counties - regridded directly onto the target (HRRR) grid instead,
    since a grid-level score can use the raw per-cell fuel classification
    rather than a county-level dominant vote. Nearest-neighbor regridding
    (not linear - `fuel_model_fbfm40` is a categorical code, interpolating
    between two different fuel codes numerically is meaningless). Returns
    None if the static bundle isn't available in this environment - the
    caller falls back to the flat FACTOR default, not an error."""
    import sys as _sys

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in _sys.path:
        _sys.path.insert(0, str(scripts_dir))
    try:
        from build_county_fuel_type import _find_latest_bundle, classify_fuel_group
    except Exception:
        return None

    try:
        bundle_path = _find_latest_bundle(None)
    except SystemExit:
        return None

    with xr.open_dataset(bundle_path) as ds:
        fuel_codes = np.asarray(ds["fuel_model_fbfm40"].values, dtype="float64")
        valid = np.asarray(ds["static_valid_mask"].values) == 1
        source_lat = np.asarray(ds["latitude"].values)
        source_lon = np.asarray(ds["longitude"].values)
    fuel_codes = np.where(valid, fuel_codes, np.nan)

    regridded_codes = regrid_to_grid(source_lat, source_lon, fuel_codes, target_lat, target_lon, method="nearest")
    vectorized_classify = np.vectorize(classify_fuel_group, otypes=[object])
    fuel_groups = vectorized_classify(regridded_codes)
    threshold_lookup = factors.PRECIP_RELIEF_FULL_MM_BY_FUEL_GROUP
    default_threshold = factors.PRECIP_RELIEF_FULL_MM
    return np.vectorize(lambda group: threshold_lookup.get(group, default_threshold))(fuel_groups)


FACTOR_GRID_KEYS = ("rh_min_afternoon", "wind_kts_max", "vpd_kpa_max", "precip_24h_mm")


def compute_blended_factor_grids(hrrr_path: Path, rrfs_path: Optional[Path] = None,
                                  fv3hires_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """HRRR's own factor grids (backbone, own native resolution, never
    resampled itself) blended with RRFS/FV3-HIRES factor grids REGRIDDED
    onto HRRR's own (lat, lon) - same "HRRR is never overridden, only
    averaged with the others when they're also present" philosophy as
    fire_weather_index/build_county_days.py::blend_sources(), just at
    pixel resolution instead of county-day resolution. rrfs_path/
    fv3hires_path are optional - a caller with no matching cached file for
    the date passes None and gets an HRRR-only result, not an error.

    Also returns rrfs_coverage_fraction/fv3hires_coverage_fraction - the
    fraction of HRRR's own domain that actually got a finite regridded
    value from that source - for callers to report real coverage rather
    than silently claiming a blend that mostly didn't land."""
    hrrr_grids = compute_factor_grids(hrrr_path)
    target_lat, target_lon = hrrr_grids["lat"], hrrr_grids["lon"]

    blended = {key: hrrr_grids[key].copy() for key in FACTOR_GRID_KEYS}
    counts = {key: np.isfinite(hrrr_grids[key]).astype("float64") for key in FACTOR_GRID_KEYS}
    coverage = {"rrfs_coverage_fraction": 0.0, "fv3hires_coverage_fraction": 0.0}

    for label, path in (("rrfs", rrfs_path), ("fv3hires", fv3hires_path)):
        if path is None:
            continue
        source_grids = compute_factor_grids(path)
        source_lat, source_lon = source_grids["lat"], source_grids["lon"]
        any_valid = None
        for key in FACTOR_GRID_KEYS:
            regridded = regrid_to_grid(source_lat, source_lon, source_grids[key], target_lat, target_lon)
            valid = np.isfinite(regridded)
            any_valid = valid if any_valid is None else (any_valid | valid)
            blended[key] = np.where(valid, np.nan_to_num(blended[key]) + np.nan_to_num(regridded), blended[key])
            counts[key] = counts[key] + valid.astype("float64")
        coverage[f"{label}_coverage_fraction"] = float(np.mean(any_valid)) if any_valid is not None else 0.0

    for key in FACTOR_GRID_KEYS:
        with np.errstate(invalid="ignore", divide="ignore"):
            blended[key] = np.where(counts[key] > 0, blended[key] / counts[key], np.nan)

    return {**blended, "lat": target_lat, "lon": target_lon, **coverage}


def compute_score_grid(grids: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None,
                        precip_relief_threshold: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted average of available factor grids, renormalized per-cell
    over whichever factors are finite there - same logic as
    factors.compute_score, vectorized over the whole grid at once.

    precip_relief_threshold: scalar or per-pixel array of "24h precip (mm)
    for full relief" - pass compute_fuel_threshold_grid()'s output for
    fuel-type-aware relief (fine grass fuels need much less rain than
    heavy timber), or omit for the flat factors.PRECIP_RELIEF_FULL_MM
    default (same as before fuel-type awareness existed)."""
    threshold = factors.PRECIP_RELIEF_FULL_MM if precip_relief_threshold is None else precip_relief_threshold
    weights = weights or {k: v for k, v in factors.FACTOR_WEIGHTS.items() if k in ("rh", "wind", "vpd", "precip_relief")}
    # _ramp_grid divides by (extreme - benign); a per-pixel threshold array
    # needs the same division done element-wise, not via the scalar-benign
    # signature - inline rather than reusing _ramp_grid when thresholds vary.
    precip_fraction = np.asarray(grids["precip_24h_mm"], dtype="float64") / np.asarray(threshold, dtype="float64")
    factor_grids = {
        "rh": _ramp_grid(grids["rh_min_afternoon"], factors.RH_BENIGN, factors.RH_EXTREME),
        "wind": _ramp_grid(grids["wind_kts_max"], factors.WIND_BENIGN, factors.WIND_EXTREME),
        "vpd": _ramp_grid(grids["vpd_kpa_max"], factors.VPD_BENIGN_KPA, factors.VPD_EXTREME_KPA),
        "precip_relief": np.clip(precip_fraction, 0.0, 1.0),
    }
    numerator = np.zeros_like(next(iter(factor_grids.values())))
    denominator = np.zeros_like(numerator)
    for name, weight in weights.items():
        value = factor_grids[name]
        valid = np.isfinite(value)
        signed_weight = -weight if name == "precip_relief" else weight
        numerator = np.where(valid, numerator + signed_weight * np.nan_to_num(value), numerator)
        denominator = np.where(valid, denominator + weight, denominator)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(denominator > 0, numerator / denominator, np.nan)
    # Same absolute-ceiling rescale as factors.compute_score - see
    # factors.RAW_SCORE_CEILING's docstring for what it's anchored to.
    score = score / factors.RAW_SCORE_CEILING
    return np.clip(score, 0.0, 1.0)


def smooth_score_grid(score: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Light Gaussian smoothing so the rendered map reads as the same kind
    of continuous surface the production Peak Fire Danger map shows,
    rather than raw HRRR's native per-pixel noise. NaN-safe: fills gaps
    with the domain mean before smoothing (a plain gaussian_filter would
    otherwise bleed NaN outward from every masked/off-grid cell), then
    restores the original NaN mask afterward - smoothing never invents
    real data where there wasn't any."""
    from scipy.ndimage import gaussian_filter

    valid = np.isfinite(score)
    if not valid.any():
        return score
    filled = np.where(valid, score, np.nanmean(score))
    smoothed = gaussian_filter(filled, sigma=sigma)
    return np.where(valid, smoothed, np.nan)


def mask_outside_polygon(lat: np.ndarray, lon: np.ndarray, geometry) -> np.ndarray:
    """True where a grid cell's center falls inside `geometry` (a shapely
    polygon/multipolygon in the same lon/lat convention as `lon`/`lat`,
    e.g. Missouri's state boundary) - vectorized via shapely.vectorized,
    same technique scripts/build_county_cells.py uses per-cell in a loop,
    just without the loop."""
    import shapely.vectorized

    return shapely.vectorized.contains(geometry, lon, lat)


def score_to_category_grid(score: np.ndarray, thresholds: list) -> np.ndarray:
    """Same calibrated-cutpoint lookup as model_bundle.score_to_category,
    vectorized via np.digitize. NaN scores map to -1 (no data), never a
    fabricated category."""
    category = np.digitize(np.nan_to_num(score, nan=-np.inf), thresholds, right=False)
    category = np.where(np.isnan(score), -1, np.clip(category, 0, 4))
    return category.astype(int)
