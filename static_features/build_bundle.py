from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyproj
import rasterio
import xarray as xr
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from static_features.schema import (CATEGORICAL_CHANNELS, CONTINUOUS_CHANNELS, GRID_SIZE, SCHEMA_VERSION,
                                    grid_fingerprint, sha256_file, validate_bundle)

BBOX = (-96.8, -88.1, 34.8, 41.8)
SOURCE_NAMES = {
    "elevation_m": "dem", "nlcd_class": "nlcd_class", "nlcd_confidence": "nlcd_confidence",
    "fuel_model_fbfm40": "fbfm40", "fuel_vegetation_type": "fvt", "canopy_cover_pct": "canopy_cover",
}


def _hrrr_crs(ds):
    for variable in ds.variables.values():
        if "grid_mapping_name" in variable.attrs:
            return pyproj.CRS.from_cf(variable.attrs)
    for name in ("gribfile_projection", "metpy_crs"):
        if name in ds and "grid_mapping_name" in ds[name].attrs:
            return pyproj.CRS.from_cf(ds[name].attrs)
    raise ValueError("HRRR file has no CF grid mapping")


def reference_grid(hrrr_path: Path):
    with xr.open_dataset(hrrr_path) as ds:
        lon = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
        west, east, south, north = BBOX
        mask = (lon >= west) & (lon <= east) & (ds.latitude >= south) & (ds.latitude <= north)
        rows, cols = np.where(mask.values)
        if not len(rows): raise ValueError("HRRR file does not intersect static domain")
        ydim, xdim = mask.dims
        x_values, y_values = np.asarray(ds[xdim].values), np.asarray(ds[ydim].values)
        xmin, xmax = float(x_values[cols.min()]), float(x_values[cols.max()])
        ymin, ymax = float(y_values[rows.min()]), float(y_values[rows.max()])
        if xmax < xmin: xmin, xmax = xmax, xmin
        if ymax < ymin: ymin, ymax = ymax, ymin
        crs = _hrrr_crs(ds)
    transform = from_bounds(xmin, ymin, xmax, ymax, GRID_SIZE, GRID_SIZE)
    x = transform.c + (np.arange(GRID_SIZE) + .5) * transform.a
    y = transform.f + (np.arange(GRID_SIZE) + .5) * transform.e
    xx, yy = np.meshgrid(x, y)
    lon, lat = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xx, yy)
    return {"x": x, "y": y, "latitude": lat, "longitude": lon, "crs": crs, "transform": transform}


def _warp(path, grid, categorical=False):
    destination = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=destination, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_transform=grid["transform"], dst_crs=grid["crs"], dst_nodata=np.nan,
                  resampling=Resampling.nearest if categorical else Resampling.bilinear)
    return destination


def _encode_categories(values):
    valid = np.isfinite(values) & (values > 0)
    raw_codes = sorted(int(value) for value in np.unique(values[valid]))
    mapping = {raw: index + 1 for index, raw in enumerate(raw_codes)}
    encoded = np.zeros(values.shape, dtype="int16")
    rounded = np.zeros(values.shape, dtype="int64"); rounded[valid] = np.rint(values[valid]).astype("int64")
    for raw, index in mapping.items(): encoded[valid & (rounded == raw)] = index
    return encoded, {"0": 0, **{str(raw): index for raw, index in mapping.items()}}


def build(hrrr_path: Path, version: str):
    source_manifest_path = paths.STATIC_SOURCE_DIR / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    products = source_manifest.get("products", {})
    missing = set(SOURCE_NAMES.values()) - set(products)
    if missing: raise ValueError(f"source products missing: {sorted(missing)}")
    grid = reference_grid(hrrr_path)
    raw = {}
    for channel, product in SOURCE_NAMES.items():
        raw[channel] = _warp(Path(products[product]["path"]), grid, categorical=channel in CATEGORICAL_CHANNELS)

    elevation = raw["elevation_m"]
    dx, dy = abs(grid["transform"].a), abs(grid["transform"].e)
    dzdy, dzdx = np.gradient(elevation, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = np.arctan2(-dzdx, dzdy)
    mean = uniform_filter(np.nan_to_num(elevation, nan=0.0), size=3)
    mean2 = uniform_filter(np.nan_to_num(elevation, nan=0.0) ** 2, size=3)
    ruggedness = np.sqrt(np.maximum(mean2 - mean ** 2, 0))
    valid = np.isfinite(elevation)
    continuous = {
        "elevation_m": elevation, "slope_degrees": slope, "aspect_sin": np.sin(aspect), "aspect_cos": np.cos(aspect),
        "terrain_ruggedness_m": ruggedness, "nlcd_confidence": raw["nlcd_confidence"],
        "canopy_cover_pct": raw["canopy_cover_pct"], "latitude": grid["latitude"], "longitude": grid["longitude"],
        "static_valid_mask": valid.astype("float32"),
    }
    categorical, mappings = {}, {}
    for name in CATEGORICAL_CHANNELS:
        categorical[name], mappings[name] = _encode_categories(raw[name])
    data_vars = {name: (("y", "x"), np.asarray(values, dtype="float32")) for name, values in continuous.items()}
    data_vars.update({name: (("y", "x"), values) for name, values in categorical.items()})
    crs_wkt = grid["crs"].to_wkt()
    fingerprint = grid_fingerprint(grid["x"], grid["y"], crs_wkt)
    ds = xr.Dataset(data_vars, coords={"x": grid["x"], "y": grid["y"]}, attrs={
        "schema_version": SCHEMA_VERSION, "bundle_version": version, "grid_fingerprint": fingerprint,
        "crs_wkt": crs_wkt, "transform": tuple(grid["transform"]), "bbox": BBOX,
    })
    bundle = paths.STATIC_BUNDLE_DIR / f"static_features_{version}.nc"
    if bundle.exists(): raise FileExistsError(f"immutable bundle already exists: {bundle}")
    ds.to_netcdf(bundle, engine="netcdf4")
    stats = {name: {"mean": float(np.nanmean(values)), "std": float(max(np.nanstd(values), 1e-6))}
             for name, values in continuous.items() if name not in ("latitude", "longitude", "static_valid_mask")}
    manifest = {
        "schema_version": SCHEMA_VERSION, "bundle_version": version, "sha256": sha256_file(bundle),
        "grid_fingerprint": fingerprint, "continuous_channels": list(CONTINUOUS_CHANNELS),
        "categorical_channels": list(CATEGORICAL_CHANNELS), "category_mappings": mappings,
        "normalization": stats, "source_manifest": source_manifest, "built_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = bundle.with_suffix(".json"); manifest_path.write_text(json.dumps(manifest, indent=2))
    validate_bundle(bundle, manifest_path); print(bundle); return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--hrrr", required=True, type=Path); parser.add_argument("--version", required=True)
    args = parser.parse_args(); build(args.hrrr, args.version)
