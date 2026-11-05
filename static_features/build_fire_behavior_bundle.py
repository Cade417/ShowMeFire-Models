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
from static_features.fire_behavior_schema import (
    CATEGORICAL_CHANNELS,
    CONTINUOUS_CHANNELS,
    FB_SCHEMA_VERSION,
    GRID_SIZE,
    grid_fingerprint,
    sha256_file,
    validate_bundle,
)

BBOX = (-96.8, -88.1, 34.8, 41.8)
SOURCE_NAMES = {
    "elevation_m": "dem",
    "fuel_model_fbfm40": "fbfm40",
    "canopy_cover_pct": "canopy_cover",
    "canopy_height_m": "canopy_height",
}
SOURCE_UNITS = {
    "dem": "m",
    "fbfm40": "code",
    "canopy_cover": "percent",
    "canopy_height": "m",
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
        if not len(rows):
            raise ValueError("HRRR file does not intersect static domain")
        ydim, xdim = mask.dims
        crs = _hrrr_crs(ds)
        if xdim in ds.coords and ydim in ds.coords:
            x_values, y_values = np.asarray(ds[xdim].values), np.asarray(ds[ydim].values)
        else:
            # Herbie/cfgrib can serialize y/x as dimensions without
            # projected coordinate variables. Derive them from the
            # two-dimensional geographic coordinates instead of treating
            # dimension indices as metre coordinates.
            projected_x, projected_y = pyproj.Transformer.from_crs(
                "EPSG:4326", crs, always_xy=True
            ).transform(lon.values, ds.latitude.values)
            x_values = np.asarray(projected_x)[rows.min(), :]
            y_values = np.asarray(projected_y)[:, cols.min()]
        xmin, xmax = float(x_values[cols.min()]), float(x_values[cols.max()])
        ymin, ymax = float(y_values[rows.min()]), float(y_values[rows.max()])
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin
    transform = from_bounds(xmin, ymin, xmax, ymax, GRID_SIZE, GRID_SIZE)
    x = transform.c + (np.arange(GRID_SIZE) + 0.5) * transform.a
    y = transform.f + (np.arange(GRID_SIZE) + 0.5) * transform.e
    xx, yy = np.meshgrid(x, y)
    lon, lat = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xx, yy)
    return {"x": x, "y": y, "latitude": lat, "longitude": lon, "crs": crs, "transform": transform}


def _warp(path, grid, categorical=False):
    destination = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.nearest if categorical else Resampling.bilinear,
        )
    return destination


def build(hrrr_path: Path, version: str):
    source_manifest_path = paths.STATIC_SOURCE_DIR / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    products = source_manifest.get("products", {})
    missing = set(SOURCE_NAMES.values()) - set(products)
    if missing:
        raise ValueError(f"source products missing for fire behavior bundle: {sorted(missing)}")
    landfire_releases = {
        products[name].get("source_release")
        for name in ("fbfm40", "canopy_cover", "canopy_height")
    }
    if None in landfire_releases or len(landfire_releases) != 1:
        raise ValueError(
            "FBFM40, canopy cover, and canopy height must come from the same "
            "declared LANDFIRE release"
        )
    for product, expected_units in SOURCE_UNITS.items():
        actual_units = products[product].get("units")
        if actual_units != expected_units:
            raise ValueError(
                f"{product} source units must be {expected_units!r}, got {actual_units!r}; "
                "convert the source raster before building"
            )
        source_path = Path(products[product]["path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"{product} source raster not found: {source_path}")
        if sha256_file(source_path) != products[product].get("sha256"):
            raise ValueError(f"{product} source checksum differs from source_manifest.json")
    if not hrrr_path.is_file():
        raise FileNotFoundError(f"representative HRRR NetCDF not found: {hrrr_path}")

    grid = reference_grid(hrrr_path)
    raw = {}
    for channel, product in SOURCE_NAMES.items():
        raw[channel] = _warp(
            Path(products[product]["path"]),
            grid,
            categorical=channel == "fuel_model_fbfm40",
        )

    elevation = raw["elevation_m"]
    dx, dy = abs(grid["transform"].a), abs(grid["transform"].e)
    dzdy, dzdx = np.gradient(elevation, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = np.arctan2(-dzdx, dzdy)
    finite_fuel = np.isfinite(raw["fuel_model_fbfm40"])
    fuel_codes = np.zeros(raw["fuel_model_fbfm40"].shape, dtype="int32")
    fuel_codes[finite_fuel] = np.rint(raw["fuel_model_fbfm40"][finite_fuel]).astype("int32")
    valid = (
        np.isfinite(elevation)
        & np.isfinite(slope)
        & np.isfinite(aspect)
        & finite_fuel
        & np.isfinite(raw["canopy_cover_pct"])
        & np.isfinite(raw["canopy_height_m"])
        & (raw["canopy_cover_pct"] >= 0.0)
        & (raw["canopy_cover_pct"] <= 100.0)
        & (raw["canopy_height_m"] >= 0.0)
    )
    continuous = {
        "elevation_m": elevation,
        "slope_degrees": slope,
        "aspect_sin": np.sin(aspect),
        "aspect_cos": np.cos(aspect),
        "canopy_cover_pct": raw["canopy_cover_pct"],
        "canopy_height_m": raw["canopy_height_m"],
        "latitude": grid["latitude"],
        "longitude": grid["longitude"],
        "static_valid_mask": valid.astype("float32"),
    }
    categorical = {"fuel_model_fbfm40": fuel_codes.astype("float32")}
    data_vars = {name: (("y", "x"), np.asarray(values, dtype="float32")) for name, values in continuous.items()}
    data_vars.update({name: (("y", "x"), values) for name, values in categorical.items()})
    crs_wkt = grid["crs"].to_wkt()
    fingerprint = grid_fingerprint(grid["x"], grid["y"], crs_wkt)
    ds = xr.Dataset(
        data_vars,
        coords={"x": grid["x"], "y": grid["y"]},
        attrs={
            "schema_version": FB_SCHEMA_VERSION,
            "bundle_version": version,
            "grid_fingerprint": fingerprint,
            "crs_wkt": crs_wkt,
            "transform": tuple(grid["transform"]),
            "bbox": BBOX,
        },
    )
    units = {
        "elevation_m": "m",
        "slope_degrees": "degrees",
        "aspect_sin": "unitless",
        "aspect_cos": "unitless",
        "canopy_cover_pct": "percent",
        "canopy_height_m": "m",
        "latitude": "degrees_north",
        "longitude": "degrees_east",
        "static_valid_mask": "1",
        "fuel_model_fbfm40": "code",
    }
    for name, value in units.items():
        ds[name].attrs["units"] = value
    bundle = paths.STATIC_BUNDLE_DIR / f"fire_behavior_static_{version}.nc"
    if bundle.exists():
        raise FileExistsError(f"immutable fire behavior bundle already exists: {bundle}")
    ds.to_netcdf(bundle, engine="netcdf4")
    stats = {
        name: {"mean": float(np.nanmean(values)), "std": float(max(np.nanstd(values), 1e-6))}
        for name, values in continuous.items()
        if name not in ("latitude", "longitude", "static_valid_mask")
    }
    manifest = {
        "schema_version": FB_SCHEMA_VERSION,
        "bundle_version": version,
        "sha256": sha256_file(bundle),
        "grid_fingerprint": fingerprint,
        "continuous_channels": list(CONTINUOUS_CHANNELS),
        "categorical_channels": list(CATEGORICAL_CHANNELS),
        "normalization": stats,
        "source_manifest": source_manifest,
        "reference_hrrr": {
            "path": str(hrrr_path.resolve()),
            "sha256": sha256_file(hrrr_path),
        },
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = bundle.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest = validate_bundle(bundle, manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(bundle)
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build immutable fire-behavior static NetCDF bundle")
    parser.add_argument("--hrrr", required=True, type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    build(args.hrrr, args.version)
