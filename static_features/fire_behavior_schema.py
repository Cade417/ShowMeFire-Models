from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

FB_SCHEMA_VERSION = 1
GRID_SIZE = 256
CONTINUOUS_CHANNELS = (
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "canopy_cover_pct",
    "canopy_height_m",
    "latitude",
    "longitude",
    "static_valid_mask",
)
CATEGORICAL_CHANNELS = ("fuel_model_fbfm40",)
VALID_FBFM40_CODES = frozenset(
    set(range(1, 14))
    | set(range(91, 100))
    | set(range(101, 110))
    | set(range(121, 125))
    | set(range(141, 150))
    | set(range(161, 166))
    | set(range(181, 190))
    | set(range(201, 205))
)
NON_BURNABLE_FBFM40_CODES = frozenset(range(91, 100))

EXPECTED_UNITS = {
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_fingerprint(x, y, crs_wkt: str) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(x, dtype="float64").tobytes())
    digest.update(np.asarray(y, dtype="float64").tobytes())
    digest.update(crs_wkt.encode())
    digest.update(f"schema={FB_SCHEMA_VERSION};size={GRID_SIZE}".encode())
    return digest.hexdigest()


def validate_bundle(bundle_path: Path, manifest_path: Path | None = None):
    manifest_path = manifest_path or bundle_path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != FB_SCHEMA_VERSION:
        raise ValueError("unsupported fire behavior static bundle schema")
    if manifest["sha256"] != sha256_file(bundle_path):
        raise ValueError("fire behavior static bundle checksum mismatch")
    with xr.open_dataset(bundle_path) as ds:
        missing = set(CONTINUOUS_CHANNELS + CATEGORICAL_CHANNELS) - set(ds.data_vars)
        if missing:
            raise ValueError(f"fire behavior bundle channels missing: {sorted(missing)}")
        if ds.sizes.get("x") != GRID_SIZE or ds.sizes.get("y") != GRID_SIZE:
            raise ValueError("fire behavior static bundle must be 256x256")
        if not ds.attrs.get("crs_wkt"):
            raise ValueError("fire behavior static bundle is missing crs_wkt")
        actual = grid_fingerprint(ds.x.values, ds.y.values, ds.attrs["crs_wkt"])
        if actual != manifest["grid_fingerprint"]:
            raise ValueError("fire behavior bundle grid fingerprint mismatch")
        if ds.attrs.get("grid_fingerprint") != actual:
            raise ValueError("dataset and manifest grid fingerprints differ")

        for name, expected in EXPECTED_UNITS.items():
            if ds[name].attrs.get("units") != expected:
                raise ValueError(
                    f"{name} units must be {expected!r}, got {ds[name].attrs.get('units')!r}"
                )
            if ds[name].dims != ("y", "x"):
                raise ValueError(f"{name} must use y,x dimensions")

        valid = np.asarray(ds["static_valid_mask"].values) > 0.5
        if not valid.any():
            raise ValueError("fire behavior static bundle has no valid cells")
        for name in CONTINUOUS_CHANNELS:
            if name == "static_valid_mask":
                continue
            if not np.isfinite(ds[name].values[valid]).all():
                raise ValueError(f"{name} contains non-finite values in valid cells")

        slope = np.asarray(ds["slope_degrees"].values)
        cover = np.asarray(ds["canopy_cover_pct"].values)
        height = np.asarray(ds["canopy_height_m"].values)
        if np.any((slope[valid] < 0.0) | (slope[valid] >= 90.0)):
            raise ValueError("slope_degrees must be in [0, 90)")
        if np.any((cover[valid] < 0.0) | (cover[valid] > 100.0)):
            raise ValueError("canopy_cover_pct must be in [0, 100]")
        if np.any(height[valid] < 0.0):
            raise ValueError("canopy_height_m cannot be negative")

        aspect_norm = np.hypot(ds["aspect_sin"].values[valid], ds["aspect_cos"].values[valid])
        if not np.allclose(aspect_norm, 1.0, atol=1e-3):
            raise ValueError("aspect sine/cosine values are not unit vectors")

        raw_codes = np.asarray(ds["fuel_model_fbfm40"].values)
        rounded_codes = np.rint(raw_codes).astype("int32")
        if not np.allclose(raw_codes[valid], rounded_codes[valid], atol=1e-4):
            raise ValueError("fuel_model_fbfm40 contains non-integer values")
        present_codes = set(np.unique(rounded_codes[valid]).tolist())
        unknown_codes = present_codes - VALID_FBFM40_CODES
        if unknown_codes:
            raise ValueError(f"unknown FBFM40 codes: {sorted(unknown_codes)}")
        burnable = valid & ~np.isin(rounded_codes, list(NON_BURNABLE_FBFM40_CODES))
        burnable_fraction = float(burnable.sum() / valid.sum())
        if burnable_fraction <= 0.0:
            raise ValueError("fire behavior static bundle contains no burnable cells")

    manifest["validation"] = {
        "valid_cell_fraction": float(valid.mean()),
        "burnable_cell_fraction": burnable_fraction,
        "fuel_model_codes": sorted(present_codes),
        "crs_validated": True,
        "units_validated": True,
        "nodata_validated": True,
    }
    return manifest
