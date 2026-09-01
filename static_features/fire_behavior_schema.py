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
        actual = grid_fingerprint(ds.x.values, ds.y.values, ds.attrs["crs_wkt"])
        if actual != manifest["grid_fingerprint"]:
            raise ValueError("fire behavior bundle grid fingerprint mismatch")
        codes = ds["fuel_model_fbfm40"].values
        if np.nanmin(codes) < 0:
            raise ValueError("negative fuel model codes in fuel_model_fbfm40")
    return manifest
