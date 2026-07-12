from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from static_features.schema import validate_bundle

FEATURE_SETS = {
    "dynamic": {"continuous": [], "categorical": []},
    "terrain": {"continuous": ["elevation_m", "slope_degrees", "aspect_sin", "aspect_cos", "terrain_ruggedness_m", "latitude", "longitude", "static_valid_mask"], "categorical": []},
    "terrain_nlcd_canopy": {"continuous": ["elevation_m", "slope_degrees", "aspect_sin", "aspect_cos", "terrain_ruggedness_m", "nlcd_confidence", "canopy_cover_pct", "latitude", "longitude", "static_valid_mask"], "categorical": ["nlcd_class"]},
    "terrain_landfire": {"continuous": ["elevation_m", "slope_degrees", "aspect_sin", "aspect_cos", "terrain_ruggedness_m", "canopy_cover_pct", "latitude", "longitude", "static_valid_mask"], "categorical": ["fuel_model_fbfm40", "fuel_vegetation_type"]},
    "all": {"continuous": ["elevation_m", "slope_degrees", "aspect_sin", "aspect_cos", "terrain_ruggedness_m", "nlcd_confidence", "canopy_cover_pct", "latitude", "longitude", "static_valid_mask"], "categorical": ["nlcd_class", "fuel_model_fbfm40", "fuel_vegetation_type"]},
}


def load_static_inputs(bundle_path: Path, feature_set="all"):
    manifest = validate_bundle(bundle_path)
    contract = FEATURE_SETS[feature_set]
    with xr.open_dataset(bundle_path) as ds:
        continuous = np.stack([ds[name].values for name in contract["continuous"]]).astype("float32") if contract["continuous"] else np.zeros((1, 256, 256), dtype="float32")
        categorical = np.stack([ds[name].values for name in contract["categorical"]]).astype("int64") if contract["categorical"] else np.zeros((1, 256, 256), dtype="int64")
    means, stds = [], []
    continuous_names = contract["continuous"] or ["__none__"]
    categorical_names = contract["categorical"] or ["__none__"]
    for index, name in enumerate(continuous_names):
        if name == "__none__": means.append(0.0); stds.append(1.0); continue
        stats = manifest.get("normalization", {}).get(name)
        means.append(stats["mean"] if stats else float(np.nanmean(continuous[index])))
        stds.append(stats["std"] if stats else float(max(np.nanstd(continuous[index]), 1e-6)))
    if means:
        continuous = (continuous - np.asarray(means, dtype="float32")[:, None, None]) / np.asarray(stds, dtype="float32")[:, None, None]
        continuous = np.nan_to_num(continuous)
    sizes = [max(manifest["category_mappings"][name].values()) + 1 for name in contract["categorical"]] or [1]
    return continuous, categorical, {
        "feature_set": feature_set, "continuous_channels": continuous_names, "categorical_channels": categorical_names,
        "category_sizes": sizes, "normalization_mean": means, "normalization_std": stds,
        "bundle_file": bundle_path.name, "bundle_sha256": manifest["sha256"], "grid_fingerprint": manifest["grid_fingerprint"],
        "schema_version": manifest["schema_version"], "category_mappings": {name: manifest["category_mappings"][name] for name in contract["categorical"]},
    }
