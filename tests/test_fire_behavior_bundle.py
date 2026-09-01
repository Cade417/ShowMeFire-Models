import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from models import versioning
from pipelines import publish_release
from pipelines.register_fire_behavior_static import register
from static_features.fire_behavior_schema import (
    FB_SCHEMA_VERSION,
    GRID_SIZE,
    grid_fingerprint,
    sha256_file,
    validate_bundle,
)


def _write_bundle(root: Path, *, fuel_code: int = 101):
    root.mkdir(parents=True, exist_ok=True)
    bundle = root / "fire_behavior_static_test.nc"
    manifest_path = bundle.with_suffix(".json")
    x = np.arange(GRID_SIZE, dtype=float)
    y = np.arange(GRID_SIZE, dtype=float)
    crs = 'PROJCRS["test"]'
    fingerprint = grid_fingerprint(x, y, crs)
    shape = (GRID_SIZE, GRID_SIZE)
    variables = {
        "elevation_m": (("y", "x"), np.full(shape, 250.0, dtype="float32")),
        "slope_degrees": (("y", "x"), np.full(shape, 5.0, dtype="float32")),
        "aspect_sin": (("y", "x"), np.zeros(shape, dtype="float32")),
        "aspect_cos": (("y", "x"), np.ones(shape, dtype="float32")),
        "canopy_cover_pct": (("y", "x"), np.full(shape, 20.0, dtype="float32")),
        "canopy_height_m": (("y", "x"), np.full(shape, 8.0, dtype="float32")),
        "latitude": (("y", "x"), np.full(shape, 38.0, dtype="float32")),
        "longitude": (("y", "x"), np.full(shape, -92.0, dtype="float32")),
        "static_valid_mask": (("y", "x"), np.ones(shape, dtype="float32")),
        "fuel_model_fbfm40": (("y", "x"), np.full(shape, fuel_code, dtype="float32")),
    }
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
    dataset = xr.Dataset(
        variables,
        coords={"x": x, "y": y},
        attrs={
            "schema_version": FB_SCHEMA_VERSION,
            "bundle_version": "test-2026.1",
            "grid_fingerprint": fingerprint,
            "crs_wkt": crs,
        },
    )
    for name, value in units.items():
        dataset[name].attrs["units"] = value
    dataset.to_netcdf(bundle)
    manifest = {
        "schema_version": FB_SCHEMA_VERSION,
        "bundle_version": "test-2026.1",
        "sha256": sha256_file(bundle),
        "grid_fingerprint": fingerprint,
        "source_manifest": {"release": "test-source"},
    }
    manifest_path.write_text(json.dumps(manifest))
    return bundle, manifest_path


def test_fire_behavior_bundle_contract_records_validation(tmp_path):
    bundle, manifest = _write_bundle(tmp_path)
    validated = validate_bundle(bundle, manifest)
    assert validated["validation"]["burnable_cell_fraction"] == 1.0
    assert validated["validation"]["fuel_model_codes"] == [101]


def test_fire_behavior_bundle_rejects_unknown_fuel_code(tmp_path):
    bundle, manifest = _write_bundle(tmp_path, fuel_code=999)
    try:
        validate_bundle(bundle, manifest)
    except ValueError as exc:
        assert "unknown FBFM40 codes" in str(exc)
    else:
        raise AssertionError("unknown fuel code was accepted")


def test_registration_creates_asset_only_beta_candidate(tmp_path):
    bundle, manifest = _write_bundle(tmp_path / "input")
    models = tmp_path / "registry" / "models"
    versions = models / "versions"
    config = models / "config.json"
    with patch.multiple(
        versioning,
        MODELS_DIR=models,
        VERSIONS_DIR=versions,
        CONFIG_PATH=config,
        API_DIR=tmp_path / "registry",
    ):
        assigned = register(bundle, manifest)
        candidate = versioning.get_model_entry("fire_behavior_static")["beta"]

    assert assigned.endswith("-beta.1")
    assert candidate["file"].endswith("_static_bundle.nc")
    assert set(candidate["assets"]) == {"static_bundle", "static_manifest"}
    assert candidate["metadata"]["promotion_gates"]["static_bundle_validated"] is True


def test_asset_only_candidate_publishes_without_model_path(tmp_path):
    bundle = tmp_path / "bundle.nc"
    manifest = tmp_path / "bundle.json"
    bundle.write_bytes(b"bundle")
    manifest.write_text("{}")
    candidate = {
        "version": "0.0.1-beta.1",
        "performance": {"bundle_version": "2026.1"},
        "trained_at": "2026-08-31",
        "metadata": {"grid_fingerprint": "abc"},
        "assets": {
            "static_bundle": {
                "file": bundle.name,
                "sha256": sha256_file(bundle),
            },
            "static_manifest": {
                "file": manifest.name,
                "sha256": sha256_file(manifest),
            },
        },
    }
    captured = {}

    def capture_release(command, check):
        assert check is True
        metadata_path = Path(next(item for item in command if item.endswith("/metadata.json")))
        captured.update(json.loads(metadata_path.read_text()))

    with (
        patch.object(publish_release.paths, "DATA_ROOT", tmp_path),
        patch.object(
            publish_release,
            "get_model_entry",
            return_value={"beta": candidate},
        ),
        patch.object(
            publish_release,
            "load_active_model_path",
            side_effect=AssertionError("asset-only release requested a model path"),
        ),
        patch.object(publish_release.subprocess, "run", side_effect=capture_release),
    ):
        publish_release.publish(
            "fire_behavior_static",
            version="0.0.1-beta.1",
            repo="owner/models",
        )

    assert set(captured["assets"]) == {"static_bundle", "static_manifest"}
    assert captured["model_metadata"]["grid_fingerprint"] == "abc"
