"""Validate and register a fire-behavior static bundle as a beta release.

This is an asset-only release. It contains immutable LANDFIRE/topography
inputs for Rothermel calculations, not a fitted machine-learning model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.versioning import register_trained_model
from static_features.fire_behavior_schema import validate_bundle


def register(bundle: Path, manifest: Path | None = None, bump: str = "patch") -> str:
    bundle = bundle.resolve()
    manifest = (manifest or bundle.with_suffix(".json")).resolve()
    if not bundle.is_file():
        raise FileNotFoundError(f"Bundle not found: {bundle}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    validated = validate_bundle(bundle, manifest)
    version = register_trained_model(
        model_type="fire_behavior_static",
        performance={
            "bundle_version": validated["bundle_version"],
            "burnable_cell_fraction": validated["validation"]["burnable_cell_fraction"],
        },
        bump=bump,
        channel="beta",
        assets={
            "static_bundle": {
                "path": bundle,
                "grid_fingerprint": validated["grid_fingerprint"],
                "schema_version": validated["schema_version"],
            },
            "static_manifest": {"path": manifest},
        },
        metadata={
            "bundle_schema_version": validated["schema_version"],
            "bundle_version": validated["bundle_version"],
            "grid_fingerprint": validated["grid_fingerprint"],
            "source_release": (validated.get("source_manifest") or {}).get("release"),
            "promotion_gates": {"static_bundle_validated": True},
            "shadow_required": False,
        },
    )
    print(json.dumps({
        "model_type": "fire_behavior_static",
        "version": version,
        "bundle": str(bundle),
        "manifest": str(manifest),
    }, indent=2))
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate and register an immutable fire-behavior static bundle"
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bump", choices=["major", "minor", "patch"], default="patch")
    args = parser.parse_args()
    register(args.bundle, args.manifest, args.bump)
