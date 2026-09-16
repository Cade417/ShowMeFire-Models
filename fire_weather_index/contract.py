"""
Split/manifest contract for fire_weather_index - reuses risk_fusion's
episode/block split logic wholesale (it's source-agnostic: contiguous
14-day episodes grouped into chronological blocks, so the same synoptic
dry/windy pattern never leaks across train/test regardless of which model
consumes the split). Only the manifest PATH is separate, so registering a
fire_weather_index split never touches risk_fusion's own manifest.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths
from risk_fusion.risk_fusion_contract import (  # noqa: F401 - re-exported on purpose
    DEFAULT_N_BLOCKS,
    EPISODE_LENGTH_DAYS,
    add_split_columns,
    assign_blocks,
    assign_episodes,
    crossfit_indices,
    load_manifest,
    save_manifest,
)
from risk_fusion.risk_fusion_contract import create_manifest as _create_manifest

SPLIT_VERSION = "fire-weather-index-county-day-split-v1"
SPLIT_MANIFEST_PATH = paths.FIRE_WEATHER_INDEX_SPLIT_MANIFEST


def create_manifest(panel, date_column: str = "valid_local_date", n_blocks: int = DEFAULT_N_BLOCKS) -> dict:
    manifest = _create_manifest(panel, date_column=date_column, n_blocks=n_blocks)
    manifest["split_version"] = SPLIT_VERSION  # own version string, not risk_fusion's
    # manifest_sha256 was computed before the line above changed the content -
    # recompute so it actually matches what gets saved (same method risk_fusion_contract uses).
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest
