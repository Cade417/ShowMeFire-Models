"""
Split/manifest contract for fire_risk_fusion, mirroring
spatial/v5_contract.py's manifest/split/forbidden-runs pattern.

SPLIT_VERSION = "risk-fusion-county-day-split-v1"

Blocking, both dimensions:
- Temporal: contiguous 14-day episodes (a synoptic dry/windy pattern
  lasts days, so a day-level split leaks it), grouped into 5
  chronological blocks by episode order (not calendar-date modulo, which
  would scatter adjacent episodes across blocks and defeat the point).
- Spatial: region_id from risk_fusion.county_geometry (REGION_METHOD) -
  not used to hold out counties entirely (every county appears in every
  block), but recorded per row so a region-stratified metric breakdown
  is possible later.

A day-level random split would be badly optimistic here: fire weather
and fire counts are autocorrelated for days at a stretch (a single dry
windy pattern drives elevated counts across a whole region for most of a
week), so a naive shuffle leaks the same episode's information across
train and test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd

SPLIT_VERSION = "risk-fusion-county-day-split-v1"
EPISODE_LENGTH_DAYS = 14
DEFAULT_N_BLOCKS = 5
EPOCH = pd.Timestamp("2000-01-01")


def assign_episodes(dates: pd.Series, episode_length_days: int = EPISODE_LENGTH_DAYS) -> pd.Series:
    """episode_id = days since a fixed epoch // episode_length_days - stable across panels/date ranges."""
    parsed = pd.to_datetime(dates)
    days_since_epoch = (parsed - EPOCH).dt.days
    return (days_since_epoch // episode_length_days).astype("int64")


def assign_blocks(episode_ids: pd.Series, n_blocks: int = DEFAULT_N_BLOCKS) -> pd.Series:
    """
    Chronological block assignment: sort UNIQUE episodes by id (id order
    equals time order, since episode_id is derived from a monotonic day
    count), split into n_blocks contiguous groups of as-equal-as-possible
    episode COUNT, and map every row to its episode's block. Chronological
    grouping (not calendar-date modulo n_blocks) is what keeps each block
    a contiguous stretch of calendar time.
    """
    unique_episodes = np.sort(episode_ids.unique())
    block_of_episode = {}
    boundaries = np.array_split(unique_episodes, n_blocks)
    for block_index, episodes_in_block in enumerate(boundaries):
        for episode in episodes_in_block:
            block_of_episode[int(episode)] = block_index
    return episode_ids.map(block_of_episode).astype("int64")


def add_split_columns(panel: pd.DataFrame, date_column: str = "valid_local_date",
                      n_blocks: int = DEFAULT_N_BLOCKS) -> pd.DataFrame:
    """Adds episode_id and block columns to a copy of panel."""
    panel = panel.copy()
    panel["episode_id"] = assign_episodes(panel[date_column])
    panel["block"] = assign_blocks(panel["episode_id"], n_blocks=n_blocks)
    return panel


def crossfit_indices(panel: pd.DataFrame, n_blocks: int = DEFAULT_N_BLOCKS) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    panel must already have a 'block' column (see add_split_columns).
    Yields (train_index, test_index) once per block, holding that block
    out - grouped 5-fold cross-validation over chronological episode
    blocks, not a single fixed train/test split.
    """
    if "block" not in panel.columns:
        raise ValueError("panel must have a 'block' column - call add_split_columns first")
    for held_out in range(n_blocks):
        test_mask = panel["block"] == held_out
        train_mask = ~test_mask
        yield panel.index[train_mask].to_numpy(), panel.index[test_mask].to_numpy()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_manifest(panel: pd.DataFrame, date_column: str = "valid_local_date",
                    n_blocks: int = DEFAULT_N_BLOCKS) -> Dict:
    """
    Summarizes the split actually applied to `panel` (which must already
    have episode_id/block columns) - block date ranges, row/episode
    counts, region distribution per block if a region_id column exists.
    """
    for column in ("episode_id", "block"):
        if column not in panel.columns:
            raise ValueError(f"panel is missing '{column}' - call add_split_columns first")

    blocks = []
    for block_index in range(n_blocks):
        subset = panel[panel["block"] == block_index]
        if subset.empty:
            blocks.append({"block": block_index, "row_count": 0, "episode_count": 0})
            continue
        entry = {
            "block": block_index,
            "row_count": int(len(subset)),
            "episode_count": int(subset["episode_id"].nunique()),
            "date_min": str(subset[date_column].min()),
            "date_max": str(subset[date_column].max()),
        }
        if "region_id" in subset.columns:
            entry["region_counts"] = subset["region_id"].value_counts().sort_index().to_dict()
        blocks.append(entry)

    manifest = {
        "split_version": SPLIT_VERSION,
        "episode_length_days": EPISODE_LENGTH_DAYS,
        "n_blocks": n_blocks,
        "total_rows": int(len(panel)),
        "total_episodes": int(panel["episode_id"].nunique()),
        "blocks": blocks,
    }
    manifest["manifest_sha256"] = _sha256_bytes(json.dumps(manifest, sort_keys=True).encode())
    return manifest


def save_manifest(manifest: Dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def load_manifest(path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
