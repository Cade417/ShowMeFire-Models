"""
Split/manifest contract for fire_weather_ml, independently implemented but
following the same shape as risk_fusion/risk_fusion_contract.py (chronological,
block-based, never a row-level random split).

Why blocking still matters here even though the label is a deterministic
physics calculation, not a fire count: the INPUT state is autocorrelated -
10-hour/100-hour dead-fuel moisture is an exponential-memory process (see
api/services/spread_rate_moisture.py's TAU_10_HR/TAU_100_HR), so a station's
readings a few hours apart share almost the same fuel-moisture state. A
row-level random split would let near-duplicate rows leak across train/test.
Default episode length is 5 days (~120 hours), comfortably longer than the
100-hour fuel memory this model's slowest input reflects.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd

SPLIT_VERSION = "fire-weather-ml-station-split-v1"
EPISODE_LENGTH_DAYS = 5
DEFAULT_N_BLOCKS = 5
EPOCH = pd.Timestamp("2000-01-01")


def assign_episodes(timestamps: pd.Series, episode_length_days: int = EPISODE_LENGTH_DAYS) -> pd.Series:
    """episode_id = hours since a fixed epoch // episode length - stable across panels/date ranges."""
    parsed = pd.to_datetime(timestamps)
    hours_since_epoch = (parsed - EPOCH) / pd.Timedelta(hours=1)
    return (hours_since_epoch // (episode_length_days * 24)).astype("int64")


def assign_blocks(episode_ids: pd.Series, n_blocks: int = DEFAULT_N_BLOCKS) -> pd.Series:
    """Chronological block assignment: contiguous groups of as-equal-as-possible episode count, in time order."""
    unique_episodes = np.sort(episode_ids.unique())
    block_of_episode = {}
    boundaries = np.array_split(unique_episodes, min(n_blocks, len(unique_episodes)) or 1)
    for block_index, episodes_in_block in enumerate(boundaries):
        for episode in episodes_in_block:
            block_of_episode[int(episode)] = block_index
    return episode_ids.map(block_of_episode).astype("int64")


def add_split_columns(panel: pd.DataFrame, timestamp_column: str = "valid_time",
                      n_blocks: int = DEFAULT_N_BLOCKS) -> pd.DataFrame:
    panel = panel.copy()
    panel["episode_id"] = assign_episodes(panel[timestamp_column])
    panel["block"] = assign_blocks(panel["episode_id"], n_blocks=n_blocks)
    return panel


def crossfit_indices(panel: pd.DataFrame, n_blocks: int = DEFAULT_N_BLOCKS) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """panel must already have a 'block' column (see add_split_columns). Yields (train_index, test_index) once per block."""
    if "block" not in panel.columns:
        raise ValueError("panel must have a 'block' column - call add_split_columns first")
    for held_out in sorted(panel["block"].unique()):
        test_mask = panel["block"] == held_out
        yield panel.index[~test_mask].to_numpy(), panel.index[test_mask].to_numpy()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_manifest(panel: pd.DataFrame, timestamp_column: str = "valid_time",
                    n_blocks: int = DEFAULT_N_BLOCKS) -> Dict:
    for column in ("episode_id", "block"):
        if column not in panel.columns:
            raise ValueError(f"panel is missing '{column}' - call add_split_columns first")

    blocks = []
    for block_index in sorted(panel["block"].unique()):
        subset = panel[panel["block"] == block_index]
        entry = {
            "block": int(block_index),
            "row_count": int(len(subset)),
            "episode_count": int(subset["episode_id"].nunique()),
            "time_min": str(subset[timestamp_column].min()),
            "time_max": str(subset[timestamp_column].max()),
        }
        if "station_id" in subset.columns:
            entry["station_counts"] = subset["station_id"].value_counts().sort_index().to_dict()
        blocks.append(entry)

    manifest = {
        "split_version": SPLIT_VERSION,
        "episode_length_days": EPISODE_LENGTH_DAYS,
        "n_blocks": n_blocks,
        "total_rows": int(len(panel)),
        "total_episodes": int(panel["episode_id"].nunique()),
        "blocks": blocks,
    }
    manifest["manifest_sha256"] = _sha256_bytes(json.dumps(manifest, sort_keys=True, default=str).encode())
    return manifest


def save_manifest(manifest: Dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def load_manifest(path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
