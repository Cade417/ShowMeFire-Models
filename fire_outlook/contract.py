"""
Verification-record schema and train/test split for the seven-day outlook.

One record = one issue time x six-hour window x station, written by the
fire-outlook worker once the window's observations are in. The split holds out
contiguous periods AND stations (never random rows): neighbouring windows,
leads and stations share the same synoptic pattern, so a random split would
leak near-duplicates across it.

  period   issue days are cut into PERIOD_BLOCK_DAYS-day blocks; every
           TEST_EVERY_BLOCK-th block (from the second) is test. Train issue days
           within PURGE_DAYS of a test block are dropped ("purged"), because a
           168 h forecast issued before the block verifies inside it.
  station  a stable hash of station_id puts ~1/STATION_TEST_MOD of stations in test.

A row is "train" only if its day AND station are train, "test" only if both are
test; mixed rows are "excluded" so no train row shares a period or a station
with a test row.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import paths

SPLIT_VERSION = "fire_outlook-block-station-1"
N_WINDOWS = 28
WINDOW_HOURS = 6
PERIOD_BLOCK_DAYS = 14
TEST_EVERY_BLOCK = 4
PURGE_DAYS = 7
STATION_TEST_MOD = 5
LEAD_GROUPS = ("d1-3", "d4-5", "d6-7")
WEATHER_VARIABLES = ("t2m_c", "rh_pct", "wind_ms", "precip_mm")
INTERVAL_NOMINAL = 0.80  # p10-p90

REQUIRED_COLUMNS = (
    "issue_time", "window_idx", "station_id", "n_members",
    "agreement_elevated", "agreement_critical", "obs_elevated", "obs_critical", "qc_ok",
    "fm10_p10", "fm10_p50", "fm10_p90", "fm10_obs",
)


def lead_group(window_idx) -> str:
    """Days 1-3 = windows 0-11, days 4-5 = 12-19, days 6-7 = 20-27."""
    idx = int(window_idx)
    return "d1-3" if idx < 12 else ("d4-5" if idx < 20 else "d6-7")


def variable_columns(name: str) -> Dict[str, str]:
    """Column names for one verified variable. 'persist' / 'nbm' may be absent."""
    if name == "fm10":
        return {"center": "fm10_p50", "lo": "fm10_p10", "hi": "fm10_p90", "obs": "fm10_obs",
                "persist": "fm10_init_obs", "nbm": None}
    return {"center": f"{name}_mean", "lo": f"{name}_p10", "hi": f"{name}_p90", "obs": f"{name}_obs",
            "persist": f"{name}_persist", "nbm": f"{name}_nbm"}


def load_records(source: Path | str | None = None) -> Tuple[pd.DataFrame, dict]:
    """Read every .parquet/.csv under `source` (or one file), validate, and derive keys."""
    source = Path(source) if source else paths.FIRE_OUTLOOK_RECORDS_DIR
    files = [source] if source.is_file() else sorted(
        p for p in source.rglob("*") if p.suffix in (".parquet", ".csv"))
    if not files:
        raise FileNotFoundError(f"no verification records (.parquet/.csv) under {source}")
    frames = [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files]
    return prepare(pd.concat(frames, ignore_index=True))


def prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"verification records missing required columns: {missing}")
    df = df.copy()
    issues = {"rows_in": int(len(df))}
    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    df["window_idx"] = df["window_idx"].astype(int)
    df["station_id"] = df["station_id"].astype(str)
    df["qc_ok"] = df["qc_ok"].fillna(False).astype(bool)

    bad_window = ~df["window_idx"].between(0, N_WINDOWS - 1)
    issues["dropped_bad_window_idx"] = int(bad_window.sum())
    df = df[~bad_window]
    before = len(df)
    df = df.drop_duplicates(["issue_time", "window_idx", "station_id"], keep="last")
    issues["dropped_duplicates"] = int(before - len(df))

    bad_range = ~(df["agreement_elevated"].between(0, 1) & df["agreement_critical"].between(0, 1))
    issues["dropped_agreement_out_of_range"] = int(bad_range.sum())
    df = df[~bad_range]
    # Critical-or-higher can never exceed Elevated-or-higher; a row that says otherwise is a worker bug.
    inverted = df["agreement_critical"] > df["agreement_elevated"] + 1e-6
    issues["critical_exceeds_elevated_rows"] = int(inverted.sum())
    df = df[~inverted]

    df["issue_day"] = df["issue_time"].dt.strftime("%Y-%m-%d")
    df["lead_group"] = df["window_idx"].map(lead_group)
    valid_start = df["issue_time"] + pd.to_timedelta(df["window_idx"] * WINDOW_HOURS, unit="h")
    df["valid_start"] = valid_start
    df["valid_month"] = valid_start.dt.month
    df["valid_bucket"] = (valid_start.dt.hour // WINDOW_HOURS).astype(int)
    issues["rows_out"] = int(len(df))
    return df.reset_index(drop=True), issues


def _block_index(day: str, origin: date) -> int:
    return (date.fromisoformat(day) - origin).days // PERIOD_BLOCK_DAYS


def build_day_split(days: Iterable[str]) -> Dict[str, str]:
    """day -> 'train' | 'test' | 'purged'. Stable as later days are appended."""
    days = sorted(set(days))
    if not days:
        return {}
    origin = date.fromisoformat(days[0])
    blocks = sorted({_block_index(d, origin) for d in days})
    test_blocks = {b for i, b in enumerate(blocks) if i % TEST_EVERY_BLOCK == 1}
    split: Dict[str, str] = {}
    test_days = [d for d in days if _block_index(d, origin) in test_blocks]
    for d in days:
        if _block_index(d, origin) in test_blocks:
            split[d] = "test"
            continue
        here = date.fromisoformat(d)
        near = any(abs((here - date.fromisoformat(t)).days) <= PURGE_DAYS for t in test_days)
        split[d] = "purged" if near else "train"
    return split


def station_side(station_id: str) -> str:
    h = int(hashlib.sha1(str(station_id).encode()).hexdigest(), 16)
    return "test" if h % STATION_TEST_MOD == 0 else "train"


def assign_split(df: pd.DataFrame) -> Tuple[pd.Series, dict]:
    day_split = build_day_split(df["issue_day"].unique())
    stations = sorted(df["station_id"].unique())
    station_split = {s: station_side(s) for s in stations}
    day_side = df["issue_day"].map(day_split)
    stn_side = df["station_id"].map(station_split)
    side = np.where((day_side == "train") & (stn_side == "train"), "train",
                    np.where((day_side == "test") & (stn_side == "test"), "test", "excluded"))
    manifest = {
        "split_version": SPLIT_VERSION,
        "period_block_days": PERIOD_BLOCK_DAYS, "test_every_block": TEST_EVERY_BLOCK,
        "purge_days": PURGE_DAYS, "station_test_mod": STATION_TEST_MOD,
        "days": day_split, "stations": station_split,
        "n_train_days": sum(v == "train" for v in day_split.values()),
        "n_test_days": sum(v == "test" for v in day_split.values()),
        "n_purged_days": sum(v == "purged" for v in day_split.values()),
        "n_train_stations": sum(v == "train" for v in station_split.values()),
        "n_test_stations": sum(v == "test" for v in station_split.values()),
        "n_rows": {k: int((side == k).sum()) for k in ("train", "test", "excluded")},
    }
    manifest["sha256"] = hashlib.sha256(
        json.dumps({"d": day_split, "s": station_split}, sort_keys=True).encode()).hexdigest()
    return pd.Series(side, index=df.index, name="split"), manifest


def write_manifest(manifest: dict, path: Path = paths.FIRE_OUTLOOK_SPLIT_MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
