"""
Train/test split for ensemble_fire_danger: whole calendar weeks, never
individual days or cells.

Fire weather regimes persist for days and every Missouri cell on one day
shares the same synoptic pattern, so a random row (or even day) split
would leak nearly-identical samples across the split and make the
calibration look better than it is. Weeks (ISO year-week) are assigned to
test deterministically - every TEST_EVERY-th week in chronological order,
starting from the second - so the split is stable as new weeks are
appended (existing weeks never change sides) and test weeks are spread
across seasons instead of all being the most recent ones.

The manifest (paths.ENSEMBLE_FD_SPLIT_MANIFEST) records every day's side
and a checksum; evaluate.py refuses to report on a panel whose days
disagree with it.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Dict, Iterable

import paths

SPLIT_VERSION = "ensemble_fire_danger-weekly-1"
TEST_EVERY = 4  # ~25% of weeks held out


def week_key(day: str) -> str:
    iso = date.fromisoformat(day).isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def build_split(days: Iterable[str]) -> Dict[str, str]:
    days = sorted(set(days))
    weeks = sorted({week_key(d) for d in days})
    test_weeks = {w for i, w in enumerate(weeks) if i % TEST_EVERY == 1}
    return {d: ("test" if week_key(d) in test_weeks else "train") for d in days}


def write_manifest(split: Dict[str, str], path: Path = paths.ENSEMBLE_FD_SPLIT_MANIFEST) -> dict:
    payload = {
        "split_version": SPLIT_VERSION,
        "test_every_nth_week": TEST_EVERY,
        "days": split,
        "n_train_days": sum(v == "train" for v in split.values()),
        "n_test_days": sum(v == "test" for v in split.values()),
    }
    payload["sha256"] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def read_manifest(path: Path = paths.ENSEMBLE_FD_SPLIT_MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
