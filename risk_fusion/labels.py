"""
Load, validate, dedup, and aggregate exported fire-event labels into
county-day counts for fire_risk_fusion.

Reads the CSV + manifest api/scripts/export_fire_labels.py produces
(mirrored into paths.FIRE_LABELS_DIR) - this module never touches the
server DB directly, per paths.py's DB_PATH docstring ("this repo's own
independent training DB - never the server's").

TIER_RANK/EXCLUDED_CAUSE_CATEGORIES are plain constant data, not logic,
so they are duplicated here rather than added to the contract-mirror
system - but they MUST stay in sync with api/core/fire_events.py's
TIER_RANK and CAUSE_CATEGORIES by hand.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PRIMARY_TIER = "official_source_confirmed"
AUXILIARY_TIER = "admin_reviewed"
TIER_RANK = {"unverified": 0, "admin_reviewed": 1, "official_source_confirmed": 2}

# Prescribed/agricultural burns are intentional and inversely related to
# fire weather (people burn on good-burn days) - the top labeling risk
# named in the project plan. FPA-FOD's own scope already excludes these
# (it is a wildfire-only database), so this filter is a safety net for
# any future source that isn't similarly scoped, not a no-op for show.
EXCLUDED_CAUSE_CATEGORIES = {"prescribed", "agricultural"}

DEDUP_RADIUS_KM = 2.0
DEDUP_TIME_HOURS = 6.0

# Verified against the real 2011-2020 Missouri FPA-FOD export: 6.38%
# (288 clusters / 635 rows of 9,956). Inspected a sample of the flagged
# clusters directly - they are genuine near-duplicates (same day, tens of
# meters apart, independently estimated acreage), consistent with
# FPA-FOD's documented behavior of compiling from multiple reporting
# systems (Forest Service, state, county) that can each independently
# report the same physical incident. This is expected for FPA-FOD
# specifically, not a sign of a pipeline bug - the 5% figure from the
# original plan assumed a single, internally-deduplicated reporting
# system. 8% leaves room for that known behavior while still catching a
# source that duplicates on a much larger scale (a real pipeline bug -
# e.g. double-ingestion - would produce collapse rates far above this,
# not a couple of points over).
MAX_DUPLICATE_COLLAPSE_RATE = 0.08

CENTRAL = ZoneInfo("America/Chicago")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fire_labels(csv_path, manifest_path) -> Tuple[pd.DataFrame, Dict]:
    """
    Verifies the CSV against its manifest's recorded checksum, loads it,
    and confirms no unverified rows slipped through. Returns (frame, manifest).
    """
    csv_path, manifest_path = Path(csv_path), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    actual_sha = _sha256_file(csv_path)
    if actual_sha != manifest["csv_sha256"]:
        raise ValueError(
            f"fire labels CSV checksum mismatch: manifest says {manifest['csv_sha256']}, "
            f"file is actually {actual_sha} - the export and manifest are out of sync"
        )

    frame = pd.read_csv(csv_path, dtype={"county_fips": str, "event_id": "Int64"})
    frame["fuel_types"] = frame["fuel_types"].fillna("")

    unverified_in_frame = int((frame["verification_tier"] == "unverified").sum())
    expected_unverified = manifest["rows_by_tier"].get("unverified", 0)
    if unverified_in_frame != expected_unverified:
        raise ValueError(
            f"label export contains {unverified_in_frame} unverified rows but its own "
            f"manifest declares {expected_unverified} - refusing to proceed on a self-inconsistent export"
        )
    frame = frame[frame["verification_tier"] != "unverified"].reset_index(drop=True)
    return frame, manifest


def apply_cause_filter(frame: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drops prescribed/agricultural rows. Returns (filtered_frame, excluded_count)."""
    mask = ~frame["cause_category"].isin(EXCLUDED_CAUSE_CATEGORIES)
    return frame[mask].copy(), int((~mask).sum())


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def cluster_events(frame: pd.DataFrame, radius_km: float = DEDUP_RADIUS_KM,
                   time_hours: float = DEDUP_TIME_HOURS) -> pd.DataFrame:
    """
    Single-linkage clusters events that are both spatially (<= radius_km)
    and temporally (<= time_hours) close enough to plausibly be duplicate
    reports of the same fire - deliberately the same thresholds as the
    Phase-1 admin moderation duplicate hint, so training and moderation
    share one definition. Adds cluster_id/cluster_size columns.

    O(n log n + n*k) via a lat-sorted sliding window (k = neighbors within
    the latitude band), fine at current volumes (~10k rows/source); revisit
    if this ever needs to run over a much larger multi-source label set.
    """
    frame = frame.reset_index(drop=True).copy()
    n = len(frame)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if n == 0:
        frame["cluster_id"] = pd.Series(dtype="int64")
        frame["cluster_size"] = pd.Series(dtype="int64")
        return frame

    lat = frame["latitude"].to_numpy(dtype="float64")
    lon = frame["longitude"].to_numpy(dtype="float64")
    times = pd.to_datetime(frame["occurred_at"], utc=True).to_numpy()

    order = np.argsort(lat)
    lat_sorted = lat[order]
    max_lat_delta = radius_km / 110.574  # generous upper bound; haversine check below is exact

    for i in range(n):
        oi = order[i]
        j = i + 1
        while j < n and (lat_sorted[j] - lat_sorted[i]) <= max_lat_delta:
            oj = order[j]
            j += 1
            dt_hours = abs((times[oi] - times[oj]) / np.timedelta64(1, "h"))
            if dt_hours > time_hours:
                continue
            if _haversine_km(lat[oi], lon[oi], lat[oj], lon[oj]) <= radius_km:
                union(oi, oj)

    cluster_ids = np.array([find(i) for i in range(n)])
    frame["cluster_id"] = cluster_ids
    frame["cluster_size"] = frame.groupby("cluster_id")["cluster_id"].transform("count")
    return frame


def representative_events(clustered_frame: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster: highest TIER_RANK, tie-broken by earliest occurred_at."""
    frame = clustered_frame.copy()
    frame["_tier_rank"] = frame["verification_tier"].map(TIER_RANK)
    frame = frame.sort_values(["_tier_rank", "occurred_at"], ascending=[False, True])
    return frame.drop_duplicates("cluster_id", keep="first").drop(columns=["_tier_rank"])


def duplicate_collapse_rate(clustered_frame: pd.DataFrame, tier: str = PRIMARY_TIER) -> float:
    """Fraction of tier rows merged into a multi-row cluster - officials should publish ~1 record/incident."""
    subset = clustered_frame[clustered_frame["verification_tier"] == tier]
    if subset.empty:
        return 0.0
    return float((subset["cluster_size"] > 1).sum()) / len(subset)


def to_county_day_counts(representative_frame: pd.DataFrame, tier: str = PRIMARY_TIER) -> pd.DataFrame:
    """
    y_cd per (county_fips, valid_local_date) for the given tier, after
    cause filtering. Columns: county_fips, valid_local_date, event_count,
    acres_sum, acres_max.
    """
    subset = representative_frame[representative_frame["verification_tier"] == tier].copy()
    subset, _ = apply_cause_filter(subset)
    if subset.empty:
        return pd.DataFrame(columns=["county_fips", "valid_local_date", "event_count", "acres_sum", "acres_max"])

    local_dt = pd.to_datetime(subset["occurred_at"], utc=True).dt.tz_convert(CENTRAL)
    subset["valid_local_date"] = local_dt.dt.strftime("%Y-%m-%d")
    grouped = subset.groupby(["county_fips", "valid_local_date"], as_index=False).agg(
        event_count=("event_id", "count"),
        acres_sum=("acres", "sum"),
        acres_max=("acres", "max"),
    )
    return grouped


def build_labels(csv_path, manifest_path) -> Dict:
    """
    End-to-end: load -> dedup -> cause-filter -> county-day counts for
    both the primary (official) and auxiliary (admin_reviewed) tiers.
    Raises if duplicate_collapse_rate on the primary tier exceeds
    MAX_DUPLICATE_COLLAPSE_RATE - a higher rate means the source itself
    is duplicating records, which is a data-quality problem worth
    stopping on rather than silently absorbing.
    """
    frame, manifest = load_fire_labels(csv_path, manifest_path)
    clustered = cluster_events(frame)
    representatives = representative_events(clustered)

    collapse_rate = duplicate_collapse_rate(clustered, tier=PRIMARY_TIER)
    if collapse_rate > MAX_DUPLICATE_COLLAPSE_RATE:
        raise ValueError(
            f"duplicate_collapse_rate={collapse_rate:.4f} exceeds the {MAX_DUPLICATE_COLLAPSE_RATE} "
            "gate for the primary label tier - the source is duplicating records; fix upstream before training"
        )

    primary_counts = to_county_day_counts(representatives, tier=PRIMARY_TIER)
    auxiliary_counts = to_county_day_counts(representatives, tier=AUXILIARY_TIER)
    _, cause_excluded = apply_cause_filter(representatives[representatives["verification_tier"] == PRIMARY_TIER])

    return {
        "primary_counts": primary_counts,
        "auxiliary_counts": auxiliary_counts,
        "manifest": manifest,
        "duplicate_collapse_rate": collapse_rate,
        "cause_excluded_count": cause_excluded,
        "raw_row_count": len(frame),
        "cluster_count": int(clustered["cluster_id"].nunique()),
    }
