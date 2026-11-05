"""Independence groups and protected splits for the vegetation-fire confidence experiments.

Detections are not independent: nearby observations describe the same fire and repeat visits describe the same
facility. A row's group is the connected component of (a) detections within link_km and link_days of each other and
(b) detections sharing a persistent site. Everything in a group stays on one side of every split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_KM = 6371.0088


def _hours(series: pd.Series) -> np.ndarray:
    t = pd.to_datetime(series, utc=True, errors="coerce")
    out = t.astype("int64").to_numpy().astype(float) / 3.6e12
    out[t.isna().to_numpy()] = np.nan
    return out


def assign_groups(det: pd.DataFrame, link_km: float = 2.0, link_days: float = 3.0, site_min_days: int = 5,
                  site_cell_deg: float = 0.02) -> pd.DataFrame:
    """event_id, site_id (NaN unless a persistent site) and group_id for every row, index-aligned.

    A persistent site is a confirmed recurring source, or a ~2 km cell that has detections on >= site_min_days distinct
    days. All thresholds are arguments, none depend on calendar dates."""
    from sklearn.neighbors import BallTree
    n = len(det)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    lat = pd.to_numeric(det["latitude"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(det["longitude"], errors="coerce").to_numpy(float)
    t = _hours(det["occurred_at"])
    ok = ~(np.isnan(lat) | np.isnan(lon) | np.isnan(t))
    idx_ok = np.flatnonzero(ok)
    event_parent = None
    if len(idx_ok):
        tree = BallTree(np.radians(np.c_[lat[idx_ok], lon[idx_ok]]), metric="haversine")
        neigh = tree.query_radius(np.radians(np.c_[lat[idx_ok], lon[idx_ok]]), r=link_km / EARTH_KM)
        for a, nb in enumerate(neigh):
            ia = idx_ok[a]
            for b in nb:
                ib = idx_ok[b]
                if ib > ia and abs(t[ia] - t[ib]) <= link_days * 24.0:
                    union(ia, ib)
    event_root = np.array([find(i) for i in range(n)])

    site = np.full(n, np.nan)
    rec = pd.to_numeric(det.get("recurring_source_id", pd.Series(np.nan, index=det.index)), errors="coerce").to_numpy(float)
    cell = (np.floor(lat / site_cell_deg) * 1_000_000 + np.floor(lon / site_cell_deg)).astype(float)
    cell[~ok] = np.nan
    days = pd.DataFrame({"cell": cell, "day": np.floor(t / 24.0)}).dropna()
    persistent = days.groupby("cell")["day"].nunique()
    persistent = set(persistent[persistent >= site_min_days].index)
    for i in range(n):
        if not np.isnan(rec[i]):
            site[i] = 1e9 + rec[i]
        elif ok[i] and cell[i] in persistent:
            site[i] = cell[i]
    by_site: dict[float, int] = {}
    for i in range(n):
        if not np.isnan(site[i]):
            if site[i] in by_site:
                union(i, by_site[site[i]])
            else:
                by_site[site[i]] = i
    group = np.array([find(i) for i in range(n)])
    return pd.DataFrame({"event_id": event_root, "site_id": site, "group_id": group}, index=det.index)


def time_holdout(times: pd.Series, groups: pd.Series, holdout_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray, int]:
    """(train_mask, test_mask, dropped). Test = groups that begin at or after the cut (the latest holdout_frac of rows by
    acquisition time); train = groups that end before it. Groups that straddle the cut are dropped from both sides so
    no fire or site informs both."""
    t = pd.to_datetime(times, utc=True, errors="coerce")
    cut = t.quantile(1.0 - holdout_frac)
    frame = pd.DataFrame({"t": t, "g": groups.to_numpy()})
    span = frame.groupby("g")["t"].agg(["min", "max"])
    test_groups = set(span.index[span["min"] >= cut])
    train_groups = set(span.index[span["max"] < cut])
    g = frame["g"]
    train, test = g.isin(train_groups).to_numpy(), g.isin(test_groups).to_numpy()
    return train, test, int((~(train | test)).sum())


def group_sizes(groups: pd.Series) -> pd.Series:
    """Rows per group for each row, used to stop one huge fire or facility dominating training."""
    return groups.map(groups.value_counts())
