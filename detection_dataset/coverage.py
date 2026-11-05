"""Where does NERIS actually report? County-level coverage map.

A detection with no NERIS call nearby only says "no fire" where the local departments report to NERIS at all.
This counts NERIS calls per county, compares with the detections there, and marks each county reporting or silent.
Silent counties are left out of the positive-vs-unlabeled comparison, and the table shows where NERIS can be trusted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A county counts as reporting with at least this many NERIS calls inside the dataset's NERIS date range.
MIN_COUNTY_CALLS = 10


def assign_county(lat, lon, counties=None) -> pd.Series:
    """County FIPS for each point (NaN outside every county). Boundary points go to the first polygon that holds them."""
    import shapely
    from shapely import STRtree
    if counties is None:
        from risk_fusion.county_geometry import load_counties
        counties = load_counties()
    geoms = [c["geometry_wgs84"] for c in counties]
    tree = STRtree(geoms)
    pts = shapely.points(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    hit_pts, hit_geoms = tree.query(pts, predicate="within")
    fips = np.full(len(pts), None, dtype=object)
    for p, g in sorted(zip(hit_pts, hit_geoms), reverse=True):  # reversed so the first polygon wins
        fips[p] = counties[g]["fips"]
    return pd.Series(fips, index=getattr(lat, "index", None))


def county_coverage(det: pd.DataFrame, neris: pd.DataFrame, counties=None, min_calls: int = MIN_COUNTY_CALLS) -> pd.DataFrame:
    """One row per county: NERIS calls, detections in the NERIS date range, how many matched, and whether it reports.
    det needs latitude, longitude, occurred_local_date, neris_matched."""
    from risk_fusion.county_geometry import load_counties
    counties = counties or load_counties()
    names = {c["fips"]: c["name"] for c in counties}
    neris_fips = assign_county(neris["lat"], neris["lon"], counties)
    calls = neris_fips.value_counts()

    lo, hi = neris["call_date"].min(), neris["call_date"].max()
    in_range = det["occurred_local_date"].between(lo, hi)
    d = det[in_range]
    det_fips = assign_county(d["latitude"], d["longitude"], counties)
    n_det = det_fips.value_counts()
    n_match = det_fips[d["neris_matched"].to_numpy()].value_counts()

    out = pd.DataFrame({"fips": list(names), "county": list(names.values())})
    out["neris_calls"] = out["fips"].map(calls).fillna(0).astype(int)
    out["detections"] = out["fips"].map(n_det).fillna(0).astype(int)
    out["matched"] = out["fips"].map(n_match).fillna(0).astype(int)
    out["match_rate"] = (out["matched"] / out["detections"].where(out["detections"] > 0)).round(4)
    out["reporting"] = out["neris_calls"] >= min_calls
    return out.sort_values("neris_calls", ascending=False).reset_index(drop=True)


def reporting_mask(det: pd.DataFrame, cov: pd.DataFrame, counties=None) -> pd.Series:
    """True for detections inside a county that reports to NERIS."""
    fips = assign_county(det["latitude"], det["longitude"], counties)
    return fips.map(cov.set_index("fips")["reporting"].to_dict()).eq(True).set_axis(det.index)


def summarize(cov: pd.DataFrame) -> dict:
    rep = cov[cov["reporting"]]
    silent = cov[~cov["reporting"]]
    return {
        "counties": int(len(cov)), "reporting": int(len(rep)), "silent": int(len(silent)),
        "min_calls_to_report": MIN_COUNTY_CALLS,
        "detections_in_reporting": int(rep["detections"].sum()), "detections_in_silent": int(silent["detections"].sum()),
        "matched_in_reporting": int(rep["matched"].sum()), "matched_in_silent": int(silent["matched"].sum()),
    }
