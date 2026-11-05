"""Load the API export + NERIS and attach nullable per-detection labels.

null = no evidence. A 0 is only ever written for an explicit not_a_fire
verdict (label_smf). A detection with no NERIS dispatch nearby stays unlabeled:
a missing report is not a verdict, and some areas never report to NERIS. Those
detections are only flagged (neris_unmatched_reporting) when they sit in a county
that does report, so the positive-vs-unlabeled comparison can use them.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from detection_dataset import coverage

EARTH_KM = 6371.0088
MO_BBOX = (-95.9, 35.9, -88.9, 40.7)  # lon_min, lat_min, lon_max, lat_max (slightly padded)

# A month counts as NERIS coverage when it holds at least this fraction of the
# median monthly call count (relative, so no calendar dates are baked in).
MIN_COVERAGE_FRACTION = 0.25
LOCAL_UTC_OFFSET_H = 5  # NERIS call dates were shifted UTC-5 upstream (see the .shp.xml provenance)


def date_span(dates: pd.Series):
    """(min, max) day of a series, or None when empty. Used to report each label source's own coverage."""
    d = dates.dropna()
    return None if d.empty else (d.min().normalize(), d.max().normalize())


def reports_frame(user_reports: pd.DataFrame) -> pd.DataFrame:
    """User-submitted reports as point evidence (lat, lon, call_date). Rejected/deleted/expired
    reports are dropped; everything else counts as someone saying a fire was there."""
    if user_reports is None or user_reports.empty:
        return pd.DataFrame(columns=["lat", "lon", "call_date"])
    r = user_reports[~user_reports["status"].astype(str).str.lower().isin({"rejected", "deleted", "expired"})]
    t = pd.to_datetime(r["occurred_at"], utc=True, errors="coerce") - pd.Timedelta(hours=LOCAL_UTC_OFFSET_H)
    out = pd.DataFrame({"lat": r["latitude"].astype(float), "lon": r["longitude"].astype(float),
                        "call_date": t.dt.tz_localize(None).dt.normalize()})
    return out.dropna().reset_index(drop=True)


def match_reports(det: pd.DataFrame, reports: pd.DataFrame, radius_km: float = 3.0, day_window: int = 1) -> pd.Series:
    """True where a user report lies within radius_km and +-day_window days of the detection."""
    from sklearn.neighbors import BallTree
    hit = pd.Series(False, index=det.index)
    if reports.empty or det.empty:
        return hit
    tree = BallTree(np.radians(reports[["lat", "lon"]].to_numpy()), metric="haversine")
    nbrs = tree.query_radius(np.radians(det[["latitude", "longitude"]].to_numpy(dtype=float)), r=radius_km / EARTH_KM)
    d = det["occurred_local_date"].to_numpy(dtype="datetime64[D]")
    rd = reports.call_date.to_numpy(dtype="datetime64[D]")
    for i, idx in enumerate(nbrs):
        if len(idx) and not np.isnat(d[i]) and (np.abs((rd[idx] - d[i]).astype(int)) <= day_window).any():
            hit.iloc[i] = True
    return hit

POSITIVE_FEEDBACK = {"confirmed_fire", "controlled_burn"}
BURN_TYPE_BY_FEEDBACK = {"confirmed_fire": "wildfire", "controlled_burn": "prescribed", "not_a_fire": "not_a_fire"}
BURN_TYPE_BY_CAUSE = {"wildfire": "wildfire", "prescribed": "prescribed", "agricultural": "agricultural",
                      "debris_burn": "agricultural", "not_a_fire": "not_a_fire"}

WEIGHTS = {"smf_feedback": 1.0, "smf_tier": 0.6, "user_report": 0.7, "neris_pos": 0.8}
# label_source values: smf_feedback, smf_tier, user_report, neris_pos
REPORT_RADIUS_KM = 3.0


def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            d.update(chunk)
    return d.hexdigest()


def _api_root() -> Path:
    return Path(os.getenv("SMF_API_REPO_ROOT", paths.REPO_ROOT.parent / "api"))


def resolve_db_path(db: str | Path | None = None) -> Path:
    """--db, else SMF_API_DB_PATH (point this at a production copy), else the API repo's local DB."""
    chosen = db or os.getenv("SMF_API_DB_PATH") or _api_root() / "data" / "showmefire.db"
    return Path(chosen)


def refresh_export(db: str | Path | None = None, since: str | None = None, directory: Path | None = None) -> Path:
    """Re-run the API's read-only export straight into SMF_DATA_ROOT/fire_labels so the build
    always works from a fresh snapshot (no manual copy). Returns the manifest path."""
    script = _api_root() / "scripts" / "export_detection_training_data.py"
    if not script.exists():
        raise FileNotFoundError(f"{script} not found - set SMF_API_REPO_ROOT to the api/ repo")
    spec = importlib.util.spec_from_file_location("_smf_export_detection_training", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    db_path = resolve_db_path(db)
    print(f"Refreshing export from {db_path}")
    return module.export(since, db_path, directory or paths.FIRE_LABELS_DIR)


def import_bundle(zip_path: str | Path, directory: Path | None = None) -> Path:
    """Unpack a detection_training_export.zip made on the server into SMF_DATA_ROOT/fire_labels
    and verify every file against its manifest checksum before accepting it."""
    import shutil
    import tempfile
    import zipfile
    directory = directory or paths.FIRE_LABELS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():  # flat files only - refuse paths that could escape the folder
                if Path(name).name != name:
                    raise ValueError(f"unexpected path in bundle: {name}")
            z.extractall(staged)
        load_export(staged)  # raises on a missing file or checksum mismatch
        directory.mkdir(parents=True, exist_ok=True)
        for f in staged.iterdir():
            shutil.copy2(f, directory / f.name)
    print(f"Imported bundle into {directory}")
    return directory / "detection_training_manifest.json"


def load_export(directory: Path | None = None) -> dict:
    """Load the API export, verifying every file against its manifest checksum."""
    directory = directory or paths.FIRE_LABELS_DIR
    manifest_path = directory / "detection_training_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found - run with --refresh to export from the API DB")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = {"manifest": manifest}
    for key, info in manifest["files"].items():
        path = directory / info["filename"]
        if _sha256(path) != info["sha256"]:
            raise ValueError(f"checksum mismatch for {path} - re-export")
        out[key] = pd.read_csv(path, low_memory=False)
    return out


def load_neris(shp: Path | None = None) -> pd.DataFrame:
    """NERIS outdoor-fire points inside Missouri, with a normalised call date.
    Columns: lat, lon, call_date (midnight), n_records, unit_response, duration, mutual_aid, dept."""
    import geopandas as gpd
    shp = shp or next(paths.NERIS_DIR.glob("*.shp"))
    gdf = gpd.read_file(shp)
    out = pd.DataFrame({
        "lon": gdf.geometry.x, "lat": gdf.geometry.y,
        "call_date": pd.to_datetime(gdf["MIN_call_d"], errors="coerce").dt.tz_localize(None).dt.normalize(),
        "n_records": gdf.get("COUNT_neri"), "unit_response": gdf.get("SUM_unit_r"),
        "duration": gdf.get("MAX_durati"), "mutual_aid": gdf.get("CONCATEN_1"),
        "dept": gdf.get("CONCATENAT"),
    })
    lon0, lat0, lon1, lat1 = MO_BBOX  # drops geocode outliers (~257 points)
    keep = out.lon.between(lon0, lon1) & out.lat.between(lat0, lat1) & out.call_date.notna()
    return out[keep].reset_index(drop=True)


def neris_coverage_months(neris: pd.DataFrame) -> set:
    counts = neris.call_date.dt.to_period("M").value_counts()
    floor = MIN_COVERAGE_FRACTION * counts.median()
    return {str(p) for p, n in counts.items() if n >= floor}


def match_neris(det: pd.DataFrame, neris: pd.DataFrame, radius_km: float = 2.0, day_window: int = 1) -> pd.DataFrame:
    """Nearest NERIS dispatch within radius_km and +-day_window calendar days of
    the detection's local date. NERIS is day-resolution and tz-shifted, hence
    the day tolerance. Returns one row per detection (index-aligned)."""
    from sklearn.neighbors import BallTree
    out = pd.DataFrame(index=det.index, columns=["neris_matched", "neris_distance_km", "neris_dt_days", "neris_n_calls",
                                                   "neris_unit_response", "neris_duration", "neris_mutual_aid", "neris_dept"])
    out["neris_matched"] = False
    out["neris_n_calls"] = 0
    if neris.empty or det.empty:
        return out
    tree = BallTree(np.radians(neris[["lat", "lon"]].to_numpy()), metric="haversine")
    det_pts = np.radians(det[["latitude", "longitude"]].to_numpy())
    det_dates = det["occurred_local_date"].to_numpy(dtype="datetime64[D]")
    neris_dates = neris.call_date.to_numpy(dtype="datetime64[D]")
    neighbours, distances = tree.query_radius(det_pts, r=radius_km / EARTH_KM, return_distance=True)
    for i, (idx, dist) in enumerate(zip(neighbours, distances)):
        if len(idx) == 0 or np.isnat(det_dates[i]):
            continue
        dt = (neris_dates[idx] - det_dates[i]).astype(int)
        ok = np.abs(dt) <= day_window
        if not ok.any():
            continue
        idx, dist, dt = idx[ok], dist[ok] * EARTH_KM, dt[ok]
        best = idx[np.argmin(dist)]
        row = det.index[i]
        out.loc[row, "neris_matched"] = True
        out.loc[row, "neris_distance_km"] = float(dist.min())
        out.loc[row, "neris_dt_days"] = int(dt[np.argmin(dist)])
        out.loc[row, "neris_n_calls"] = int(len(idx))
        out.loc[row, "neris_unit_response"] = neris.at[best, "unit_response"]
        out.loc[row, "neris_duration"] = neris.at[best, "duration"]
        out.loc[row, "neris_mutual_aid"] = neris.at[best, "mutual_aid"]
        out.loc[row, "neris_dept"] = neris.at[best, "dept"]
    return out


def add_smf_labels(det: pd.DataFrame, incidents: pd.DataFrame, feedback: pd.DataFrame) -> pd.DataFrame:
    """label_smf / burn_type from approved feedback on the detection's (merge-resolved)
    incident, falling back to an admin-reviewed tier + non-unknown cause_category."""
    det = det.copy()
    inc = incidents.set_index("id")

    def resolve(i):
        seen = 0
        while pd.notna(i) and i in inc.index and pd.notna(inc.at[i, "merged_into_id"]) and seen < 10:
            i, seen = inc.at[i, "merged_into_id"], seen + 1
        return i

    det["incident_id_resolved"] = det["incident_id"].map(resolve) if "incident_id" in det else np.nan

    approved = feedback[feedback["status"] == "approved"].copy()
    approved["incident_id"] = approved["incident_id"].map(resolve)
    # newest approved verdict per incident wins
    approved = approved.sort_values("reviewed_at").groupby("incident_id").tail(1).set_index("incident_id")
    cls = det["incident_id_resolved"].map(approved["classification"]) if len(approved) else pd.Series(index=det.index, dtype=object)

    label = pd.Series(np.nan, index=det.index, dtype=float)
    label[cls.isin(POSITIVE_FEEDBACK)] = 1.0
    label[cls == "not_a_fire"] = 0.0
    source = pd.Series(np.where(label.notna(), "smf_feedback", None), index=det.index, dtype=object)
    burn = cls.map(BURN_TYPE_BY_FEEDBACK)

    # admin-reviewed tier with an explicit cause (and incident review_decision as a tiebreak)
    tiered = det["verification_tier"].isin(["admin_reviewed", "official_source_confirmed"]) & \
        det["cause_category"].fillna("unknown").ne("unknown") & label.isna()
    cause_burn = det["cause_category"].map(BURN_TYPE_BY_CAUSE)
    label[tiered & cause_burn.notna()] = np.where(cause_burn[tiered & cause_burn.notna()] == "not_a_fire", 0.0, 1.0)
    source[tiered & cause_burn.notna()] = "smf_tier"
    burn = burn.where(burn.notna(), cause_burn.where(tiered))

    det["label_smf"] = label
    det["label_smf_source"] = source
    det["burn_type"] = burn
    return det


def build_labels(det: pd.DataFrame, export: dict, neris: pd.DataFrame, radius_km: float = 2.0) -> pd.DataFrame:
    det = add_smf_labels(det, export["incidents"], export["feedback"])
    det["occurred_dt"] = pd.to_datetime(det["occurred_at"], utc=True, errors="coerce")
    # NERIS dates were shifted to local time (UTC-5 per provenance); mirror that.
    det["occurred_local_date"] = (det["occurred_dt"] - pd.Timedelta(hours=LOCAL_UTC_OFFSET_H)).dt.tz_localize(None).dt.normalize()

    det = det.join(match_neris(det, neris, radius_km))
    # NERIS only speaks for days inside its own date range AND in a dense month.
    covered_months = neris_coverage_months(neris)
    span = date_span(neris["call_date"])
    in_range = det["occurred_local_date"].between(*span) if span else pd.Series(False, index=det.index)
    in_cov = in_range & det["occurred_local_date"].dt.to_period("M").astype(str).isin(covered_months)
    det["neris_in_coverage"] = in_cov

    det["label_neris"] = np.where(det["neris_matched"], 1.0, np.nan)
    # No dispatch nearby is NOT a negative. It is only "unlabeled", and only worth comparing where NERIS reports at all.
    cov = coverage.county_coverage(det, neris)
    det["neris_county_reporting"] = coverage.reporting_mask(det, cov)
    det["neris_unmatched_reporting"] = in_cov & ~det["neris_matched"] & det["neris_county_reporting"]
    det["label_report"] = np.where(match_reports(det, reports_frame(export.get("user_reports")), REPORT_RADIUS_KM), 1.0, np.nan)

    label = det["label_smf"].copy()
    src = det["label_smf_source"].copy()
    for col, name in (("label_report", "user_report"), ("label_neris", "neris_pos")):
        fill = label.isna() & det[col].notna()
        label[fill] = det.loc[fill, col]
        src[fill] = name
    det["label"] = label
    det["label_source"] = src
    det["sample_weight"] = src.map(WEIGHTS).fillna(0.0)
    det["neris_year_month"] = det["occurred_local_date"].dt.to_period("M").astype(str)
    return det
