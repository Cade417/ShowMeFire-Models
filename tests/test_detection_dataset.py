import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection_dataset import derive, features, labels


def _det(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-01-10T18:00:00Z")
    return pd.DataFrame({
        "id": range(n), "source": rng.choice(["viirs", "modis", "ngfs"], n), "satellite": "N",
        "occurred_at": [(t0 + pd.Timedelta(hours=int(h))).isoformat() for h in rng.integers(0, 24 * 60, n)],
        "latitude": rng.uniform(37, 39, n), "longitude": rng.uniform(-93, -91, n),
        "frp": rng.gamma(2, 3, n), "confidence": rng.choice(["nominal", "high", "low"], n),
        "bright_t7": rng.uniform(300, 360, n), "bright_t13": rng.uniform(280, 310, n), "pixel_area": 0.14,
        "land_cover": "Cropland:80,Trees:20", "daynight": "D", "incident_id": rng.integers(1, 40, n),
        "verification_tier": "unverified", "cause_category": "unknown", "recurring_source_id": np.nan,
        "exclusion_zone_id": np.nan, "first_seen_at": "2026-02-01T00:00:00Z", "weather_danger_category": "low",
    })


def _neris(rows):
    return pd.DataFrame(rows, columns=["lat", "lon", "call_date", "n_records", "unit_response", "duration", "mutual_aid", "dept"])


def test_neris_match_radius_and_day_tolerance():
    det = _det(3)
    det["occurred_local_date"] = pd.Timestamp("2026-01-10")
    det.loc[:, ["latitude", "longitude"]] = [[38.0, -92.0], [38.0, -92.0], [38.0, -92.0]]
    ne = _neris([
        (38.001, -92.0, pd.Timestamp("2026-01-11"), 1, 1, 10, "NEITHER", "A"),   # within 2 km, +1 day -> match
    ])
    m = labels.match_neris(det, ne, radius_km=2, day_window=1)
    assert m["neris_matched"].all() and (m["neris_dt_days"] == 1).all()
    far = _neris([(38.5, -92.0, pd.Timestamp("2026-01-10"), 1, 1, 1, "", "A")])
    assert not labels.match_neris(det, far)["neris_matched"].any()
    late = _neris([(38.0, -92.0, pd.Timestamp("2026-01-13"), 1, 1, 1, "", "A")])
    assert not labels.match_neris(det, late)["neris_matched"].any()


def test_smf_labels_only_not_a_fire_is_negative_and_merges_resolve():
    det = _det(4)
    det["incident_id"] = [1, 2, 3, 9]
    det["verification_tier"] = ["unverified", "unverified", "unverified", "admin_reviewed"]
    det["cause_category"] = ["unknown", "unknown", "unknown", "agricultural"]
    inc = pd.DataFrame({"id": [1, 2, 3, 9], "merged_into_id": [np.nan, 1, np.nan, np.nan]})
    fb = pd.DataFrame({"incident_id": [1, 3], "classification": ["controlled_burn", "not_a_fire"],
                       "status": ["approved", "approved"], "reviewed_at": ["2026-01-02", "2026-01-02"]})
    out = labels.add_smf_labels(det, inc, fb)
    assert list(out["label_smf"]) == [1.0, 1.0, 0.0, 1.0]  # prescribed + ag = real fire; merged inc 2 -> 1
    assert out["burn_type"].tolist() == ["prescribed", "prescribed", "not_a_fire", "agricultural"]


def test_missing_neris_call_is_unlabeled_not_negative():
    det = _det(50)
    det["incident_id"] = np.nan
    det.loc[:9, ["latitude", "longitude"]] = [37.03, -90.0]  # same county as the NERIS calls, but beyond the 2 km match radius
    det.loc[:9, "occurred_at"] = "2026-01-15T18:00:00Z"
    export = {"incidents": pd.DataFrame({"id": [], "merged_into_id": []}),
              "feedback": pd.DataFrame({"incident_id": [], "classification": [], "status": [], "reviewed_at": []})}
    dense = [(37.0, -90.0, pd.Timestamp("2026-01-05") + pd.Timedelta(days=i % 25), 1, 1, 1, "", "A") for i in range(300)]
    dense += [(37.0, -90.0, pd.Timestamp("2026-02-05"), 1, 1, 1, "", "A")] * 3  # sparse month
    out = labels.build_labels(det, export, _neris(dense))
    assert not (out["label"] == 0).any()  # absence of a call never becomes "not a fire"
    in_jan = out["occurred_local_date"].dt.to_period("M").astype(str) == "2026-01"
    assert (out["neris_in_coverage"] == in_jan).all()  # only dense months count as coverage
    here = out.index <= 9
    flagged = out["neris_unmatched_reporting"]
    assert not flagged[~here].any()  # every other county has no NERIS calls: silent, so nothing is said about it
    assert (flagged[here] == (in_jan[here] & ~out.loc[here, "neris_matched"])).all()
    assert flagged.any()


def test_county_coverage_marks_silent_counties():
    from detection_dataset import coverage
    det = _det(20)
    det["occurred_local_date"] = pd.Timestamp("2026-01-10")
    det["neris_matched"] = False
    ne = _neris([(37.0, -90.0, pd.Timestamp("2026-01-10"), 1, 1, 1, "", "A")] * 12)
    cov = coverage.county_coverage(det, ne)
    assert cov["reporting"].sum() == 1 and cov["neris_calls"].max() == 12
    assert cov.loc[~cov["reporting"], "neris_calls"].sum() == 0
    assert cov["detections"].sum() <= 20


def test_date_span_is_data_derived():
    s = pd.Series(pd.to_datetime(["2026-04-10", None, "2026-09-01"]))
    lo, hi = labels.date_span(s)
    assert (str(lo.date()), str(hi.date())) == ("2026-04-10", "2026-09-01")
    assert labels.date_span(pd.Series([], dtype="datetime64[ns]")) is None


def test_user_report_labels_detection_outside_neris_dates():
    det = _det(20)
    det["incident_id"] = np.nan
    det["occurred_at"] = "2026-10-05T18:00:00Z"  # long after NERIS ends
    det.loc[:, "latitude"], det.loc[:, "longitude"] = 38.0, -92.0
    det.loc[0, ["latitude", "longitude"]] = [38.0, -92.0]
    det.loc[1:, "latitude"] = 39.5  # far from the report
    export = {"incidents": pd.DataFrame({"id": [], "merged_into_id": []}),
              "feedback": pd.DataFrame({"incident_id": [], "classification": [], "status": [], "reviewed_at": []}),
              "user_reports": pd.DataFrame({"latitude": [38.01, 38.0], "longitude": [-92.0, -92.0],
                                            "occurred_at": ["2026-10-05T20:00:00Z", "2026-10-05T20:00:00Z"],
                                            "status": ["approved", "rejected"]})}
    neris = _neris([(37.0, -90.0, pd.Timestamp("2026-01-05"), 1, 1, 1, "", "A")] * 400)
    out = labels.build_labels(det, export, neris)
    assert out.loc[0, "label"] == 1.0 and out.loc[0, "label_source"] == "user_report"
    assert out.loc[1:, "label"].isna().all()  # no NERIS coverage in October -> unlabeled
    assert not out["neris_in_coverage"].any()


def test_features_and_ml_columns_end_to_end():
    det = _det(400)
    det["label"] = np.where(det.index % 3 == 0, det["frp"] > 5, np.nan)
    det["label"] = det["label"].astype(float)
    det["sample_weight"] = np.where(det["label"].notna(), 1.0, 0.0)
    det["burn_type"] = np.where(det["label"] == 1, np.where(det.index % 2 == 0, "wildfire", "prescribed"), None)
    det["label_neris"] = np.nan
    det["label_smf"] = det["label"]
    X, groups = features.build_features(det)
    assert not any(c.startswith(("neris_matched", "label")) for c in X.columns)
    ml, report = derive.derive_all(det, X, features.spatial_group(det), groups)
    assert {"ml_cluster_id", "ml_anomaly_score", "ml_embed_1", "ml_neighbor_label_rate"} <= set(ml.columns)
    assert "ml_p_fire_oof" in ml.columns and ml["ml_p_fire_oof"].notna().all()
    assert report["p_fire"]["oof_auc"] > 0.6  # label is a function of frp
