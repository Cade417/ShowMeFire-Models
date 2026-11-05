import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from detection_dataset import groups as grp, labels, review_labels
from detection_dataset.confidence_v2 import model as cm, train


def _detections(n_clusters=80, per=4, seed=0):
    """Clusters of detections ('fires'); half are real (higher FRP), spread over ~100 days and the Missouri map."""
    rng = np.random.default_rng(seed)
    rows, i = [], 1
    base = pd.Timestamp("2026-03-01T18:00:00Z")
    for c in range(n_clusters):
        real = c % 2 == 0
        lat, lon = 37.0 + (c % 10) * 0.25, -94.0 + (c // 10) * 0.4
        day = base + pd.Timedelta(days=int(c * 100 / n_clusters))
        for k in range(per):
            t = day + pd.Timedelta(minutes=int(rng.integers(0, 120)))
            rows.append({"id": i, "source": "viirs" if k % 2 == 0 else "modis", "external_id": f"e{i}", "satellite": "N",
                         "occurred_at": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "first_seen_at": (t + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "latitude": lat + rng.normal(0, 0.002), "longitude": lon + rng.normal(0, 0.002),
                         "frp": float(rng.gamma(6 if real else 1.5, 3)), "confidence": rng.choice(["n", "h", "l"]),
                         "bright_t7": 330.0 + (15 if real else 0) + rng.normal(0, 3), "bright_t13": 295.0, "pixel_area": 0.14,
                         "solar_zenith_angle": 40.0, "satellite_zenith_angle": 30.0, "daynight": "D",
                         "land_cover": "Trees:60,Cropland:40" if real else "Developed:70,Trees:30",
                         "exclusion_zone_id": np.nan, "recurring_source_id": np.nan,
                         "detection_confidence_pct": float(rng.uniform(40, 55)), "incident_id": np.nan,
                         "verification_tier": "unverified", "cause_category": "unknown", "status": "pending", "county_fips": ""})
            i += 1
    return pd.DataFrame(rows)


def _export(det, reviewed_frac=1.0, protected_every=0):
    ids = det["id"].tolist()
    cluster_real = {int(r.id): ((int(r.id) - 1) // 4) % 2 == 0 for r in det.itertuples()}
    chosen = ids[: int(len(ids) * reviewed_frac)]
    outcomes = pd.DataFrame({
        "id": range(1, len(chosen) + 1), "event_id": chosen, "target_version": "veg-v1",
        "outcome": ["vegetation_fire" if cluster_real[e] else "not_vegetation_fire" for e in chosen],
        "reason": "", "evidence": "", "reviewer": "r1", "adjudication_status": "single", "reviewed_at": "2026-06-01"})
    selection = pd.DataFrame({"id": range(1, len(chosen) + 1), "event_id": chosen, "stream": "random", "bucket": "",
                              "sampling_version": "random-v1", "population_n": len(ids),
                              "protected": [1 if protected_every and (e % protected_every == 0) else 0 for e in chosen],
                              "selected_at": "2026-06-01"})
    return {"manifest": {"generated_at": "test"}, "detections": det, "incidents": pd.DataFrame({"id": [], "merged_into_id": []}),
            "feedback": pd.DataFrame({"incident_id": [], "classification": [], "status": [], "reviewed_at": []}),
            "user_reports": pd.DataFrame({"latitude": [], "longitude": [], "occurred_at": [], "status": []}),
            "review_outcomes": outcomes, "review_selection": selection,
            "recurring_sources": pd.DataFrame(columns=["id", "latitude", "longitude", "detection_count", "status", "reviewed_at"]),
            "feature_snapshots": pd.DataFrame(), "score_history": pd.DataFrame()}


def _neris():
    return pd.DataFrame([(37.0, -93.0, pd.Timestamp("2026-03-02"), 1, 1, 10, "NEITHER", "A")] * 3,
                        columns=["lat", "lon", "call_date", "n_records", "unit_response", "duration", "mutual_aid", "dept"])


# ---------------------------------------------------------------- groups

def test_groups_link_nearby_same_fire_and_keep_far_apart_fires_separate():
    det = pd.DataFrame({"latitude": [38.0, 38.001, 38.5, 38.0], "longitude": [-92.0, -92.0, -92.0, -92.0],
                        "occurred_at": ["2026-05-01T10:00:00Z", "2026-05-02T10:00:00Z", "2026-05-01T10:00:00Z", "2026-07-01T10:00:00Z"]})
    g = grp.assign_groups(det)["group_id"].tolist()
    assert g[0] == g[1] and len({g[0], g[2], g[3]}) == 3


def test_persistent_site_pulls_its_visits_into_one_group():
    days = pd.date_range("2026-05-01", periods=6, freq="10D")
    det = pd.DataFrame({"latitude": 38.0, "longitude": -92.0, "occurred_at": [d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in days]})
    out = grp.assign_groups(det)
    assert out["group_id"].nunique() == 1 and out["site_id"].notna().all()


def test_time_holdout_drops_groups_that_straddle_the_cut():
    t = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01", "2026-03-01", "2026-03-02", "2026-04-01"], utc=True))
    g = pd.Series([1, 1, 2, 3, 3, 4])
    tr, te, dropped = grp.time_holdout(t, g, holdout_frac=0.3)
    assert not (tr & te).any()
    # the cut falls between the two March days, so group 3 (both days) straddles it and is dropped from both sides
    assert set(g[te]) == {4} and set(g[tr]) == {1, 2} and dropped == 2
    far = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-02-01", "2026-03-01", "2026-03-01", "2026-04-01"], utc=True))
    tr2, te2, dropped2 = grp.time_holdout(far, pd.Series([1, 1, 2, 3, 3, 4]), holdout_frac=0.3)
    assert not (tr2 & te2).any() and dropped2 == 0


# ---------------------------------------------------------------- labels

def test_review_labels_carry_provenance():
    det = _detections(4)
    out = review_labels.build_label_v2(det, _export(det), _neris())
    assert (out["label_provenance"] == "review").all()
    assert out.loc[out["id"] == 1, "label_v2"].item() == 1.0   # cluster 0 is real
    assert out.loc[out["id"] == 5, "label_v2"].item() == 0.0   # cluster 1 is not


def test_neris_is_positive_only_and_a_disagreeing_review_becomes_a_conflict():
    det = _detections(2)
    det[["latitude", "longitude"]] = [[37.0, -93.0]] * len(det)
    det["occurred_at"] = "2026-03-02T15:00:00Z"
    out = review_labels.build_label_v2(det, _export(det, reviewed_frac=0.0), _neris())
    assert (out["label_provenance"] == "neris_outdoor_unverified_type").all() and (out["label_v2"] == 1).all()
    export = _export(det)
    export["review_outcomes"]["outcome"] = "not_vegetation_fire"
    out = review_labels.build_label_v2(det, export, _neris())
    assert (out["label_conflict"] == 1).all() and out["label_v2"].isna().all()  # reviewer says no, NERIS says yes: left for review


def test_cannot_determine_is_not_a_label():
    det = _detections(2)
    export = _export(det)
    export["review_outcomes"]["outcome"] = "cannot_determine"
    out = review_labels.build_label_v2(det, export, _neris())
    assert out["label_v2"].isna().all() and (out["review_state"] == "cannot_determine").all()
    assert review_labels.label_summary(out)["cannot_determine"] == len(det)


# ---------------------------------------------------------------- model + bundle parity

def test_json_bundle_scores_exactly_like_the_fitted_model_including_missing_and_log1p():
    rng = np.random.default_rng(1)
    n = 400
    X = pd.DataFrame({"frp_log": rng.normal(2, 1, n), "n_1km_24h": rng.poisson(3, n).astype(float), "lc_tree": rng.uniform(0, 1, n)})
    X.loc[rng.choice(n, 60, replace=False), "lc_tree"] = np.nan
    y = (X["frp_log"] + 0.3 * X["n_1km_24h"] + rng.normal(0, 1, n) > 2.5).astype(int)
    fit = cm.LogRegFit(list(X.columns)).fit(X, y)
    cal = cm.fit_platt(fit.logit(X), y)
    bundle = json.loads(json.dumps(fit.to_bundle("t", ["viirs"], "schema", cal)))
    api = cm.bundle_module()
    assert "n_1km_24h" in bundle["log1p"] and bundle["missing_indicators"] == ["lc_tree"]
    expected = cm.apply_platt(cal, fit.logit(X))
    got = [api.score_bundle(bundle, {k: (None if pd.isna(v) else float(v)) for k, v in row.items()})[0] for row in X.to_dict("records")]
    assert np.allclose(got, expected, atol=1e-9)


# ---------------------------------------------------------------- end to end

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DETECTION_CONFIDENCE_V2_DIR", tmp_path / "v2")
    monkeypatch.setattr(paths, "REPORTS_DIR", tmp_path / "reports")
    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(train, "append_experiment", lambda *a, **k: None)
    return tmp_path


def _run(monkeypatch, export, **kw):
    monkeypatch.setattr(labels, "load_export", lambda d=None: export)
    monkeypatch.setattr(labels, "load_neris", lambda p=None: _neris())
    return train.run(register=False, **kw)


def test_gate_fits_nothing_without_human_negatives(sandbox, monkeypatch):
    det = _detections(20)
    export = _export(det)
    export["review_outcomes"] = export["review_outcomes"][export["review_outcomes"]["outcome"] == "vegetation_fire"]
    out = _run(monkeypatch, export)
    assert out["status"] == "insufficient_labels"
    assert not list((sandbox / "v2" / "bundles").glob("*.json"))  # nothing fitted, nothing exported
    assert "NERIS" in out["message"]


def test_end_to_end_with_reviewed_negatives_protects_holdout_and_writes_a_bundle(sandbox, monkeypatch):
    export = _export(_detections(80), protected_every=5)
    out = _run(monkeypatch, export, min_per_class=30, min_groups=10)
    assert out["status"] == "evaluated"
    v = out["variants"]["reviewed_only"]
    cv = v["grouped_cv"]["logreg_calibrated"]
    assert cv["average_precision"] > 0.7 and cv["n_groups"] >= 10  # the synthetic signal is learnable
    assert v["protected"]["rows"] > 0 and "logreg_calibrated" in v["protected"]
    assert v["grouped_cv"]["prevalence_only"]["auc"] == 0.5
    assert v["n"] + v["protected"]["rows"] <= out["frame"]["targets"]  # protected rows are not in the fitting set
    bundles = list((sandbox / "v2" / "bundles").glob("*.json"))
    assert len(bundles) == 1
    bundle = json.loads(bundles[0].read_text())
    assert bundle["supported_sources"] == ["viirs", "modis"] and bundle["calibrator"] is not None
    assert list((sandbox / "reports").glob("detection_confidence_v2_*.md"))


def test_unsupported_sensor_rows_are_never_in_the_training_frame():
    det = _detections(20)
    det.loc[det["id"] <= 8, "source"] = "ngfs"
    frame, _, _, _ = train.build_frame(_export(det), _neris())
    assert set(frame["source"]) <= {"viirs", "modis"}


# ---------------------------------------------------------------- shadow report

def test_shadow_report_compares_models_on_random_stream_reviews_only():
    from detection_dataset import shadow_report
    det = _detections(60)
    export = _export(det, protected_every=5)
    real = {int(r.id): ((int(r.id) - 1) // 4) % 2 == 0 for r in det.itertuples()}
    rng = np.random.default_rng(3)
    rows, k = [], 1
    for eid, is_real in real.items():
        rows.append({"id": k, "event_id": eid, "model_name": "v1_heuristic", "model_version": "formula", "mode": "initial",
                     "score": float(rng.uniform(0.4, 0.55)), "calibrated": 0, "snapshot_id": k, "scored_at": "x"}); k += 1
        rows.append({"id": k, "event_id": eid, "model_name": "veg_conf_logreg", "model_version": "t", "mode": "initial",
                     "score": float(0.8 if is_real else 0.2), "calibrated": 1, "snapshot_id": k, "scored_at": "x"}); k += 1
    export["score_history"] = pd.DataFrame(rows)
    # an enriched-stream review must not count
    extra = pd.DataFrame({"id": [9999], "event_id": [1], "stream": ["enriched"], "bucket": ["masked"], "sampling_version": ["x"],
                          "population_n": [1], "protected": [0], "selected_at": ["x"]})
    export["review_selection"] = pd.concat([export["review_selection"], extra], ignore_index=True)
    rep = shadow_report.build_report(export)
    assert rep["status"] == "reported" and rep["reviewed_random_stream"] == len(det)
    good = rep["models"]["veg_conf_logreg@t"]["random_stream"]
    assert good["auc"] == 1.0 and rep["models"]["v1_heuristic@formula"]["random_stream"]["auc"] < 0.7
    assert "v1_cross_calibrated" in rep["models"]["v1_heuristic@formula"]
    assert "Shadow scoring report" in shadow_report.markdown(rep)


def test_shadow_report_is_honest_when_there_is_nothing_yet():
    from detection_dataset import shadow_report
    export = _export(_detections(4), reviewed_frac=0.0)
    export["score_history"] = pd.DataFrame()
    rep = shadow_report.build_report(export)
    assert rep["status"] == "nothing_to_report"
