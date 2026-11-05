"""ML-generated columns. Every supervised column is out-of-fold on labeled rows
(spatial-block GroupKFold), and unlabeled rows get the mean of the fold models,
so no row is ever scored by a model that trained on it."""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SAMPLES = 30  # matches api/detection-confidence-model/config.py MIN_TRAINING_SAMPLES
N_SPLITS = 5
XGB_PARAMS = dict(max_depth=3, learning_rate=0.1, n_estimators=100, subsample=0.9, colsample_bytree=0.8, n_jobs=2, verbosity=0)


# Features that mostly encode *when* a detection happened. Label sources cover different dates
# (e.g. weak negatives only in NERIS's months, feedback only later), so a model can score well
# just by knowing the date. They are excluded from the model inputs and used as a confound check.
TIME_CONFOUNDS = ["doy_sin", "doy_cos", "month", "neris_prior_month_calls"]

# Never model inputs: artifacts of ingestion or of the previous scorer, not properties of the detection.
ALWAYS_EXCLUDED = ["ingest_lag_h", "detection_confidence_pct"]

# Only exist for detections that live in the database (incident clustering, the scorer's weather/static context,
# footprints, zone flags). Detections read from a --history-csv have none of them, so when the dataset mixes the two
# origins a model could "predict" the label just by noticing which origin a row came from. Excluded in that case.
PROVENANCE_ONLY = ["weather_danger_prob", "weather_danger_cat", "incident_size", "incident_span_h", "incident_n_sources",
                   "incident_max_frp", "in_recurring_source", "in_exclusion_zone", "has_footprint", "fuel_model_fbfm40",
                   "canopy_cover_frac"]
MAX_MISSING_GAP = 0.3  # any other feature whose missing-rate differs this much between the two origins is dropped too
PROVENANCE_AUC_WARN = 0.75


def _enough(y) -> bool:
    y = pd.Series(y).dropna()
    return len(y) >= MIN_SAMPLES and y.nunique() >= 2 and y.value_counts().min() >= 3


def _folds(groups, n):
    from sklearn.model_selection import GroupKFold
    return GroupKFold(n_splits=max(2, min(N_SPLITS, len(np.unique(groups)))))


def oof_binary(X: pd.DataFrame, y: pd.Series, w: pd.Series, groups, cols=None):
    """Returns (scores for every row, metrics, importances) or (None, reason, None)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from xgboost import XGBClassifier
    cols = cols or list(X.columns)
    lab = y.notna().to_numpy()
    if not _enough(y):
        return None, {"skipped": f"need >= {MIN_SAMPLES} labels with both classes (have {int(lab.sum())})"}, None
    Xl, yl, wl, gl = X.loc[lab, cols], y[lab].astype(int), w[lab], groups[lab]
    oof = pd.Series(np.nan, index=X.index)
    unlabeled_scores, imps = [], []
    for tr, te in _folds(gl, N_SPLITS).split(Xl, yl, gl):
        if yl.iloc[tr].nunique() < 2:
            continue
        pos, neg = (yl.iloc[tr] == 1).sum(), (yl.iloc[tr] == 0).sum()
        m = XGBClassifier(**XGB_PARAMS, scale_pos_weight=neg / max(pos, 1), eval_metric="logloss")
        m.fit(Xl.iloc[tr], yl.iloc[tr], sample_weight=wl.iloc[tr])
        oof.loc[Xl.index[te]] = m.predict_proba(Xl.iloc[te])[:, 1]
        if (~lab).any():
            unlabeled_scores.append(m.predict_proba(X.loc[~lab, cols])[:, 1])
        imps.append(m.feature_importances_)
    if unlabeled_scores:
        oof.loc[~lab] = np.mean(unlabeled_scores, axis=0)
    scored = oof[lab].notna()
    metrics = {"n_labeled": int(lab.sum()), "n_pos": int((yl == 1).sum()), "n_neg": int((yl == 0).sum()), "folds": len(imps)}
    if scored.sum() >= MIN_SAMPLES and yl[scored.values].nunique() == 2:
        metrics["oof_auc"] = float(roc_auc_score(yl[scored.values], oof[lab][scored], sample_weight=wl[scored.values]))
        metrics["oof_pr_auc"] = float(average_precision_score(yl[scored.values], oof[lab][scored], sample_weight=wl[scored.values]))
    importance = dict(sorted(zip(cols, np.mean(imps, axis=0).tolist()), key=lambda kv: -kv[1])) if imps else {}
    return oof, metrics, importance


def oof_burn_type(X: pd.DataFrame, burn: pd.Series, groups, cols=None):
    """Multi-class (wildfire/prescribed/agricultural) probabilities among real fires."""
    from xgboost import XGBClassifier
    cols = cols or list(X.columns)
    real = burn.isin(["wildfire", "prescribed", "agricultural"])
    classes = sorted(burn[real].unique())
    if real.sum() < MIN_SAMPLES or len(classes) < 2 or burn[real].value_counts().min() < 3:
        return None, {"skipped": f"need >= {MIN_SAMPLES} typed real fires across >=2 burn types (have {int(real.sum())})"}
    code = {c: i for i, c in enumerate(classes)}
    yl, Xl, gl = burn[real].map(code), X.loc[real, cols], groups[real.to_numpy()]
    out = pd.DataFrame(np.nan, index=X.index, columns=[f"ml_p_burn_{c}" for c in classes])
    other_scores = []
    for tr, te in _folds(gl, N_SPLITS).split(Xl, yl, gl):
        if yl.iloc[tr].nunique() < len(classes):
            continue
        m = XGBClassifier(**XGB_PARAMS, objective="multi:softprob", num_class=len(classes))
        m.fit(Xl.iloc[tr], yl.iloc[tr])
        out.loc[Xl.index[te]] = m.predict_proba(Xl.iloc[te])
        if (~real).any():
            other_scores.append(m.predict_proba(X.loc[~real, cols]))
    if other_scores:
        out.loc[~real] = np.mean(other_scores, axis=0)
    return out, {"classes": classes, "n_typed": int(real.sum())}


def unsupervised(X: pd.DataFrame, n_clusters: int = 8, n_components: int = 8, seed: int = 0) -> pd.DataFrame:
    """Cluster id/distance, isolation-forest anomaly, PCA embeddings - fit on every row."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    keep = X.columns[X.notna().any()]
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X[keep]))
    Z = np.nan_to_num(Z)
    out = pd.DataFrame(index=X.index)
    k = max(1, min(n_clusters, len(X) // 5))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    d = km.transform(Z)
    out["ml_cluster_id"] = km.labels_
    out["ml_cluster_dist"] = d.min(axis=1)
    out["ml_anomaly_score"] = -IsolationForest(n_estimators=200, random_state=seed).fit(Z).score_samples(Z)  # higher = more unusual
    nc = max(1, min(n_components, Z.shape[1], len(X)))
    emb = PCA(n_components=nc, random_state=seed).fit_transform(Z)
    for i in range(nc):
        out[f"ml_embed_{i + 1}"] = emb[:, i]
    return out


def neighbor_label_rate(det: pd.DataFrame, y: pd.Series, groups, radius_km: float = 10.0, window_days: int = 14) -> pd.Series:
    """Smoothed label rate among nearby labeled detections in time, computed only from labels in
    *other* spatial blocks (so it can't leak a detection's own block's labels)."""
    from sklearn.neighbors import BallTree
    EARTH_KM = 6371.0088
    lab = y.notna().to_numpy()
    out = pd.Series(np.nan, index=det.index)
    if lab.sum() < 2:
        return out
    pts = np.radians(det[["latitude", "longitude"]].to_numpy(dtype=float))
    days = pd.to_datetime(det["occurred_at"], utc=True, errors="coerce").astype("int64").to_numpy() / 8.64e13
    li = np.flatnonzero(lab)
    tree = BallTree(pts[li], metric="haversine")
    nbrs = tree.query_radius(pts, r=radius_km / EARTH_KM)
    yv = y.to_numpy()
    for i, nb in enumerate(nbrs):
        cand = li[nb]
        cand = cand[(groups[cand] != groups[i]) & (np.abs(days[cand] - days[i]) <= window_days)]
        if len(cand):
            out.iloc[i] = (yv[cand].sum() + 0.5) / (len(cand) + 1.0)  # Laplace-smoothed
    return out


def derive_all(det: pd.DataFrame, X: pd.DataFrame, groups_arr: np.ndarray, feature_groups: dict) -> tuple[pd.DataFrame, dict]:
    """Returns (ml_* columns, report dict). Neris-derived columns are never in X."""
    report: dict = {}
    out = pd.DataFrame(index=det.index)
    y, w = det["label"], det["sample_weight"]
    drop = set(TIME_CONFOUNDS) | set(ALWAYS_EXCLUDED)

    # Mixed origins: some detections come from a --history-csv, which has no incident/weather/fuel fields (see
    # PROVENANCE_ONLY). Model VIIRS/MODIS only then: they are what the CSV covers, so every such row has the same sensor
    # fields. NGFS has no history and different fields, so it is not scored here. Within that group, whether a row is also
    # in the database must not be predictable from the features (positives are enriched there, because the NERIS matches
    # were the rows ingested): origin_auc measures exactly that.
    mixed = "from_history_csv" in det and det["from_history_csv"].fillna(0).nunique() > 1
    train_rows = det["source"].isin(["viirs", "modis"]) if mixed else pd.Series(True, index=det.index)
    if mixed:
        in_db = (pd.to_numeric(det["id"], errors="coerce") > 0).astype(int)
        drop |= set(PROVENANCE_ONLY)
        if in_db[train_rows].nunique() > 1:
            gap = (X[train_rows & (in_db == 0)].isna().mean() - X[train_rows & (in_db == 1)].isna().mean()).abs()
            drop |= set(gap[gap > MAX_MISSING_GAP].index)
            remaining = [c for c in X.columns if c not in drop]
            Xs = X[train_rows]
            _, pm, _ = oof_binary(Xs, in_db[train_rows].astype(float), pd.Series(1.0, index=Xs.index),
                                  groups_arr[train_rows.to_numpy()], cols=remaining)
            report["provenance_check"] = {"dropped": sorted(drop - set(TIME_CONFOUNDS) - set(ALWAYS_EXCLUDED)),
                                          "origin_auc": pm.get("oof_auc"), "n_in_database": int(in_db[train_rows].sum()),
                                          "n_csv_only": int((in_db[train_rows] == 0).sum()), "modelled_sources": ["viirs", "modis"]}
            if (pm.get("oof_auc") or 0) > PROVENANCE_AUC_WARN:
                report["warning_provenance"] = ("the remaining features still reveal whether a detection is in the database, "
                                                "and positives are enriched there, so scores may reflect that rather than fire.")
    cols = [c for c in X.columns if c not in drop]
    report["model_inputs"] = len(cols)
    report["model_rows"] = int(train_rows.sum())
    y = y.where(train_rows)  # labels outside the modelled sources are kept in the dataset but never trained on

    # Positive-unlabeled target: NERIS-matched detections (1) vs detections with no nearby call in counties that report to
    # NERIS (0 = unlabeled, NOT "not a fire"). Anything SMF has a verdict on is left out of it.
    unl = det["neris_unmatched_reporting"].fillna(False).astype(bool) if "neris_unmatched_reporting" in det else pd.Series(False, index=det.index)
    pu_y = pd.Series(np.where(det["label_neris"] == 1, 1.0, np.where(unl & det["label_smf"].isna(), 0.0, np.nan)), index=det.index).where(train_rows)
    ones = pd.Series(1.0, index=det.index)

    # Confound check: how well do date-like features alone separate the labels?
    time_cols = [c for c in TIME_CONFOUNDS if c in X.columns]
    _, base, _ = oof_binary(X, y, w, groups_arr, cols=time_cols)
    report["time_only_baseline"] = base
    _, pu_base, _ = oof_binary(X, pu_y, ones, groups_arr, cols=time_cols)
    report["pu_time_only_baseline"] = pu_base
    if max(base.get("oof_auc", 0), pu_base.get("oof_auc", 0)) > 0.8:
        report["warning"] = ("labels are strongly separable by date alone - label sources cover different periods, "
                             "so scores partly reflect when a detection occurred. Treat OOF scores as optimistic.")

    # Is there signal in the detection itself? Sensor/geometry/land features only: no location, season, time of day or
    # neighbour counts, which can mirror where and when NERIS reports rather than whether a detection is a fire.
    intrinsic = [c for c in cols if feature_groups.get(c) in ("base", "sensor", "land")]
    _, sensor_only, _ = oof_binary(X, y, w, groups_arr, cols=intrinsic)
    report["sensor_only_baseline"] = {**sensor_only, "n_features": len(intrinsic)}
    _, pu_sensor, _ = oof_binary(X, pu_y, ones, groups_arr, cols=intrinsic)
    report["pu_sensor_only_baseline"] = {**pu_sensor, "n_features": len(intrinsic)}

    p, metrics, imp = oof_binary(X, y, w, groups_arr, cols=cols)
    report["p_fire"] = metrics
    report["feature_importance"] = imp
    if p is not None:  # (scores for every row, incl. unlabeled)
        out["ml_p_fire_oof"] = p
        out["ml_scored_by_final"] = y.isna().astype(int)

    bt, meta = oof_burn_type(X, det["burn_type"].where(train_rows), groups_arr, cols=cols)
    report["burn_type"] = meta
    if bt is not None:
        out = out.join(bt)

    # Positive-vs-unlabeled score (target built above)
    pu, pu_metrics, _ = oof_binary(X, pu_y, ones, groups_arr, cols=cols)
    report["pu_neris"] = pu_metrics
    if pu is not None:
        out["ml_pu_score"] = pu

    out = out.join(unsupervised(X[cols]))
    out["ml_neighbor_label_rate"] = neighbor_label_rate(det, y, groups_arr)
    supervised = [c for c in out.columns if c.startswith(("ml_p_", "ml_pu_"))]
    out.loc[~train_rows, supervised] = np.nan  # no score for sources that were not modelled (NGFS when history is mixed in)
    return out, report
