"""Vegetation-fire confidence candidate (v2): reproducible experiment, not a deployment.

    python -m detection_dataset.confidence_v2.train [--history-csv F ...] [--seed 0] [--no-register]

Reads the API export (detections, reviews, feedback, snapshots), labels with target veg-v1, builds scoring-time features
(the API's own causal builder), groups rows into independent fires/sites, then compares reviewed-only and
NERIS-augmented logistic regression (plus a shallow XGBoost for reference) against v1 and a prevalence baseline, with
grouped cross-validation, a later-time holdout and the protected random-review subset. Fits NOTHING unless human-reviewed
labels cover both outcomes with enough independent groups; it never falls back to NERIS-only. A passing run writes a
JSON bundle and a beta registration; nothing is activated and nothing in the API reads it until it is copied there.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import paths
from detection_dataset import groups as grp, labels, review_labels
from detection_dataset.confidence_v2 import metrics as M
from detection_dataset.confidence_v2.model import (LogRegFit, apply_platt, bundle_module, features_module, fit_platt)
from pipelines.experiment_log import append_experiment

MODEL_TYPE = "detection_confidence_v2"
SUPPORTED = ["viirs", "modis"]
HUMAN = {"review", "feedback_vegetation_unspecified", "admin_tier_cause"}
INITIAL_DELAY_MIN = 15  # same rule as the API's shadow pass
MIN_PER_CLASS = 30
MIN_GROUPS_PER_CLASS = 10
GROUP_WEIGHT_POWER = 0.5  # sample weight divided by group_size ** 0.5 so one big fire/facility cannot dominate


# ------------------------------------------------------------------ data

def merge_history(det: pd.DataFrame, files) -> pd.DataFrame:
    """Add FIRMS history CSV rows that are not in the export (negative ids, unknown arrival time)."""
    if not files:
        return det
    hist = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates("external_id")
    new = hist[~hist["external_id"].isin(det["external_id"])].copy()
    new["id"] = -(pd.RangeIndex(len(new)) + 1)
    new["first_seen_at"] = np.nan
    return pd.concat([det, new], ignore_index=True)


def build_frame(export: dict, neris: pd.DataFrame, history_csv=None, radius_km: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    """(labeled frame with meta + feature columns, all detections, feature names, info)."""
    fm = features_module()
    det = merge_history(export["detections"].copy(), history_csv)
    det = review_labels.build_label_v2(det, export, neris, radius_km).reset_index(drop=True)

    targets = det[det["source"].isin(SUPPORTED) & det["label_v2"].notna()].copy()
    arrival = pd.to_datetime(targets["first_seen_at"], utc=True, errors="coerce")
    targets["as_of"] = (arrival + pd.Timedelta(minutes=INITIAL_DELAY_MIN)).where(arrival.notna())
    rec = export.get("recurring_sources")
    rec = rec[rec["status"] == "confirmed"] if rec is not None and len(rec) else None
    X, avail = fm.build_features(targets, det, rec)

    snap = export.get("feature_snapshots")
    used = 0
    if snap is not None and len(snap):
        first = snap[snap["mode"] == "initial"].sort_values("id").drop_duplicates("event_id").set_index("event_id")
        for pos, eid in zip(targets.index, pd.to_numeric(targets["id"], errors="coerce")):
            if eid in first.index and first.at[eid, "schema_version"] == fm.FEATURE_SCHEMA:
                stored = json.loads(first.at[eid, "features_json"])
                X.loc[pos] = [np.nan if stored.get(f) is None else stored[f] for f in fm.FEATURE_NAMES]
                used += 1
    frame = pd.concat([targets.drop(columns=["as_of"]), X.add_prefix("f_")], axis=1)
    g = grp.assign_groups(frame)
    frame = pd.concat([frame, g], axis=1)
    frame["arrival_known"] = avail["arrival_known"]
    info = {"snapshot_rows_used": used, "targets": len(frame), "history_rows": len(det),
            "unknown_arrival_share": round(float(1 - avail["arrival_known"].mean()), 3) if len(avail) else None}
    return frame, det, list(fm.FEATURE_NAMES), info


def gate(frame: pd.DataFrame, min_per_class: int = MIN_PER_CLASS, min_groups: int = MIN_GROUPS_PER_CLASS) -> dict:
    """Human-reviewed labels (review, feedback, admin tier) must cover both outcomes with enough independent groups."""
    human = frame[frame["label_provenance"].isin(HUMAN) & (frame["protected"] == 0)]
    out = {}
    for name, cls in (("negative", 0.0), ("positive", 1.0)):
        sub = human[human["label_v2"] == cls]
        out[name] = {"rows": int(len(sub)), "groups": int(sub["group_id"].nunique())}
    out["negatives_ok"] = out["negative"]["rows"] >= min_per_class and out["negative"]["groups"] >= min_groups
    out["positives_ok"] = out["positive"]["rows"] >= min_per_class and out["positive"]["groups"] >= min_groups
    out["min_per_class"], out["min_groups"] = min_per_class, min_groups
    return out


# ------------------------------------------------------------------ modelling helpers

def _weights(frame: pd.DataFrame, neris_weight: float) -> np.ndarray:
    base = np.where(frame["label_provenance"] == review_labels.PROVENANCE["neris"], neris_weight, 1.0)
    gs = grp.group_sizes(frame["group_id"]).to_numpy(dtype=float)
    return base / np.power(gs, GROUP_WEIGHT_POWER)


def variant_mask(frame: pd.DataFrame, name: str, neris_weight: float) -> pd.Series:
    human = frame["label_provenance"].isin(HUMAN)
    if name == "reviewed_only":
        return human
    return human | ((frame["label_provenance"] == review_labels.PROVENANCE["neris"]) & (neris_weight > 0))


def _splitter(y, groups, n_splits, seed):
    from sklearn.model_selection import StratifiedGroupKFold
    k = max(2, min(n_splits, int(pd.Series(groups[y == 1]).nunique()), int(pd.Series(groups[y == 0]).nunique())))
    return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)


def oof_logits(X: pd.DataFrame, y, w, groups, features, seed, kind="logreg", n_splits=5):
    """Out-of-fold scores (logit for logreg, probability for xgb) and the fold count, grouped so related rows never straddle."""
    out = np.full(len(y), np.nan)
    folds = 0
    for tr, te in _splitter(y, groups, n_splits, seed).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        if kind == "logreg":
            out[te] = LogRegFit(features, seed=seed).fit(X.iloc[tr], y[tr], w[tr]).logit(X.iloc[te])
        else:
            from xgboost import XGBClassifier
            m = XGBClassifier(max_depth=2, n_estimators=100, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, n_jobs=2, verbosity=0,
                              eval_metric="logloss", random_state=seed)
            m.fit(X.iloc[tr][features], y[tr], sample_weight=w[tr])
            out[te] = m.predict_proba(X.iloc[te][features])[:, 1]
        folds += 1
    return out, folds


def cross_calibrate(scores_logit, y, groups, seed):
    """Calibrated out-of-fold probabilities: each fold's sigmoid is fitted on the other folds' out-of-fold logits."""
    p = np.full(len(y), np.nan)
    ok = ~np.isnan(scores_logit)
    idx = np.flatnonzero(ok)
    if len(idx) < 20:
        return p
    yy, gg = y[idx], groups[idx]
    for tr, te in _splitter(yy, gg, 5, seed).split(idx, yy, gg):
        cal = fit_platt(scores_logit[idx[tr]], yy[tr])
        p[idx[te]] = apply_platt(cal, scores_logit[idx[te]])
    return p


def fit_calibrated(X, y, w, groups, features, seed):
    """(fitted model, sigmoid calibrator from grouped inner out-of-fold logits). Calibrator is None if it cannot be fitted."""
    inner, _ = oof_logits(X, y, w, groups, features, seed, n_splits=3)
    ok = ~np.isnan(inner)
    cal = fit_platt(inner[ok], y[ok]) if ok.sum() >= 20 else None
    return LogRegFit(features, seed=seed).fit(X, y, w), cal


def _v1_logit(frame: pd.DataFrame) -> np.ndarray:
    p = np.clip(pd.to_numeric(frame["detection_confidence_pct"], errors="coerce").to_numpy(float) / 100.0, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _evaluate_sets(name_to_p: dict, y, groups, sources, mask=None) -> dict:
    out = {}
    for name, p in name_to_p.items():
        ok = ~np.isnan(p) if mask is None else (~np.isnan(p) & mask)
        res = M.evaluate(y[ok], p[ok], groups[ok])
        res["scored_rows"] = int(ok.sum())
        if len(set(np.asarray(sources)[ok])) > 1:
            res["by_sensor"] = M.by_subset(y[ok], p[ok], groups[ok], np.asarray(sources)[ok])
        out[name] = res
    return out


# ------------------------------------------------------------------ one variant

def run_variant(frame: pd.DataFrame, features: list[str], name: str, neris_weight: float, seed: int, holdout_frac: float) -> dict:
    fcols = [f"f_{f}" for f in features]
    sel = variant_mask(frame, name, neris_weight)
    dev = frame[sel & (frame["protected"] == 0) & (frame["label_conflict"] == 0)].reset_index(drop=True)
    y = dev["label_v2"].to_numpy(dtype=int)
    g = dev["group_id"].to_numpy()
    src = dev["source"].to_numpy()
    w = _weights(dev, neris_weight)
    X = dev[fcols].rename(columns=lambda c: c[2:])
    out: dict = {"variant": name, "neris_weight": neris_weight, "n": int(len(dev)), "positives": int(y.sum()),
                 "groups": int(len(np.unique(g))), "by_provenance": dev["label_provenance"].value_counts().to_dict()}
    if len(np.unique(y)) < 2 or len(np.unique(g)) < 5:
        out["status"] = "not enough rows/groups to evaluate"
        return out

    # grouped cross-validation (unseen fires and sites)
    lg, folds = oof_logits(X, y, w, g, features, seed, "logreg")
    xg, _ = oof_logits(X, y, w, g, features, seed, "xgb")
    v1 = _v1_logit(dev)
    v1p = cross_calibrate(np.where(np.isnan(v1), np.nan, v1), y, g, seed)
    prevalence = np.full(len(y), y.mean())
    out["grouped_cv"] = {"folds": folds, **_evaluate_sets(
        {"logreg_calibrated": cross_calibrate(lg, y, g, seed), "logreg_raw": apply_platt(None, lg), "xgb_shallow_raw": xg,
         "v1_calibrated": v1p, "prevalence_only": prevalence}, y, g, src)}

    # optimistic contrast: random split lets a site or fire sit on both sides (familiar-site performance)
    from sklearn.model_selection import StratifiedKFold
    k = max(2, min(5, int(min(y.sum(), (1 - y).sum()))))
    rnd = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        rnd[te] = LogRegFit(features, seed=seed).fit(X.iloc[tr], y[tr], w[tr]).predict(X.iloc[te])
    out["random_split_familiar_sites_optimistic"] = _evaluate_sets({"logreg_raw": rnd}, y, g, src)["logreg_raw"]

    # later-time holdout: train on early groups, test on groups that begin after the cut
    tr_m, te_m, dropped = grp.time_holdout(dev["occurred_at"], dev["group_id"], holdout_frac)
    th: dict = {"train_rows": int(tr_m.sum()), "test_rows": int(te_m.sum()), "straddling_rows_dropped": dropped}
    if tr_m.sum() >= 20 and te_m.sum() >= 1 and len(np.unique(y[tr_m])) == 2:
        fit, cal = fit_calibrated(X[tr_m].reset_index(drop=True), y[tr_m], w[tr_m], g[tr_m], features, seed)
        p = apply_platt(cal, fit.logit(X[te_m]))
        v1cal = fit_platt(v1[tr_m][~np.isnan(v1[tr_m])], y[tr_m][~np.isnan(v1[tr_m])])
        th.update(_evaluate_sets({"logreg_calibrated": p, "v1_calibrated": apply_platt(v1cal, v1[te_m]) if v1cal else np.full(te_m.sum(), np.nan),
                                  "prevalence_only": np.full(te_m.sum(), y[tr_m].mean())}, y[te_m], g[te_m], src[te_m]))
    else:
        th["note"] = "not enough rows/outcomes on both sides of the time split"
    out["time_holdout"] = th

    # protected random-review subset: never used to fit, calibrate or choose anything above
    prot = frame[(frame["protected"] == 1) & frame["label_provenance"].isin(HUMAN) & (frame["label_conflict"] == 0)].reset_index(drop=True)
    pr: dict = {"rows": int(len(prot))}
    if len(prot) and len(np.unique(prot["label_v2"])) == 2 and len(np.unique(y)) == 2:
        fit, cal = fit_calibrated(X, y, w, g, features, seed)
        py = prot["label_v2"].to_numpy(dtype=int)
        pg = prot["group_id"].to_numpy()
        pX = prot[fcols].rename(columns=lambda c: c[2:])
        v1cal = fit_platt(v1[~np.isnan(v1)], y[~np.isnan(v1)])
        pv1 = _v1_logit(prot)
        pr.update(_evaluate_sets({"logreg_calibrated": apply_platt(cal, fit.logit(pX)),
                                  "v1_calibrated": apply_platt(v1cal, pv1) if v1cal else np.full(len(prot), np.nan),
                                  "prevalence_only": np.full(len(prot), y.mean())}, py, pg, prot["source"].to_numpy()))
        pr["note"] = "prevalence in the review sample, not the live stream, unless the subset came from the random stream"
    else:
        pr["note"] = "no protected reviewed subset with both outcomes yet"
    out["protected"] = pr
    out["status"] = "evaluated"
    return out


# ------------------------------------------------------------------ run

def run(history_csv=None, seed: int = 0, holdout_frac: float = 0.2, neris_weights=(0.25, 0.5, 1.0), register: bool = True,
        min_per_class: int = MIN_PER_CLASS, min_groups: int = MIN_GROUPS_PER_CLASS, neris_path=None, export_dir=None) -> dict:
    export = labels.load_export(export_dir)
    neris = labels.load_neris(neris_path)
    frame, det, features, info = build_frame(export, neris, history_csv)
    summary = review_labels.label_summary(det)
    g = gate(frame, min_per_class, min_groups)
    report: dict = {"generated": datetime.now().isoformat(timespec="seconds"), "target_version": review_labels.TARGET_VERSION,
                    "feature_schema": features_module().FEATURE_SCHEMA, "features": features, "labels": summary,
                    "gate": g, "frame": info, "seed": seed,
                    "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}}
    if not g["negatives_ok"]:
        report["status"] = "insufficient_labels"
        report["message"] = ("Not enough human-reviewed NEGATIVE detections across independent groups "
                             f"(have {g['negative']['rows']} rows / {g['negative']['groups']} groups, need {min_per_class} / {min_groups}). "
                             "Nothing was fitted. NERIS matches are positives only and are never used on their own.")
        return _finish(report, None)

    variants = {"reviewed_only": run_variant(frame, features, "reviewed_only", 0.0, seed, holdout_frac)} if g["positives_ok"] else {}
    for w in neris_weights:
        variants[f"reviewed_plus_neris@{w}"] = run_variant(frame, features, "reviewed_plus_neris", w, seed, holdout_frac)
    report["variants"] = variants
    report["status"] = "evaluated"

    chosen_name = "reviewed_only" if "reviewed_only" in variants else f"reviewed_plus_neris@{neris_weights[len(neris_weights) // 2]}"
    chosen_w = 0.0 if chosen_name == "reviewed_only" else float(chosen_name.split("@")[1])
    sel = variant_mask(frame, chosen_name.split("@")[0], chosen_w)
    dev = frame[sel & (frame["protected"] == 0) & (frame["label_conflict"] == 0)].reset_index(drop=True)
    fcols = [f"f_{f}" for f in features]
    X = dev[fcols].rename(columns=lambda c: c[2:])
    fit, cal = fit_calibrated(X, dev["label_v2"].to_numpy(dtype=int), _weights(dev, chosen_w), dev["group_id"].to_numpy(), features, seed)
    version = datetime.now().strftime("%Y%m%d-%H%M")
    bundle = fit.to_bundle(version, SUPPORTED, features_module().FEATURE_SCHEMA, cal, metadata={
        "variant": chosen_name, "target_version": review_labels.TARGET_VERSION, "rows": int(len(dev)), "seed": seed,
        "label_provenance": dev["label_provenance"].value_counts().to_dict(),
        "export_generated_at": export["manifest"].get("generated_at"), "calibration": "sigmoid over logit, grouped inner OOF",
        "note": "candidate for SHADOW scoring only; not validated for operational use"})
    report["candidate"] = {"variant": chosen_name, "version": version, "calibrated": bool(cal)}
    return _finish(report, bundle, register)


def _finish(report: dict, bundle: dict | None, register: bool = False) -> dict:
    out_dir = paths.DETECTION_CONFIDENCE_V2_DIR
    (out_dir / "bundles").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    if bundle is not None:
        bundle_path = out_dir / "bundles" / f"veg_conf_model_{bundle['version']}.json"
        bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        report["candidate"]["bundle"] = str(bundle_path)
        if register:
            try:
                from models import versioning
                report["candidate"]["registered_version"] = versioning.register_trained_model(
                    MODEL_TYPE, assets={"model": str(bundle_path)}, channel="beta",
                    metadata={"target_version": report["target_version"], "feature_schema": report["feature_schema"],
                              "shadow_required": True, "supported_sources": SUPPORTED})
            except Exception as exc:  # registry problems must not lose the run
                report["candidate"]["register_error"] = str(exc)
    (out_dir / f"confidence_v2_report_{stamp}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    paths.REPORTS_DIR.joinpath(f"detection_confidence_v2_{stamp}.md").write_text(_markdown(report), encoding="utf-8")
    metrics = {"status": report["status"], "targets": report["frame"]["targets"]}
    best = _headline(report)
    if best:
        metrics.update(best)
    append_experiment(MODEL_TYPE, {k: v for k, v in metrics.items() if isinstance(v, (int, float))}, params={"seed": report["seed"]})
    print(json.dumps({k: report[k] for k in ("status", "gate", "labels") if k in report}, indent=2, default=str))
    return report


def _headline(report: dict) -> dict:
    for name, v in (report.get("variants") or {}).items():
        cv = (v.get("grouped_cv") or {}).get("logreg_calibrated") or {}
        if cv.get("average_precision") is not None:
            return {"cv_average_precision": cv["average_precision"], "cv_auc": cv.get("auc", 0.0)}
    return {}


def _fmt(m: dict) -> str:
    if not m:
        return "n/a"
    if "note" in m:
        return m["note"] + f" (n={m.get('n')})"
    ci = f" [{m['average_precision_ci95'][0]}-{m['average_precision_ci95'][1]}]" if "average_precision_ci95" in m else ""
    return f"AP {m.get('average_precision')}{ci}, AUC {m.get('auc')}, Brier {m.get('brier')}, log loss {m.get('log_loss')} (n={m['n']}, pos={m['positives']}, groups={m.get('n_groups')})"


def _markdown(r: dict) -> str:
    lines = [f"# Vegetation-fire confidence v2 experiment ({r['generated']})", "",
             f"Target `{r['target_version']}`, feature schema `{r['feature_schema']}`, seed {r['seed']}. "
             "Experimental: shadow evaluation only; nothing here is a validated probability.", "",
             f"## Status: {r['status']}", ""]
    if r.get("message"):
        lines += [r["message"], ""]
    g = r["gate"]
    lines += ["## Human-reviewed label gate", "",
              f"- negatives: {g['negative']['rows']} rows / {g['negative']['groups']} groups (need {g['min_per_class']} / {g['min_groups']}) -> {'ok' if g['negatives_ok'] else 'NOT ENOUGH'}",
              f"- positives: {g['positive']['rows']} rows / {g['positive']['groups']} groups -> {'ok' if g['positives_ok'] else 'not enough (reviewed-only variant skipped)'}", "",
              "## Labels", "", "```json", json.dumps(r["labels"], indent=2), "```", "",
              f"Frame: {r['frame']}", ""]
    for name, v in (r.get("variants") or {}).items():
        lines += [f"## Variant {name}", "", f"rows {v['n']}, positives {v['positives']}, groups {v['groups']}, provenance {v['by_provenance']}", ""]
        if v.get("status") != "evaluated":
            lines += [v.get("status", ""), ""]
            continue
        lines.append("Grouped cross-validation (unseen fires and sites):")
        for k, m in v["grouped_cv"].items():
            if isinstance(m, dict):
                lines.append(f"- {k}: {_fmt(m)}")
        lines += ["", f"Random split, familiar sites (optimistic contrast): {_fmt(v['random_split_familiar_sites_optimistic'])}", "",
                  f"Later-time holdout ({v['time_holdout'].get('train_rows')} train / {v['time_holdout'].get('test_rows')} test rows):"]
        for k, m in v["time_holdout"].items():
            if isinstance(m, dict) and "n" in m:
                lines.append(f"- {k}: {_fmt(m)}")
        lines += ["", f"Protected reviewed subset ({v['protected'].get('rows')} rows): {v['protected'].get('note', '')}"]
        for k, m in v["protected"].items():
            if isinstance(m, dict) and "n" in m:
                lines.append(f"- {k}: {_fmt(m)}")
        lines.append("")
        m = (v["grouped_cv"].get("logreg_calibrated") or {})
        if m.get("thresholds"):
            lines += ["Alert trade-off (logistic, grouped CV):", ""] + [
                f"- threshold {t['threshold']}: {t['alerts']} alerts ({t['alert_share']}), precision {t['precision']}, recall {t['recall']}" for t in m["thresholds"]] + [""]
    if r.get("candidate"):
        lines += ["## Candidate artifact", "", json.dumps(r["candidate"], indent=2),
                  "", "To score it in shadow: copy the bundle to `api/detection-confidence-model/models/v2/veg_conf_model.json` and restart. "
                  "No automatic promotion exists.", ""]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-csv", type=Path, nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--neris-weights", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--min-per-class", type=int, default=MIN_PER_CLASS)
    p.add_argument("--min-groups", type=int, default=MIN_GROUPS_PER_CLASS)
    p.add_argument("--neris-path", type=Path, default=None)
    p.add_argument("--export-dir", type=Path, default=None)
    p.add_argument("--no-register", action="store_true")
    a = p.parse_args()
    run(a.history_csv, a.seed, a.holdout_frac, tuple(a.neris_weights), not a.no_register, a.min_per_class, a.min_groups, a.neris_path, a.export_dir)


if __name__ == "__main__":
    main()
