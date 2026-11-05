"""Vegetation-fire labels (target veg-v1) from every evidence source, with provenance and conflicts kept.

Priority is review > feedback/admin tier > NERIS, but a lower source that disagrees with a higher one is a CONFLICT: the
row is left unlabeled and flagged for review rather than quietly taking the most convenient answer. Source weights are
only sample weights for sensitivity analysis; they are not claims about how often a label is right.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from detection_dataset import labels

TARGET_VERSION = "veg-v1"
NERIS_WEIGHT_SWEEP = (0.0, 0.25, 0.5, 1.0)
PROVENANCE = {"review": "review", "smf_feedback": "feedback_vegetation_unspecified", "smf_tier": "admin_tier_cause",
              "neris": "neris_outdoor_unverified_type"}


def review_final(export: dict) -> pd.DataFrame:
    """One row per reviewed event: the latest outcome, whether it is a settled binary label, and why not if not."""
    out = export.get("review_outcomes")
    cols = ["event_id", "review_label", "review_state", "review_reviewers"]
    if out is None or out.empty:
        return pd.DataFrame(columns=cols)
    out = out.sort_values("id")
    rows = []
    for eid, g in out.groupby("event_id"):
        last = g.iloc[-1]
        if last["adjudication_status"] == "conflict":
            state, label = "conflict", np.nan
        elif last["outcome"] == "cannot_determine":
            state, label = "cannot_determine", np.nan
        else:
            state, label = "settled", 1.0 if last["outcome"] == "vegetation_fire" else 0.0
        rows.append({"event_id": int(eid), "review_label": label, "review_state": state, "review_reviewers": int(g["reviewer"].nunique())})
    return pd.DataFrame(rows, columns=cols)


def protected_events(export: dict) -> set:
    sel = export.get("review_selection")
    if sel is None or sel.empty:
        return set()
    s = sel[(sel["stream"] == "random") & (sel["protected"] == 1)]
    return set(s["event_id"].astype(int))


def build_label_v2(det: pd.DataFrame, export: dict, neris: pd.DataFrame, radius_km: float = 2.0,
                   neris_weight: float = 0.5) -> pd.DataFrame:
    """det with label_v2 (1/0/NaN), label_provenance, label_conflict, review_state, weight_v2, protected, stream."""
    det = labels.build_labels(det, export, neris, radius_km).copy()
    rev = review_final(export).set_index("event_id")
    eid = pd.to_numeric(det["id"], errors="coerce")
    det["review_label"] = eid.map(rev["review_label"]) if len(rev) else np.nan
    det["review_state"] = eid.map(rev["review_state"]) if len(rev) else None

    candidates = {"review": det["review_label"], "smf": det["label_smf"], "neris": det["label_neris"]}
    present = pd.DataFrame(candidates)
    disagree = present.notna().sum(axis=1).ge(2) & (present.nunique(axis=1, dropna=True) > 1)
    det["label_conflict"] = disagree.astype(int)

    label = pd.Series(np.nan, index=det.index)
    prov = pd.Series(None, index=det.index, dtype=object)
    weight = pd.Series(0.0, index=det.index)
    take_neris = det["label_neris"].notna()
    take_smf = det["label_smf"].notna()
    take_rev = det["review_label"].notna()
    label[take_neris] = det.loc[take_neris, "label_neris"]
    prov[take_neris] = PROVENANCE["neris"]
    weight[take_neris] = neris_weight
    label[take_smf] = det.loc[take_smf, "label_smf"]
    prov[take_smf] = det.loc[take_smf, "label_smf_source"].map(PROVENANCE).fillna(PROVENANCE["smf_feedback"])
    weight[take_smf] = 1.0
    label[take_rev] = det.loc[take_rev, "review_label"]
    prov[take_rev] = PROVENANCE["review"]
    weight[take_rev] = 1.0
    label[disagree] = np.nan  # keep the conflict visible instead of choosing a side
    prov[disagree] = None
    weight[disagree] = 0.0
    det["label_v2"], det["label_provenance"], det["weight_v2"] = label, prov, weight

    det["protected"] = eid.isin(protected_events(export)).astype(int)
    sel = export.get("review_selection")
    stream = pd.Series(None, index=det.index, dtype=object)
    if sel is not None and len(sel):
        first = sel.sort_values("id").drop_duplicates("event_id").set_index("event_id")["stream"]
        stream = eid.map(first)
    det["review_stream"] = stream
    return det


def label_summary(det: pd.DataFrame) -> dict:
    """Counts by provenance and class, plus the cases that need a human (conflicts, cannot-determine)."""
    lab = det[det["label_v2"].notna()]
    return {
        "by_provenance": {str(k): {"positive": int((g["label_v2"] == 1).sum()), "negative": int((g["label_v2"] == 0).sum())}
                          for k, g in lab.groupby("label_provenance")},
        "conflicts_to_review": int(det["label_conflict"].sum()),
        "cannot_determine": int((det["review_state"] == "cannot_determine").sum()),
        "unresolved_review_conflicts": int((det["review_state"] == "conflict").sum()),
        "protected_labeled": int(((det["protected"] == 1) & det["label_v2"].notna()).sum()),
    }
