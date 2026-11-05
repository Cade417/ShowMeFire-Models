"""Evaluation for the vegetation-fire confidence candidate: ranking, alert trade-offs and probability quality.

No single number is a promotion criterion. Every result carries its n, positives and (where there are enough groups) a
group-resampled interval, because rows from one fire or site are not independent.
"""
from __future__ import annotations

import numpy as np

THRESHOLDS = (0.3, 0.5, 0.7)
MIN_FOR_METRICS = 30
MIN_GROUPS_FOR_CI = 10


def _safe(fn, y, p):
    try:
        return float(fn(y, p))
    except ValueError:
        return float("nan")


def _round(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def reliability(y, p, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()), "mean_predicted": _round(p[m].mean()), "observed_rate": _round(y[m].mean())})
    return out


def evaluate(y, p, groups=None, thresholds=THRESHOLDS, n_boot: int = 200, seed: int = 0) -> dict:
    """y in {0,1}, p in [0,1]. Returns counts, AP/AUC (+ group interval), Brier, log loss, threshold table, reliability."""
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    out: dict = {"n": int(len(y)), "positives": int(y.sum()), "prevalence": _round(y.mean()) if len(y) else None}
    if len(y) < MIN_FOR_METRICS or len(np.unique(y)) < 2:
        out["note"] = f"too few rows to conclude (need >= {MIN_FOR_METRICS} with both outcomes)"
        return out
    out["average_precision"] = _round(_safe(average_precision_score, y, p))
    out["auc"] = _round(_safe(roc_auc_score, y, p))
    out["brier"] = _round(brier_score_loss(y, p))
    out["log_loss"] = _round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    rows = []
    for t in thresholds:
        flagged = p >= t
        tp = int((flagged & (y == 1)).sum())
        rows.append({"threshold": t, "alerts": int(flagged.sum()), "alert_share": _round(flagged.mean()),
                     "precision": _round(tp / flagged.sum()) if flagged.any() else None, "recall": _round(tp / y.sum())})
    out["thresholds"] = rows
    out["reliability"] = reliability(y, p)
    if groups is not None:
        g = np.asarray(groups)
        uniq = np.unique(g)
        out["n_groups"] = int(len(uniq))
        if len(uniq) >= MIN_GROUPS_FOR_CI:
            rng = np.random.default_rng(seed)
            by_group = {u: np.flatnonzero(g == u) for u in uniq}
            aps, aucs = [], []
            for _ in range(n_boot):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                idx = np.concatenate([by_group[u] for u in pick])
                if len(np.unique(y[idx])) < 2:
                    continue
                aps.append(average_precision_score(y[idx], p[idx]))
                aucs.append(roc_auc_score(y[idx], p[idx]))
            if len(aps) >= 20:
                out["average_precision_ci95"] = [_round(np.percentile(aps, 2.5)), _round(np.percentile(aps, 97.5))]
                out["auc_ci95"] = [_round(np.percentile(aucs, 2.5)), _round(np.percentile(aucs, 97.5))]
        else:
            out["ci_note"] = f"fewer than {MIN_GROUPS_FOR_CI} independent groups: no interval"
    return out


def by_subset(y, p, groups, labels_per_row, **kwargs) -> dict:
    """evaluate() for each distinct value of labels_per_row (e.g. sensor)."""
    labels_per_row = np.asarray(labels_per_row)
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    return {str(v): evaluate(y[labels_per_row == v], p[labels_per_row == v], groups[labels_per_row == v], **kwargs)
            for v in np.unique(labels_per_row)}
