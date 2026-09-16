"""Paired statistical evidence and safety gates for V5."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

POLICY_VERSION = "v5-promotion-policy-v2"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 417


def _metric(actual, prediction, name):
    error = np.asarray(prediction, float) - np.asarray(actual, float)
    if name == "mae":
        return float(np.mean(np.abs(error)))
    if name == "rmse":
        return float(np.sqrt(np.mean(error ** 2)))
    if name == "absolute_bias":
        return abs(float(np.mean(error)))
    raise ValueError(f"unsupported paired metric: {name}")


def paired_block_bootstrap(frame, metric="mae", *, mask=None, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """Compare predictions while resampling complete forecast-initialization blocks."""
    selected = frame if mask is None else frame.loc[np.asarray(mask, bool)]
    selected = selected.dropna(subset=["actual_fm", "candidate_fm", "incumbent_fm", "bootstrap_block"])
    if selected.empty or selected.bootstrap_block.nunique() < 2:
        return None
    actual = selected.actual_fm.to_numpy(float)
    candidate = selected.candidate_fm.to_numpy(float)
    incumbent = selected.incumbent_fm.to_numpy(float)
    table = pd.DataFrame({
        "block": selected.bootstrap_block.astype(str),
        "candidate_error": candidate - actual,
        "incumbent_error": incumbent - actual,
    })
    if metric == "mae":
        table["candidate_value"] = table.candidate_error.abs()
        table["incumbent_value"] = table.incumbent_error.abs()
    elif metric == "rmse":
        table["candidate_value"] = table.candidate_error ** 2
        table["incumbent_value"] = table.incumbent_error ** 2
    elif metric == "absolute_bias":
        table["candidate_value"] = table.candidate_error
        table["incumbent_value"] = table.incumbent_error
    else:
        raise ValueError(f"unsupported paired metric: {metric}")
    grouped = table.groupby("block", sort=True).agg(
        candidate_sum=("candidate_value", "sum"), incumbent_sum=("incumbent_value", "sum"), rows=("candidate_value", "size")
    )
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(grouped), size=(samples, len(grouped)))
    rows = grouped.rows.to_numpy(float)[draw].sum(axis=1)
    candidate_draw = grouped.candidate_sum.to_numpy(float)[draw].sum(axis=1) / rows
    incumbent_draw = grouped.incumbent_sum.to_numpy(float)[draw].sum(axis=1) / rows
    if metric == "rmse":
        candidate_draw, incumbent_draw = np.sqrt(candidate_draw), np.sqrt(incumbent_draw)
    elif metric == "absolute_bias":
        candidate_draw, incumbent_draw = np.abs(candidate_draw), np.abs(incumbent_draw)
    delta = candidate_draw - incumbent_draw
    incumbent_point = _metric(actual, incumbent, metric)
    candidate_point = _metric(actual, candidate, metric)
    relative = delta / max(incumbent_point, 1e-12)
    return {
        "metric": metric,
        "rows": int(len(selected)),
        "blocks": int(len(grouped)),
        "candidate": candidate_point,
        "incumbent": incumbent_point,
        "delta": candidate_point - incumbent_point,
        "relative_delta": (candidate_point - incumbent_point) / max(incumbent_point, 1e-12),
        "probability_candidate_better": float(np.mean(delta < 0)),
        "delta_ci95": [float(np.quantile(delta, .025)), float(np.quantile(delta, .975))],
        "relative_delta_ci95": [float(np.quantile(relative, .025)), float(np.quantile(relative, .975))],
        "one_sided_95_relative_upper": float(np.quantile(relative, .95)),
        "samples": int(samples),
        "seed": int(seed),
    }


def _category_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, int), np.asarray(predicted, int)
    elevated, predicted_elevated = actual >= 2, predicted >= 2
    tp = int(np.sum(elevated & predicted_elevated)); fn = int(np.sum(elevated & ~predicted_elevated))
    fp = int(np.sum(~elevated & predicted_elevated))
    f1 = []
    for category in range(5):
        cat_actual, cat_prediction = actual == category, predicted == category
        cat_tp = int(np.sum(cat_actual & cat_prediction)); cat_fp = int(np.sum(~cat_actual & cat_prediction))
        cat_fn = int(np.sum(cat_actual & ~cat_prediction)); denominator = 2 * cat_tp + cat_fp + cat_fn
        f1.append(2 * cat_tp / denominator if denominator else 0.0)
    return {
        "elevated_support": int(elevated.sum()),
        "elevated_recall": tp / (tp + fn) if tp + fn else None,
        "elevated_false_alarm_ratio": fp / (tp + fp) if tp + fp else None,
        "macro_f1": float(np.mean(f1)),
        "over_one_category_fraction": float(np.mean(np.abs(actual - predicted) > 1)),
    }


def evaluate_policy(frame, *, probability_required=.90, minimum_days=None, early_checkpoint=False):
    required = {"bootstrap_block", "actual_fm", "candidate_fm", "incumbent_fm", "p10", "p50", "p90",
                "actual_category", "candidate_category", "incumbent_category", "summer", "critical", "rain_event"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"paired evidence columns missing: {missing}")
    overall = paired_block_bootstrap(frame, "mae")
    rmse = paired_block_bootstrap(frame, "rmse")
    bias = paired_block_bootstrap(frame, "absolute_bias")
    summer = paired_block_bootstrap(frame, "mae", mask=frame.summer)
    critical = paired_block_bootstrap(frame, "mae", mask=frame.critical)
    candidate_categories = _category_metrics(frame.actual_category, frame.candidate_category)
    incumbent_categories = _category_metrics(frame.actual_category, frame.incumbent_category)
    coverage = float(np.mean((frame.actual_fm >= frame.p10) & (frame.actual_fm <= frame.p90)))
    ordering = int(np.sum((frame.p10 > frame.p50) | (frame.p50 > frame.p90)))
    critical_mask = frame.critical.to_numpy(bool)
    candidate_fnr = float(np.mean(frame.candidate_fm.to_numpy()[critical_mask] > 6)) if critical_mask.any() else None
    incumbent_fnr = float(np.mean(frame.incumbent_fm.to_numpy()[critical_mask] > 6)) if critical_mask.any() else None
    support = {
        "rows": int(len(frame)), "days": int(frame.bootstrap_block.astype(str).nunique()),
        "summer": int(np.sum(frame.summer)), "rain_event": int(np.sum(frame.rain_event)),
        "critical": int(np.sum(frame.critical)), "elevated": candidate_categories["elevated_support"],
        "elevated_dates": int(frame.loc[frame.actual_category >= 2, "bootstrap_block"].astype(str).nunique()),
    }
    minimum_elevated = 100 if early_checkpoint or minimum_days is None else 30
    minimum_elevated_dates = 2 if early_checkpoint else 1
    prospective = minimum_days is not None
    critical_supported = support["critical"] >= 100
    if prospective:
        # Prospective Critical evidence is conditional: no cases is inconclusive,
        # while any observed cases must not show a point regression.
        critical_safe = True if support["critical"] == 0 else critical is not None and critical["delta"] <= 0
        critical_fnr_safe = True if support["critical"] == 0 else candidate_fnr <= incumbent_fnr + .02
    else:
        critical_safe = critical_supported and critical is not None and critical["delta"] <= 0
        critical_fnr_safe = critical_supported and candidate_fnr is not None and candidate_fnr <= incumbent_fnr + .02
    checks = {
        "paired_mae_lower": overall is not None and overall["delta"] < 0,
        "mae_probability": overall is not None and overall["probability_candidate_better"] >= probability_required,
        "mae_noninferiority_1pct": overall is not None and overall["one_sided_95_relative_upper"] <= .01,
        "rmse_no_worse": rmse is not None and rmse["delta"] <= 0,
        "absolute_bias_no_worse": bias is not None and bias["delta"] <= 0,
        "summer_mae_no_worse": summer is not None and summer["delta"] <= 0,
        "critical_mae_no_worse": critical_safe,
        "critical_fnr": critical_fnr_safe,
        "elevated_recall": candidate_categories["elevated_recall"] is not None and candidate_categories["elevated_recall"] >= incumbent_categories["elevated_recall"] - .02,
        "elevated_far": candidate_categories["elevated_false_alarm_ratio"] is not None and candidate_categories["elevated_false_alarm_ratio"] <= incumbent_categories["elevated_false_alarm_ratio"] + .02,
        "macro_f1": candidate_categories["macro_f1"] >= incumbent_categories["macro_f1"] - .01,
        "over_one_category": candidate_categories["over_one_category_fraction"] <= incumbent_categories["over_one_category_fraction"] + .005,
        "interval_coverage": .78 <= coverage <= .82,
        "interval_ordering": ordering == 0,
        "summer_support": support["summer"] >= 1_000 if not prospective else True,
        "rain_support": support["rain_event"] >= 300 if (not prospective or early_checkpoint) else True,
        "critical_support": critical_supported if not prospective else True,
        "elevated_support": support["elevated"] >= minimum_elevated,
        "elevated_dates": support["elevated_dates"] >= minimum_elevated_dates,
    }
    if minimum_days is not None:
        checks["minimum_days"] = support["days"] >= minimum_days
    return {
        "policy_version": POLICY_VERSION, "pass": all(checks.values()), "checks": checks,
        "support": support, "bootstrap": {"mae": overall, "rmse": rmse, "absolute_bias": bias,
                                             "summer_mae": summer, "critical_mae": critical},
        "category_metrics": {"candidate": candidate_categories, "incumbent": incumbent_categories},
        "critical_fnr": {"candidate": candidate_fnr, "incumbent": incumbent_fnr,
                         "supported": critical_supported},
        "interval": {"coverage": coverage, "order_violations": ordering},
    }


def dataframe_sha256(frame):
    columns = sorted(frame.columns)
    value = pd.util.hash_pandas_object(frame[columns].sort_values(columns).reset_index(drop=True), index=False).values.tobytes()
    return hashlib.sha256(value).hexdigest()


def policy_sha256():
    payload = {"version": POLICY_VERSION, "bootstrap_samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
