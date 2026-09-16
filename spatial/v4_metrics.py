"""Fuel-moisture, category, and probability metrics for V4."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error

from spatial.rule_contract import category
from spatial.station_contract import basic_metrics

MPS_TO_KNOTS = 1.9438444924406


def categories(fm, rh, wind_ms):
    return np.asarray([category(f, r, w * MPS_TO_KNOTS) for f, r, w in zip(fm, rh, wind_ms)], dtype=object)


def category_metrics(actual_fm, actual_rh, actual_wind, predicted_fm, forecast_rh, forecast_wind):
    actual = categories(actual_fm, actual_rh, actual_wind)
    predicted = categories(predicted_fm, forecast_rh, forecast_wind)
    mask = np.asarray([(a is not None and p is not None) for a, p in zip(actual, predicted)])
    if not mask.any(): return {"samples": 0, "supported": False}
    actual, predicted = actual[mask].astype(int), predicted[mask].astype(int)
    high_actual, high_predicted = actual >= 2, predicted >= 2
    tp = int(np.sum(high_actual & high_predicted)); fn = int(np.sum(high_actual & ~high_predicted))
    fp = int(np.sum(~high_actual & high_predicted))
    return {"samples": int(len(actual)), "supported": int(high_actual.sum()) >= 100,
            "confusion_matrix": confusion_matrix(actual, predicted, labels=range(5)).tolist(),
            "elevated_support": int(high_actual.sum()),
            "elevated_recall": tp / (tp + fn) if tp + fn else None,
            "elevated_false_alarm_ratio": fp / (tp + fp) if tp + fp else None,
            "elevated_precision": tp / (tp + fp) if tp + fp else None,
            "macro_f1": float(f1_score(actual, predicted, labels=range(5), average="macro", zero_division=0)),
            "over_one_category_fraction": float(np.mean(np.abs(actual - predicted) > 1)),
            "critical_recall": float(np.mean(predicted[actual >= 3] >= 3)) if np.any(actual >= 3) else None}


def probability_metrics(actual_high, probability, bins=10):
    actual, probability = np.asarray(actual_high, bool), np.clip(np.asarray(probability, float), 0, 1)
    if not len(actual): return {"samples": 0, "brier": None, "ece": None}
    edges = np.linspace(0, 1, bins + 1); ece = 0.0; reliability = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if selected.any():
            observed, predicted = float(actual[selected].mean()), float(probability[selected].mean())
            ece += selected.mean() * abs(observed - predicted)
            reliability.append({"low": low, "high": high, "count": int(selected.sum()),
                                "predicted": predicted, "observed": observed})
    climatology = np.full(len(actual), actual.mean())
    brier = mean_squared_error(actual, probability); reference = mean_squared_error(actual, climatology)
    return {"samples": int(len(actual)), "brier": float(brier), "ece": float(ece),
            "brier_skill": float(1 - brier / reference) if reference else None,
            "reliability": reliability}


def basic_and_tail(actual, predicted):
    report = basic_metrics(actual, predicted); error = np.abs(np.asarray(actual) - np.asarray(predicted))
    report["tail_error"] = {"p90": float(np.quantile(error, .9)), "p95": float(np.quantile(error, .95)),
                            "over_5_fraction": float(np.mean(error > 5))}
    return report
