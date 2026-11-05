"""Out-of-fold lead guard for bounded residual corrections."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_lead_guard(scored, minimum_rows=1000):
    """Fit per-lead blend weights using only OOF development predictions."""
    weights = {}
    for lead, group in scored.groupby("lead_hour"):
        folds = sorted(group.fold.unique()); base_mae = np.mean(np.abs(group.base - group.actual))
        base_rmse = np.sqrt(np.mean((group.base - group.actual) ** 2)); best = (float("inf"), 0.0)
        if len(group) >= minimum_rows:
            for alpha in np.linspace(0, 1, 21):
                prediction = group.base + alpha * (group.residual_p50 - group.base)
                mae = np.mean(np.abs(prediction - group.actual)); rmse = np.sqrt(np.mean((prediction - group.actual) ** 2))
                fold_safe = True
                for fold in folds:
                    part = group[group.fold == fold]; candidate = part.base + alpha * (part.residual_p50 - part.base)
                    fold_safe &= np.sqrt(np.mean((candidate - part.actual) ** 2)) <= 1.01 * np.sqrt(np.mean((part.base - part.actual) ** 2))
                objective = mae / max(base_mae, 1e-9) + rmse / max(base_rmse, 1e-9)
                if fold_safe and mae < base_mae and objective < best[0]: best = (objective, float(alpha))
        weights[float(lead)] = best[1]
    ordered = sorted(weights)
    # Enforce smoothness only by reducing the larger neighbor. This preserves
    # every safety-forced zero rather than resurrecting a harmful correction.
    for _ in range(len(ordered)):
        for index in range(1, len(ordered)):
            left, right = ordered[index-1], ordered[index]
            if weights[left] > weights[right] + .25: weights[left] = weights[right] + .25
            elif weights[right] > weights[left] + .25: weights[right] = weights[left] + .25
    return {str(lead): weights[lead] for lead in ordered}


def sequence_weights(leads, guard):
    return np.vectorize(lambda value: float(guard.get(str(float(value)), 0.0)))(leads).astype("float32")
