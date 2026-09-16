"""Out-of-fold regime/lead guard for bounded V5 residual corrections."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _metrics(actual, prediction):
    error = np.asarray(prediction) - np.asarray(actual)
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error ** 2))), abs(float(np.mean(error)))


def fit_guard(scored: pd.DataFrame, minimum_rows=300):
    guard = {}
    for (regime, lead), group in scored.groupby(["regime", "lead_hour"]):
        key = f"{regime}|{float(lead)}"
        base = _metrics(group.actual, group.base)
        best = None
        if len(group) >= minimum_rows:
            for cap in (0.5, 1.0, 1.5, 2.0):
                bounded = np.clip(group.raw_correction.to_numpy(float), -cap, cap)
                for alpha in np.linspace(0, 1, 21):
                    candidate = group.base.to_numpy(float) + alpha * bounded
                    metrics = _metrics(group.actual, candidate)
                    fold_safe = all(
                        _metrics(part.actual, part.base + alpha * np.clip(part.raw_correction, -cap, cap))[1]
                        <= 1.01 * _metrics(part.actual, part.base)[1]
                        for _, part in group.groupby("fold")
                    )
                    critical = group.actual.to_numpy(float) <= 6
                    critical_safe = not critical.any() or np.mean(np.abs(candidate[critical] - group.actual.to_numpy()[critical])) <= (
                        np.mean(np.abs(group.base.to_numpy()[critical] - group.actual.to_numpy()[critical])) + 0.02
                    )
                    if metrics[0] < base[0] and fold_safe and critical_safe:
                        objective = metrics[0] / max(base[0], 1e-9) + metrics[1] / max(base[1], 1e-9)
                        if best is None or objective < best[0]:
                            best = (objective, float(alpha), float(cap), metrics)
        guard[key] = {
            "weight": 0.0 if best is None else best[1],
            "cap": 0.0 if best is None else best[2],
            "support": int(len(group)),
            "reason": "unsupported_or_not_consistently_better" if best is None else "oof_improvement",
            "base_mae": base[0],
            "guarded_mae": base[0] if best is None else best[3][0],
        }
    # Smooth weights within a regime by reducing a larger neighbor. Zeros stay zero.
    for regime in sorted(scored.regime.unique()):
        leads = sorted(float(value) for value in scored.loc[scored.regime == regime, "lead_hour"].unique())
        for _ in range(len(leads)):
            for pos in range(1, len(leads)):
                left, right = f"{regime}|{leads[pos-1]}", f"{regime}|{leads[pos]}"
                if left not in guard or right not in guard: continue
                lw, rw = guard[left]["weight"], guard[right]["weight"]
                if lw > rw + .25: guard[left]["weight"] = rw + .25
                elif rw > lw + .25: guard[right]["weight"] = lw + .25
    return guard


def apply_guard(base, correction, leads, regimes, guard, available=None):
    base, correction, leads = map(np.asarray, (base, correction, leads))
    available = np.ones(len(base), bool) if available is None else np.asarray(available, bool)
    prediction = base.astype(float).copy(); weights = np.zeros(len(base)); caps = np.zeros(len(base)); reasons = []
    for index, (lead, regime) in enumerate(zip(leads, regimes)):
        record = guard.get(f"{regime}|{float(lead)}", {})
        weight, cap = float(record.get("weight", 0)), float(record.get("cap", 0))
        reason = record.get("reason", "missing_guard")
        if not available[index]: weight, cap, reason = 0.0, 0.0, "unavailable_features"
        weights[index], caps[index] = weight, cap
        prediction[index] += weight * np.clip(correction[index], -cap, cap)
        reasons.append(reason)
    return prediction, weights, caps, reasons


def fit_uncertainty(scored, guard, minimum_rows=300, target=.8):
    prediction, _, _, _ = apply_guard(scored.base, scored.raw_correction, scored.lead_hour, scored.regime, guard)
    errors = np.abs(scored.actual.to_numpy(float) - prediction)
    result = {"target_coverage": target, "global": float(np.quantile(errors, target, method="higher")), "regimes": {}}
    for regime, group in scored.assign(error=errors).groupby("regime"):
        if len(group) >= minimum_rows:
            result["regimes"][regime] = {"half_width": float(np.quantile(group.error, target, method="higher")), "support": int(len(group))}
    return result


def intervals(prediction, regimes, uncertainty):
    widths = np.asarray([uncertainty.get("regimes", {}).get(str(regime), {}).get("half_width", uncertainty["global"]) for regime in regimes])
    prediction = np.asarray(prediction, float)
    return np.column_stack((prediction - widths, prediction, prediction + widths))
