"""
Diagnostics for the fire_risk_fusion count model.

poisson_binary_consistency is the check named in the project plan: fires
cluster within a county-day (one escaped burn spawns several records; one
weather episode hits several counties), which makes a naive Poisson count
head overconfident. Comparing its implied P(Y>=1) against an independently
fit binary head catches that before it reaches a promotion gate.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import statsmodels.api as sm

MAX_DECILE_GAP_THRESHOLD = 0.05
PEARSON_DISPERSION_THRESHOLD = 1.25


def poisson_binary_consistency(
    y: np.ndarray,
    lam: np.ndarray,
    n_deciles: int = 10,
    count_model_params: int = 1,
) -> Dict:
    """
    y: observed county-day counts.
    lam: the count model's fitted rate (lambda) for each row - same length as y.
    count_model_params: number of parameters the count model used (for the
        Pearson dispersion degrees-of-freedom correction). Defaults to 1
        (intercept-only / offset-only model).

    Fits an independent binary head - GLM(1{y>=1} ~ 1, offset=log(lam),
    family=Binomial(link=cloglog)) - and compares its implied P(Y>=1)
    against the count model's 1 - exp(-lam). A cloglog binary model with
    offset=log(lam) and a fitted intercept of exactly 0 would reproduce
    1 - exp(-lam) exactly; a nonzero intercept or large per-decile gaps
    mean the binary data disagrees with what the Poisson rate implies -
    the overdispersion signal that triggers a negative-binomial count
    family instead.

    Returns mean_abs_gap, max_decile_gap, pearson_dispersion,
    binary_recalibration_intercept, and recommended_count_family
    ("poisson" or "negative_binomial").
    """
    y = np.asarray(y, dtype="float64")
    lam = np.asarray(lam, dtype="float64")
    if y.shape != lam.shape:
        raise ValueError("y and lam must be the same shape")
    if len(y) < n_deciles:
        raise ValueError(f"need at least {n_deciles} rows for {n_deciles}-decile binning, got {len(y)}")

    lam_safe = np.clip(lam, 1e-10, None)
    p_count = 1.0 - np.exp(-lam_safe)
    indicator = (y >= 1).astype("float64")

    offset = np.log(lam_safe)
    exog = np.ones((len(lam_safe), 1))
    binary_model = sm.GLM(
        indicator, exog, offset=offset,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    )
    binary_result = binary_model.fit()
    intercept = float(binary_result.params[0])
    p_binary = 1.0 - np.exp(-lam_safe * np.exp(intercept))

    abs_gap = np.abs(p_count - p_binary)
    mean_abs_gap = float(np.mean(abs_gap))

    order = np.argsort(lam_safe)
    decile_indices = np.array_split(order, n_deciles)
    decile_gaps = [
        abs(float(np.mean(p_count[idx])) - float(np.mean(p_binary[idx])))
        for idx in decile_indices if len(idx) > 0
    ]
    max_decile_gap = float(np.max(decile_gaps))

    pearson_resid = (y - lam_safe) / np.sqrt(lam_safe)
    df = max(1, len(y) - count_model_params)
    pearson_dispersion = float(np.sum(pearson_resid ** 2) / df)

    overdispersed = max_decile_gap > MAX_DECILE_GAP_THRESHOLD or pearson_dispersion > PEARSON_DISPERSION_THRESHOLD
    recommended_count_family = "negative_binomial" if overdispersed else "poisson"

    return {
        "mean_abs_gap": mean_abs_gap,
        "max_decile_gap": max_decile_gap,
        "pearson_dispersion": pearson_dispersion,
        "binary_recalibration_intercept": intercept,
        "recommended_count_family": recommended_count_family,
    }


def calibration_diagnostics(y: np.ndarray, lam: np.ndarray, n_bins: int = 5) -> Dict:
    """
    calibration_slope: the coefficient on logit(p_count) in a logistic
    regression of 1{y>=1} on it. 1.0 means the count model's implied
    P(Y>=1) is exactly as confident as the data supports; <1 means
    overconfident (predicted probabilities too extreme), >1 underconfident.

    reliability_max_bin_gap: p_count vs. the observed event frequency,
    grouped into n_bins EQUAL-COUNT bins (not equal-width - rare-event
    probability mass sits near zero, so equal-width bins would be mostly
    empty at the high end) - the largest |predicted - observed| gap across
    bins.
    """
    y = np.asarray(y, dtype="float64")
    lam = np.asarray(lam, dtype="float64")
    p_count = 1.0 - np.exp(-np.clip(lam, 1e-10, None))
    indicator = (y >= 1).astype("float64")

    logit_p = np.log(p_count / (1.0 - p_count))
    exog = sm.add_constant(logit_p)
    calibration_model = sm.GLM(indicator, exog, family=sm.families.Binomial())
    calibration_result = calibration_model.fit()
    slope = float(calibration_result.params[1])

    order = np.argsort(p_count)
    bins = np.array_split(order, n_bins)
    bin_gaps = [abs(float(np.mean(p_count[idx])) - float(np.mean(indicator[idx]))) for idx in bins if len(idx) > 0]

    return {"calibration_slope": slope, "reliability_max_bin_gap": float(np.max(bin_gaps))}
