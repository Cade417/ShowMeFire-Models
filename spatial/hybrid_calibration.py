"""Bias and conformal interval calibration shared by fit and inference."""
from __future__ import annotations

import numpy as np


def fit_calibration(actual, quantiles, target_coverage=0.8, bias_clip=1.5):
    actual, quantiles = np.asarray(actual), np.asarray(quantiles)
    bias_offset = float(np.clip(np.mean(actual - quantiles[:, 1]), -bias_clip, bias_clip))
    shifted = _bias_adjusted_quantiles(quantiles, bias_offset)
    scores = np.maximum.reduce((shifted[:, 0] - actual, actual - shifted[:, 2], np.zeros(len(actual))))
    expansion = float(np.quantile(scores, target_coverage, method="higher"))
    return {"bias_offset": bias_offset, "conformal_expansion": expansion,
            "target_interval_coverage": target_coverage, "bias_clip": bias_clip}


def apply_calibration(quantiles, calibration):
    result = _bias_adjusted_quantiles(
        np.asarray(quantiles), float(calibration["bias_offset"])
    )
    result[:, 0] -= float(calibration["conformal_expansion"])
    result[:, 2] += float(calibration["conformal_expansion"])
    return result


def _bias_adjusted_quantiles(quantiles, bias_offset):
    """Shift P50 while retaining non-negative distances to its interval bounds."""
    quantiles = np.asarray(quantiles)
    median = quantiles[:, 1] + bias_offset
    lower_distance = np.maximum(quantiles[:, 1] - quantiles[:, 0], 0.0)
    upper_distance = np.maximum(quantiles[:, 2] - quantiles[:, 1], 0.0)
    return np.column_stack((median - lower_distance, median, median + upper_distance))
