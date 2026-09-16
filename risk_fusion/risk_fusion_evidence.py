"""
Evaluation metrics and paired-bootstrap comparison for fire_risk_fusion
count models, mirroring spatial/v5_evidence.py's approach (paired block
bootstrap over episodes, not rows) but with probabilistic count metrics
instead of v5's mae/rmse/absolute_bias, which don't apply to a count model.

Both metrics here are "lower is better" (proper scoring rules / deviance),
matching the sign convention of every *_lower_than_* check in the project
plan's promotion-policy design.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Dict

import numpy as np
from scipy.special import gammaln

POLICY_VERSION = "risk-fusion-promotion-policy-v1"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 811


def policy_sha256() -> str:
    """
    Hashes the policy identity (version + bootstrap sampling parameters) -
    mirrors spatial/v5_evidence.py's policy_sha256() exactly. A change to
    any of these invalidates every prior evaluation report that cited the
    old hash, which is the point: registration refuses a report whose
    policy_sha256 doesn't match this function's current output.
    """
    payload = {"version": POLICY_VERSION, "bootstrap_samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def log_score(y: np.ndarray, lam: np.ndarray) -> float:
    """
    Mean Poisson negative log-likelihood: -log P(y | lam) per row,
    averaged. Lower is better. Uses the full Poisson pmf (including the
    log-factorial normalizing term) so this is comparable in an absolute
    sense across models, not just as a ranking.
    """
    y = np.asarray(y, dtype="float64")
    lam = np.clip(np.asarray(lam, dtype="float64"), 1e-10, None)
    nll = lam - y * np.log(lam) + gammaln(y + 1.0)
    return float(np.mean(nll))


def poisson_deviance(y: np.ndarray, lam: np.ndarray) -> float:
    """
    Mean Poisson deviance: 2 * mean(y*log(y/lam) - (y - lam)), with the
    y*log(y/lam) term defined as 0 when y=0 (its analytic limit). Lower
    is better; 0 is a perfect fit.
    """
    y = np.asarray(y, dtype="float64")
    lam = np.clip(np.asarray(lam, dtype="float64"), 1e-10, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_log_ratio = np.where(y > 0, y * np.log(y / lam), 0.0)
    deviance_per_row = 2.0 * (y_log_ratio - (y - lam))
    return float(np.mean(deviance_per_row))


def paired_block_bootstrap(
    y: np.ndarray,
    lam_candidate: np.ndarray,
    lam_baseline: np.ndarray,
    block_ids: np.ndarray,
    metric_fn: Callable = log_score,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict:
    """
    Resamples whole blocks (e.g. episode_id) with replacement - not
    individual rows - since rows within a block are autocorrelated and
    row-level bootstrap would understate the real uncertainty. Reports
    the fraction of bootstrap draws where the candidate model's metric is
    strictly lower (better) than the baseline's, plus the mean delta.
    """
    y = np.asarray(y, dtype="float64")
    lam_candidate = np.asarray(lam_candidate, dtype="float64")
    lam_baseline = np.asarray(lam_baseline, dtype="float64")
    block_ids = np.asarray(block_ids)

    unique_blocks = np.unique(block_ids)
    if len(unique_blocks) < 2:
        raise ValueError("paired_block_bootstrap needs at least 2 distinct blocks to resample")

    rows_by_block = {block: np.flatnonzero(block_ids == block) for block in unique_blocks}
    rng = np.random.default_rng(seed)

    deltas = np.empty(samples, dtype="float64")
    for i in range(samples):
        drawn_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([rows_by_block[block] for block in drawn_blocks])
        deltas[i] = metric_fn(y[idx], lam_candidate[idx]) - metric_fn(y[idx], lam_baseline[idx])

    point_estimate = metric_fn(y, lam_candidate) - metric_fn(y, lam_baseline)
    return {
        "point_estimate_delta": float(point_estimate),
        "probability_candidate_better": float(np.mean(deltas < 0)),
        "mean_bootstrap_delta": float(np.mean(deltas)),
        "samples": samples,
    }
