"""Lightweight, dependency-free training-run history.

One JSONL file per model_type under paths.EXPERIMENTS_DIR, one append-only
row per training run. This exists so "did retraining on two more weeks of
data actually help?" has an answer without eyeballing printed metrics across
terminal sessions - not a replacement for the model registry (config.json
already tracks what's stable/beta), just a history of what was tried.
"""
import json
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from models.versioning import get_model_entry


def _log_path(model_type):
    return paths.EXPERIMENTS_DIR / f"{model_type}.jsonl"


def append_experiment(model_type, metrics, beta_version=None, params=None):
    """Append one row describing a completed training run.

    `metrics` should be whatever performance dict was computed for this run
    (e.g. {"mae": ..., "r2_score": ..., "training_samples": ..., "test_samples": ...}).
    `params` is optional (e.g. chosen hyperparameters from a search).
    """
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "beta_version": beta_version,
        "metrics": metrics,
        "params": params or {},
    }
    path = _log_path(model_type)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def read_experiments(model_type):
    path = _log_path(model_type)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compare_to_stable(model_type, new_metrics, metric="mae", lower_is_better=True):
    """Compare a freshly trained candidate's metric against the currently
    stable model's recorded performance (from models/config.json).

    Returns a human-readable line and a bool (True if the candidate is
    better), or (line, None) if there's no stable model yet to compare to.
    """
    stable = get_model_entry(model_type).get("stable")
    if not stable or metric not in (stable.get("performance") or {}):
        return f"No stable {model_type!r} model on record yet - nothing to compare against.", None

    stable_value = stable["performance"][metric]
    new_value = new_metrics.get(metric)
    if new_value is None:
        return f"Candidate has no {metric!r} metric to compare.", None

    better = (new_value < stable_value) if lower_is_better else (new_value > stable_value)
    delta = new_value - stable_value
    pct = (abs(delta) / stable_value * 100) if stable_value else 0.0
    direction = "better" if better else "worse"
    line = (
        f"beta {metric}={new_value:.4f} vs stable {metric}={stable_value:.4f} "
        f"({pct:.1f}% {direction})"
    )
    return line, better
