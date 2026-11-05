"""Regularised logistic regression for vegetation-fire confidence, fitted by hand around sklearn so every preprocessing
step is explicit and exportable as a plain JSON bundle (scored without sklearn by api/services/detection_bundle.py).

Steps, all fitted on the training rows only: log1p of skewed count features -> training median for missing values (plus a
missing-indicator for each feature that had gaps) -> standardise -> L2 logistic regression. Probabilities from this model
alone are not assumed calibrated; a sigmoid over the logit is fitted separately (fit_platt).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from detection_dataset import labels

BUNDLE_SCHEMA = "veg-conf-model-v1"


def load_api_module(name: str, filename: str):
    """Load an api/services module by path (the training repo must not import the API as a package)."""
    path = labels._api_root() / "services" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def features_module():
    return load_api_module("detection_features_v2", "detection_features_v2.py")


def bundle_module():
    return load_api_module("detection_bundle", "detection_bundle.py")


def log1p_features(names: list[str]) -> list[str]:
    """Counts and other heavy-tailed non-negative inputs (documented transform)."""
    return [n for n in names if n.startswith("n_") or n in ("distinct_days_1km_365d", "max_other_frp_3h", "recurring_source_detections")]


class LogRegFit:
    def __init__(self, features: list[str], C: float = 0.5, seed: int = 0):
        self.features, self.C, self.seed = list(features), C, seed

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.features].astype(float).copy()
        for f in self.log1p:
            out[f] = np.log1p(out[f].clip(lower=0))
        return out

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        P = self._prep(X)
        missing = P[self.indicators].isna().astype(float).to_numpy() if self.indicators else np.empty((len(P), 0))
        filled = P.fillna(pd.Series(self.impute)).to_numpy()
        return np.hstack([filled, missing])

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "LogRegFit":
        from sklearn.linear_model import LogisticRegression
        self.log1p = log1p_features(self.features)
        P = self._prep(X)
        self.impute = {f: (float(P[f].median()) if P[f].notna().any() else 0.0) for f in self.features}
        self.indicators = [f for f in self.features if P[f].isna().any()]
        D = self._design(X)
        self.mean = D.mean(axis=0)
        self.scale = D.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        self.clf = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed)
        self.clf.fit((D - self.mean) / self.scale, np.asarray(y, dtype=int), sample_weight=sample_weight)
        return self

    def logit(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.decision_function((self._design(X) - self.mean) / self.scale)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(X)))

    def to_bundle(self, version: str, supported_sources: list[str], feature_schema: str, calibrator: dict | None,
                  metadata: dict | None = None) -> dict:
        return {
            "schema": BUNDLE_SCHEMA, "model_name": "veg_conf_logreg", "version": version, "feature_schema": feature_schema,
            "features": self.features, "log1p": self.log1p, "impute": self.impute, "missing_indicators": self.indicators,
            "scaler": {"mean": self.mean.tolist(), "scale": self.scale.tolist()},
            "coef": self.clf.coef_[0].tolist(), "intercept": float(self.clf.intercept_[0]),
            "calibrator": calibrator, "supported_sources": supported_sources, "metadata": metadata or {},
        }


def fit_platt(logits: np.ndarray, y, sample_weight=None) -> dict | None:
    """Sigmoid calibrator over a model's logit: p = sigmoid(a * logit + b). None when it cannot be fitted."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2 or len(y) < 10:
        return None
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(np.asarray(logits, dtype=float).reshape(-1, 1), y, sample_weight=sample_weight)
    return {"type": "sigmoid", "a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}


def apply_platt(cal: dict | None, logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    if cal:
        z = cal["a"] * z + cal["b"]
    return 1.0 / (1.0 + np.exp(-z))
