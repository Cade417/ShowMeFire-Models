"""Checksum-validated runtime for a local V5 experimental bundle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
from spatial.rule_contract import RULE_SPEC_SHA256
from spatial.station_contract import sha256_file
from spatial.v5_features import BASE_FEATURES, FEATURES, SPECIALIST_FEATURES, regime_labels
from spatial.evaluate_baselines import FEATURES as INCUMBENT_FEATURES
from spatial.v5_guard import apply_guard, intervals
from spatial.v5_training import predict_base, predict_incumbent, predict_specialist, prepare


def load_bundle(directory):
    directory = Path(directory); contract = json.loads((directory / "contract.json").read_text())
    paths = {"base": directory / "base_xgboost.json", "specialist": directory / "specialist_xgboost.json",
             "guard": directory / "guard.json", "uncertainty": directory / "uncertainty.json"}
    expected = {"base": "base_model_sha256", "specialist": "specialist_model_sha256",
                "guard": "guard_sha256", "uncertainty": "uncertainty_sha256"}
    for name, path in paths.items():
        if not path.exists() or sha256_file(path) != contract.get(expected[name]):
            raise RuntimeError(f"V5 bundle checksum mismatch: {path.name}")
    expected_base = INCUMBENT_FEATURES if contract.get("base_family") == "incumbent" else BASE_FEATURES
    if contract.get("features") != FEATURES or contract.get("base_features") != expected_base or contract.get("specialist_features") != SPECIALIST_FEATURES:
        raise RuntimeError("V5 feature contract mismatch")
    if contract.get("rule_spec_sha256") != RULE_SPEC_SHA256:
        raise RuntimeError("V5 rule contract mismatch")
    if (contract.get("precipitation_contract_version") != PRECIPITATION_CONTRACT_VERSION or
            contract.get("precipitation_contract_sha256") != PRECIPITATION_CONTRACT_SHA256):
        raise RuntimeError("V5 precipitation contract mismatch")
    base = xgb.XGBRegressor(); base.load_model(paths["base"])
    specialist = xgb.XGBRegressor(); specialist.load_model(paths["specialist"])
    return {"contract": contract, "base": base, "specialist": specialist,
            "guard": json.loads(paths["guard"].read_text()),
            "uncertainty": json.loads(paths["uncertainty"].read_text())}


def score(runtime, frame):
    base = (predict_incumbent(runtime["base"], frame) if runtime["contract"].get("base_family") == "incumbent"
            else predict_base(runtime["base"], frame, runtime["contract"]["physics_variant"]))
    prepared = prepare(frame, base, runtime["contract"]["physics_variant"])
    correction = predict_specialist(runtime["specialist"], prepared); regimes = regime_labels(prepared)
    available = prepared.precip_available.fillna(0).to_numpy() > 0
    prediction, weights, caps, reasons = apply_guard(base, correction, prepared.lead_hour, regimes,
                                                     runtime["guard"], available)
    quantiles = intervals(prediction, regimes, runtime["uncertainty"])
    return prepared, {"base": np.asarray(base), "raw_correction": np.asarray(correction),
                      "prediction": prediction, "intervals": quantiles, "regimes": regimes,
                      "guard_weights": weights, "guard_caps": caps, "guard_reasons": reasons,
                      "fallback": weights == 0}
