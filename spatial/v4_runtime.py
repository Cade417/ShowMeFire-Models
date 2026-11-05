"""Load and score a local, non-registered V4 bundle."""
from __future__ import annotations

import json

import numpy as np
import torch
import xgboost as xgb

from spatial.station_contract import sha256_file
from spatial.v4_features import add_v4_features
from spatial.v4_guard import sequence_weights
from spatial.v4_training import load_residual, predict, predict_base, sequence_arrays
from spatial.v4_features import FEATURES
from spatial.rule_contract import RULE_SPEC_SHA256
from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION


def load_bundle(directory, require_calibration=False, device=None):
    contract=json.loads((directory/"contract.json").read_text());base_path=directory/"base_xgboost.json";residual_path=directory/"guarded_gru.pt"
    guard_path=directory/"lead_guard.json"
    expected={base_path:contract["base_model_sha256"],residual_path:contract["residual_model_sha256"],guard_path:contract["lead_guard_sha256"]}
    for path,digest in expected.items():
        if not path.exists() or sha256_file(path)!=digest:raise RuntimeError(f"V4 bundle checksum mismatch: {path.name}")
    base=xgb.XGBRegressor();base.load_model(base_path);checkpoint=torch.load(residual_path,map_location="cpu",weights_only=False)
    if contract.get("features")!=FEATURES or checkpoint.get("features")!=FEATURES:
        raise RuntimeError("V4 feature contract mismatch")
    if contract.get("rule_spec_sha256")!=RULE_SPEC_SHA256:
        raise RuntimeError("V4 rule contract mismatch")
    if (contract.get("precipitation_contract_version") != PRECIPITATION_CONTRACT_VERSION or
            contract.get("precipitation_contract_sha256") != PRECIPITATION_CONTRACT_SHA256):
        raise RuntimeError("V4 precipitation contract mismatch")
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    runtime={"contract":contract,"base":base,"checkpoint":checkpoint,"model":load_residual(checkpoint,device),
             "guard":json.loads(guard_path.read_text()),"device":device}
    calibration_path=directory/"calibration.json"
    if calibration_path.exists():runtime["calibration"]=json.loads(calibration_path.read_text())
    elif require_calibration:raise RuntimeError("V4 calibration asset missing")
    return runtime


def score(runtime, frame):
    base_prediction=predict_base(runtime["base"],frame,runtime["contract"]["physics_variant"])
    prepared=add_v4_features(frame,base_prediction,runtime["contract"]["physics_variant"])
    arrays,indices,keys=sequence_arrays(prepared);weights=sequence_weights(arrays[4],runtime["guard"])
    quantiles=predict(runtime["model"],arrays,runtime["checkpoint"]["mean"],runtime["checkpoint"]["std"],runtime["device"],weights)
    return prepared,arrays,indices,keys,quantiles
