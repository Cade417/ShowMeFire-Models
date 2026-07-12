"""Export fixed-step four-input spatial ONNX and require runtime parity."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spatial.model import SpatialQuantileModel
from spatial.static_inputs import load_static_inputs


def export(checkpoint_path: Path, sample_path: Path, static_bundle: Path, atol=1e-4):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False); contract = dict(checkpoint["static_contract"])
    contract["dynamic_mean"] = np.asarray(checkpoint["dynamic_mean"]).reshape(-1).tolist()
    contract["dynamic_std"] = np.asarray(checkpoint["dynamic_std"]).reshape(-1).tolist()
    contract["sequence_steps"] = int(checkpoint["sequence_steps"])
    if contract["bundle_sha256"] != load_static_inputs(static_bundle, contract["feature_set"])[2]["bundle_sha256"]: raise ValueError("checkpoint/static bundle mismatch")
    scont, scat, _ = load_static_inputs(static_bundle, contract["feature_set"])
    model = SpatialQuantileModel(checkpoint["dynamic_channels"], checkpoint["static_continuous_channels"], checkpoint["category_sizes"], embedding_dims=checkpoint["embedding_dims"])
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    with np.load(sample_path) as item:
        dynamic = ((item["dynamic_sequence"] - checkpoint["dynamic_mean"]) / checkpoint["dynamic_std"])[None].astype("float32")
        physics = item["physics"][None].astype("float32")
    inputs = (torch.from_numpy(dynamic), torch.from_numpy(scont[None]), torch.from_numpy(scat[None]), torch.from_numpy(physics))
    output_path = checkpoint_path.with_suffix(".onnx")
    names = ["dynamic_sequence", "static_continuous", "static_categorical", "physics_trajectory"]
    torch.onnx.export(model, inputs, output_path, input_names=names, output_names=["output_quantiles"], opset_version=18,
                      dynamic_axes={name: {0: "batch"} for name in [*names, "output_quantiles"]})
    document = onnx.load(output_path)
    for key, value in {"static_contract": json.dumps(contract), "feature_set": contract["feature_set"], "grid_fingerprint": contract["grid_fingerprint"]}.items():
        item = document.metadata_props.add(); item.key = key; item.value = value
    onnx.save(document, output_path)
    feed = {name: tensor.numpy() for name, tensor in zip(names, inputs)}
    with torch.no_grad(): reference = model(*inputs).numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"]); actual = session.run(None, feed)[0]
    difference = float(np.max(np.abs(reference - actual)))
    if difference > atol: raise RuntimeError(f"ONNX parity failed: {difference} > {atol}")
    smoke = checkpoint_path.with_name(checkpoint_path.stem + "_smoke.npz")
    np.savez_compressed(smoke, **feed, expected=reference)
    print(f"{output_path} parity={difference}; smoke={smoke}"); return output_path, smoke


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--static-bundle", required=True, type=Path); parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args(); export(args.checkpoint, args.sample, args.static_bundle, args.atol)
