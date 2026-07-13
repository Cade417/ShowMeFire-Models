"""Export only the leakage-safe HRRR student and require ONNX parity."""
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
from spatial.model import StudentExport, TeacherStudentSpatialModel
from spatial.static_inputs import load_static_inputs


def _normalize(values, contract):
    return ((values - np.asarray(contract["mean"])) / np.asarray(contract["std"])).astype("float32")


def export(checkpoint_path: Path, sample_path: Path, static_bundle: Path, atol=1e-4):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False); contract = dict(checkpoint["static_contract"])
    if contract["bundle_sha256"] != load_static_inputs(static_bundle, contract["feature_set"])[2]["bundle_sha256"]:
        raise ValueError("checkpoint/static bundle mismatch")
    scont, scat, _ = load_static_inputs(static_bundle, contract["feature_set"])
    full = TeacherStudentSpatialModel(checkpoint["static_continuous_channels"], checkpoint["category_sizes"],
        checkpoint.get("hidden_channels", 32), checkpoint["embedding_dims"])
    full.load_state_dict(checkpoint["state_dict"]); model = StudentExport(full).eval()
    with np.load(sample_path) as item:
        inputs_np = {
            "antecedent_rtma": _normalize(item["antecedent_rtma"], checkpoint["normalizers"]["antecedent"])[None],
            "hrrr_forecast": _normalize(item["hrrr_forecast"], checkpoint["normalizers"]["hrrr"])[None],
            "current_fm_state": item["current_fm_state"][None].astype("float32"),
            "static_continuous": scont[None].astype("float32"), "static_categorical": scat[None].astype("int64"),
            "physics_trajectory": item["physics_trajectory"][None].astype("float32")}
    names = list(inputs_np); inputs = tuple(torch.from_numpy(inputs_np[name]) for name in names)
    output_path = checkpoint_path.with_suffix(".onnx")
    torch.onnx.export(model, inputs, output_path, input_names=names, output_names=["output_quantiles"], opset_version=18,
                      dynamic_axes={name: {0: "batch"} for name in [*names, "output_quantiles"]})
    release_contract = {**contract, "teacher_used_for_training": bool(checkpoint["teacher_used_for_training"]),
        "teacher_exported": False, "antecedent_length": checkpoint["antecedent_length"],
        "allowed_missing_antecedent": checkpoint["allowed_missing_antecedent"], "hrrr_leads": checkpoint["hrrr_leads"],
        "normalizers": {key: {part: np.asarray(value[part]).reshape(-1).tolist() for part in ("mean", "std")}
                        for key, value in checkpoint["normalizers"].items()},
        "inference_inputs": names, "training_only_inputs": ["realized_rtma_future"]}
    document = onnx.load(output_path)
    for key, value in {"static_contract": json.dumps(release_contract), "feature_set": contract["feature_set"],
                       "grid_fingerprint": contract["grid_fingerprint"], "teacher_exported": "false"}.items():
        prop = document.metadata_props.add(); prop.key = key; prop.value = value
    onnx.save(document, output_path)
    feed = {name: tensor.numpy() for name, tensor in zip(names, inputs)}
    with torch.no_grad(): reference = model(*inputs).numpy()
    actual = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"]).run(None, feed)[0]
    difference = float(np.max(np.abs(reference - actual)))
    if difference > atol: raise RuntimeError(f"ONNX parity failed: {difference} > {atol}")
    smoke = checkpoint_path.with_name(checkpoint_path.stem + "_smoke.npz")
    np.savez_compressed(smoke, **feed, expected=reference)
    print(f"{output_path} student-only parity={difference}; smoke={smoke}"); return output_path, smoke


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--static-bundle", required=True, type=Path); parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args(); export(args.checkpoint, args.sample, args.static_bundle, args.atol)
