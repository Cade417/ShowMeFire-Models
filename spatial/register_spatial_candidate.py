from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from models.versioning import register_trained_model
from spatial.export_onnx import export
from static_features.schema import validate_bundle


def register(bundle: Path, sample: Path):
    ablation_path = paths.REPORTS_DIR / "spatial_ablation.json"; ablation = json.loads(ablation_path.read_text())
    if not ablation["static_candidate_gate"]: raise RuntimeError("static ablation promotion gate did not pass")
    selected = ablation["selected"]; checkpoint = Path(ablation["artifacts"][selected]); bundle_manifest = bundle.with_suffix(".json")
    static = validate_bundle(bundle, bundle_manifest); onnx_path, smoke = export(checkpoint, sample, bundle)
    metrics = ablation["results"][selected]
    if not metrics["temporal"]["quantile_order_valid"]: raise RuntimeError("quantile ordering gate failed")
    version = register_trained_model("fuel_moisture_spatial", performance=metrics, channel="beta", assets={
        "model": onnx_path, "checkpoint": checkpoint,
        "static_bundle": {"path": bundle, "schema_version": static["schema_version"], "grid_fingerprint": static["grid_fingerprint"]},
        "static_manifest": bundle_manifest, "evaluation": ablation_path, "smoke": smoke,
    })
    print(f"Registered fuel_moisture_spatial beta {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--static-bundle", required=True, type=Path); parser.add_argument("--sample", required=True, type=Path)
    args = parser.parse_args(); register(args.static_bundle, args.sample)
