"""Fit the search-selected V3 bundle on the full development partition."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.hybrid_contract import FEATURE_SCHEMA_VERSION, load_or_create_manifest
from spatial.hybrid_features import FEATURES
from spatial.hybrid_training import load_frame, prepare_final_fit, sequence_arrays, train_model
from spatial.station_contract import sha256_file


def main():
    parser = argparse.ArgumentParser(description="Fit selected hybrid V3 on development data")
    parser.add_argument("--dataset", type=Path, default=paths.ALIGNED_DIR / "station_leads.csv")
    parser.add_argument("--manifest", type=Path, default=paths.REPORTS_DIR / "hybrid_v3_split_manifest.json")
    parser.add_argument("--search", type=Path, default=paths.REPORTS_DIR / "hybrid_v3_search.json")
    parser.add_argument("--output-dir", type=Path, default=paths.MODELS_DIR / "hybrid_v3_candidate")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite candidate directory: {args.output_dir}")
    frame = load_frame(args.dataset); manifest = load_or_create_manifest(args.manifest, frame, args.dataset, FEATURES, create=False)
    search = json.loads(args.search.read_text())
    if search.get("dataset_sha256") != manifest["dataset_sha256"] or search.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise SystemExit("Search/split contract mismatch")
    if search.get("holdout_accessed") or search.get("calibration_accessed"):
        raise SystemExit("Search report indicates forbidden partition access")
    print("preparing development OOF base predictions", flush=True)
    prepared, base_model, provenance = prepare_final_fit(frame, manifest["development_runs"])
    config = search["selected_config"]; fixed_epochs = int(search["selected_fixed_epochs"])
    _, checkpoint, report = train_model(sequence_arrays(prepared)[0], None, fixed_epochs=fixed_epochs,
                                         epochs=fixed_epochs, patience=5, seed=417, **config)
    checkpoint.update({"dataset_sha256": manifest["dataset_sha256"], "manifest_sha256": manifest["manifest_sha256"],
                       "split_version": manifest["version"], "feature_schema_version": FEATURE_SCHEMA_VERSION})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.output_dir / "base_xgboost.json"; residual_path = args.output_dir / "residual_gru.pt"
    contract_path = args.output_dir / "contract.json"
    base_model.save_model(base_path); torch.save(checkpoint, residual_path)
    contract = {"status": "fitted_uncalibrated", "dataset_sha256": manifest["dataset_sha256"],
                "manifest_sha256": manifest["manifest_sha256"], "split_version": manifest["version"],
                "feature_schema_version": FEATURE_SCHEMA_VERSION, "features": FEATURES,
                "base_model_sha256": sha256_file(base_path), "residual_model_sha256": sha256_file(residual_path),
                "development_runs": len(manifest["development_runs"]), "crossfit_blocks": 5,
                "crossfit_scored_runs": sum(len(item["scored_runs"]) for item in provenance),
                "selected_config": config, "fixed_epochs": fixed_epochs, "training_report": report,
                "historical_relock": True, "prospective_shadow_required": True,
                "created_at": datetime.now(timezone.utc).isoformat()}
    contract_path.write_text(json.dumps(contract, indent=2))
    print(json.dumps({"base_model": str(base_path), "residual_model": str(residual_path),
                      "contract": str(contract_path), "calibrated": False}, indent=2))


if __name__ == "__main__":
    main()
