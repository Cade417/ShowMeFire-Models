"""Register a station checkpoint only after a passing locked evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from models.versioning import register_trained_model
from spatial.station_contract import sha256_file


def main():
    parser = argparse.ArgumentParser(description="Register a gate-approved station candidate as beta")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, default=paths.REPORTS_DIR / "final_station_evaluation.json")
    args = parser.parse_args()
    report = json.loads(args.evaluation.read_text())
    required = ["checks", "dataset_sha256", "split_version", "feature_schema_version", "checkpoint_sha256", "metrics"]
    missing = [key for key in required if key not in report]
    if missing:
        raise SystemExit(f"Evaluation metadata missing: {', '.join(missing)}")
    if not report.get("pass") or not all(report["checks"].values()):
        raise SystemExit("Candidate failed the locked offline gate; registration refused")
    if sha256_file(args.checkpoint) != report["checkpoint_sha256"]:
        raise SystemExit("Checkpoint checksum does not match the evaluation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    contracts = {
        "dataset_sha256": report["dataset_sha256"],
        "split_version": report["split_version"],
        "feature_schema_version": report["feature_schema_version"],
    }
    mismatches = [key for key, value in contracts.items() if checkpoint.get(key) != value]
    if mismatches:
        raise SystemExit(f"Checkpoint/evaluation contract mismatch: {', '.join(mismatches)}")
    performance = {
        "offline_gate": report["checks"], "candidate": report["metrics"]["candidate"],
        "incumbent_control": report["metrics"]["incumbent_control"],
        "dataset_sha256": report["dataset_sha256"], "split_version": report["split_version"],
        "feature_schema_version": report["feature_schema_version"],
        "prospective_shadow_required": True,
    }
    version = register_trained_model("fuel_moisture_station_sequence", args.checkpoint,
                                     performance=performance, channel="beta")
    print(json.dumps({"registered_version": version, "channel": "beta", "production_changed": False}, indent=2))


if __name__ == "__main__":
    main()
