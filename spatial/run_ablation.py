from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.static_inputs import FEATURE_SETS
from spatial.train_spatial import train


def run(bundle, epochs, batch_size, force):
    results = {}; artifacts = {}
    for feature_set in FEATURE_SETS:
        artifact, metrics = train(bundle, feature_set, epochs, batch_size, force); artifacts[feature_set] = str(artifact); results[feature_set] = metrics
    best_mae = min(value["temporal"]["mae"] for value in results.values()); eligible = [name for name, value in results.items() if value["temporal"]["mae"] <= best_mae * 1.01 and value["temporal"]["quantile_order_valid"]]
    order = list(FEATURE_SETS); selected = min(eligible, key=order.index); dynamic = results["dynamic"]
    selected_low = results[selected]["temporal"]["critical_low_fm_mae"]; dynamic_low = dynamic["temporal"]["critical_low_fm_mae"]
    low_gate = selected_low is None or dynamic_low is None or selected_low <= dynamic_low * 1.01
    gate = selected != "dynamic" and results[selected]["temporal"]["mae"] < dynamic["temporal"]["mae"] and results[selected]["temporal"]["interval_coverage"] >= dynamic["temporal"]["interval_coverage"] - .02 and low_gate
    report = {"results": results, "artifacts": artifacts, "selected": selected, "static_candidate_gate": gate,
              "rule": "smallest candidate within 1% of best; must beat dynamic MAE without >2pp interval-coverage regression"}
    output = paths.REPORTS_DIR / "spatial_ablation.json"; output.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2)); return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--static-bundle", required=True, type=Path); parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); run(args.static_bundle, args.epochs, args.batch_size, args.force)
