"""Apply the frozen promotion gate to distilled and non-distilled students."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.train_spatial import distillation_gate


def compare(feature_set="all"):
    distilled_path = paths.REPORTS_DIR / f"spatial_{feature_set}_distilled.json"
    control_path = paths.REPORTS_DIR / f"spatial_{feature_set}_control.json"
    distilled, control = json.loads(distilled_path.read_text()), json.loads(control_path.read_text())
    gate = distillation_gate(distilled, control)
    gate.update({"feature_set": feature_set, "distilled_report": distilled_path.name, "control_report": control_path.name,
                 "quantile_order_valid": all(distilled[name]["quantile_order_valid"] for name in ("temporal", "station", "region")),
                 "rule": "At least 2% temporal MAE improvement, or better calibration without MAE increase; no holdout/critical-low regression above 1%."})
    gate["pass"] = gate["pass"] and gate["quantile_order_valid"]
    output = paths.REPORTS_DIR / f"spatial_{feature_set}_distillation_gate.json"
    output.write_text(json.dumps(gate, indent=2)); print(json.dumps(gate, indent=2)); return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--feature-set", default="all")
    compare(parser.parse_args().feature_set)
