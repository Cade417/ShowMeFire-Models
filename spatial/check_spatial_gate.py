import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

coverage = json.loads((paths.REPORTS_DIR / "coverage.json").read_text())
sequence = json.loads((paths.REPORTS_DIR / "sequence_metrics.json").read_text())
baseline = json.loads((paths.REPORTS_DIR / "baseline_metrics.json").read_text())
incumbent_mae = baseline["temporal"]["incumbent_control"]["mae"]
passes = coverage["spatial_model_data_gate"]["pass"] and sequence["mae"] <= .95 * min(sequence["persistence_mae"], incumbent_mae)
result = {"pass": passes, "coverage_gate": coverage["spatial_model_data_gate"], "sequence_mae": sequence["mae"],
          "required_max_mae": .95 * min(sequence["persistence_mae"], incumbent_mae)}
print(json.dumps(result, indent=2))
raise SystemExit(0 if passes else 2)
