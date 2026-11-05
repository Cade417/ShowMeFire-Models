"""Report the locked station gate; never infer success from stale aggregate files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

evaluation = paths.REPORTS_DIR / "final_station_evaluation.json"
if not evaluation.exists():
    print(json.dumps({"pass": False, "reason": "final_station_evaluation.json is unavailable"}, indent=2))
    raise SystemExit(2)
report = json.loads(evaluation.read_text())
result = {"pass": bool(report.get("pass")), "checks": report.get("checks", {}),
          "checkpoint_sha256": report.get("checkpoint_sha256"),
          "dataset_sha256": report.get("dataset_sha256"),
          "prospective_shadow_required": report.get("prospective_shadow_required", True)}
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["pass"] and all(result["checks"].values()) else 2)
