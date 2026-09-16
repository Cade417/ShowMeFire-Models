import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

report_path = paths.REPORTS_DIR / "hybrid_v3_final_evaluation.json"
if not report_path.exists():
    print(json.dumps({"pass": False, "reason": "hybrid_v3_final_evaluation.json is unavailable"}, indent=2))
    raise SystemExit(2)
report = json.loads(report_path.read_text())
result = {"pass": bool(report.get("pass")) and all(report.get("checks", {}).values()),
          "checks": report.get("checks", {}), "historical_relock": report.get("historical_relock"),
          "prospective_shadow_required": report.get("prospective_shadow_required", True),
          "dataset_sha256": report.get("dataset_sha256"), "manifest_sha256": report.get("manifest_sha256")}
print(json.dumps(result, indent=2)); raise SystemExit(0 if result["pass"] else 2)
