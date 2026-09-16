"""Evaluate immutable API shadow evidence at the predeclared day-14/day-30 checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.v5_evidence import POLICY_VERSION, dataframe_sha256, evaluate_policy, policy_sha256


def load_evidence(root):
    rows, run_results, bundles = [], [], set()
    for prediction_path in sorted(Path(root).glob("*.prediction.json")):
        prediction = json.loads(prediction_path.read_text()); run_id = str(prediction["run_id"])
        observation_path = prediction_path.with_name(f"{run_id}.observation.json")
        run_results.append({"run_id": run_id, "success": True})
        if not observation_path.exists(): continue
        observation = json.loads(observation_path.read_text())
        observed = observation.get("observations", {})
        if isinstance(observed, list): observed = {str(item["row_key"]): item for item in observed}
        bundles.add(prediction.get("bundle_manifest_sha256"))
        for index, row_key in enumerate(prediction["row_keys"]):
            target = observed.get(str(row_key))
            if not target or not target.get("available", True): continue
            actual = target.get("target_fm"); actual_category = target.get("actual_category")
            if actual is None or actual_category is None: continue
            stable_category = prediction["stable_category"][index]; candidate_category = prediction["v5_category"][index]
            interval = prediction["v5_p10_p50_p90"][index]
            valid_time = target.get("valid_time") or str(row_key).rsplit("|", 1)[-1]
            timestamp = pd.to_datetime(valid_time, utc=True, errors="coerce")
            rows.append({"row_key": row_key, "bootstrap_block": timestamp.strftime("%Y-%m-%d"),
                "actual_fm": actual, "candidate_fm": prediction["v5_fm"][index],
                "incumbent_fm": prediction["stable_fm"][index], "p10": interval[0], "p50": interval[1], "p90": interval[2],
                "actual_category": actual_category, "candidate_category": candidate_category,
                "incumbent_category": stable_category, "summer": timestamp.month in (6, 7, 8),
                "critical": actual <= 6, "rain_event": bool(target.get("rain_event", False)),
                "run_id": run_id, "observation_time": target.get("observation_time"),
                "match_age_minutes": target.get("match_age_minutes")})
    if len(bundles) > 1: raise RuntimeError("shadow evidence contains multiple bundle checksums")
    return pd.DataFrame(rows), run_results, next(iter(bundles), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("day14", "day30"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); minimum_days = 14 if args.checkpoint == "day14" else 30
    output = args.output or paths.REPORTS_DIR / f"v5_shadow_{args.checkpoint}_evaluation.json"
    if output.exists(): raise SystemExit(f"Refusing to overwrite checkpoint report {output}")
    frame, runs, bundle_sha = load_evidence(args.evidence_root)
    if frame.empty: raise SystemExit("No verified V5 shadow rows are available")
    evidence = evaluate_policy(frame, probability_required=.95 if minimum_days == 14 else .90,
                               minimum_days=minimum_days, early_checkpoint=minimum_days == 14)
    state_path = args.evidence_root / "shadow-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    attempted = int(state.get("runs", len(runs))); successful = int(state.get("successful_runs", len(runs)))
    success_rate = successful / max(attempted, 1)
    operational = {"shadow_success_rate": success_rate >= .95,
                   "public_forecast_failures": int(state.get("public_forecast_failures", 0)) == 0}
    report = {"status": "prospective_checkpoint", "checkpoint": args.checkpoint,
              "policy_version": POLICY_VERSION, "policy_sha256": policy_sha256(),
              "pass": evidence["pass"] and all(operational.values()), "production_eligible": False,
              "authorization_allowed": evidence["pass"] and all(operational.values()),
              "evidence": evidence, "operational_checks": operational, "shadow_success_rate": success_rate,
              "bundle_manifest_sha256": bundle_sha, "paired_dataframe_sha256": dataframe_sha256(frame),
              "evaluated_at": datetime.now(timezone.utc).isoformat()}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": report["pass"], "checkpoint": args.checkpoint, "days": evidence["support"]["days"],
                      "failed_checks": [key for key, value in {**evidence["checks"], **operational}.items() if not value],
                      "report": str(output)}, indent=2))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__": main()
