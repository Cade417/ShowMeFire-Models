"""
One-command (re)training for ensemble_fire_danger:

    capture -> build_panel -> split -> fit -> evaluate -> register_beta

    python -m ensemble_fire_danger.retrain                      # routine: last 3 days captured, new days scored
    python -m ensemble_fire_danger.retrain --capture-days-back 45 --workers 6   # first-time backfill
    python -m ensemble_fire_danger.retrain --skip-capture --no-register         # experiment on existing data

Prints the new calibration's headline numbers next to the currently
registered beta's (from the local registry), and appends one row per run to
<SMF_DATA_ROOT>/reports/experiments/ensemble_fire_danger.jsonl, the same
tracking convention as pipelines/retrain_fuel_moisture.py. Registration is
skipped (with the reason printed) if any gate FAILS; deferred gates are
normal while history accumulates. Nothing here touches the server - publish
with pipelines/publish_release.py --model ensemble_fire_danger when happy.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402

from . import build_panel, capture, contract, evaluate, fit, members, model_bundle  # noqa: E402

logger = logging.getLogger(__name__)
EXPERIMENT_LOG = paths.EXPERIMENTS_DIR / "ensemble_fire_danger.jsonl"


def _headline(report: dict) -> dict:
    out = {}
    for track, t in report.get("tracks", {}).items():
        for k, entry in t["categories"].items():
            nb = entry["neighborhood"]
            out[f"{track}:{k}"] = {"events": nb["events"], "brier_cal": nb["brier_cal"], "bss_vs_det": nb["bss_vs_det"]}
    return out


def _current_beta() -> dict:
    try:
        from models.versioning import get_model_entry
        beta = get_model_entry("ensemble_fire_danger").get("beta") or {}
        return {"version": beta.get("version"), "performance": beta.get("performance")}
    except Exception:
        return {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-days-back", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--rescore", action="store_true", help="re-score every captured day (after a code change)")
    parser.add_argument("--fm-model", help="XGBoost fuel-moisture model - use the server's stable one")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    config = members.load_config()

    if not args.skip_capture:
        today = datetime.now(timezone.utc).date()
        days = [today - timedelta(days=i) for i in range(args.capture_days_back, -1, -1)]
        print("capture:", json.dumps(capture.capture_days(days, workers=args.workers)))

    build_panel.main((["--rescore"] if args.rescore else []) + (["--fm-model", args.fm_model] if args.fm_model else []))
    panel = pd.read_parquet(paths.ENSEMBLE_FD_PANEL)
    split_manifest = contract.write_manifest(contract.build_split(panel["day"].unique()))
    split = split_manifest["days"]
    print(f"split: {split_manifest['n_train_days']} train days / {split_manifest['n_test_days']} test days")

    fitted = fit.fit(panel, split, float(config.get("default_categorical_threshold", 0.5)))
    model_bundle.save(paths.ENSEMBLE_FD_CANDIDATE_DIR, fitted)
    report = evaluate.evaluate(panel, split, fitted, config)
    paths.ENSEMBLE_FD_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    model_bundle.save(paths.ENSEMBLE_FD_CANDIDATE_DIR, fitted, evaluation=report)

    previous = _current_beta()
    headline = _headline(report)
    print(json.dumps({"overall_pass": report["overall_pass"],
                      "failed_gates": [g["name"] for g in report["gates"] if g["status"] == "fail"],
                      "deferred_gates": [g["name"] for g in report["gates"] if g["status"] == "deferred"],
                      "track_comparison": report["track_comparison"],
                      "new": headline, "previous_beta": previous.get("version")}, indent=2))

    version = None
    if args.no_register:
        print("registration skipped (--no-register)")
    elif not report["overall_pass"]:
        print("registration skipped: at least one gate FAILED (see the report)")
    else:
        from . import register_beta
        register_beta.main([])
        version = json.loads((paths.ENSEMBLE_FD_CANDIDATE_DIR / "registered_version.json").read_text())["version"]

    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(), "registered_version": version,
            "train_days": split_manifest["n_train_days"], "test_days": split_manifest["n_test_days"],
            "overall_pass": report["overall_pass"], "track_comparison": report["track_comparison"],
            "headline": headline,
        }) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
