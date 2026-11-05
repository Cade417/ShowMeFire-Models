"""
One-command (re)training for the seven-day fire outlook:

    load verification records -> split (periods + stations) -> fit -> evaluate -> candidate bundle

    python -m fire_outlook.retrain                       # records from $SMF_DATA_ROOT/fire_outlook/verification
    python -m fire_outlook.retrain --records path/to/records.parquet
    python -m fire_outlook.retrain --no-save             # evaluate only

Writes the candidate bundle to paths.FIRE_OUTLOOK_CANDIDATE_DIR (calibration.json,
evaluation.json), the split manifest and offline report under reports/, and appends
one row to reports/experiments/fire_outlook.jsonl (same convention as the other
families). It never registers, publishes, or touches the server: the worker does not
consume a calibration yet, and deferred gates are normal while history accumulates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402

from . import contract, evaluate, fit  # noqa: E402

EXPERIMENT_LOG = paths.EXPERIMENTS_DIR / "fire_outlook.jsonl"


def save_candidate(candidate_dir: Path, fitted: dict, report: dict, split: dict) -> None:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    calibration = dict(fitted, fitted_at=stamp, split_version=split["split_version"], split_sha256=split["sha256"])
    cal_text = json.dumps(calibration, indent=2)
    (candidate_dir / "calibration.json").write_text(cal_text, encoding="utf-8")
    (candidate_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (candidate_dir / "checksums.json").write_text(json.dumps(
        {"calibration.json": hashlib.sha256(cal_text.encode()).hexdigest()}, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", help="file or directory of .parquet/.csv verification records")
    parser.add_argument("--no-save", action="store_true", help="evaluate only; write no bundle or report")
    args = parser.parse_args(argv)

    try:
        df, issues = contract.load_records(args.records)
    except FileNotFoundError as exc:
        print(f"{exc}\nThe fire-outlook worker exports verification records once windows have observations; "
              f"copy them to {paths.FIRE_OUTLOOK_RECORDS_DIR}.")
        return 2
    side, split = contract.assign_split(df)
    print(f"records: {json.dumps(issues)}")
    print(f"split: {split['n_train_days']} train / {split['n_test_days']} test / {split['n_purged_days']} purged days; "
          f"{split['n_train_stations']} train / {split['n_test_stations']} test stations; rows {split['n_rows']}")

    fitted = fit.fit(df, side)
    report = evaluate.evaluate(df, side, fitted)
    report["record_checks"] = issues
    failed = [g["name"] for g in report["gates"] if g["status"] == "fail"]
    deferred = [g["name"] for g in report["gates"] if g["status"] == "deferred"]
    print(json.dumps({"overall_pass": report["overall_pass"], "failed_gates": failed,
                      "n_deferred_gates": len(deferred)}, indent=2))

    if args.no_save:
        print("nothing saved (--no-save)")
        return 0
    contract.write_manifest(split)
    paths.FIRE_OUTLOOK_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_candidate(paths.FIRE_OUTLOOK_CANDIDATE_DIR, fitted, report, split)
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(), "registered_version": None,
            "train_days": split["n_train_days"], "test_days": split["n_test_days"],
            "overall_pass": report["overall_pass"], "failed_gates": failed, "n_deferred_gates": len(deferred),
            "split_sha256": split["sha256"],
        }) + "\n")
    print(f"candidate bundle: {paths.FIRE_OUTLOOK_CANDIDATE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
