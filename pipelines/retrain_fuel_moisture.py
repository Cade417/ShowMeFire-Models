#!/usr/bin/env python3
"""
Biweekly fuel-moisture retraining orchestrator.

Replaces the manual "run these 8 scripts in order" workflow (still
documented, and still working, as pipelines/trainnewmodel.sh) with a single
in-process call chain: each phase's real function is imported and called
directly rather than shelling out to a subprocess per script, so one Python
traceback tells you exactly which phase failed instead of a shell script
exiting silently on `set -e`.

This does not change what any phase does - it only removes the "remember to
run these 8 commands in this order, and know which env-relative directory
you're supposed to be in" friction, and adds a comparison of the new
candidate against whatever is currently `stable` so a retrain's value is
visible immediately instead of requiring a manual before/after metrics
comparison.

Usage:
    python pipelines/retrain_fuel_moisture.py
    python pipelines/retrain_fuel_moisture.py --full-retrain
    python pipelines/retrain_fuel_moisture.py --no-search --skip-ingest

Promotion is intentionally NOT automated here - this script always leaves
the result in the `beta` channel. Publishing/promoting to `stable` stays a
deliberate, separate human step (see pipelines/publish_release.py and
pipelines/promote_model.py), consistent with how every other model family
in this repo is promoted.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# This is the entrypoint meant for unattended/scheduled runs; several called
# modules print emoji status markers (✅/⚠️/❌), which crashes on Windows
# consoles whose default stdout codepage isn't UTF-8. Reconfigure defensively
# so a cosmetic print never takes down an otherwise-successful (or
# already-failed, and about to be reported) run.
for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths


class PhaseError(RuntimeError):
    """Raised when a named phase fails, so the top-level handler can report
    which phase broke without a bare traceback."""


def _run_phase(name, fn, *args, **kwargs):
    print(f"\n=== {name} ===")
    start = time.monotonic()
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        raise PhaseError(f"Phase {name!r} failed: {exc}") from exc
    elapsed = time.monotonic() - start
    print(f"--- {name} done in {elapsed:.1f}s ---")
    return result


def main():
    parser = argparse.ArgumentParser(description="Biweekly fuel-moisture retrain orchestrator")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip RAWS observation ingest")
    parser.add_argument("--skip-index", action="store_true", help="Skip station->HRRR-grid indexing")
    parser.add_argument("--skip-snapshots", action="store_true", help="Skip snapshot creation for new HRRR files")
    parser.add_argument("--skip-extract", action="store_true", help="Skip HRRR feature extraction")
    parser.add_argument("--full-retrain", action="store_true",
                         help="Reset all snapshots first, so every HRRR file is reprocessed from scratch")
    parser.add_argument("--extract-days-back", type=int, default=21,
                         help="Only extract HRRR features for snapshots dated in the last N days "
                              "(default: 21). Without this cutoff, extraction rescans EVERY unprocessed "
                              "snapshot regardless of age - a snapshots table can accumulate years of "
                              "never-mined rows (e.g. from building the risk_fusion historical panel) "
                              "that don't even contribute training rows if there's no matching "
                              "observation, just wasted time. Ignored (no cutoff applied) when "
                              "--full-retrain or --extract-full-history is set.")
    parser.add_argument("--extract-full-history", action="store_true",
                         help="Ignore --extract-days-back and extract every unprocessed snapshot "
                              "regardless of age. Only intended for a genuine one-time historical "
                              "backfill, not routine biweekly runs.")
    parser.add_argument("--no-search", action="store_true",
                         help="Skip hyperparameter search; use the historical fixed defaults")
    parser.add_argument("--channel", choices=["beta", "stable"], default="beta",
                         help="Channel to register the trained model under (default: beta - do not change this "
                              "for routine biweekly runs; promotion is a separate, deliberate step)")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], default="patch")
    args = parser.parse_args()

    # Imported lazily (after argument parsing) so `--help` doesn't pay the
    # cost of loading xgboost/xarray/etc., and so a missing optional
    # dependency only breaks the phase that actually needs it.
    from pipelines.ingest_obs import init_database, clear_tables, ingest_archive
    from pipelines.index_stations import index_stations
    from scripts.create_snapshots import create_snapshots_from_hrrr
    from scripts.reset_snapshots import reset_snapshots
    from pipelines.extract_hrrr import run_miner
    from pipelines.generate_training_set import generate_training_set
    from pipelines.prepare_features import prepare_features, enhance_features_with_lags
    from pipelines.train_model import train_fuel_moisture_model

    overall_start = time.monotonic()

    if not args.skip_ingest:
        _run_phase("Phase 1: Ingest RAWS observations", init_database)
        _run_phase("Phase 1: Ingest RAWS observations", clear_tables)
        _run_phase("Phase 1: Ingest RAWS observations", ingest_archive)
    else:
        print("Skipping ingest (--skip-ingest)")

    if not args.skip_index:
        _run_phase("Phase 2: Index stations to HRRR grid", index_stations)
    else:
        print("Skipping station indexing (--skip-index)")

    if args.full_retrain:
        print("\n⚠️  --full-retrain: resetting all snapshots for complete reprocessing")
        _run_phase("Phase 3: Reset snapshots", reset_snapshots, confirm=True)

    if not args.skip_snapshots:
        _run_phase("Phase 3: Create snapshots for new HRRR files", create_snapshots_from_hrrr)
    else:
        print("Skipping snapshot creation (--skip-snapshots)")

    if not args.skip_extract:
        if args.full_retrain or args.extract_full_history:
            extract_since_date = None
        else:
            extract_since_date = (datetime.now(timezone.utc) - timedelta(days=args.extract_days_back)).strftime("%Y-%m-%d")
        _run_phase("Phase 4: Extract HRRR weather features", run_miner, since_date=extract_since_date)
    else:
        print("Skipping HRRR extraction (--skip-extract)")

    _run_phase("Phase 5: Generate training set", generate_training_set)
    _run_phase("Phase 6: Prepare features", prepare_features)
    _run_phase("Phase 6: Add lagged/rolling features", enhance_features_with_lags)

    training_result = _run_phase(
        "Phase 7: Train model",
        train_fuel_moisture_model,
        channel=args.channel,
        bump=args.bump,
        search=not args.no_search,
    )

    _run_phase("Phase 8: Summarize training data", _print_data_summary)

    total_elapsed = time.monotonic() - overall_start
    print(f"\n✅ Retrain pipeline completed in {total_elapsed / 60:.1f} min")
    print(f"   Registered {training_result['channel']} version {training_result['version']}")
    print(f"   {training_result['comparison']}")
    if training_result["channel"] == "beta":
        if training_result["is_better_than_stable"] is False:
            print("   ⚠️  This candidate is WORSE than the current stable model on the holdout metric - "
                  "review before publishing/promoting.")
        print(f"\n   Next: python pipelines/publish_release.py --model fuel_moisture")
        print(f"         python pipelines/promote_model.py --model fuel_moisture --version {training_result['version']}")


def _print_data_summary():
    import pandas as pd
    path = paths.TRAINING_DATA_DIR / "final_training_data.csv"
    try:
        df = pd.read_csv(path)
        print(f"   Total training samples: {len(df):,}")
        print(f"   Date range: {df['obs_time'].min()} to {df['obs_time'].max()}")
        print(f"   Stations: {df['station_id'].nunique()}")
    except FileNotFoundError:
        print(f"   No training data found at {path}")


if __name__ == "__main__":
    try:
        main()
    except PhaseError as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)
