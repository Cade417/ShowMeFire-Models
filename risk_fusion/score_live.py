"""
Score every Missouri county's fire-ignition risk for TODAY (or a chosen
recent date) using live HRRR data - the fire_risk_fusion monthly-baseline
+ weather-residual model (risk_fusion/fit_glm.py), fit fresh each run on
the real 2014-2020 labeled panel.

Needs exactly one fresh HRRR run, not a long antecedent window: the
model's feature set (fit_glm.MONTH_DUMMY_COLUMNS + FAST_WEATHER_FEATURES)
uses only calendar month/weekday plus same-day RH/wind/VPD/precip - no
KBDI, no GDD, no fuel moisture. This is what makes scoring "today"
tractable without a 90-day spin-up fetch.

This is a quick-look tool, not the registered model: no bundle is
persisted (fit_risk_fusion.py/register_risk_fusion_beta.py do that), and
scores are advisory-only. They report expected fire count and P(>=1 fire)
for a live forward day, not a training-consistent evaluation.

Usage:
    python -m risk_fusion.score_live
    python -m risk_fusion.score_live --date 2026-08-08
    python -m risk_fusion.score_live --date 2020-04-15 --no-fetch
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths
from risk_fusion import build_county_days, model_bundle
from spatial.hrrr_capture import DEFAULT_LEAD_HOURS, classify_fetch_failure, clear_local_cache, fetch_hrrr

TRAINING_PANEL_PATH = paths.RISK_FUSION_DIR / "labeled_panel_2014_2020.csv"
DISPLAY_COLUMNS = ["county_fips", "county_name", "lam", "p_ge1_fire",
                    "rh_min_afternoon", "wind_kts_max", "precip_24h_mm"]


def resolve_target_run(max_lookback_days: int = 3, cache_dir: Optional[Path] = None) -> Tuple[date, datetime]:
    """
    Tries today's 12Z HRRR run first, stepping back one calendar day at a
    time when a run is "unavailable" (or retries on a "transient" failure
    are exhausted) - e.g. run too early in the day for NOAA to have
    published the 12Z cycle yet. A same-day 12Z run's f04 lead lands on
    the same America/Chicago calendar date as the run itself, so "target
    date" and "run date" are identical by construction. Returns
    (target_date, run_dt actually used) so callers can report which one.
    """
    cache_dir = cache_dir or paths.CACHE_HRRR_DIR
    today_12z = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    last_exc: Optional[Exception] = None

    for day_offset in range(max_lookback_days + 1):
        run_dt = today_12z - timedelta(days=day_offset)
        for attempt in range(3):
            try:
                clear_local_cache(run_dt)
                fetch_hrrr(run_dt, cache_dir=cache_dir)
                return run_dt.date(), run_dt
            except Exception as exc:
                last_exc = exc
                if classify_fetch_failure(exc) != "transient":
                    break
                time.sleep(2 ** (attempt + 1))

    raise RuntimeError(
        f"Could not fetch any HRRR run in the last {max_lookback_days + 1} days "
        f"(back to {(today_12z - timedelta(days=max_lookback_days)).date()}): {last_exc}"
    )


def target_run_file(run_dt: datetime, cache_dir: Optional[Path] = None) -> Path:
    """Reproduces fetch_hrrr's own filename convention exactly, so this always names the file it just fetched."""
    cache_dir = cache_dir or paths.CACHE_HRRR_DIR
    lead_hours = list(DEFAULT_LEAD_HOURS)
    return cache_dir / f"hrrr_{run_dt:%Y%m%d_%H}z_f{lead_hours[0]:02d}-{lead_hours[-1]:02d}.nc"


def build_target_panel(run_dt: datetime) -> pd.DataFrame:
    """
    Builds features from exactly the target run's file - not build()'s
    default (the whole archive) or its since= filter (open-ended, would
    pull in every later cached day too). One HRRR run in, one day's
    county-day panel out.
    """
    return build_county_days.build(run_files=[target_run_file(run_dt)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: resolve automatically, trying today first)")
    parser.add_argument("--max-lookback-days", type=int, default=3,
                        help="How many days to step back if the newest run isn't fetchable yet (default: 3)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Score only from an already-cached run; requires --date, errors if that run isn't cached")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: training-data/risk_fusion/live_scores_<date>.csv)")
    args = parser.parse_args()

    if args.no_fetch and not args.date:
        parser.error("--no-fetch requires --date")

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        run_dt = datetime(target_date.year, target_date.month, target_date.day, 12, tzinfo=timezone.utc)
        if not args.no_fetch:
            clear_local_cache(run_dt)
            fetch_hrrr(run_dt, cache_dir=paths.CACHE_HRRR_DIR)
    else:
        target_date, run_dt = resolve_target_run(max_lookback_days=args.max_lookback_days)

    fetch_note = "from cache (--no-fetch)" if args.no_fetch else "fetched live"
    print(f"Scoring {target_date} using the {run_dt:%Y%m%d}_{run_dt:%H}z HRRR run ({fetch_note})...")

    target_panel = build_target_panel(run_dt)
    if target_panel.empty:
        raise SystemExit(
            f"No county-day rows produced for {target_date} - is the "
            f"{run_dt:%Y%m%d}_{run_dt:%H}z run actually cached at {paths.CACHE_HRRR_DIR}?"
        )

    print("Fitting the monthly-baseline + weather-residual model on the historical labeled panel...")
    train_panel = model_bundle.load_training_panel(TRAINING_PANEL_PATH)
    bundle = model_bundle.fit(train_panel)
    scored = model_bundle.score(target_panel, bundle)

    output_path = args.output or (paths.RISK_FUSION_DIR / f"live_scores_{target_date}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

    print(f"\nTop 20 highest-risk counties for {target_date}:")
    print(scored[DISPLAY_COLUMNS].head(20).to_string(index=False))
    print(f"\nFull {len(scored)}-county table written to {output_path}")


if __name__ == "__main__":
    main()
