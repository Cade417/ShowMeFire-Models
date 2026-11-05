"""Compare a FIRMS history CSV (api/scripts/backfill_firms_history.py) with NERIS outdoor-fire dispatches and write
the overlapping detections, ready for api/scripts/ingest_firms_history_csv.py.

    python -m detection_dataset.overlap --history path\\to\\firms_history_2025-12-01_2026-08-08.csv
                                        [--neris-radius-km 2] [--day-window 1] [--neris-path X.shp]

A detection overlaps when a NERIS dispatch lies within the radius and +-day-window calendar days (same rule and the
same UTC-5 date shift as labels.match_neris). Only the matches are written; nothing reads or writes the database.
The unmatched rows are not lost: pass the full CSV to `build --history-csv` to use them as weak negatives.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from detection_dataset import labels


def find_overlap(history: pd.DataFrame, neris: pd.DataFrame, radius_km: float = 2.0, day_window: int = 1):
    """Returns (overlapping rows with their neris_* match columns, summary dict)."""
    det = history.reset_index(drop=True).copy()
    t = pd.to_datetime(det["occurred_at"], utc=True, errors="coerce")
    det["occurred_local_date"] = (t - pd.Timedelta(hours=labels.LOCAL_UTC_OFFSET_H)).dt.tz_localize(None).dt.normalize()
    match = labels.match_neris(det, neris, radius_km, day_window)
    hit = match["neris_matched"].astype(bool)

    span = labels.date_span(neris["call_date"])
    in_span = det["occurred_local_date"].between(*span) if span else pd.Series(False, index=det.index)
    summary = {
        "history_rows": len(det),
        "history_span": _fmt(labels.date_span(det["occurred_local_date"])),
        "neris_rows": len(neris),
        "neris_span": _fmt(span),
        "history_rows_inside_neris_dates": int(in_span.sum()),
        "overlapping": int(hit.sum()),
        "match_rate_inside_neris_dates": round(float(hit.sum() / in_span.sum()), 4) if in_span.sum() else None,
        "radius_km": radius_km, "day_window": day_window,
        "by_source": det.loc[hit, "source"].value_counts().to_dict() if "source" in det else {},
        "by_month": det.loc[hit, "occurred_local_date"].dt.to_period("M").astype(str).value_counts().sort_index().to_dict(),
    }
    out = pd.concat([history.reset_index(drop=True), match], axis=1)[hit.to_numpy()]
    return out.reset_index(drop=True), summary


def _fmt(span):
    return None if span is None else [str(span[0].date()), str(span[1].date())]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--neris-path", type=Path, default=None)
    p.add_argument("--neris-radius-km", type=float, default=2.0)
    p.add_argument("--day-window", type=int, default=1)
    p.add_argument("--out", type=Path, default=None, help="output CSV (default: SMF_DATA_ROOT/detection_dataset/firms_neris_overlap_<date>.csv)")
    a = p.parse_args()

    history = pd.read_csv(a.history)
    overlap, summary = find_overlap(history, labels.load_neris(a.neris_path), a.neris_radius_km, a.day_window)
    out = a.out or paths.DETECTION_DATASET_DIR / f"firms_neris_overlap_{datetime.now():%Y%m%d}.csv"
    overlap.to_csv(out, index=False)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out} ({len(overlap)} overlapping detections). Copy it to the server and run "
          f"scripts/ingest_firms_history_csv.py --csv <file> --dry-run first.")


if __name__ == "__main__":
    main()
