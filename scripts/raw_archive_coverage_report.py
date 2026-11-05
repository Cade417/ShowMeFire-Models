"""
Raw-archive completeness report - the gap the training-suite audit
explicitly flagged: spatial/coverage_report.py validates the built/aligned
dataset, but nothing validates the raw HRRR/RTMA/Synoptic archives (or
static rasters) for completeness independent of building a model. Reads
the backfill manifests these three scripts already maintain
(scripts/backfill_hrrr.py, backfill_rtma_for_hrrr.py, backfill_synoptic.py)
rather than re-deriving anything - they are the source of truth for what
was actually attempted and its outcome.

Mirrors spatial/coverage_report.py's own shape: build a report dict, write
it atomically to paths.REPORTS_DIR, print it, return it.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

FAILED_FRACTION_WARN_THRESHOLD = 0.10
MIN_COVERED_DAYS_FOR_GATE = 180  # matches spatial/coverage_report.py's own spatial_model_data_gate threshold


def _load_manifest(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _calendar_dates(timestamps: Iterable[str]) -> list:
    dates = set()
    for value in timestamps:
        try:
            dates.add(date.fromisoformat(value[:10]))
        except ValueError:
            continue
    return sorted(dates)


def _gap_days(covered_dates: list) -> int:
    """Count of calendar days strictly between the first and last covered date with zero entries."""
    if len(covered_dates) < 2:
        return 0
    covered_set = set(covered_dates)
    total_span = (covered_dates[-1] - covered_dates[0]).days + 1
    return total_span - len(covered_set)


def _timestamped_summary(entries: Dict[str, Dict], label: str) -> Dict:
    if not entries:
        return {"source": label, "status": "no_manifest_entries", "total": 0}
    statuses = [entry.get("status") for entry in entries.values()]
    completed = sum(1 for status in statuses if status == "complete")
    failed = sum(1 for status in statuses if status == "failed")
    covered_dates = _calendar_dates(entries.keys())
    total = len(entries)
    failed_fraction = failed / total if total else 0.0
    return {
        "source": label,
        "status": "ok",
        "total": total,
        "complete": completed,
        "failed": failed,
        "failed_fraction": round(failed_fraction, 4),
        "date_min": covered_dates[0].isoformat() if covered_dates else None,
        "date_max": covered_dates[-1].isoformat() if covered_dates else None,
        "covered_calendar_days": len(covered_dates),
        "gap_days_within_range": _gap_days(covered_dates),
        "healthy": failed_fraction <= FAILED_FRACTION_WARN_THRESHOLD,
    }


def hrrr_summary(manifest_path: Path = None) -> Dict:
    manifest = _load_manifest(manifest_path or (paths.CACHE_HRRR_DIR / "backfill_manifest.json"))
    if manifest is None:
        return {"source": "hrrr", "status": "no_manifest_found", "total": 0}
    return _timestamped_summary(manifest.get("runs", {}), "hrrr")


def rtma_summary(manifest_path: Path = None) -> Dict:
    manifest = _load_manifest(manifest_path or (paths.CACHE_RTMA_DIR / "backfill_manifest.json"))
    if manifest is None:
        return {"source": "rtma", "status": "no_manifest_found", "total": 0}
    return _timestamped_summary(manifest.get("analyses", {}), "rtma")


def raw_data_summary(manifest_path: Path = None) -> Dict:
    """Synoptic station observations. A zero-station day is a legitimate outcome (per the runbook), reported separately from real failures."""
    manifest = _load_manifest(manifest_path or (paths.ARCHIVE_RAW_DATA_DIR / "backfill_manifest.json"))
    if manifest is None:
        return {"source": "raw_data", "status": "no_manifest_found", "total": 0}
    days = manifest.get("days", {})
    if not days:
        return {"source": "raw_data", "status": "no_manifest_entries", "total": 0}
    statuses = [entry.get("status") for entry in days.values()]
    completed = sum(1 for status in statuses if status == "complete")
    failed = sum(1 for status in statuses if status == "failed")
    zero_station_days = sum(1 for entry in days.values() if entry.get("status") == "complete" and entry.get("stations") == 0)
    covered_dates = _calendar_dates(days.keys())
    total = len(days)
    failed_fraction = failed / total if total else 0.0
    return {
        "source": "raw_data",
        "status": "ok",
        "total": total,
        "complete": completed,
        "failed": failed,
        "failed_fraction": round(failed_fraction, 4),
        "zero_station_days": zero_station_days,
        "date_min": covered_dates[0].isoformat() if covered_dates else None,
        "date_max": covered_dates[-1].isoformat() if covered_dates else None,
        "covered_calendar_days": len(covered_dates),
        "gap_days_within_range": _gap_days(covered_dates),
        "healthy": failed_fraction <= FAILED_FRACTION_WARN_THRESHOLD,
    }


def static_raster_summary() -> Dict:
    """Reports not_yet_acquired rather than treating the (expected-empty-until-run) directory as an error - see task tracking for real acquisition."""
    manifest_path = paths.STATIC_SOURCE_DIR / "source_manifest.json"
    if not manifest_path.exists():
        return {"source": "static_rasters", "status": "not_yet_acquired",
                "note": "static_features/download_sources.py has not been run - see docs/spatial_fuel_moisture_runbook.md section 6"}
    manifest = _load_manifest(manifest_path)
    products = list((manifest or {}).get("products", {}).keys()) if manifest else []
    return {"source": "static_rasters", "status": "acquired", "products": products}


def generate(output: Optional[Path] = None) -> Dict:
    sources = {
        "hrrr": hrrr_summary(),
        "rtma": rtma_summary(),
        "raw_data": raw_data_summary(),
        "static_rasters": static_raster_summary(),
    }
    gate_inputs = [sources["hrrr"], sources["rtma"], sources["raw_data"]]
    gate_pass = all(
        source.get("status") == "ok" and source.get("healthy") and source.get("covered_calendar_days", 0) >= MIN_COVERED_DAYS_FOR_GATE
        for source in gate_inputs
    )
    failing_reasons = [
        f"{source['source']}: {source.get('status')}"
        + ("" if source.get("status") != "ok" else
           f" (healthy={source.get('healthy')}, covered_days={source.get('covered_calendar_days')})")
        for source in gate_inputs
        if not (source.get("status") == "ok" and source.get("healthy")
                and source.get("covered_calendar_days", 0) >= MIN_COVERED_DAYS_FOR_GATE)
    ]
    report = {
        "sources": sources,
        "raw_archive_ready": {
            "pass": gate_pass,
            "requires_covered_days": MIN_COVERED_DAYS_FOR_GATE,
            "requires_failed_fraction_at_most": FAILED_FRACTION_WARN_THRESHOLD,
            "failing_reasons": failing_reasons,
        },
    }
    output = Path(output) if output else paths.REPORTS_DIR / "raw_archive_coverage.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str))
    temporary.replace(output)
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report completeness of the raw HRRR/RTMA/Synoptic/static-raster archives, independent of any built dataset.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    generate(args.output)
