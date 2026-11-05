"""Model Lab schedule + runtime catalog.

Tracks:
- Production/runtime reference (when live forecast & archive jobs run)
- Local lab jobs the Streamlit app can run now or on a cron schedule
  (CDN pull, unpack, score latest day)

Schedule state lives under SMF_DATA_ROOT/model_lab/ so it survives app restarts.
"""
from __future__ import annotations

import json
import threading
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from paths import DATA_ROOT, REPORTS_DIR

LAB_STATE_DIR = DATA_ROOT / "model_lab"
SCHEDULE_PATH = LAB_STATE_DIR / "schedule.json"
HISTORY_PATH = LAB_STATE_DIR / "job_history.jsonl"
MAX_HISTORY = 200

# Reference card for what the live API host runs. Times are America/Chicago.
PRODUCTION_RUNTIME = [
    {
        "id": "daily_forecast_production",
        "name": "DailyForecast (production)",
        "where": "api host cron → scripts/forecasts.sh",
        "typical_time": "After 12Z HRRR availability (usually late morning CT)",
        "script": "api/forecast/DailyForecast.py",
        "outputs": "archive/forecasts/station_forecasts_YYYYMMDD_12.json + maps/CDN",
        "notes": "Loads stable fuel_moisture; public product path.",
    },
    {
        "id": "daily_forecast_beta",
        "name": "DailyForecast_ModelFD (beta)",
        "where": "api host cron → scripts/forecasts.sh (after production)",
        "typical_time": "Same window as production, sequential after it",
        "script": "api/forecast/DailyForecast_ModelFD.py",
        "outputs": "station_forecasts_beta_YYYYMMDD_12.json + *-beta.png (CDN off by default)",
        "notes": "Parallel beta product for comparison; does not replace public maps.",
    },
    {
        "id": "forecast_ai",
        "name": "Forecast AI text",
        "where": "scripts/forecasts.sh",
        "typical_time": "Immediately after DailyForecast",
        "script": "api/forecast/forecast_ai.py",
        "outputs": "Narrative/text products",
        "notes": "",
    },
    {
        "id": "end_of_day_archive",
        "name": "End-of-day archive → CDN",
        "where": "api APScheduler",
        "typical_time": "23:45 America/Chicago",
        "script": "api/services/archive_bundler.py",
        "outputs": "R2/CDN data-archive/YYYYMMDD.zip",
        "notes": "Bundles HRRR + forecasts + raw_data; removes local server copies after upload.",
    },
    {
        "id": "v5_shadow_verify",
        "name": "V5 shadow observation attach",
        "where": "api APScheduler",
        "typical_time": "Every 3 hours",
        "script": "api/services/v5_verification.py",
        "outputs": "data/model-shadow/v5 evidence",
        "notes": "Prospective evidence only; never changes public P50.",
    },
    {
        "id": "rtma_capture",
        "name": "RTMA capture",
        "where": "api APScheduler",
        "typical_time": ":50 every hour",
        "script": "api/services/rtma_capture.py",
        "outputs": "cache/rtma/",
        "notes": "",
    },
]


@dataclass
class LabJobSpec:
    id: str
    name: str
    description: str
    default_enabled: bool = False
    default_hour: int = 1
    default_minute: int = 15
    params: dict[str, Any] = field(default_factory=dict)


LAB_JOBS = [
    LabJobSpec(
        id="sync_cdn_archives",
        name="Pull + unpack CDN archives",
        description="Download missing data-archive/YYYYMMDD.zip files from the CDN and unpack forecasts/obs into SMF_DATA_ROOT.",
        default_enabled=False,
        default_hour=1,
        default_minute=30,
        params={"days": 7, "unpack": True},
    ),
    LabJobSpec(
        id="unpack_local_zips",
        name="Unpack local archive_zips",
        description="Run unpack on every zip already in archive_zips/ (no download).",
        default_enabled=False,
        default_hour=2,
        default_minute=0,
        params={},
    ),
    LabJobSpec(
        id="score_latest_day",
        name="Score latest production vs beta day",
        description="Score the newest shared production/beta forecast date against observations and write a JSON summary under reports/model_lab/.",
        default_enabled=False,
        default_hour=3,
        default_minute=0,
        params={},
    ),
]


_scheduler = None
_scheduler_lock = threading.Lock()
_job_lock = threading.Lock()


def _ensure_state_dir():
    LAB_STATE_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_lab").mkdir(parents=True, exist_ok=True)


def default_schedule_config() -> dict:
    jobs = {}
    for spec in LAB_JOBS:
        jobs[spec.id] = {
            "enabled": spec.default_enabled,
            "hour": spec.default_hour,
            "minute": spec.default_minute,
            "params": dict(spec.params),
        }
    return {"timezone": "America/Chicago", "jobs": jobs}


def load_schedule_config() -> dict:
    _ensure_state_dir()
    if not SCHEDULE_PATH.exists():
        cfg = default_schedule_config()
        save_schedule_config(cfg)
        return cfg
    with open(SCHEDULE_PATH, encoding="utf-8") as handle:
        cfg = json.load(handle)
    # Fill any newly added jobs.
    base = default_schedule_config()
    for job_id, job_cfg in base["jobs"].items():
        cfg.setdefault("jobs", {}).setdefault(job_id, job_cfg)
    return cfg


def save_schedule_config(config: dict) -> None:
    _ensure_state_dir()
    temp = SCHEDULE_PATH.with_suffix(".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    temp.replace(SCHEDULE_PATH)


def append_history(entry: dict) -> None:
    _ensure_state_dir()
    with open(HISTORY_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")
    # Trim
    lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    if len(lines) > MAX_HISTORY:
        HISTORY_PATH.write_text("\n".join(lines[-MAX_HISTORY:]) + "\n", encoding="utf-8")


def read_history(limit: int = 50) -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    rows = []
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(rows))


def _run_sync_cdn(params: dict) -> dict:
    from model_lab.cdn_sync import sync_missing_since

    days = int(params.get("days", 7))
    unpack = bool(params.get("unpack", True))
    results = sync_missing_since(days=days, unpack=unpack)
    return {
        "days": days,
        "results": [item.to_dict() for item in results],
        "ok": all(item.error is None for item in results),
        "downloaded": sum(1 for item in results if item.downloaded),
        "unpacked": sum(1 for item in results if item.unpacked),
        "errors": [item.to_dict() for item in results if item.error],
    }


def _run_unpack_local(_params: dict) -> dict:
    from model_lab.cdn_sync import list_local_zips, unpack_local_zip

    summaries = []
    for path in list_local_zips():
        stats = unpack_local_zip(path)
        summaries.append({"zip": path.name, **stats})
    return {"zips": len(summaries), "summaries": summaries, "ok": True}


def _run_score_latest(_params: dict) -> dict:
    from model_lab.data import discover_forecast_dates, list_series_for_date, load_scored_series
    from model_lab.metrics import calculate_metrics

    dates = list(discover_forecast_dates())
    chosen = None
    for date_token in reversed(dates):
        paths = list_series_for_date(date_token, ["production", "beta"])
        if paths.get("production") and paths.get("beta"):
            chosen = date_token
            break
    if chosen is None and dates:
        chosen = dates[-1]
    if chosen is None:
        return {"ok": False, "error": "No local forecast archives to score"}

    report = {"date": chosen, "series": {}, "ok": True}
    for series in ("production", "beta"):
        merged, meta = load_scored_series(chosen, series)
        metrics = calculate_metrics(merged) if not merged.empty else {}
        report["series"][series] = {"meta": meta, "metrics": metrics, "rows": len(merged)}
    out = REPORTS_DIR / "model_lab" / f"score_{chosen}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(out)
    return report


JOB_RUNNERS: dict[str, Callable[[dict], dict]] = {
    "sync_cdn_archives": _run_sync_cdn,
    "unpack_local_zips": _run_unpack_local,
    "score_latest_day": _run_score_latest,
}


def run_job(job_id: str, params: dict | None = None) -> dict:
    if job_id not in JOB_RUNNERS:
        raise KeyError(f"Unknown lab job: {job_id}")
    cfg = load_schedule_config()
    job_cfg = cfg.get("jobs", {}).get(job_id, {})
    merged = dict(job_cfg.get("params") or {})
    if params:
        merged.update(params)

    started = datetime.now().isoformat(timespec="seconds")
    entry = {"job_id": job_id, "started_at": started, "params": merged}
    try:
        with _job_lock:
            result = JOB_RUNNERS[job_id](merged)
        entry.update({
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "ok": bool(result.get("ok", True)),
            "result": result,
        })
    except Exception as exc:  # noqa: BLE001
        entry.update({
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc()[-2000:],
        })
    append_history(entry)
    return entry


def get_scheduler():
    global _scheduler
    created = False
    cfg = None
    with _scheduler_lock:
        if _scheduler is None:
            from apscheduler.schedulers.background import BackgroundScheduler
            from pytz import timezone

            cfg = load_schedule_config()
            tz_name = cfg.get("timezone") or "America/Chicago"
            _scheduler = BackgroundScheduler(timezone=timezone(tz_name))
            _scheduler.start()
            created = True
    # Register jobs outside the lock to avoid deadlock with apply_schedule().
    if created:
        apply_schedule(cfg)
    return _scheduler


def apply_schedule(config: dict | None = None) -> None:
    """(Re)register cron jobs from config onto the background scheduler."""
    from apscheduler.triggers.cron import CronTrigger
    from pytz import timezone

    config = config or load_schedule_config()
    save_schedule_config(config)

    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            from apscheduler.schedulers.background import BackgroundScheduler
            tz_name = config.get("timezone") or "America/Chicago"
            _scheduler = BackgroundScheduler(timezone=timezone(tz_name))
            _scheduler.start()
        scheduler = _scheduler

    tz = timezone(config.get("timezone") or "America/Chicago")
    for spec in LAB_JOBS:
        job_id = spec.id
        job_cfg = config.get("jobs", {}).get(job_id, {})
        existing = scheduler.get_job(job_id)
        if existing:
            scheduler.remove_job(job_id)
        if not job_cfg.get("enabled"):
            continue
        hour = int(job_cfg.get("hour", spec.default_hour))
        minute = int(job_cfg.get("minute", spec.default_minute))
        scheduler.add_job(
            run_job,
            trigger=CronTrigger(hour=hour, minute=minute, timezone=tz),
            id=job_id,
            kwargs={"job_id": job_id},
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )


def scheduler_status() -> list[dict]:
    scheduler = get_scheduler()
    cfg = load_schedule_config()
    rows = []
    for spec in LAB_JOBS:
        job_cfg = cfg.get("jobs", {}).get(spec.id, {})
        job = scheduler.get_job(spec.id)
        rows.append({
            "id": spec.id,
            "name": spec.name,
            "enabled": bool(job_cfg.get("enabled")),
            "hour": job_cfg.get("hour"),
            "minute": job_cfg.get("minute"),
            "params": job_cfg.get("params") or {},
            "next_run": job.next_run_time.isoformat() if job and job.next_run_time else None,
            "description": spec.description,
        })
    return rows
