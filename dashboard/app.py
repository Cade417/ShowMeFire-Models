#!/usr/bin/env python3
"""Local FastAPI dashboard: data-source freshness, model registry, and
one-click triggers for both, backed entirely by pipeline/ - no new state,
everything here reads/writes the same manifests, models/config.json, and
experiment JSONLs the CLI scripts already use.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.jobs import JobManager
from pipeline.registry import DATA_SOURCES_BY_KEY, TRAIN_TARGETS_BY_TYPE, detection_confidence_target
from pipeline.status import all_model_status, all_source_status, experiments_for

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="ShowMeFire model-training dashboard")
jobs = JobManager()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/sources")
def list_sources():
    return all_source_status()


@app.post("/api/sources/{key}/run")
def run_source(key: str, dry_run: bool = False, days_back: int | None = None, limit: int | None = None,
                force: bool = False):
    source = DATA_SOURCES_BY_KEY.get(key)
    if source is None:
        raise HTTPException(404, f"Unknown source: {key!r}")
    if not source.available():
        raise HTTPException(409, f"{source.label} is not available (missing script or SMF_SSH_TARGET/SMF_REMOTE_ARCHIVE_DIR)")
    command = source.build_command(dry_run=dry_run, days_back=days_back or source.default_days_back,
                                    limit=limit, force=force)
    job = jobs.start(source.label, command, source.script.parent.parent)
    return job.as_dict()


@app.get("/api/models")
def list_models():
    return all_model_status()


@app.get("/api/models/{model_type}/experiments")
def model_experiments(model_type: str):
    return experiments_for(model_type)


@app.post("/api/models/{model_type}/train")
def train_model(model_type: str):
    target = TRAIN_TARGETS_BY_TYPE.get(model_type)
    if target is None and model_type == "detection_confidence":
        target = detection_confidence_target()
    if target is None:
        raise HTTPException(404, f"No single-command retrain wired up for {model_type!r} yet "
                                  f"(see docs/pipeline_dashboard.md for how to add one)")
    if not target.available():
        raise HTTPException(409, f"{target.label}'s training script was not found on disk")
    job = jobs.start(target.label, target.build_command(), target.cwd)
    return job.as_dict()


@app.get("/api/jobs")
def list_jobs():
    return [job.as_dict() for job in jobs.list()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job: {job_id!r}")
    return {**job.as_dict(), "log": job.tail()}


def main():
    import uvicorn
    uvicorn.run("dashboard.app:app", host="127.0.0.1", port=8787, reload=False)


if __name__ == "__main__":
    main()
