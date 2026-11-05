"""Generic background-subprocess job runner.

Shared by the CLI orchestrator (pipeline/run.py) and the dashboard
(dashboard/app.py) so a "run this puller" and "run this training script"
button do exactly the same thing a terminal invocation would - launch the
script's own CLI as a subprocess, with its own manifest/retry logic
untouched, and just capture stdout/stderr to a logfile this process can
tail. Nothing here is persisted across process restarts - job history is
the logfiles on disk, re-enumerated at startup.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

JOBS_DIR = paths.REPORTS_DIR / "dashboard_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Job:
    id: str
    label: str
    command: List[str]
    cwd: Path
    log_path: Path
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    status: str = "running"  # running | ok | failed
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    returncode: Optional[int] = None

    def refresh(self):
        if self.process is None or self.status != "running":
            return
        code = self.process.poll()
        if code is not None:
            self.returncode = code
            self.status = "ok" if code == 0 else "failed"
            self.finished_at = time.time()

    def tail(self, max_bytes: int = 20000) -> str:
        if not self.log_path.exists():
            return ""
        data = self.log_path.read_bytes()
        return data[-max_bytes:].decode("utf-8", errors="replace")

    def as_dict(self) -> dict:
        self.refresh()
        return {
            "id": self.id, "label": self.label, "command": self.command,
            "status": self.status, "started_at": self.started_at,
            "finished_at": self.finished_at, "returncode": self.returncode,
            "log_path": str(self.log_path),
        }


class JobManager:
    """One per process. The dashboard keeps a module-level instance; the CLI
    orchestrator creates its own short-lived instance and runs steps
    synchronously (wait_for_completion) instead of polling."""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, label: str, command: List[str], cwd: Path) -> Job:
        job_id = uuid.uuid4().hex[:12]
        log_path = JOBS_DIR / f"{job_id}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=str(cwd), stdout=log_file, stderr=subprocess.STDOUT,
            env=_subprocess_env(),
        )
        job = Job(id=job_id, label=label, command=command, cwd=cwd,
                  log_path=log_path, process=process)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.started_at, reverse=True)

    def wait(self, job: Job, timeout: Optional[float] = None) -> Job:
        if job.process is not None:
            job.process.wait(timeout=timeout)
        job.refresh()
        return job


def _subprocess_env():
    import os
    # Same fix documented in scripts/sync_training_data.py's _run(): a
    # non-interactive child process falls back to the ANSI codepage and
    # raises UnicodeEncodeError on any emoji/checkmark print (Herbie does
    # this on every fetch attempt) without these two set.
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
