"""Explicit, confirmation-gated operations against the API checkout."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import paths

API_ROOT = Path(os.getenv("SMF_API_ROOT", Path(__file__).resolve().parents[2] / "api"))
AUDIT_PATH = paths.DATA_ROOT / "model_lab" / "live_audit.jsonl"


def _confirm(token: str):
    if token != "LIVE":
        raise PermissionError("Type LIVE exactly to authorize an API/CDN operation")


def _run(command: list[str], *, token: str, timeout: int = 1800) -> dict:
    _confirm(token)
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(
        command,
        cwd=str(API_ROOT),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    record = {
        "timestamp": started,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def import_release(model_type: str, tag: str, repo: str, token: str, bump: str = "patch") -> dict:
    return _run(
        ["python", "pipelines/import_model.py", "--model", model_type, "--tag", tag, "--repo", repo, "--bump", bump],
        token=token,
    )


def promote_api(model_type: str, token: str, version: str | None = None) -> dict:
    command = ["python", "pipelines/promote_model.py", "--model", model_type]
    if version:
        command.extend(["--version", version])
    return _run(command, token=token)


def shadow_diagnostics(token: str) -> dict:
    # Diagnostics are read-only but kept behind LIVE so the panel has one
    # unambiguous boundary and cannot be mistaken for the local ladder.
    return _run(["python", "-c", "from services.model_shadow import diagnostics; import json; print(json.dumps(diagnostics()))"], token=token)


def upload_maps(files: list[str], keys: list[str], token: str) -> dict:
    _confirm(token)
    if len(files) != len(keys):
        raise ValueError("files and keys must have equal length")
    script = (
        "from scripts.upload_cdn import upload_to_cdn; "
        f"upload_to_cdn({files!r}, {keys!r})"
    )
    return _run(["python", "-c", script], token=token)
