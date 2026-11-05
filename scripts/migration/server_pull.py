#!/usr/bin/env python3
"""Pulls the VPS's own archive directory (separate from R2) via the
existing pull_archives.sh + unpack_archive_zip.py - that path already
works and is unchanged here; this just wraps it with migration_status.json
progress updates so the dashboard shows this phase alongside the R2 pull.

Usage:
    python scripts/migration/server_pull.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[2]))
import status as mstatus

REPO_ROOT = Path(__file__).resolve().parents[2]

# On Windows, plain "bash" can resolve to the WSL launcher in
# System32 ahead of Git Bash on PATH, depending on shell/session - WSL
# doesn't understand drive-letter paths like M:/..., so pull_archives.sh
# (which needs Git Bash's rsync/ssh) silently fails with "No such file
# or directory" even though the script is right there. Prefer a real
# Git-for-Windows bash.exe if one exists before falling back to PATH.
_GIT_BASH_CANDIDATES = [
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files\Git\usr\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
]


def _find_bash():
    for candidate in _GIT_BASH_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return shutil.which("bash") or "bash"


def run():
    ssh_target = os.getenv("SMF_SSH_TARGET")
    remote_dir = os.getenv("SMF_REMOTE_ARCHIVE_DIR")
    if not ssh_target or not remote_dir:
        mstatus.update_phase("server_pull", status="skipped",
                              message="SMF_SSH_TARGET / SMF_REMOTE_ARCHIVE_DIR not set")
        print("Skipped: SMF_SSH_TARGET / SMF_REMOTE_ARCHIVE_DIR not set")
        return 0
    mstatus.update_phase("server_pull", status="running",
                          message=f"rsync {ssh_target}:{remote_dir} -> data/archive_zips/")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    script_path = (REPO_ROOT / "scripts" / "pull_archives.sh").as_posix()
    result = subprocess.run([_find_bash(), script_path], cwd=REPO_ROOT, env=env)
    final_status = "completed" if result.returncode == 0 else "failed"
    mstatus.update_phase(
        "server_pull", status=final_status,
        message="Server archive pull finished" if final_status == "completed"
        else f"pull_archives.sh exited with code {result.returncode}",
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(run())
