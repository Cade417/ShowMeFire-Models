"""Declarative registry of data sources and trainable models.

Adding a new puller or a new trainable model means adding one entry here -
nothing else in this package needs to change.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

REPO_ROOT = Path(__file__).resolve().parent.parent

# Steps that accept --days-back (RTMA derives its window from local HRRR
# valid times instead - see scripts/backfill_rtma_for_hrrr.py).
STEPS_SUPPORTING_DAYS_BACK = {"synoptic", "hrrr", "rrfs", "fv3hires"}


@dataclass
class DataSource:
    key: str
    label: str
    script: Path  # python script, or a .sh script run via bash
    manifest_path: Path
    manifest_key: str  # top-level dict key inside the manifest holding per-item records ("days"/"runs"/"analyses"/"cycles")
    default_days_back: Optional[int] = None
    supports_force: bool = True
    supports_window_args: bool = True  # --start/--end/--days-back/--limit/--workers; r2_restore supports none of these
    note: str = ""

    def available(self) -> bool:
        if self.script.suffix == ".sh":
            return self.script.exists() and bool(os.getenv("SMF_SSH_TARGET")) and bool(os.getenv("SMF_REMOTE_ARCHIVE_DIR"))
        return self.script.exists()

    def build_command(self, *, dry_run=False, days_back=None, start=None, end=None,
                       limit=None, workers=None, force=False) -> List[str]:
        if self.script.suffix == ".sh":
            # Git Bash / MSYS's bash.exe re-parses the raw Windows command
            # line itself (there's no real argv on Windows, every process
            # gets one string and parses it per its own rules) and treats
            # "\" + letter as an escape sequence, same as an interactive
            # shell would - confirmed live 2026-10-08 running this exact
            # command from the dashboard's subprocess.Popen (no shell
            # involved, so this isn't Python's doing). Forward slashes sidestep
            # it entirely and work identically on Windows.
            return ["bash", str(self.script).replace("\\", "/")]
        command = [sys.executable, str(self.script)]
        if self.supports_window_args:
            if start:
                command += ["--start", start]
            if end:
                command += ["--end", end]
            if days_back is not None and self.key in STEPS_SUPPORTING_DAYS_BACK:
                command += ["--days-back", str(days_back)]
            if limit is not None:
                command += ["--limit", str(limit)]
            if workers is not None:
                command += ["--workers", str(workers)]
        if dry_run:
            command.append("--dry-run")
        if force and self.supports_force:
            command.append("--force")
        return command


DATA_SOURCES: List[DataSource] = [
    DataSource(
        key="r2_restore", label="Restore from local R2 mirror (no network)",
        script=REPO_ROOT / "pipeline" / "r2_restore.py",
        manifest_path=paths.DATA_ROOT / "r2_restore_manifest.json",
        manifest_key="restored", supports_force=False, supports_window_args=False,
        note="Materializes whatever scripts/migration/r2_pull.py already pulled from R2 into the "
             "working cache dirs, before any of the sources below hit the network. Covers hrrr/rrfs/"
             "observations only - RTMA and FV3-HIRES were never archived to R2 (see pipeline/r2_restore.py).",
    ),
    DataSource(
        key="synoptic", label="Synoptic station observations",
        script=REPO_ROOT / "scripts" / "backfill_synoptic.py",
        manifest_path=paths.ARCHIVE_RAW_DATA_DIR / "backfill_manifest.json",
        manifest_key="days", default_days_back=14,
    ),
    DataSource(
        key="hrrr", label="HRRR runs (cropped to Missouri)",
        script=REPO_ROOT / "scripts" / "backfill_hrrr.py",
        manifest_path=paths.CACHE_HRRR_DIR / "backfill_manifest.json",
        manifest_key="runs", default_days_back=21,
    ),
    DataSource(
        key="rtma", label="RTMA analyses for cached HRRR runs",
        script=REPO_ROOT / "scripts" / "backfill_rtma_for_hrrr.py",
        manifest_path=paths.CACHE_RTMA_DIR / "backfill_manifest.json",
        manifest_key="analyses", default_days_back=None,
        note="Window is derived from local HRRR valid times, not a standalone lookback.",
    ),
    DataSource(
        key="rrfs", label="RRFS cycles (noaa-rrfs-ops-pds)",
        script=REPO_ROOT / "scripts" / "backfill_rrfs.py",
        manifest_path=paths.CACHE_RRFS_DIR / "backfill_manifest.json",
        manifest_key="cycles", default_days_back=21,
        note="Only the 00/06/12/18 UTC cycles exist; bucket starts 2026-08-12, nothing before that.",
    ),
    DataSource(
        key="fv3hires", label="FV3-HIRES cycles (NOMADS)",
        script=REPO_ROOT / "scripts" / "capture_fv3hires.py",
        manifest_path=paths.CACHE_FV3HIRES_DIR / "capture_manifest.json",
        manifest_key="cycles", default_days_back=3, supports_force=False,
        note="NOMADS short rolling window - a missed cycle cannot be recovered later.",
    ),
    DataSource(
        key="archives", label="Archive zip pull (server rsync)",
        script=REPO_ROOT / "scripts" / "pull_archives.sh",
        manifest_path=paths.ARCHIVE_ZIPS_DIR,  # not a JSON manifest - status.py lists files directly
        manifest_key="",
        note="Requires SMF_SSH_TARGET/SMF_REMOTE_ARCHIVE_DIR; not applicable when the training machine has no upstream server.",
    ),
]

DATA_SOURCES_BY_KEY = {source.key: source for source in DATA_SOURCES}

# Order used by pipeline/run.py: r2_restore first (pure local file copy, no
# network - fills in whatever's already sitting in the pulled R2 mirror so
# the network-bound steps below have less to actually fetch), then Synoptic
# obs -> HRRR -> RTMA-for-HRRR (RTMA's window depends on locally cached HRRR
# valid times, so HRRR must run first). RRFS/FV3-HIRES/archives are independent.
SEQUENCED_STEPS = ("r2_restore", "synoptic", "hrrr", "rtma")
INDEPENDENT_STEPS = ("rrfs", "fv3hires", "archives")


@dataclass
class TrainTarget:
    model_type: str
    label: str
    # Either a single script (list of [python_exe, script_path, *extra_args])
    # or a python -m module invocation - both run as a plain subprocess.
    module: Optional[str] = None
    script: Optional[Path] = None
    extra_args: List[str] = field(default_factory=list)
    cwd: Path = REPO_ROOT
    note: str = ""

    def available(self) -> bool:
        if self.module:
            return True  # resolved at run time by -m; failure surfaces in the job log
        return bool(self.script and self.script.exists())

    def build_command(self) -> List[str]:
        if self.module:
            return [sys.executable, "-m", self.module, *self.extra_args]
        return [sys.executable, str(self.script), *self.extra_args]


TRAIN_TARGETS: List[TrainTarget] = [
    TrainTarget(
        model_type="fuel_moisture", label="Fuel moisture (XGBoost)",
        script=REPO_ROOT / "pipelines" / "retrain_fuel_moisture.py",
    ),
    TrainTarget(
        model_type="ensemble_fire_danger", label="Ensemble fire danger (HREF/REFS)",
        module="ensemble_fire_danger.retrain",
    ),
    TrainTarget(
        model_type="fire_outlook", label="Seven-day fire outlook (verification + calibration)",
        module="fire_outlook.retrain",
        note="Needs verification records exported by the fire-outlook worker in SMF_DATA_ROOT/fire_outlook/verification",
    ),
    TrainTarget(
        model_type="fire_weather_index", label="Fire weather danger index (calibration)",
        script=REPO_ROOT / "fire_weather_index" / "calibrate.py",
    ),
    TrainTarget(
        model_type="fire_weather_ml", label="Fire weather ML (station-hour emulator)",
        script=REPO_ROOT / "fire_weather_ml" / "fit_model.py",
    ),
    TrainTarget(
        model_type="fire_risk_fusion", label="Fire risk fusion (Poisson GLM)",
        script=REPO_ROOT / "risk_fusion" / "fit_glm.py",
    ),
    TrainTarget(
        model_type="detection_dataset", label="Detection confidence dataset (SMF feedback + NERIS + ML variables)",
        module="detection_dataset.build",
        note="Needs the API's export_detection_training_data.py output in SMF_DATA_ROOT/fire_labels",
    ),
    TrainTarget(
        model_type="detection_confidence_v2", label="Vegetation-fire confidence v2 (reviewed labels, shadow candidate)",
        module="detection_dataset.confidence_v2.train",
        note="Fits nothing until human-reviewed negatives exist; writes a beta bundle for SHADOW scoring only",
    ),
]
# Model families without a single-command retrain (spatial v4/v5/hybrid/
# station-sequence are each a multi-step fit -> calibrate -> evaluate ->
# register pipeline) are intentionally left out of TRAIN_TARGETS rather than
# guessing a step order. Add them here once a specific pipeline step order
# is decided - see docs/pipeline_dashboard.md for how.

TRAIN_TARGETS_BY_TYPE = {target.model_type: target for target in TRAIN_TARGETS}


def detection_confidence_target() -> Optional[TrainTarget]:
    """The detection-confidence model lives in the sibling api/ repo, not
    here. Only returns a target if SMF_API_REPO_ROOT is set and points at a
    real checkout - otherwise callers should show it as "not configured"
    rather than erroring."""
    api_root = os.getenv("SMF_API_REPO_ROOT", "").strip()
    if not api_root:
        return None
    script = Path(api_root) / "detection-confidence-model" / "retrain.py"
    if not script.exists():
        return None
    return TrainTarget(
        model_type="detection_confidence", label="Detection confidence (XGBoost)",
        script=script, cwd=script.parent.parent,
        note=f"Resolved from SMF_API_REPO_ROOT={api_root}",
    )
