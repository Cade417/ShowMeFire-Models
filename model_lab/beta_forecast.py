"""Run tagged local beta forecast products with CDN upload disabled."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import paths

API_ROOT = Path(__file__).resolve().parents[2] / "api"
FORECAST_SCRIPT = API_ROOT / "forecast" / "DailyForecast_ModelFD.py"
MAP_NAMES = (
    "mo-forecastfiredanger-beta",
    "mo-forecastfuelmoisture-beta",
    "mo-forecastminrh-beta",
    "mo-forecastmaxwind-beta",
    "mo-forecastmaxtemp-beta",
    "mo-forecastrainfall-beta",
    "mo-forecastswe-beta",
)


def run_beta_forecast(
    *,
    fm_model_path: str | Path | None = None,
    fd_model_path: str | Path | None = None,
    fd_meta_path: str | Path | None = None,
    tag: str,
    python_executable: str | None = None,
    timeout_seconds: int = 3600,
) -> dict:
    """Generate a local tagged ModelFD forecast and copy map PNGs to data root.

    The API script is run in a subprocess with `uploadForecast=false`,
    `SMF_BETA_OUTPUT_ROOT`, and model path overrides. No public CDN writes occur.
    """
    if not tag or not tag.replace("-", "").replace("_", "").isalnum():
        raise ValueError("tag must contain only letters, numbers, '-' or '_'")
    if not fm_model_path:
        raise ValueError(
            "A fuel-moisture model artifact is required. "
            "Select a trained FM model or register a local stable model."
        )
    if not Path(fm_model_path).exists():
        raise FileNotFoundError(f"Fuel-moisture model artifact not found: {fm_model_path}")
    if not FORECAST_SCRIPT.exists():
        raise FileNotFoundError(f"Missing API forecast script: {FORECAST_SCRIPT}")

    output_root = paths.DATA_ROOT
    map_dir = paths.DATA_ROOT / "model_lab" / "beta_maps" / tag
    map_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "SMF_BETA_SERIES_TAG": tag,
        "SMF_BETA_OUTPUT_ROOT": str(output_root),
        "SMF_HRRR_CACHE_DIR": str(paths.CACHE_HRRR_DIR),
        # RAWS observations must come from the deployed API for local beta
        # forecasts; callers can override this for a local API checkout.
        "SMF_API_BASE_URL": os.getenv("SMF_API_BASE_URL", "https://api.showmefire.org"),
        "uploadForecast": "false",
    })
    if fm_model_path:
        env["SMF_FM_MODEL_PATH"] = str(Path(fm_model_path).resolve())
    if fd_model_path:
        env["SMF_FD_MODEL_PATH"] = str(Path(fd_model_path).resolve())
    if fd_meta_path:
        env["SMF_FD_MODEL_META_PATH"] = str(Path(fd_meta_path).resolve())

    executable = python_executable or sys.executable
    cache_files = list(paths.CACHE_HRRR_DIR.glob("hrrr_*.nc")) if paths.CACHE_HRRR_DIR.exists() else []
    if not cache_files:
        return {
            "ok": False,
            "tag": tag,
            "returncode": None,
            "maps": [],
            "archive_dir": str(paths.ARCHIVE_FORECASTS_DIR),
            "error": (
                f"No local HRRR cache found at {paths.CACHE_HRRR_DIR}. "
                "Use Data sync to download and unpack an archive containing "
                "the required HRRR run before generating a local beta."
            ),
            "stdout": "",
            "stderr": "",
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

    completed = subprocess.run(
        [executable, str(FORECAST_SCRIPT)],
        cwd=str(API_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )

    copied = []
    api_images = API_ROOT / "images"
    suffix = f"-{tag}.png"
    if api_images.exists():
        for base in MAP_NAMES:
            source = api_images / f"{base}{suffix}"
            if source.exists():
                destination = map_dir / source.name
                shutil.copy2(source, destination)
                copied.append(str(destination))

    return {
        "ok": completed.returncode == 0,
        "tag": tag,
        "returncode": completed.returncode,
        "maps": copied,
        "archive_dir": str(paths.ARCHIVE_FORECASTS_DIR),
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-8000:],
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
