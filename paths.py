"""
Central data-location config for the ShowMeFire-Models training repo.

Every data path in this repo is anchored under SMF_DATA_ROOT (env var),
defaulting to a `data/` folder inside this repo (gitignored) if unset. This
is what makes the repo portable across machines - point SMF_DATA_ROOT at
wherever the data actually lives (this machine's disk, an external drive,
a desktop's local storage) and every script resolves paths from it without
any other change.

Mirrors the directory names already used on the server (api/cache/hrrr,
api/archive/raw_data, api/archive/forecasts) so unpack_archive_zip.py's
zip-member routing needs no changes.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("SMF_DATA_ROOT", REPO_ROOT / "data")).resolve()

CACHE_DIR = DATA_ROOT / "cache"
CACHE_HRRR_DIR = CACHE_DIR / "hrrr"

ARCHIVE_DIR = DATA_ROOT / "archive"
ARCHIVE_RAW_DATA_DIR = ARCHIVE_DIR / "raw_data"
ARCHIVE_FORECASTS_DIR = ARCHIVE_DIR / "forecasts"

ARCHIVE_ZIPS_DIR = DATA_ROOT / "archive_zips"  # where zips from the server's data_archive_day get placed for unpacking

TRAINING_DATA_DIR = DATA_ROOT / "data"  # CSVs / intermediates (training_set_mo.csv, final_training_data.csv, etc.)
SNAPSHOTS_DIR = TRAINING_DATA_DIR / "snapshots"

DB_PATH = DATA_ROOT / "showmefire.db"  # this repo's own independent training DB - never the server's

MODELS_DIR = DATA_ROOT / "models"  # this repo's own independent model registry (see models/versioning.py)

PLOTS_DIR = DATA_ROOT / "plots"  # diagnostic plots (feature importance, station previews, etc.)

for _d in (CACHE_HRRR_DIR, ARCHIVE_RAW_DATA_DIR, ARCHIVE_FORECASTS_DIR, ARCHIVE_ZIPS_DIR,
           TRAINING_DATA_DIR, SNAPSHOTS_DIR, MODELS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
