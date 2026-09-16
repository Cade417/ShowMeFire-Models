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
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass
DATA_ROOT = Path(os.getenv("SMF_DATA_ROOT", REPO_ROOT / "data")).resolve()

CACHE_DIR = DATA_ROOT / "cache"
CACHE_HRRR_DIR = CACHE_DIR / "hrrr"
CACHE_RTMA_DIR = CACHE_DIR / "rtma"
CACHE_RRFS_DIR = CACHE_DIR / "rrfs"
CACHE_FV3HIRES_DIR = CACHE_DIR / "fv3hires"

ARCHIVE_DIR = DATA_ROOT / "archive"
ARCHIVE_RAW_DATA_DIR = ARCHIVE_DIR / "raw_data"
ARCHIVE_FORECASTS_DIR = ARCHIVE_DIR / "forecasts"

ARCHIVE_ZIPS_DIR = DATA_ROOT / "archive_zips"  # where zips from the server's data_archive_day get placed for unpacking

TRAINING_DATA_DIR = DATA_ROOT / "data"  # CSVs / intermediates (training_set_mo.csv, final_training_data.csv, etc.)
SNAPSHOTS_DIR = TRAINING_DATA_DIR / "snapshots"

DB_PATH = DATA_ROOT / "showmefire.db"  # this repo's own independent training DB - never the server's

MODELS_DIR = DATA_ROOT / "models"  # this repo's own independent model registry (see models/versioning.py)

PLOTS_DIR = DATA_ROOT / "plots"  # diagnostic plots (feature importance, station previews, etc.)
ALIGNED_DIR = DATA_ROOT / "aligned"
PRECIP_ALIGNED_DATASET = ALIGNED_DIR / "station_leads_precipitation-v1.csv"
V4_ALIGNED_DATASET = ALIGNED_DIR / "station_leads_v4_precipitation-v1.csv"
REPORTS_DIR = DATA_ROOT / "reports"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"  # one JSONL per model_type, one row per training run (see pipelines/experiment_log.py)
V4_SPLIT_MANIFEST = REPORTS_DIR / "v4_precipitation-v1_split_manifest.json"
V4_CANDIDATE_DIR = MODELS_DIR / "v4_precipitation-v1_shadow_candidate"
V5_SPLIT_MANIFEST = REPORTS_DIR / "v5_summer_guarded_split_manifest.json"
V5_CANDIDATE_DIR = MODELS_DIR / "v5_summer_guarded_shadow_candidate-r2"
STATIC_DIR = DATA_ROOT / "static"
STATIC_SOURCE_DIR = STATIC_DIR / "source"
STATIC_BUNDLE_DIR = STATIC_DIR / "bundles"

# fire_risk_fusion (county-day ignition model) - a separate model family
# from the fuel_moisture lineage above, see risk_fusion/risk_fusion_contract.py.
FIRE_LABELS_DIR = DATA_ROOT / "fire_labels"  # pulled from the API's export_fire_labels.py output, never a live DB read
RISK_FUSION_DIR = DATA_ROOT / "risk_fusion"
RISK_FUSION_PANEL = RISK_FUSION_DIR / "county_days.csv"
RISK_FUSION_SPLIT_MANIFEST = REPORTS_DIR / "risk_fusion_split_manifest.json"
RISK_FUSION_CANDIDATE_DIR = MODELS_DIR / "risk_fusion_shadow_candidate"

# fire_weather_ml (station-hour physical fire-behavior emulator) - a third,
# independent model family, trained against the same Rothermel calculation
# api/services/spread_rate.py runs live, not against fire-occurrence
# reports. See fire_weather_ml/__init__.py and docs/fire_weather_ml_plan.md.
FIRE_WEATHER_ML_DIR = DATA_ROOT / "fire_weather_ml"
FIRE_WEATHER_ML_PANEL = FIRE_WEATHER_ML_DIR / "station_panel.csv"
FIRE_WEATHER_ML_SPLIT_MANIFEST = REPORTS_DIR / "fire_weather_ml_split_manifest.json"
FIRE_WEATHER_ML_CANDIDATE_DIR = MODELS_DIR / "fire_weather_ml_shadow_candidate"

# fire_weather_index (numeric, non-rule fire-weather danger score, blending
# HRRR/RRFS/FV3-HIRES) - a fourth, independent model family. See
# fire_weather_index/__init__.py for how this differs from risk_fusion.
FIRE_WEATHER_INDEX_DIR = DATA_ROOT / "fire_weather_index"
FIRE_WEATHER_INDEX_PANEL = FIRE_WEATHER_INDEX_DIR / "county_days.csv"
FIRE_WEATHER_INDEX_SPLIT_MANIFEST = REPORTS_DIR / "fire_weather_index_split_manifest.json"
FIRE_WEATHER_INDEX_CANDIDATE_DIR = MODELS_DIR / "fire_weather_index_shadow_candidate"

for _d in (CACHE_HRRR_DIR, CACHE_RTMA_DIR, CACHE_RRFS_DIR, CACHE_FV3HIRES_DIR, ARCHIVE_RAW_DATA_DIR, ARCHIVE_FORECASTS_DIR, ARCHIVE_ZIPS_DIR,
           TRAINING_DATA_DIR, SNAPSHOTS_DIR, MODELS_DIR, PLOTS_DIR, ALIGNED_DIR, REPORTS_DIR, EXPERIMENTS_DIR,
           STATIC_SOURCE_DIR, STATIC_BUNDLE_DIR, FIRE_LABELS_DIR, RISK_FUSION_DIR, FIRE_WEATHER_ML_DIR, FIRE_WEATHER_INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)
