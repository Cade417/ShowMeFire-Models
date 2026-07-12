import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

sys.path.append(str(BASE_DIR.parent))
import paths

# Standalone/experimental - kept separate from the main fuel_moisture registry,
# but anchored under the same SMF_DATA_ROOT so nothing ends up inside the git repo.
SOURCE_DATA_PATH = paths.TRAINING_DATA_DIR / "final_training_data.csv"
OUTPUT_DATA_DIR = paths.DATA_ROOT / "fire-danger-model" / "data"
MODELS_DIR = paths.DATA_ROOT / "fire-danger-model" / "models"
REPORTS_DIR = paths.DATA_ROOT / "fire-danger-model" / "reports"
for _d in (OUTPUT_DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_DATA_PATH = OUTPUT_DATA_DIR / "prepared_train.csv"
TEST_DATA_PATH = OUTPUT_DATA_DIR / "prepared_test.csv"
SPLIT_META_PATH = OUTPUT_DATA_DIR / "split_metadata.json"

LATEST_MODEL_PATH = MODELS_DIR / "fire_danger_model.json"
LATEST_MODEL_META_PATH = MODELS_DIR / "fire_danger_model_meta.json"

RANDOM_STATE = 42
TRAIN_FRACTION = 0.8

# Candidate predictors for the standalone fire danger model.
# We prioritize columns already present in data/final_training_data.csv.
FEATURE_CANDIDATES = [
    "temp_c",
    "rel_humidity",
    "wind_speed_ms",
    "hour",
    "month",
    "emc_baseline",
    "temp_mean_3h",
    "rh_mean_3h",
    "temp_mean_6h",
    "rh_mean_6h",
    "precip_1h",
    "precip_3h",
    "precip_6h",
    "precip_24h",
    "hours_since_rain",
]

TARGET_SCORE_COL = "fire_danger_score"
TARGET_CATEGORY_COL = "fire_danger_category"

CATEGORY_LABELS = ["Low", "Moderate", "Elevated", "Critical", "Extreme"]
ALL_CATEGORY_IDS = [0, 1, 2, 3, 4]
HIGH_IMPACT_CATEGORY_IDS = [2, 3, 4]

# Minimum support expected for elevated-risk classes in the test split.
MIN_HIGH_IMPACT_SUPPORT = 25

# Default score boundaries for 5-class mapping if no calibration metadata is found.
DEFAULT_CATEGORY_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]

# Gate policy for standalone evaluation.
PRIMARY_GATE_MIN_MACRO_F1 = 0.55
BASELINE_MAX_MACRO_F1_DEGRADATION = 0.05
