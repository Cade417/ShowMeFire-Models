from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths


def generate():
    frame = pd.read_csv(paths.ALIGNED_DIR / "station_leads.csv", parse_dates=["forecast_init_time", "valid_time"])
    observed = frame[frame.target_mask == 1]
    per_day = observed.groupby(observed.forecast_init_time.dt.date).agg(targets=("target_fm", "size"), stations=("station_id", "nunique"))
    report = {
        "rows": int(len(frame)), "observed_targets": int(len(observed)),
        "usable_initialization_dates": int(frame.forecast_init_time.dt.date.nunique()),
        "stations": int(observed.station_id.nunique()),
        "date_start": frame.forecast_init_time.min().isoformat() if len(frame) else None,
        "date_end": frame.forecast_init_time.max().isoformat() if len(frame) else None,
        "median_stations_per_day": float(per_day.stations.median()) if len(per_day) else 0,
        "initial_age_hours": frame.initial_age_hours.describe().replace({np.nan: None}).to_dict(),
        "targets_by_lead": {str(k): int(v) for k, v in observed.groupby("lead_hour").size().items()},
        "targets_by_month": {str(k): int(v) for k, v in observed.groupby(observed.forecast_init_time.dt.month).size().items()},
        "critical_low_fm_targets": int((observed.target_fm <= 6).sum()),
        "missing_fraction": {col: float(frame[col].isna().mean()) for col in ["rtma_temp_c", "rtma_rh", "hrrr_temp_c", "hrrr_rh", "target_fm"]},
    }
    report["spatial_model_data_gate"] = {
        "pass": report["usable_initialization_dates"] >= 180 and report["median_stations_per_day"] >= 20,
        "requires_dates": 180, "requires_median_stations": 20,
    }
    output = paths.REPORTS_DIR / "coverage.json"
    output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    generate()
