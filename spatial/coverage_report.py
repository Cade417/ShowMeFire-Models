from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths


def generate(dataset=None, output=None):
    dataset = Path(dataset) if dataset else (paths.PRECIP_ALIGNED_DATASET if paths.PRECIP_ALIGNED_DATASET.exists()
                                              else paths.ALIGNED_DIR / "station_leads.csv")
    frame = pd.read_csv(dataset, parse_dates=["forecast_init_time", "valid_time"])
    observed = frame[frame.target_mask == 1]
    per_day = observed.groupby(observed.forecast_init_time.dt.date).agg(targets=("target_fm", "size"), stations=("station_id", "nunique"))
    report = {
        "dataset": str(dataset), "rows": int(len(frame)), "observed_targets": int(len(observed)),
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
    if {"target_rh", "target_wind_ms"}.issubset(frame.columns):
        category_mask = observed.target_rh.notna() & observed.target_wind_ms.notna()
        report["observed_category_labels"] = int(category_mask.sum())
        report["observed_category_label_fraction"] = float(category_mask.mean()) if len(observed) else 0.0
        report["target_match_age_minutes"] = observed.target_match_age_minutes.describe().replace({np.nan: None}).to_dict()
    precip_column = "hrrr_precip_increment_mm" if "hrrr_precip_increment_mm" in frame else None
    if precip_column:
        available = frame.get("precip_available", frame[precip_column].notna()).astype(bool)
        amounts = frame[precip_column]
        unique = frame.loc[available].drop_duplicates(["run_id", "lead_hour", "station_id"])
        bands = pd.cut(unique[precip_column].fillna(-1), bins=[-np.inf, 0, 0.1, 2, 10, np.inf],
                       labels=["none", "trace", "light", "moderate", "heavy"])
        report["precipitation"] = {
            "available_fraction": float(available.mean()),
            "nonzero_rows": int((amounts.fillna(0) > 0).sum()),
            "maximum_interval_mm": float(amounts.max()),
            "by_month": {str(k): int(v) for k, v in unique.groupby(unique.valid_time.dt.month).size().items()},
            "rain_rows_by_lead": {str(k): int(v) for k, v in unique[unique[precip_column] > 0].groupby("lead_hour").size().items()},
            "intensity_bands": {str(k): int(v) for k, v in bands.value_counts(sort=False).items()},
            "stations_with_rain": int(unique.loc[unique[precip_column] > 0, "station_id"].nunique()),
        }
    report["spatial_model_data_gate"] = {
        "pass": report["usable_initialization_dates"] >= 180 and report["median_stations_per_day"] >= 17,
        "requires_dates": 180, "requires_median_stations": 17,
    }
    output = Path(output) if output else paths.REPORTS_DIR / "coverage.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str))
    temporary.replace(output)
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    generate(args.dataset, args.output)
