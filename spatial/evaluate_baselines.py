from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.physics import evolve_fm

FEATURES = ["initial_fm", "initial_age_hours", "lead_hour", "rtma_temp_c", "rtma_rh", "rtma_wind_ms", "hrrr_temp_c", "hrrr_rh", "hrrr_wind_ms", "hrrr_precip_mm", "lat", "lon"]


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {"mae": float(mean_absolute_error(actual, predicted)), "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
            "bias": float(np.mean(predicted - actual)), "r2": float(r2_score(actual, predicted)), "samples": int(len(actual))}


def add_physics(frame):
    outputs = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.sort_values("lead_hour").groupby(["run_id", "station_id"]):
        outputs.loc[group.index] = evolve_fm(group.initial_fm.iloc[0], group.hrrr_temp_c.values, group.hrrr_rh.values)
    return outputs


def _evaluate_split(train, test):
    train, test = train.dropna(subset=FEATURES + ["target_fm", "physics_fm"]), test.dropna(subset=FEATURES + ["target_fm", "physics_fm"])
    model = xgb.XGBRegressor(n_estimators=300, learning_rate=.05, max_depth=5, objective="reg:squarederror", random_state=417)
    model.fit(train[FEATURES], train.target_fm)
    predictions = {"persistence": test.initial_fm.values, "physics": test.physics_fm.values, "incumbent_control": model.predict(test[FEATURES])}
    report = {name: metrics(test.target_fm, values) for name, values in predictions.items()}
    for name, values in predictions.items():
        scored = test[["target_fm", "lead_hour", "initial_fm"]].copy()
        scored["prediction"] = values
        report[name]["by_lead"] = {
            str(lead): metrics(group.target_fm, group.prediction)
            for lead, group in scored.groupby("lead_hour") if len(group) >= 2
        }
        drying = scored.target_fm < scored.initial_fm
        low = scored.target_fm <= 6
        report[name]["regimes"] = {
            "drying": metrics(scored.target_fm[drying], scored.prediction[drying]) if drying.sum() >= 2 else None,
            "wetting": metrics(scored.target_fm[~drying], scored.prediction[~drying]) if (~drying).sum() >= 2 else None,
            "critical_low_fm": metrics(scored.target_fm[low], scored.prediction[low]) if low.sum() >= 2 else None,
        }
    return report, model


def run():
    frame = pd.read_csv(paths.ALIGNED_DIR / "station_leads.csv", parse_dates=["forecast_init_time", "valid_time"])
    frame = frame[frame.target_mask == 1].copy()
    frame["physics_fm"] = add_physics(frame)
    dates = np.array(sorted(frame.forecast_init_time.dt.date.unique()))
    cutoff = dates[max(1, int(len(dates) * .8)) - 1]
    temporal_train, temporal_test = frame[frame.forecast_init_time.dt.date <= cutoff], frame[frame.forecast_init_time.dt.date > cutoff]
    stations = sorted(frame.station_id.unique())
    held_stations = {station for station in stations if sum(map(ord, station)) % 5 == 0}
    station_train, station_test = frame[~frame.station_id.isin(held_stations)], frame[frame.station_id.isin(held_stations)]
    lon_cutoff = float(frame.lon.median())
    region_train, region_test = frame[frame.lon < lon_cutoff], frame[frame.lon >= lon_cutoff]
    results = {}
    temporal_metrics, model = _evaluate_split(temporal_train, temporal_test)
    results["temporal"] = temporal_metrics
    results["station"] = _evaluate_split(station_train, station_test)[0] if len(station_test) else {}
    results["region"] = _evaluate_split(region_train, region_test)[0] if len(region_test) else {}
    results["split_manifest"] = {"temporal_cutoff": str(cutoff), "held_stations": sorted(held_stations), "region_lon_cutoff": lon_cutoff}
    model.save_model(paths.MODELS_DIR / "baseline_xgboost_control.json")
    output = paths.REPORTS_DIR / "baseline_metrics.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run()
