"""Create the V5 evidence contract and a development-only incumbent error atlas."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.evaluate_baselines import FEATURES as INCUMBENT_FEATURES
from spatial.station_contract import basic_metrics
from spatial.v5_contract import load_or_create, reject_forbidden
from spatial.v5_features import FEATURES, add_v5_features, regime_labels
from spatial.v5_training import load_frame
from spatial.search_v5 import atomic


def grouped_metrics(scored, columns, minimum=100):
    report = {}
    for column in columns:
        report[column] = {str(key): basic_metrics(group.target_fm, group.prediction)
                          for key, group in scored.groupby(column, observed=False) if len(group) >= minimum}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=paths.V4_ALIGNED_DATASET)
    parser.add_argument("--v4-manifest", type=Path, default=paths.V4_SPLIT_MANIFEST)
    parser.add_argument("--manifest", type=Path, default=paths.V5_SPLIT_MANIFEST)
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "v5_incumbent_error_atlas.json")
    args = parser.parse_args(); frame = load_frame(args.dataset)
    manifest = load_or_create(args.manifest, frame, args.dataset, args.v4_manifest, FEATURES)
    rows = []
    for fold in manifest["folds"]:
        print(f"Building incumbent error atlas {fold['name']}...", flush=True)
        reject_forbidden(manifest, fold["train_runs"] + fold["validation_runs"], "V5 error atlas")
        training = frame[frame.run_id.isin(set(fold["train_runs"])) & (frame.target_mask == 1)].dropna(subset=["target_fm"])
        validation = frame[frame.run_id.isin(set(fold["validation_runs"])) & (frame.target_mask == 1)].copy()
        model = xgb.XGBRegressor(n_estimators=300, learning_rate=.05, max_depth=5,
                                 objective="reg:squarederror", random_state=417, tree_method="hist")
        model.fit(training[INCUMBENT_FEATURES], training.target_fm)
        prepared = add_v5_features(validation, model.predict(validation[INCUMBENT_FEATURES]), "rain25_scale2")
        prepared["prediction"] = prepared.incumbent_base_fm
        prepared["fold"] = fold["name"]; prepared["regime"] = regime_labels(prepared)
        prepared["month"] = prepared.valid_time.dt.month
        prepared["rain_band"] = pd.cut(prepared.precip_intensity_mmph, [-.001, .1, 2, 10, np.inf], labels=["none_trace", "light", "moderate", "heavy"])
        prepared["fm_band"] = pd.cut(prepared.target_fm, [-np.inf, 6, 9, 15, np.inf], labels=["critical", "low", "moderate", "wet"])
        rows.append(prepared)
    scored = pd.concat(rows, ignore_index=True)
    atlas = {"status": "development_oof", "manifest_sha256": manifest["manifest_sha256"],
             "samples": len(scored), "overall": basic_metrics(scored.target_fm, scored.prediction),
             "groups": grouped_metrics(scored, ["fold", "month", "lead_hour", "station_id", "regime", "rain_band", "fm_band"]),
             "forbidden_accessed": False, "prospective_claim": False}
    atomic(args.output, atlas)
    print(json.dumps({"manifest": str(args.manifest), "atlas": str(args.output), "samples": len(scored),
                      "overall": atlas["overall"]}, indent=2))


if __name__ == "__main__": main()
