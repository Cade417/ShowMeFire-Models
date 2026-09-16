"""Create V4 evidence contract and select rain physics on development folds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.rule_contract import category,check_api_copy
from spatial.station_contract import basic_metrics
from spatial.v4_contract import load_or_create, reject_forbidden
from spatial.v4_features import FEATURES, RAIN_VARIANTS, add_v4_features
from spatial.v4_training import load_frame


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", type=Path, default=paths.V4_ALIGNED_DATASET)
    parser.add_argument("--v3-manifest", type=Path, default=paths.REPORTS_DIR / "hybrid_v3_split_manifest.json")
    parser.add_argument("--manifest", type=Path, default=paths.V4_SPLIT_MANIFEST)
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "v4_precipitation-v1_physics_selection.json")
    args = parser.parse_args(); check_api_copy()
    frame = load_frame(args.dataset); manifest = load_or_create(args.manifest, frame, args.dataset, args.v3_manifest, FEATURES)
    reusable=frame[frame.run_id.isin(set(manifest["development_runs"]+manifest["calibration_runs"])) & (frame.target_mask==1)]
    category_values=[category(f,r,w*1.9438444924406) for f,r,w in zip(reusable.target_fm,reusable.target_rh,reusable.target_wind_ms)]
    category_support=sum(value is not None for value in category_values);elevated_support=sum(value is not None and value>=2 for value in category_values)
    if category_support<1000 or elevated_support<100:
        raise RuntimeError(f"Insufficient observed category support: labels={category_support} elevated={elevated_support}")
    precipitation_available = bool((frame.hrrr_precip_mm.fillna(0) > 0).any())
    variants = list(RAIN_VARIANTS) if precipitation_available else ["legacy"]
    results = []
    for name in variants:
        prepared = add_v4_features(frame, [0] * len(frame), name)
        fold_metrics = []
        for fold in manifest["folds"]:
            reject_forbidden(manifest, fold["validation_runs"], "physics selection")
            scored = prepared[prepared.run_id.isin(set(fold["validation_runs"])) & (prepared.target_mask == 1)]
            fold_metrics.append(basic_metrics(scored.target_fm, scored.rain_physics_fm))
        results.append({"variant": name, "folds": fold_metrics,
                        "mean_mae": sum(x["mae"] for x in fold_metrics) / len(fold_metrics),
                        "mean_rmse": sum(x["rmse"] for x in fold_metrics) / len(fold_metrics),
                        "mean_absolute_bias": sum(abs(x["bias"]) for x in fold_metrics) / len(fold_metrics)})
    results.sort(key=lambda x: (x["mean_mae"], x["mean_rmse"], x["mean_absolute_bias"]))
    report = {"selected_variant": results[0]["variant"], "ranking": results,
              "observed_category_support":category_support,"observed_elevated_support":elevated_support,
              "precipitation_available": precipitation_available,
              "precipitation_note": None if precipitation_available else "Archived HRRR precipitation is constant zero; rain variants were not scored.",
              "manifest_sha256": manifest["manifest_sha256"], "forbidden_accessed": False}
    args.output.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
