"""Resumable 24-configuration development-only search for hybrid V3."""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.hybrid_contract import load_or_create_manifest
from spatial.hybrid_features import FEATURES
from spatial.hybrid_training import load_frame, prepare_fold, sequence_arrays, train_model

SEEDS = (417, 1337, 2026)


def configs():
    return [{"hidden_size": hidden, "num_layers": layers, "learning_rate": rate,
             "p50_loss": loss, "dropout": 0.0 if layers == 1 else 0.1,
             "batch_size": 32, "weight_decay": 1e-4}
            for hidden, layers, rate, loss in itertools.product((32, 64, 96), (1, 2), (3e-4, 1e-3), ("mae", "huber"))]


def config_key(config):
    return json.dumps(config, sort_keys=True)


def trial_key(record):
    return config_key(record["config"]), record["fold"], int(record["seed"])


def rank(records):
    grouped = {}
    for record in records:
        grouped.setdefault(config_key(record["config"]), []).append(record)
    result = []
    for values in grouped.values():
        config = values[0]["config"]
        result.append({"config": config,
                       "mean_mae": statistics.mean(item["mae"] for item in values),
                       "mean_absolute_bias": statistics.mean(abs(item["bias"]) for item in values),
                       "mean_rmse": statistics.mean(item["rmse"] for item in values),
                       "parameter_tiebreak": config["hidden_size"] * config["num_layers"],
                       "trials": len(values)})
    return sorted(result, key=lambda item: (item["mean_mae"], item["mean_absolute_bias"],
                                            item["mean_rmse"], item["parameter_tiebreak"]))


def save_state(path, manifest, screening, repeated, phase):
    value = {"status": "complete" if phase == "complete" else "in_progress", "phase": phase,
             "dataset_sha256": manifest["dataset_sha256"], "manifest_sha256": manifest["manifest_sha256"],
             "screening": screening, "repeated": repeated}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def summary(config, fold, seed, report):
    metrics = report["validation_metrics"]
    return {"config": config, "fold": fold, "seed": seed, "mae": metrics["mae"],
            "rmse": metrics["rmse"], "bias": metrics["bias"], "best_epoch": report["best_epoch"],
            "runtime_seconds": report["runtime_seconds"]}


def main():
    parser = argparse.ArgumentParser(description="Search hybrid V3 without calibration/test access")
    parser.add_argument("--dataset", type=Path, default=paths.ALIGNED_DIR / "station_leads.csv")
    parser.add_argument("--manifest", type=Path, default=paths.REPORTS_DIR / "hybrid_v3_split_manifest.json")
    parser.add_argument("--state", type=Path, default=paths.REPORTS_DIR / "hybrid_v3_search.state.json")
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "hybrid_v3_search.json")
    parser.add_argument("--screen-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    frame = load_frame(args.dataset)
    manifest = load_or_create_manifest(args.manifest, frame, args.dataset, FEATURES)
    forbidden = set(manifest["calibration_runs"]) | set(manifest["locked_test_runs"])
    prepared = {}
    for fold in manifest["folds"]:
        if forbidden & (set(fold["train_runs"]) | set(fold["validation_runs"])):
            raise RuntimeError(f"Calibration/test leakage into {fold['name']}")
        print(f"preparing cross-fitted base features for {fold['name']}", flush=True)
        train_frame, validation_frame, _ = prepare_fold(frame, fold["train_runs"], fold["validation_runs"])
        prepared[fold["name"]] = (sequence_arrays(train_frame)[0], sequence_arrays(validation_frame)[0])
    screening, repeated = [], []
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state.get("dataset_sha256") != manifest["dataset_sha256"] or state.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError("Resume state contract mismatch")
        screening, repeated = state.get("screening", []), state.get("repeated", [])
        print(f"resuming screening={len(screening)} repeated={len(repeated)}", flush=True)
    completed = {trial_key(item) for item in screening}
    candidates = configs()
    for number, config in enumerate(candidates, 1):
        for fold in manifest["folds"]:
            key = (config_key(config), fold["name"], 417)
            if key in completed:
                continue
            print(f"screen config={number}/24 fold={fold['name']} {config}", flush=True)
            _, _, report = train_model(*prepared[fold["name"]], epochs=args.screen_epochs,
                                       patience=args.patience, seed=417, **config)
            screening.append(summary(config, fold["name"], 417, report)); completed.add(key)
            save_state(args.state, manifest, screening, repeated, "screening")
    screen_rank = rank(screening); finalists = [item["config"] for item in screen_rank[:4]]
    latest = manifest["folds"][-1]["name"]
    completed = {trial_key(item) for item in repeated}
    for number, config in enumerate(finalists, 1):
        for seed in SEEDS:
            key = (config_key(config), latest, seed)
            if key in completed:
                continue
            print(f"repeat config={number}/4 seed={seed} fold={latest}", flush=True)
            _, _, report = train_model(*prepared[latest], epochs=40, patience=5, seed=seed, **config)
            repeated.append(summary(config, latest, seed, report)); completed.add(key)
            save_state(args.state, manifest, screening, repeated, "repeated")
    repeated_rank = rank(repeated); selected = repeated_rank[0]["config"]
    selected_trials = [item for item in repeated if config_key(item["config"]) == config_key(selected)]
    fixed_epochs = max(1, round(statistics.median(item["best_epoch"] for item in selected_trials)))
    output = {"status": "selected_not_fitted", "search_space_size": 24,
              "dataset_sha256": manifest["dataset_sha256"], "manifest_sha256": manifest["manifest_sha256"],
              "holdout_accessed": False, "calibration_accessed": False,
              "screening": screening, "screening_rank": screen_rank,
              "repeated": repeated, "repeated_rank": repeated_rank,
              "selected_config": selected, "selected_fixed_epochs": fixed_epochs}
    args.output.write_text(json.dumps(output, indent=2)); save_state(args.state, manifest, screening, repeated, "complete")
    print(json.dumps({"selected_config": selected, "fixed_epochs": fixed_epochs,
                      "search_report": str(args.output), "holdout_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
