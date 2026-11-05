"""Validation-only staged search for the station sequence model."""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.station_contract import load_or_create_manifest, sha256_file
from spatial.train_station_sequence import load_frame, train_experiment

HIDDEN_SIZES = (16, 32, 64)
LAYERS = (1, 2)
LEARNING_RATES = (3e-4, 1e-3)
SEEDS = (417, 1337, 2026)


def _configuration(hidden_size, num_layers, learning_rate):
    return {"hidden_size": hidden_size, "num_layers": num_layers,
            "learning_rate": learning_rate, "dropout": 0.0 if num_layers == 1 else 0.1,
            "weight_decay": 1e-4, "batch_size": 32}


def _summary(config, fold, seed, report):
    metrics = report["validation_metrics"]
    return {"config": config, "fold": fold, "seed": seed, "mae": metrics["mae"],
            "bias": metrics["bias"], "best_epoch": report["best_epoch"],
            "runtime_seconds": report["runtime_seconds"]}


def _rank(records):
    grouped = {}
    for record in records:
        key = json.dumps(record["config"], sort_keys=True)
        grouped.setdefault(key, []).append(record)
    ranked = []
    for values in grouped.values():
        config = values[0]["config"]
        ranked.append({"config": config, "mean_mae": statistics.mean(item["mae"] for item in values),
                       "mean_absolute_bias": statistics.mean(abs(item["bias"]) for item in values),
                       "parameter_tiebreak": config["hidden_size"] * config["num_layers"],
                       "trials": len(values)})
    return sorted(ranked, key=lambda item: (item["mean_mae"], item["mean_absolute_bias"], item["parameter_tiebreak"]))


def _trial_key(record):
    return json.dumps(record["config"], sort_keys=True), record["fold"], int(record["seed"])


def _save_state(path, manifest, screening, repeated, phase):
    state = {"status": "complete" if phase == "complete" else "in_progress", "phase": phase, "dataset_sha256": manifest["dataset_sha256"],
             "split_version": manifest["version"], "screening": screening, "repeated_trials": repeated}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Run the fixed 12-configuration station search without holdout access")
    parser.add_argument("--dataset", type=Path, default=paths.ALIGNED_DIR / "station_leads.csv")
    parser.add_argument("--split-manifest", type=Path, default=paths.REPORTS_DIR / "station_split_manifest.json")
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "station_search.json")
    parser.add_argument("--state", type=Path, default=paths.REPORTS_DIR / "station_search.state.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--screen-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()
    frame = load_frame(args.dataset)
    manifest = load_or_create_manifest(args.split_manifest, frame, args.dataset)
    holdout = set(manifest["holdout_runs"])
    for fold in manifest["folds"]:
        if set(fold["train_runs"]) & set(fold["validation_runs"]):
            raise RuntimeError(f"Overlapping train/validation runs in {fold['name']}")
        if holdout & (set(fold["train_runs"]) | set(fold["validation_runs"])):
            raise RuntimeError(f"Locked holdout leaked into {fold['name']}")

    configurations = [_configuration(*values) for values in itertools.product(HIDDEN_SIZES, LAYERS, LEARNING_RATES)]
    screening, repeated = [], []
    if args.state.exists():
        state = json.loads(args.state.read_text())
        if state.get("dataset_sha256") != manifest["dataset_sha256"] or state.get("split_version") != manifest["version"]:
            raise RuntimeError("Search resume state does not match the dataset/split contract")
        screening = state.get("screening", [])
        repeated = state.get("repeated_trials", [])
        print(f"resuming screening={len(screening)} repeated={len(repeated)}", flush=True)
    completed = {_trial_key(record) for record in screening}
    for config_index, config in enumerate(configurations, start=1):
        for fold in manifest["folds"]:
            key = (json.dumps(config, sort_keys=True), fold["name"], 417)
            if key in completed:
                continue
            print(f"screen config={config_index}/12 fold={fold['name']} {config}", flush=True)
            _, _, report = train_experiment(
                frame, manifest, fold["train_runs"], fold["validation_runs"], epochs=args.screen_epochs,
                patience=args.patience, seed=417, **config)
            screening.append(_summary(config, fold["name"], 417, report))
            completed.add(key)
            _save_state(args.state, manifest, screening, repeated, "screening")
    screening_rank = _rank(screening)
    top_configs = [item["config"] for item in screening_rank[:3]]

    completed = {_trial_key(record) for record in repeated}
    repeat_fold = manifest["folds"][-1]
    for config_index, config in enumerate(top_configs, start=1):
        for seed in SEEDS:
            key = (json.dumps(config, sort_keys=True), repeat_fold["name"], seed)
            if key in completed:
                continue
            print(f"repeat config={config_index}/3 seed={seed} fold={repeat_fold['name']}", flush=True)
            _, _, report = train_experiment(
                frame, manifest, repeat_fold["train_runs"], repeat_fold["validation_runs"], epochs=40,
                patience=5, seed=seed, **config)
            repeated.append(_summary(config, repeat_fold["name"], seed, report))
            completed.add(key)
            _save_state(args.state, manifest, screening, repeated, "repeated")
    repeated_rank = _rank(repeated)
    selected = repeated_rank[0]["config"]
    selected_key = json.dumps(selected, sort_keys=True)
    selected_trials = [item for item in repeated if json.dumps(item["config"], sort_keys=True) == selected_key]
    final_epochs = max(1, int(round(statistics.median(item["best_epoch"] for item in selected_trials))))
    print(f"final development fit epochs={final_epochs} config={selected}", flush=True)
    _, checkpoint, final_report = train_experiment(
        frame, manifest, manifest["development_runs"], None, fixed_epochs=final_epochs,
        epochs=final_epochs, patience=5, seed=417, **selected)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_path = args.checkpoint or paths.MODELS_DIR / "experiments" / f"station_selected_{stamp}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    report = {
        "status": "selected_not_evaluated", "search_space_size": 12, "screening": screening,
        "screening_rank": screening_rank, "repeated_trials": repeated, "repeated_rank": repeated_rank,
        "selected_config": selected, "selected_fixed_epochs": final_epochs,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset_sha256": manifest["dataset_sha256"], "split_version": manifest["version"],
        "holdout_accessed": False, "final_fit": final_report,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    _save_state(args.state, manifest, screening, repeated, "complete")
    print(json.dumps({"selected_config": selected, "fixed_epochs": final_epochs,
                      "checkpoint": str(checkpoint_path), "search_report": str(args.output),
                      "holdout_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
