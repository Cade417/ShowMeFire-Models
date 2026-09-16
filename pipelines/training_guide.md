# Show Me Fire Training Guide

There are two additive fuel-moisture pipelines:

1. The legacy point-based XGBoost pipeline under `pipelines/`.
2. The RTMA/HRRR probabilistic station and spatial pipeline under `spatial/`.

The spatial workflow, static terrain/land-cover/fuel inputs, release process,
API activation, fallback, and troubleshooting are documented in the
[Spatial Fuel-Moisture Operator Runbook](../docs/spatial_fuel_moisture_runbook.md).

## Legacy XGBoost workflow

The XGBoost model remains the mandatory production fallback. Intended to be
rerun roughly every two weeks as new observations accumulate.

**Preferred: single orchestrator.**

```bash
python pipelines/retrain_fuel_moisture.py
```

This runs the same 8 phases below as one process instead of one command per
phase, and:

- Runs a rolling-origin (`TimeSeriesSplit`) hyperparameter search over a
  small grid before the final fit (`--no-search` to skip it and use the
  historical fixed defaults instead).
- After registering the new beta candidate, prints a comparison against
  whatever is currently `stable` (`pipelines/experiment_log.py`), so you can
  see at a glance whether two more weeks of data actually helped.
- Appends one row per run to `<SMF_DATA_ROOT>/reports/experiments/fuel_moisture.jsonl`
  - a lightweight training history, not a replacement for the model registry.
- Still only ever registers to `beta`. Publishing/promoting to `stable`
  remains a separate, deliberate step - this script never does it for you.

Useful flags: `--skip-ingest` (data already ingested), `--skip-index`,
`--skip-snapshots`, `--skip-extract` (mirrors `trainnewmodel.sh`'s flags),
`--full-retrain` (reset all snapshots for a complete reprocess).

**Equivalent manual phases** (what the orchestrator runs, in-process, in this
order - useful if you need to debug or rerun a single phase):

```bash
python pipelines/ingest_obs.py
python pipelines/index_stations.py
python scripts/create_snapshots.py
python pipelines/extract_hrrr.py
python pipelines/generate_training_set.py
python pipelines/prepare_features.py
python pipelines/train_model.py --channel beta
```

`pipelines/trainnewmodel.sh` still works as a bash equivalent of the same
8 phases (one `python3` subprocess per phase) for anyone with existing
muscle memory, but prefer the Python orchestrator above for new or
scheduled runs - a single Python traceback tells you exactly which phase
failed, instead of a shell script silently exiting on the first error.

Review the printed chronological holdout metrics (and the beta-vs-stable
comparison line) before publishing.

```bash
python pipelines/publish_release.py --model fuel_moisture
```

Import and promotion happen separately on the API server. Do not manually edit
feature lists, registry JSON, or model files as described by older versions of
this document.

## Data sync

To pull the latest observations/HRRR/RTMA data before retraining, see the
pullers under `scripts/` (`pull_archives.sh`, `backfill_synoptic.py`,
`backfill_rtma_for_hrrr.py`) - each is independently incremental/resumable.
