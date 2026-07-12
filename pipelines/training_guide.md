# Show Me Fire Training Guide

There are two additive fuel-moisture pipelines:

1. The legacy point-based XGBoost pipeline under `pipelines/`.
2. The RTMA/HRRR probabilistic station and spatial pipeline under `spatial/`.

The spatial workflow, static terrain/land-cover/fuel inputs, release process,
API activation, fallback, and troubleshooting are documented in the
[Spatial Fuel-Moisture Operator Runbook](../docs/spatial_fuel_moisture_runbook.md).

## Legacy XGBoost workflow

The XGBoost model remains the mandatory production fallback. To rebuild it:

```bash
python pipelines/ingest_obs.py
python pipelines/index_stations.py
python scripts/create_snapshots.py
python pipelines/extract_hrrr.py
python pipelines/generate_training_set.py
python pipelines/prepare_features.py
python pipelines/train_model.py --channel beta
```

Review the printed chronological holdout metrics before publishing. Training
registers beta; it does not replace stable production automatically.

```bash
python pipelines/publish_release.py --model fuel_moisture
```

Import and promotion happen separately on the API server. Do not manually edit
feature lists, registry JSON, or model files as described by older versions of
this document.
