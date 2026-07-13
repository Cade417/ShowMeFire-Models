# ShowMeFire-Models

Standalone training environment for the Show Me Fire fuel-moisture / fire-danger models, split out of the main `Show Me Fire/api` repo so training can run on a different machine (e.g. a desktop with more storage and a GPU) without needing the server's disk or touching what's live.

For the complete RTMA/HRRR spatial workflow—including static data acquisition,
Windows CUDA setup, training gates, release/import, API activation, rollback,
and troubleshooting—use the
[Spatial Fuel-Moisture Operator Runbook](docs/spatial_fuel_moisture_runbook.md).

No bulk training data is ever pushed to this repo - see `paths.py`.

## Setup

```bash
pip install -r requirements.txt

# Point this at wherever training data should actually live. Defaults to
# a gitignored ./data folder inside this repo if unset.
export SMF_DATA_ROOT=/path/to/your/data   # e.g. an external drive, or a desktop's local disk
```

## Getting training data in

Training input comes from unpacking the daily zip bundles the server produces (`api/services/archive_bundler.py`, written to `data_archive_day/` on the server). `scripts/pull_archives.sh` handles both steps - rsyncs new zips down (resumable, only pulls what's new on repeat runs) and unpacks them:

```bash
./scripts/pull_archives.sh user@server /remote/path/to/api/data_archive_day/

# or set these once (e.g. in ~/.bashrc) and just run with no args:
export SMF_SSH_TARGET=user@server
export SMF_REMOTE_ARCHIVE_DIR=/remote/path/to/api/data_archive_day/
./scripts/pull_archives.sh
```

Plain bash + rsync + ssh - works on Ubuntu (including WSL), macOS, or any Linux box. Fresh Ubuntu/WSL installs may need `sudo apt install -y rsync openssh-client` first. This populates `$SMF_DATA_ROOT/archive_zips/`, then unpacks HRRR, observations, and forecasts. RTMA from older ZIPs is still accepted, but new historical RTMA is downloaded locally by `scripts/backfill_rtma_for_hrrr.py`.

To just unpack zips you already have locally (no download), run `python pipelines/unpack_archive_zip.py` directly.

## Running the pipeline

The commands in this section are the legacy point-based XGBoost fallback
pipeline. They do not train the spatial PyTorch model.

```bash
python pipelines/ingest_obs.py          # raw JSON -> this repo's own SQLite DB (never the server's)
python pipelines/index_stations.py      # map stations to the HRRR grid
python scripts/create_snapshots.py      # register new HRRR files as snapshots
python pipelines/extract_hrrr.py        # extract weather features for unprocessed snapshots
python pipelines/generate_training_set.py
python pipelines/prepare_features.py
python pipelines/train_model.py         # registers beta by default; review before promotion
```

## RTMA + sequence fuel-moisture pipeline

After production HRRR/Synoptic archives have been pulled, backfill the causal
and realized RTMA teacher windows locally. The default window is init-12h
through the final HRRR valid hour (28 unique hours for a 12z f04-f15 run):

```bash
python scripts/backfill_rtma_for_hrrr.py --dry-run
python scripts/backfill_rtma_for_hrrr.py --limit 2
python scripts/backfill_rtma_for_hrrr.py
python spatial/build_aligned_dataset.py
python spatial/coverage_report.py
python spatial/evaluate_baselines.py

# Windows/NVIDIA: install the CUDA-matched torch wheel, then the extras.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-spatial.txt
python spatial/env_check.py
python spatial/train_station_sequence.py
python spatial/check_spatial_gate.py

# Run the static-bundle workflow below only after the gate succeeds.
```

The tensor builder separates 13 causal RTMA frames, 15 future realized RTMA
frames (training-only), and HRRR f04-f15. Train both the `--no-distillation`
control and default teacher-distilled candidate. `spatial/export_onnx.py`
exports only the HRRR student; future RTMA can never be an inference input.

Fuel-moisture loss and metrics are always masked to real station observations.
The statewide output includes quantiles, nearest-station distance, effective
station count, and confidence; RTMA is weather input and never an FM label.

### Static terrain and fuel bundle

Acquire or register official GeoTIFFs, build one immutable HRRR-grid bundle,
then reference it from every run tensor:

```bash
python static_features/download_sources.py \
  --dem-file /data/dem.tif \
  --nlcd-class-file /data/nlcd_class.tif \
  --nlcd-confidence-file /data/nlcd_confidence.tif \
  --fbfm40-file /data/landfire_fbfm40.tif \
  --fvt-file /data/landfire_fvt.tif \
  --canopy-cover-file /data/landfire_canopy.tif \
  --release 3dep-nlcd-landfire-2026

python static_features/build_bundle.py --hrrr data/cache/hrrr/<representative>.nc --version 2026.1
python spatial/build_spatial_tensors.py --static-bundle data/static/bundles/static_features_2026.1.nc
python spatial/run_ablation.py --static-bundle data/static/bundles/static_features_2026.1.nc
python spatial/register_spatial_candidate.py --static-bundle data/static/bundles/static_features_2026.1.nc --sample data/aligned/spatial_tensors/<sample>.npz
python pipelines/publish_release.py --model fuel_moisture_spatial
```

Provider URLs may be supplied instead of local files. Original rasters remain
under `SMF_DATA_ROOT` and are never committed. The API imports the published
bundle; it never downloads or rebuilds source geography.

Every run of `train_model.py` lands in this repo's own `beta` channel (`$SMF_DATA_ROOT/models/config.json`) - it never silently overwrites anything. To promote a candidate within this repo's own registry:

```bash
python pipelines/promote_model.py --model fuel_moisture --version 1.5.0-beta.1
```

## Getting a model back into production

The example below is for the legacy single-file model. Spatial candidates use
the multi-asset procedure in the operator runbook; do not adapt this example
by manually attaching or copying spatial assets.

Promotion here only affects *this repo's own* registry - it has no effect on the live server, which has its own independent copy of `models/versioning.py` and its own `config.json`. The bridge between the two is git releases:

```bash
python pipelines/publish_release.py --model fuel_moisture --version 1.5.0-beta.1   # publishes a GitHub pre-release
# ...evaluate...
gh release edit fuel_moisture-v1.5.0-beta.1 --prerelease=false                      # promote on GitHub
```

Then, on the server:

```bash
python pipelines/import_model.py --model fuel_moisture --tag fuel_moisture-v1.5.0-beta.1
python pipelines/promote_model.py --model fuel_moisture --version 1.5.0-beta.1       # the server's own promote step
```

This requires the repo to actually be pushed to GitHub (private) and `gh auth login` set up wherever `publish_release.py` / `import_model.py` run (both machines can reuse the `GITHUB_TOKEN` already in the server's `.env` instead of a fresh login).

## Layout notes

- `paths.py` - single source of truth for every data location, anchored under `SMF_DATA_ROOT`
- `core/database.py`, `models/versioning.py` - vendored copies of the server's modules, repointed at this repo's own data root. Not shared with the server directly - if the DB schema changes on the server side, mirror it here.
- `forecast/forecastverification.py`, `forecast/firedangermodel.py`, `fire-danger-model/` - training-side scripts copied over as-is; the server's copies are now vestigial
