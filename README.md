# ShowMeFire-Models

Standalone training environment for the Show Me Fire fuel-moisture / fire-danger models, split out of the main `Show Me Fire/api` repo so training can run on a different machine (e.g. a desktop with more storage and a GPU) without needing the server's disk or touching what's live.

No bulk training data is ever pushed to this repo - see `paths.py`.

## Setup

```bash
pip install -r requirements.txt

# Point this at wherever training data should actually live. Defaults to
# a gitignored ./data folder inside this repo if unset.
export SMF_DATA_ROOT=/path/to/your/data   # e.g. an external drive, or a desktop's local disk
```

## Getting training data in

Training input comes from unpacking the daily zip bundles the server produces (`api/services/archive_bundler.py`, written to `data_archive_day/` on the server). Copy those zips into `$SMF_DATA_ROOT/archive_zips/` (scp, USB drive, however's convenient), then:

```bash
python pipelines/unpack_archive_zip.py
```

This populates `cache/hrrr/`, `archive/raw_data/`, and `archive/forecasts/` under `$SMF_DATA_ROOT`.

## Running the pipeline

```bash
python pipelines/ingest_obs.py          # raw JSON -> this repo's own SQLite DB (never the server's)
python pipelines/index_stations.py      # map stations to the HRRR grid
python scripts/create_snapshots.py      # register new HRRR files as snapshots
python pipelines/extract_hrrr.py        # extract weather features for unprocessed snapshots
python pipelines/generate_training_set.py
python pipelines/prepare_features.py
python pipelines/train_model.py         # --channel beta (default) or --channel stable, --bump patch/minor/major
```

Every run of `train_model.py` lands in this repo's own `beta` channel (`$SMF_DATA_ROOT/models/config.json`) - it never silently overwrites anything. To promote a candidate within this repo's own registry:

```bash
python pipelines/promote_model.py --model fuel_moisture --version 1.5.0-beta.1
```

## Getting a model back into production

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
