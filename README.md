# ShowMeFire-Models

Standalone training environment for the Show Me Fire fuel-moisture / fire-danger models, split out of the main `Show Me Fire/api` repo so training can run on a different machine (e.g. a desktop with more storage and a GPU) without needing the server's disk or touching what's live.

For the complete RTMA/HRRR spatial workflow—including static data acquisition,
Windows CUDA setup, training gates, release/import, API activation, rollback,
and troubleshooting—use the
[Spatial Fuel-Moisture Operator Runbook](docs/spatial_fuel_moisture_runbook.md).

No bulk training data is ever pushed to this repo - see `paths.py`.

**For the full data → train → publish → import → promote workflow, across every model family (not just fuel_moisture), see the [Model Lifecycle guide](docs/model_lifecycle.md).** That's now the canonical reference for retraining, publishing, importing, and promoting/activating any model type. This README keeps only repo setup and data-sync instructions.

## Model Lab (local compare UI)

Streamlit lab for scoring production vs beta forecast archives against observations, browsing the local registry/shadow candidates, and reading offline evaluation reports:

```powershell
cd M:\_Development\ShowMeFire\model-training
.venv\Scripts\Activate.ps1
$env:SMF_DATA_ROOT = "M:\_Development\ShowMeFire\training-data"
python -m pip install -r requirements-lab.txt
streamlit run model_lab/app.py
```

Details: [`model_lab/README.md`](model_lab/README.md).

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

## Training, publishing, importing, and promoting a model

See the [Model Lifecycle guide](docs/model_lifecycle.md) for the full walkthrough — data capture, training/evaluation, registering a beta candidate, publishing a GitHub release, importing it on the server, and promoting/activating it — covering every model family (the legacy XGBoost pipeline, the RTMA/spatial sequence pipeline, the static terrain/fuel bundle, and the advisory-only shadow families) with the exact commands for each.

## Layout notes

- `paths.py` - single source of truth for every data location, anchored under `SMF_DATA_ROOT`
- `core/database.py`, `models/versioning.py` - vendored copies of the server's modules, repointed at this repo's own data root. Not shared with the server directly - if the DB schema changes on the server side, mirror it here.
- `forecast/forecastverification.py`, `forecast/firedangermodel.py`, `fire-danger-model/` - training-side scripts copied over as-is; the server's copies are now vestigial
