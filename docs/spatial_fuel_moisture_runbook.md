# Spatial Fuel-Moisture Operator Runbook

This is the canonical guide for collecting data, building static geography,
training the probabilistic station and spatial models, publishing a candidate,
and activating it in the Show Me Fire API.

## What the system predicts

The spatial model predicts 12 hourly fuel-moisture quantiles (`P10`, `P50`,
`P90`) from:

- real Synoptic station fuel-moisture observations at forecast initialization;
- RTMA analyzed temperature, relative humidity, and wind;
- the HRRR forecast sequence for temperature, RH, wind, and precipitation;
- a physics-based fuel-moisture trajectory;
- terrain, land cover, vegetation, fuel model, and canopy data.

Synoptic observations are the only fuel-moisture truth. RTMA, HRRR,
interpolation, and static rasters are model inputs and must never be written as
observed fuel-moisture labels.

The current XGBoost model remains the production fallback. A training run
cannot silently replace it.

## Repository responsibilities

| Repository | Responsibility |
| --- | --- |
| `ShowMeFire-Models` | Archive ingestion, HRRR-driven historical RTMA backfill, static-data acquisition, alignment, training, evaluation, ONNX export, release publishing |
| `api` | Live RTMA capture with seven-day retention, historical Synoptic backfill, daily HRRR/observation archives, verified release import, production inference and fallback |

The two repositories have independent model registries. A candidate crosses
between them only through a GitHub release.

## Data directory layout

All training data is under `SMF_DATA_ROOT`. If unset, it defaults to
`ShowMeFire-Models/data/`.

```text
$SMF_DATA_ROOT/
├── archive_zips/               # daily ZIPs copied from production
├── archive/raw_data/           # canonical Synoptic JSON days
├── cache/hrrr/                 # HRRR forecast NetCDF files
├── cache/rtma/                 # hourly RTMA NetCDF files
├── aligned/
│   ├── station_leads.csv
│   └── spatial_tensors/        # dynamic-only NPZ run tensors
├── static/
│   ├── source/                 # original GeoTIFFs + source manifest
│   └── bundles/                # immutable 256x256 NetCDF bundles
├── models/                     # local beta/stable registry and candidates
└── reports/                    # coverage, baseline, sequence, ablation reports
```

Never commit bulk data, source rasters, tensors, trained weights, or API tokens.

## 1. Environment setup

### Base training environment

```bash
cd ShowMeFire-Models
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set a durable data location:

```bash
# Linux/macOS/WSL
export SMF_DATA_ROOT=/path/to/showmefire-training-data

# Windows PowerShell
$env:SMF_DATA_ROOT = "D:\ShowMeFire-Training-Data"
```

### Windows NVIDIA environment

Use the CUDA index compatible with the installed NVIDIA driver. The current
project example is CUDA 12.4:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-spatial.txt
python spatial/env_check.py
```

Do not begin full training unless `cuda_available=True` and the expected GPU
and VRAM are printed. `num_workers=0` is the supported Windows default.

## 2. Production data collection and backfill

Run these commands inside the API environment on the server.

```bash
# Inspect the one-year Synoptic job. This contacts metadata but writes no days.
python scripts/backfill_synoptic.py --dry-run

# Fetch the rolling one-year entitlement in resumable UTC days.
python scripts/backfill_synoptic.py

# Atomically bundle HRRR, observations, and forecasts.
python -m services.archive_bundler
```

Requirements:

- `SYNOPTIC_API_TOKEN` must be configured on the server.
- Fuel-moisture labels remain missing when Synoptic has no valid reading.
- Backfills are resumable; failed days/hours remain visible in their manifests.
- Existing ZIPs are copied, merged, CRC-checked, and atomically replaced.

RTMA live capture runs hourly at minute `:50` through APScheduler, targets the
preceding complete UTC analysis hour, and retains seven days by default. New
RTMA files are not placed in permanent daily ZIPs. Set
`RTMA_RETENTION_DAYS` to an integer greater than zero to change the live cache.

### 2a. Local Synoptic backfill (no server access required)

`scripts/backfill_synoptic.py` also exists in this repo and can fetch
directly into `archive/raw_data/` on the training machine - useful for
extending history further back than what's already been archived on the
server, without needing SSH access at all. Unlike the production script,
it defaults to the full 9-state region (`MO, OK, AR, TN, KY, IL, IA, NE, KS`)
on Synoptic network `2`, which in practice returns far more fuel-moisture
stations than Missouri alone (~19 MO-only vs. ~90-116 across the region) -
`load_observations()` has no state filter, so the wider set flows straight
into the aligned dataset.

```bash
# Add SYNOPTIC_API_TOKEN=... to .env first (gitignored, never committed).

# Inspect a full year back from today. Writes and fetches nothing.
python scripts/backfill_synoptic.py --dry-run

# Fetch the full year, 7-day chunks, 4 concurrent requests.
python scripts/backfill_synoptic.py

# Narrower range, or re-fetch days that already exist under an older/
# narrower query (e.g. upgrading old MO-only days to the 9-state set):
python scripts/backfill_synoptic.py --start 2026-04-01 --end 2026-07-11 --force
```

This is pure HTTP/JSON (no native-library concurrency hazards), so
`--workers` here is thread-based, unlike the process-based HRRR/RTMA
backfills below. One file is written per UTC day
(`raw_data_YYYYMMDD.json`), matching what `pull_archives.sh` already
produces from server ZIPs - both feed the same `load_observations()` path.
The resumable manifest is `archive/raw_data/backfill_manifest.json`. A day
with zero surviving stations (e.g. everything ignore-filtered) still gets
written and marked complete - it's a legitimate outcome, not a gap to
retry forever. `ignored_stations` (SQLite) is applied the same as
production.

## 3. Synchronize archives to the training machine

From Linux, macOS, or WSL:

```bash
export SMF_SSH_TARGET=user@production-host
export SMF_REMOTE_ARCHIVE_DIR=/remote/path/to/api/data_archive_day/
./scripts/pull_archives.sh
```

The script transfers completed ZIPs without `--inplace`, then routes:

- `hrrr_*.nc` to `cache/hrrr/`;
- legacy `rtma_*.nc` members, when present, to `cache/rtma/`;
- `raw_data_*.json` to `archive/raw_data/`;
- station forecast JSON to `archive/forecasts/`.

Do not synchronize while a manual server backfill/merge is still running.

Old production ZIPs that already contain RTMA remain valid and are not
compacted. New ZIPs intentionally contain no RTMA.

## 4. Backfill HRRR and RTMA on the training machine

### 4a. Backfill HRRR

`scripts/backfill_hrrr.py` fetches historical HRRR runs directly via Herbie
(NOAA/AWS archive), cropped to the same Missouri-buffered bounding box RTMA
already uses - nothing downstream needs the full CONUS grid, and cropping
takes a run from ~466MB to a few MB. Confirmed against real data up to a
year back.

```bash
# Inspect a full year back from today.
python scripts/backfill_hrrr.py --dry-run

# Fetch the full year. Defaults match existing cached files: 12z, f04-f15.
python scripts/backfill_hrrr.py

# Narrower range, or a smoke test before committing to the full year:
python scripts/backfill_hrrr.py --start 2025-08-01 --end 2025-08-07 --limit 2
```

Like RTMA below, this uses process-pool concurrency, not threads: the
netCDF write and GRIB decode inside `fetch_hrrr` aren't thread-safe (an
earlier attempt with threads on the RTMA backfill segfaulted). The
resumable manifest is `cache/hrrr/backfill_manifest.json`.

### 4b. Backfill RTMA

Historical RTMA storage belongs under `SMF_DATA_ROOT`. For each local HRRR run,
the default teacher window downloads initialization minus 12 hours through the
final HRRR valid hour. A 12z f04-f15 run therefore requires 28 analyses, 00z
through 03z the next day. Overlapping hours are downloaded only once:

```bash
python scripts/backfill_rtma_for_hrrr.py --dry-run
python scripts/backfill_rtma_for_hrrr.py --limit 2
python scripts/backfill_rtma_for_hrrr.py
```

The command reads HRRR valid times, validates RTMA extracted from legacy ZIPs,
and reports existing/missing analyses plus estimated storage. `--window anchor`
retains the old one-analysis behavior for diagnostics; training requires the
default `--window teacher` data.

Optional filters:

```bash
python scripts/backfill_rtma_for_hrrr.py --start 2026-01-01 --end 2026-03-31
python scripts/backfill_rtma_for_hrrr.py --force --limit 2
```

The resumable schema-v2 manifest is
`$SMF_DATA_ROOT/cache/rtma/backfill_manifest.json`. A full run exits nonzero
when a required selected analysis remains unresolved. It records run windows
separately from globally deduplicated analyses and migrates v1 anchors. `--limit` applies only to
the selected missing/forced work and is intended for smoke testing.

## 5. Build aligned station data and required baselines

```bash
python spatial/build_aligned_dataset.py --max-initial-age-hours 3
python spatial/coverage_report.py
python spatial/evaluate_baselines.py
python spatial/tune_station_sequence.py
python spatial/evaluate_station_candidate.py --checkpoint <selected-checkpoint>
python spatial/check_spatial_gate.py
python spatial/register_station_candidate.py --checkpoint <selected-checkpoint>
```

Station tuning screens the fixed 12-configuration search space on three
expanding chronological folds, then repeats the top three configurations
across three seeds. Search and ordinary training do not access the locked
holdout and do not register artifacts. The selected development checkpoint is
evaluated once; only a passing, checksum-matched final report can be registered
as beta. Registration never changes the API server's stable model.

After the spatial gate succeeds, build teacher/student tensors, train an
identical non-distilled control and distilled candidate, and create the
held-out realized-weather association report:

```bash
python spatial/build_spatial_tensors.py --static-bundle "$STATIC_BUNDLE"
python spatial/train_spatial.py --static-bundle "$STATIC_BUNDLE" --no-distillation
python spatial/train_spatial.py --static-bundle "$STATIC_BUNDLE"
python spatial/compare_distillation.py --feature-set all
python spatial/report_realized_weather.py --static-bundle "$STATIC_BUNDLE"
python spatial/export_onnx.py --checkpoint models/fuel_moisture_spatial_all_distilled.pt \
  --sample "$SMF_DATA_ROOT/aligned/spatial_tensors/<sample>.npz" --static-bundle "$STATIC_BUNDLE"
```

The realized `+1..+15` RTMA sequence is training-only. Only the HRRR student is
exported. Reports describe associations and forecast-error patterns, never
physical causation. Production loads 13 causal RTMA frames, fills at most two
gaps from an earlier frame, and falls back to XGBoost on a third gap.

Review:

- `reports/coverage.json`
- `reports/baseline_metrics.json`
- `reports/sequence_metrics.json`

The spatial gate requires at least 180 usable initialization dates, a typical
20 stations per usable day, and at least 5% temporal MAE improvement by the
station sequence model over persistence and the incumbent control.

`check_spatial_gate.py` exits with status `2` when the gate fails. That is a
valid operational outcome: retain the station model, continue collection, and
do not use `--force` for a production candidate.

## 6. Acquire static source rasters

The bundle requires one single-band GeoTIFF for each product over the Missouri
domain plus the one-degree buffer:

| CLI name | Required content | Type/units | Resampling |
| --- | --- | --- | --- |
| `dem` | USGS 3DEP 1/3 arc-second bare-earth DEM | continuous meters | bilinear |
| `nlcd-class` | latest complete Annual NLCD land-cover class | categorical integer | nearest |
| `nlcd-confidence` | matching NLCD class confidence | continuous product value | bilinear |
| `fbfm40` | LANDFIRE Fire Behavior Fuel Model 40 | categorical integer | nearest |
| `fvt` | LANDFIRE Fuel Vegetation Type | categorical integer | nearest |
| `canopy-cover` | LANDFIRE canopy cover | continuous percent | bilinear |

Use the newest complete release intentionally; do not replace source rasters
during an existing training experiment.

Official entry points:

- USGS 3DEP/The National Map: <https://www.usgs.gov/3d-elevation-program>
- Annual NLCD/MRLC: <https://www.mrlc.gov/data/project/annual-nlcd>
- LANDFIRE data access: <https://landfire.gov/data>

The acquisition command accepts either a local file or a direct official URL
for each product. ZIP URLs are supported only when the archive contains one
GeoTIFF. Mosaic multi-tile products before registering them.

```bash
python static_features/download_sources.py \
  --dem-file /source/3dep_missouri_buffered.tif \
  --nlcd-class-file /source/nlcd_class.tif \
  --nlcd-confidence-file /source/nlcd_confidence.tif \
  --fbfm40-file /source/landfire_fbfm40.tif \
  --fvt-file /source/landfire_fvt.tif \
  --canopy-cover-file /source/landfire_canopy_cover.tif \
  --release 3dep-nlcd-landfire-2026
```

For a direct official URL, replace `--<product>-file` with
`--<product>-url`. Downloads use `.partial` files and HTTP Range resume.

To discover relevant 3DEP tile URLs without registering a DEM:

```bash
python static_features/download_sources.py --discover-3dep --release discovery-only
```

This writes the URLs to `static/source/source_manifest.json`; it does not
mosaic them. The final build still requires one `--dem-file` or `--dem-url`.

Verify `source_manifest.json` contains all six products, checksums, byte sizes,
release label, acquisition time, and resolved source URLs where applicable.

## 7. Build the immutable static bundle

Choose a normal representative HRRR file from the same product/run structure
used for spatial training:

```bash
python static_features/build_bundle.py \
  --hrrr "$SMF_DATA_ROOT/cache/hrrr/hrrr_YYYYMMDD_12z_f04-15.nc" \
  --version 2026.1
```

Outputs:

```text
static/bundles/static_features_2026.1.nc
static/bundles/static_features_2026.1.json
```

The NetCDF contains the canonical HRRR Lambert `256×256` grid, projected
coordinates, latitude/longitude, elevation, slope, aspect sine/cosine,
ruggedness, NLCD confidence, canopy cover, and encoded NLCD/FBFM40/FVT
categories. Category index `0` always means unknown/nodata.

Bundles are immutable. If the output version already exists, choose a new
version; do not overwrite it. A bundle change requires tensor rebuild,
retraining, evaluation, and a new model release.

## 8. Build dynamic tensors

```bash
python spatial/build_spatial_tensors.py \
  --static-bundle "$SMF_DATA_ROOT/static/bundles/static_features_2026.1.nc"
```

For a smoke test first:

```bash
python spatial/build_spatial_tensors.py \
  --static-bundle "$SMF_DATA_ROOT/static/bundles/static_features_2026.1.nc" \
  --limit 7
```

Run NPZ files contain dynamic weather/current-state data, physics trajectory,
sparse real targets, temporal/station/region masks, and the static bundle
checksum/grid fingerprint. Static geography is loaded separately and is not
duplicated in every NPZ.

If the bundle changes, remove or relocate tensors built against the old
fingerprint before rebuilding. Never mix fingerprints in one experiment.

## 9. Train the static-feature ablation

After the spatial gate succeeds:

```bash
python spatial/run_ablation.py \
  --static-bundle "$SMF_DATA_ROOT/static/bundles/static_features_2026.1.nc" \
  --epochs 20 \
  --batch-size 2
```

The command trains identical chronological candidates:

1. dynamic-only control;
2. terrain;
3. terrain + NLCD + canopy;
4. terrain + LANDFIRE;
5. all static features.

It evaluates future dates, held-out stations, and the held-out eastern region.
The report is `reports/spatial_ablation.json`.

Selection rules:

- choose the smallest candidate within 1% of the best temporal MAE;
- the static candidate must beat dynamic-only temporal MAE;
- interval coverage may not regress by more than two percentage points;
- critical-low-FM MAE may not regress by more than 1%;
- quantile ordering must be valid.

`--force` bypasses the prerequisite data/performance gate for development
only. A forced run must never be registered or promoted as production.

## 10. Register and publish a beta candidate

Choose any tensor from the same static fingerprint as the parity/smoke sample:

```bash
python spatial/register_spatial_candidate.py \
  --static-bundle "$SMF_DATA_ROOT/static/bundles/static_features_2026.1.nc" \
  --sample "$SMF_DATA_ROOT/aligned/spatial_tensors/spatial_YYYYMMDDHH.npz"
```

Registration is refused unless the ablation gate passes. It exports ONNX,
compares ONNX Runtime against PyTorch, writes a smoke NPZ, and registers one
beta asset contract containing:

- ONNX model;
- PyTorch checkpoint;
- static NetCDF bundle;
- static manifest;
- evaluation report;
- inference smoke sample.

Publish the beta as a GitHub prerelease:

```bash
export SMF_GITHUB_REPO=Cade417/ShowMeFire-Models
python pipelines/publish_release.py --model fuel_moisture_spatial
```

The printed tag has the form
`fuel_moisture_spatial-v<semantic-beta-version>`. Do not mark the GitHub
release final as a substitute for server import, verification, and promotion.

## 11. Import and activate on the API server

Ensure the deployed API dependencies include `onnxruntime` and restart the
container after changing requirements.

```bash
python pipelines/import_model.py \
  --model fuel_moisture_spatial \
  --tag fuel_moisture_spatial-v<version> \
  --repo Cade417/ShowMeFire-Models
```

Import downloads into a temporary directory and verifies all declared assets,
checksums, the static schema/grid, and the ONNX smoke output before registering
server-side beta. A failed import does not alter stable production state.

Review beta/shadow results, then promote the server-side version printed by
the import command:

```bash
python pipelines/promote_model.py \
  --model fuel_moisture_spatial \
  --version <server-beta-version>
```

Promotion switches the ONNX model and static bundle as one registry entry.
The forecast generators then attempt spatial P50 inference. Any failure keeps
their existing XGBoost result for that forecast.

Check runtime state:

```bash
curl http://localhost:8000/api/model/spatial/diagnostics
```

Important fields are `available`, `fallback`, `fallback_reason`,
`last_success`, `inference_ms`, `bundle`, and `feature_set`.

## Fallback and rollback

Spatial inference falls back to XGBoost when:

- no stable spatial asset contract exists;
- an asset checksum, schema, or grid fingerprint differs;
- more than two of the 13 causal RTMA history frames are absent;
- fewer than three causal station FM observations are available;
- HRRR sequence length differs from the exported model;
- ONNX Runtime fails or returns non-finite/crossed quantiles.

Fallback is safe but should be investigated through the diagnostics endpoint
and API logs.

To roll back, import the previously known-good GitHub release as a new beta and
promote the newly assigned server beta version. Do not edit registry JSON or
swap individual model/bundle files manually.

## Refresh and retraining cadence

- Ingest/synchronize observations and weather at least weekly.
- Retrain the station sequence model approximately monthly after meaningful
  new coverage accumulates.
- Run the full spatial ablation quarterly or after substantial new seasonal
  coverage.
- Review Annual NLCD and LANDFIRE releases annually.
- Refresh 3DEP only when a meaningful updated domain mosaic is available.

Every static-source refresh gets a new source release label and bundle version
and repeats steps 7–11. Never place a new static bundle under an old model.

## RTMA migration rollout

1. Stop any running API `backfill_rtma.py` process before deploying.
2. Deploy the API retention and archive-exclusion change.
3. Allow the live cleanup to remove unbundled RTMA older than seven days.
4. Do not rewrite historical ZIPs merely to remove their existing RTMA.
5. Pull current HRRR and observation archives to the training machine.
6. Run the HRRR-driven dry-run, two-anchor smoke test, and full local backfill.
7. Confirm every selected manifest record is complete before alignment.

## Troubleshooting

### `source products missing`

All six products must be present in `source_manifest.json`. Re-run
`download_sources.py` with the missing `--*-file` or `--*-url` option.

### ZIP contains more than one GeoTIFF

Extract/mosaic the provider archive outside the script, then pass the final
single-band mosaic through `--*-file`.

### Static checksum or grid mismatch

Do not edit a built NetCDF or manifest. Rebuild with a new bundle version and
rebuild every tensor/checkpoint that references it.

### Spatial gate exits `2`

Review coverage and baseline reports. Continue using the station/XGBoost path;
do not force a production run.

### CUDA is unavailable

Reinstall PyTorch from the correct CUDA index and verify the NVIDIA driver.
Do not assume plain `pip install torch` installed a CUDA wheel.

### Out of GPU memory

Reduce `--batch-size` from `2` to `1`. Do not change the canonical grid or
model width without treating it as a new experiment and rerunning ablations.

### Import succeeds but forecasts fall back

Check `/api/model/spatial/diagnostics`, then verify RTMA for the initialization
hour, station count, HRRR step count, and the active asset checksums.

### ONNX parity fails

Keep the PyTorch checkpoint as an unregistered experiment. Do not publish or
promote it. Investigate unsupported operations or numerical differences first.
# Hybrid station V3 workflow

V3 is an experimental residual bundle: an XGBoost causal trajectory plus an
ordered-quantile GRU correction. It uses `station-split-v2` (80% development,
10% calibration, 10% locked historical relock). None of these commands changes
the current stable model or the existing `0.0.1-beta.5` station candidate.

Run each stage explicitly from the repository virtual environment:

```powershell
python spatial/search_hybrid_station.py 2>&1 |
  Tee-Object -FilePath $env:SMF_DATA_ROOT\reports\hybrid-v3-search.log
python spatial/fit_hybrid_station.py
python spatial/calibrate_hybrid_station.py
python spatial/evaluate_hybrid_station.py
python spatial/check_hybrid_gate.py
```

The search writes atomic state after every trial and safely resumes from
`hybrid_v3_search.state.json`. The final evaluation refuses to overwrite an
existing report. Only a fully passing bundle can be registered:

```powershell
python spatial/register_hybrid_station.py
```

Registration uses the separate `fuel_moisture_station_hybrid` beta model type.
Stable promotion is intentionally outside this repository and still requires a
30-day prospective shadow period.

# Guarded station V4 workflow

V4 permanently excludes the exposed V3 historical relock. It uses observed
station FM/RH/wind for category verification, an enhanced XGBoost base, a
bounded/gated seven-quantile GRU correction, and an out-of-fold per-lead guard.
V4 is experimental shadow-only until 30 new prospective days and an elevated
risk period pass every fuel-moisture, category, and probability gate.

```powershell
python scripts/backfill_hrrr.py --start 2025-07-14 --end 2026-08-01 --precip-context --existing-runs-only --workers 6
python spatial/build_aligned_dataset.py --max-initial-age-hours 3
python spatial/coverage_report.py --dataset "$env:SMF_DATA_ROOT\aligned\station_leads_precipitation-v1.csv"
python spatial/validate_precip_rebuild.py
python spatial/enrich_v4_dataset.py
python spatial/prepare_v4.py
python spatial/search_v4_base.py
python spatial/search_v4_residual.py
python spatial/fit_v4.py
python spatial/calibrate_v4.py
python spatial/evaluate_v4_development.py
python spatial/shadow_export_v4.py
```

If the complete HRRR rebuild is interrupted after the previous aligned CSV was
already valid, create the V4 label dataset without repeating the grid work:

```powershell
python spatial/enrich_v4_dataset.py
```

This joins observed RH and wind back to the exact station/target timestamp,
validates fuel-moisture provenance and the 30-minute tolerance, and writes
`station_leads_v4_precipitation-v1.csv` atomically. It never modifies
`station_leads.csv` or the earlier V4 dataset.
V4 commands use this separate dataset by default.

The precipitation backfill stores compact F00-F03 precipitation-only sidecars;
it does not duplicate the large F04-F15 temperature/RH/wind files. The aligned
builder writes versioned per-run fragments and resumes them when source mtimes
and the precipitation-contract checksum still match. If a sidecar arrives
after a fragment was built, that fragment is automatically invalidated.

V4 selection includes an August development fold plus later winter and spring
folds. It prefers summer MAE only among candidates within 1% of the best global
MAE, and residual training gives summer rows a modest 1.25 weight. Evaluation
writes the complete JSON report to the reports directory and prints only a
compact gate summary to the terminal.

`prepare_v4.py` must report `precipitation_available: true`. A constant-zero or
unitless archive now fails before a dataset is written. F04-F15 cumulative
values remain available as `hrrr_precip_mm` for compatibility, while V4 uses
the explicit interval and interval-duration fields.

Configure the API-only experimental loader with `SMF_V4_SHADOW_BUNDLE`. This
does not create a registry entry and cannot affect authoritative P50 output.
Prospective predictions and observations are written as separate immutable
files. Once sufficient new evidence exists:

```powershell
python spatial/evaluate_v4_prospective.py
python spatial/register_v4.py
```

`register_v4.py` refuses calibration or historical reports. The direct-danger
`forecast/firedangermodel.py` workflow is quarantined and requires the explicit
`--allow-experimental` acknowledgement.
# Summer-guarded V5 experiment

V5 is an experimental, non-registering path. It keeps the incumbent XGBoost
prediction as its exact fallback and applies a bounded shallow-XGBoost residual
only in regime/lead cells that improve development out-of-fold MAE without an
unsafe fold RMSE or critical-fuel-moisture regression.

```powershell
$env:SMF_DATA_ROOT='M:\_Development\ShowMeFire\training-data'
$env:SMF_XGB_DEVICE='cuda'
python spatial/prepare_v5.py
python spatial/search_v5.py
python spatial/fit_v5.py
python spatial/shadow_export_v5.py
```

The expensive causal feature frame is cached as
`reports/v5_static_features.pkl` and is accepted only when its evidence
manifest checksum, physics variant, row count, and index contract match.

Set `SMF_V5_SHADOW_BUNDLE` in the API only to the directory produced by
`shadow_export_v5.py`. Shadow records are immutable: the prediction file must
exist before a separate observation file can be attached. Repeated V5 errors
disable V5 shadow collection only.

`evaluate_v5.py` reads prospective runs strictly after 2026-08-01 and refuses
to overwrite its report. `register_v5.py` refuses registration unless that
prospective report passes every gate and explicitly allows beta registration.
Neither command changes the stable registry pointer.
