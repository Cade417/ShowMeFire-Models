# Observed Rothermel Spread Rate Operator Runbook

This runbook creates and deploys the immutable static geography required by
the experimental Observed Rothermel Spread Rate product.

The spread-rate calculation is a physical Rothermel calculation, not a fitted
machine-learning model. The desktop builds its versioned LANDFIRE/topography
input bundle. The API combines that bundle with hourly RTMA weather and RAWS
fuel-moisture observations every 15 minutes.

Production must never use the synthetic bundle from
`api/scripts/build_synthetic_fire_behavior_bundle.py`.

## 1. Desktop setup

Clone or update `ShowMeFire-Models` on the machine with enough local storage:

```bash
cd ShowMeFire-Models
python -m venv .venv
source .venv/bin/activate             # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export SMF_DATA_ROOT=/path/to/large/local/disk/showmefire-data
export SMF_GITHUB_REPO=OWNER/ShowMeFire-Models
```

`SMF_DATA_ROOT` contains source rasters, bundles, model registry files, and
weather archives. It is intentionally gitignored.

Authenticate GitHub CLI before publishing:

```bash
gh auth login
# Or export GH_TOKEN using a token that can create releases in the repository.
```

## 2. Obtain official source rasters

Download all four inputs for the same domain and document their release:

| Input | Official source | Required native values |
| --- | --- | --- |
| DEM | USGS 3DEP / The National Map | elevation in metres |
| FBFM40 | LANDFIRE | raw Scott–Burgan fuel-model codes |
| Canopy cover | LANDFIRE | percent, 0–100 |
| Canopy height | LANDFIRE | metres |

Use the current official LANDFIRE release for all three LANDFIRE layers. Do
not rescale, classify, or colorize FBFM40. If a downloaded raster uses encoded
units, convert it to the required units before registration.

The repository includes an automated downloader. It uses the LANDFIRE Product
Service for a clipped three-layer request and The National Map API to discover
and mosaic the matching 3DEP DEM tiles. The LANDFIRE release is intentionally
explicit so a later release cannot silently change an experiment. The LFPS
service requires an identifying email address; it is used for request
traceability and is not stored in the source manifest.

From PowerShell, run:

```powershell
python static_features/download_fire_behavior_assets.py `
  --email "analyst@example.org" `
  --landfire-release LF2024
```

The default domain is the Missouri domain plus buffer
(`-96.8,34.8,-88.1,41.8`). Override it with `--bbox west,south,east,north`
when needed. Add `--force` only when intentionally replacing an existing
source set. The command writes `dem.tif`, `fbfm40.tif`, `canopy_cover.tif`,
and `canopy_height.tif` under `$SMF_DATA_ROOT/static/source/`, then updates
`source_manifest.json`. Add `--output-resolution 30` or another value greater
than 30 to request a coarser common LFPS resolution; omit it to retain the
provider-native resolution. Network failures can be retried; partial
downloads are retained.

3DEP tiles download concurrently with six workers by default. Adjust
`--workers` for your connection, for example `--workers 10`; use
`--workers 1` for a serial download if the provider throttles parallel
requests.

LF2025 currently exposes the FBFM40, canopy-cover, and canopy-height layers
only for its `SW` and `NW` update areas. For the default Missouri `SC` domain,
use `LF2024` unless the LFPS product table confirms that a newer release covers
the entire AOI.

LFPS jobs can take more than 15 minutes for a large native-resolution AOI. The
downloader waits up to 30 minutes by default. If the local process times out
after LFPS later reports success, reuse the existing job (the download remains
available for a limited time):

```powershell
python static_features/download_fire_behavior_assets.py `
  --email "contact@showmefire.org" `
  --landfire-release LF2024 `
  --landfire-job-id "d42e5486-2e62-4af0-944f-fab21cbf4771"
```

Inspect every raster before continuing:

```bash
gdalinfo /data/dem_m.tif
gdalinfo /data/landfire_fbfm40.tif
gdalinfo /data/landfire_canopy_cover_pct.tif
gdalinfo /data/landfire_canopy_height_m.tif
```

Each file must be a single-band georeferenced raster with a declared CRS and
nodata value. Register the files and their checksums in the source manifest:

```bash
python static_features/download_sources.py \
  --dem-file /data/dem_m.tif \
  --dem-units m \
  --fbfm40-file /data/landfire_fbfm40.tif \
  --fbfm40-units code \
  --canopy-cover-file /data/landfire_canopy_cover_pct.tif \
  --canopy-cover-units percent \
  --canopy-height-file /data/landfire_canopy_height_m.tif \
  --canopy-height-units m \
  --release 3dep-LANDFIRE-RELEASE
```

The command copies inputs below
`$SMF_DATA_ROOT/static/source/` and creates `source_manifest.json` with source
paths, checksums, CRS, dimensions, nodata, and units.

## 3. Select the reference HRRR grid

The bundle is aligned to the API's 256×256 Missouri HRRR grid. Use a
representative HRRR NetCDF previously unpacked from a server archive:

```bash
./scripts/pull_archives.sh user@server /remote/path/to/api/data_archive_day/
```

Choose a complete HRRR file that contains `x`, `y`, `latitude`, `longitude`,
and its CF grid-mapping variable:

```bash
export HRRR_REFERENCE="$SMF_DATA_ROOT/cache/hrrr/REPLACE_WITH_REAL_FILE.nc"
python -c "import xarray as xr, os; d=xr.open_dataset(os.environ['HRRR_REFERENCE']); print(d.sizes); print(d.longitude.shape)"
```

Do not use an RTMA file as the reference grid.

If no HRRR NetCDF is already available, download a small cropped reference
file from the official HRRR archive with Herbie:

```powershell
python static_features/download_reference_hrrr.py `
  --date 2026-08-31 `
  --cycle 12
```

The file is written to `$SMF_DATA_ROOT/cache/hrrr/`. Use the printed path in
the bundle command below. The date and cycle are explicit so the reference
grid used by an experiment is reproducible.

## 4. Build and validate the immutable bundle

Choose a human-readable bundle version tied to the source release:

```bash
export FIRE_BUNDLE_VERSION=2026.1

python static_features/build_fire_behavior_bundle.py \
  --hrrr "$HRRR_REFERENCE" \
  --version "$FIRE_BUNDLE_VERSION"
```

Outputs:

```text
$SMF_DATA_ROOT/static/bundles/fire_behavior_static_2026.1.nc
$SMF_DATA_ROOT/static/bundles/fire_behavior_static_2026.1.json
```

The builder fails rather than publishing bad geography. Validation covers:

- source and bundle checksums;
- source CRS, dimensions, nodata metadata, and declared units;
- exact 256×256 grid and grid fingerprint;
- finite values in valid cells;
- slope, canopy cover, and canopy height ranges;
- integer, recognized FBFM40 codes and burnable-cell coverage.

Bundles are immutable. Use a new version if any input or builder behavior
changes.

## 5. Register the desktop beta candidate

```bash
python pipelines/register_fire_behavior_static.py \
  --bundle "$SMF_DATA_ROOT/static/bundles/fire_behavior_static_${FIRE_BUNDLE_VERSION}.nc"
```

This validates the bundle again, copies both assets into the desktop registry,
and creates a `fire_behavior_static` beta entry. It does not affect production.

Inspect the candidate:

```bash
python -c "from models.versioning import get_model_entry; import json; print(json.dumps(get_model_entry('fire_behavior_static'), indent=2))"
```

## 6. Publish the GitHub prerelease

Use the beta version printed by the registration command:

```bash
python pipelines/publish_release.py \
  --model fire_behavior_static \
  --version 0.0.1-beta.1 \
  --repo "$SMF_GITHUB_REPO"
```

The release contains:

- the NetCDF bundle;
- its JSON manifest;
- `metadata.json` with asset declarations, checksums, fingerprint, and
  validation metadata.

Keep the release as a prerelease until the production import check succeeds.

## 7. Import and promote on the API server

Run from the production `api/` directory:

```bash
export SMF_GITHUB_REPO=OWNER/ShowMeFire-Models

python pipelines/import_model.py \
  --model fire_behavior_static \
  --tag fire_behavior_static-v0.0.1-beta.1 \
  --repo "$SMF_GITHUB_REPO"

python pipelines/promote_model.py --model fire_behavior_static
```

Import rejects missing checksums, synthetic bundles, invalid validation gates,
wrong dimensions, missing channels, and grid-fingerprint mismatches.
Promotion atomically switches the registry entry; individual assets must
never be copied or swapped manually.

After successful production promotion, the GitHub prerelease can be marked as
a normal release:

```bash
gh release edit fire_behavior_static-v0.0.1-beta.1 \
  --repo "$SMF_GITHUB_REPO" \
  --prerelease=false
```

## 8. Warm RTMA history and generate the first product

The product needs at least 120 complete causal hours and targets 168 hours.
Warm-up is resumable and does not block API startup:

```bash
python -c "from services.rtma_capture import warmup_rtma_cache; print(warmup_rtma_cache(days=7))"
```

Then use the Testbed admin **Rebuild spread rate** action, or wait for the
15-minute scheduler. The server retains the seven-day RTMA cache locally; a
Cloudflare cache is not required for this workflow.

Verify:

```bash
curl -fsS https://api.showmefire.org/api/testbed/spread-rate/status
curl -I https://api.showmefire.org/testbed-assets/images/spread_rate_latest.png
curl -I https://api.showmefire.org/testbed-assets/gis/spread_rate/spread_rate_latest.tif
```

Expected progression:

1. `waiting_for_rtma` while the latest analysis has not arrived;
2. `warming` until at least 120 complete hours are cached;
3. `ready` after successful Rothermel calculation and atomic publication.

## 9. Rollback

List registry history and choose the previous stable version:

```bash
python -c "from models.versioning import get_model_entry; import json; print(json.dumps(get_model_entry('fire_behavior_static'), indent=2))"

python pipelines/promote_model.py \
  --model fire_behavior_static \
  --rollback \
  --version PREVIOUS_VERSION
```

Rollback changes the complete asset contract atomically. The next scheduled
spread-rate run uses the restored bundle.

## Troubleshooting

### `No asset contract for 'fire_behavior_static' channel 'stable'`

The server has not imported and promoted a fire-behavior bundle. Complete
steps 5–7.

### Source units rejected

Convert the raster itself into metres or percent as required, then rerun
`download_sources.py`. Do not relabel encoded values by changing only the
`--*-units` argument.

### Unknown FBFM40 codes

Confirm the source is the raw Scott–Burgan FBFM40 raster, not a color table,
FVT layer, or resampled continuous surface.

### Product remains `warming`

Rerun the seven-day warm-up. Failed NOAA hours can be retried safely. Check
`conditioning_hours_available` and `error` in the public status endpoint.

### PNG remains missing

Read the status endpoint first. Artifacts are written only after static input,
RTMA history, RAWS conditioning, and Rothermel output all validate.
