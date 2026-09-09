# fire_weather_ml: First-Generation Fully-ML Fire Weather Risk Model

## Why this exists

The production system has two model families: the fuel-moisture XGBoost lineage,
and `risk_fusion/` (a Poisson GLM predicting county-day fire likelihood, fit
against real fire-occurrence reports). A request came in for a genuinely new,
fully-ML fire weather risk model - not an evolution of `risk_fusion` - living
on its own branch (`ml-fire-weather-v1`), predicting a continuous score.

## Why it is NOT trained against fire-occurrence

Explicitly rejected during design: ignition depends on human/lightning source
availability as much as weather, so a model fit to occurrence mostly learns
"where people are and when they're outside," not "how dangerous the weather
is." Training against occurrence would repeat `risk_fusion`'s exact framing
under a different name, not create something new.

## Why Rothermel-computed labels instead

`api/services/spread_rate.py` already runs a real Rothermel surface-fire
calculation live in production (via `pyretechnics`), every ~15 minutes:
real LANDFIRE fuel/canopy/slope static geography
(`data/static/bundles/fire_behavior_static_2026.4.*` - already built, not
synthetic or empty) combined with real hourly RTMA weather and RAWS-anchored
dead-fuel-moisture conditioning (`api/services/spread_rate_moisture.py`),
producing rate-of-spread, fireline-intensity, and flame-length outputs. It
only ever publishes the *latest* hour - `services/beta_products.py`'s
retention cleanup prunes older rasters, so there's no ready-made historical
label archive sitting on disk.

But it's a **deterministic physical calculation**, not an observation - its
exact logic can be re-run offline over historical weather/fuel-moisture data
to generate as much labeled training data as there is historical weather to
feed it. That turns "process more data" from a scarce-label collection
problem (like `risk_fusion` faces) into a compute problem, and gives a real,
physically-grounded, non-occurrence continuous target with effectively
unlimited volume.

This reframes the model's actual value proposition, and it's important to be
honest about it rather than oversell: **fire_weather_ml is not detecting fire
risk from nothing - it's learning to emulate/extend a real physics
calculation cheaply and at scale.** Two concrete ways it can add real value
beyond pure emulation:

1. **Feature expansion beyond what Rothermel uses** - antecedent drought
   (KBDI), accumulated warmth (GDD), and other multi-day-memory features the
   fixed physical formula has no way to incorporate (see `features.py`).
2. **Weak-signal calibration against reality** - real fire-occurrence and any
   observed-fire-behavior data are a *secondary, advisory* cross-check
   (`evaluate.py`'s `fire_occurrence_ranking_advisory` gate - always reported
   as `deferred`, never `pass`/`fail`), never the training target and never a
   promotion gate. This directly reflects the "not fire-event driven" design
   decision.

## Module map (`fire_weather_ml/`)

- `rothermel_labels.py` - independent re-implementation of
  `api/services/spread_rate.py::_compute_cell_ros`'s physics (same
  `pyretechnics` calls, same unit conversions), operating row-wise on a
  station/hour panel instead of a grid. Cross-checked against the real
  `api/services/spread_rate.py` logic in `tests/test_rothermel_labels.py`.
- `features.py` - the ML input feature set (current weather, fuel moisture,
  terrain, plus independently-implemented KBDI/GDD memory features).
- `panel.py` - joins per-station weather/fuel-moisture history to per-station
  static terrain, derives features, attaches the Rothermel label.
- `contract.py` - chronological, 5-day-episode-blocked train/test split
  (fuel-moisture memory is autocorrelated on roughly that timescale - see
  `api/services/spread_rate_moisture.py`'s `TAU_10_HR`/`TAU_100_HR`).
- `model_bundle.py` - XGBoost regressor fit/score/save/load. A real
  regression model is justified from v1 (unlike `risk_fusion`'s GLM
  restraint) because label volume here is bounded by historical weather
  data, not by scarce fire reports.
- `evaluate.py` - offline evidence report, explicit per-gate status
  (`pass`/`fail`/`deferred`/`not_applicable`, nothing silently omitted).
  Primary gates: beats a naive weather-only baseline (proves the KBDI/GDD
  features earn their keep) on held-out data. Advisory gate: occurrence
  cross-check, permanently non-gating.
- `register_beta.py` - registers a beta candidate to this repo's own
  training-side registry (`models/versioning.py`) only. `advisory_only=True`,
  not production-eligible. Does not touch `api/`.

## Independence from `risk_fusion`

No code is imported from `risk_fusion/` - KBDI/GDD, the split contract, and
the model-bundle shape are independently written here, even though the
*shape* (contract → fit → evaluate → register-beta, explicit gate statuses)
intentionally mirrors `risk_fusion`'s established, reviewed pattern.
`risk_fusion/county_precip_normals.json` (a data file, not code) is a
plausible Phase 2 input for calibrating KBDI's mean-annual-precipitation
term - reusing it as data would not violate this boundary.

## pyretechnics environment limitation - RESOLVED

`pyretechnics==2025.5.15`'s published sdist fails to *compile* against
NumPy 2.x headers (`error: 'PyArray_Descr' has no member named 'subarray'`) -
its generated Cython C source references a struct field NumPy 2.x's public
header no longer exposes directly. Confirmed on both this Windows venv and
a fresh WSL Ubuntu venv, so it's a real build-time incompatibility, not
platform-specific.

**Fix** (documented in `requirements-pyretechnics.txt`): build it against
NumPy 1.26.4 headers, then restore NumPy 2.2.6 afterward. NumPy's C-ABI is
backward compatible across this gap - confirmed by actually importing and
running `pyretechnics` (including a full `calc_surface_fire_behavior_max`
call) with NumPy 2.2.6 loaded, not just assumed from a version number.

```
pip install "numpy==1.26.4"
pip install --no-deps --no-build-isolation -r requirements-pyretechnics.txt
pip install "numpy==2.2.6"
```

With this, `tests/test_rothermel_labels.py`'s cross-check against
`api/services/spread_rate.py`'s real `_compute_cell_ros` runs (not skipped)
and passes - the concrete proof this module's physics matches production's,
not an approximation of it. That cross-check also caught two real bugs
during Phase 2 (both fixed, not worked around): `compute_row_label` was
importing `pyretechnics` before its own numeric fuel-code range check
(should short-circuit without importing anything for an out-of-range code),
and the test itself was comparing percent-unit inputs against
`_compute_cell_ros`'s fraction-unit contract (production's caller,
`compute_spread_rate_grid`, converts before calling it - the test now
converts too, for a fair comparison).

## Phase 2 - real historical data build (done)

Real files inspected and used, not assumed:

- **Weather + observed 10-hr fuel moisture**: `paths.PRECIP_ALIGNED_DATASET`
  (`training-data/aligned/station_leads_v4_precipitation-v1.csv`), filtered to
  the static bundle's Missouri bbox `(-96.8, -88.1, 34.8, 41.8)`: 41 stations
  in that rectangle, spanning 2025-07-14 to 2026-08-02 (~13 months).
- **Static terrain**: `data/static/bundles/fire_behavior_static_2026.4.nc`
  (real, already-built 256x256 LANDFIRE/3DEP grid) via a new
  `static_lookup.py` (nearest-valid-cell cKDTree match, independently
  implemented). All 41 in-bbox stations matched a valid cell - the bbox
  rectangle, not terrain coverage, was the real constraint.
- **Real county filter**: the bbox rectangle spans parts of AR/IA/KS/NE too.
  A new `precip_normals.py` nearest-county spatial join (against
  `risk_fusion/county_boundaries.geojson` + `risk_fusion/county_precip_normals.json`
  - real MO-only data, reused as data, not imported as code) is what actually
  restricts the panel to real Missouri stations: 23 of 41 stations fell
  outside every MO county polygon despite being inside the bbox rectangle,
  leaving **18 real Missouri stations**.
- **Real wind direction**: the aligned CSV only has wind speed (direction was
  lost in its per-lead-hour aggregation). A new `wind_direction.py` reads
  real `u10`/`v10` from the historical RTMA cache
  (`training-data/cache/rtma/rtma_*.nc`, ~9,500 hourly files) at each
  station's nearest grid cell - 0 rows had to be dropped for a missing cache
  hour; the RTMA archive fully covers this window.
- **Derived fm1/fm100 and live-fuel moisture**: the source data only has one
  fuel-moisture reading (treated as the observed 10-hr class) and no live-
  fuel signal. `features.derive_fm1_fm10_fm100` (Nelson-EMC free-running
  1/10/100-hr propagation, then all three shifted by the residual between
  the free-running 10-hr estimate and the REAL observed 10-hr reading - the
  same anchor-to-a-real-point-observation idea as
  `api/services/spread_rate_moisture.py`'s RAWS correction, applied
  temporally here since a real reading exists every hour) and
  `features.live_moisture_percent` (GDD-driven Scott-Burgan L2/L4 mapping)
  are explicitly flagged as approximations in their own docstrings - genuine
  derivations, not observations, and not yet calibrated against real Missouri
  green-up data.
- **Real result** (`training-data/fire_weather_ml/station_panel.csv`,
  produced by `historical_panel.py`): **18 stations, 68,244 station-hours,
  64,320 with a real Rothermel-computed label (94.25% - the other 5.75% are
  genuine non-burnable-fuel-model or degenerate-input cells, not a bug),
  spanning the full ~13-month window.** That's roughly 215x `risk_fusion`'s
  ~300 primary-tier fire-event count, sitting on disk already rather than
  newly collected - the real "process more data" number this design promised.
  Label values are physically plausible for Missouri's climate (`ros_ch_per_h`
  up to ~14.2 chains/hour - "Moderate" on production's own 6-class scale, not
  the extreme end that scale's western-wildfire-oriented upper classes target).

New modules this phase: `static_lookup.py`, `precip_normals.py`,
`wind_direction.py`, `historical_panel.py` (the orchestrator), plus
`derive_fm1_fm10_fm100`/`live_moisture_percent`/`nelson_emc` added to
`features.py`. 39 new tests (`test_static_lookup.py`, `test_precip_normals.py`,
`test_wind_direction.py`, `test_fire_weather_derived_moisture.py`,
`test_historical_panel.py`), including a full synthetic end-to-end
integration test using the REAL `risk_fusion` county/precip-normal data
(small, checked-in, no network needed).

## Phases

1. **Scaffolding** - done.
2. **Historical data build** - done (see above). Real coverage: 18 stations,
   68,244 rows, 64,320 labeled, ~13 months.
3. **Fit + evaluate** - fit against the real panel, run the gates in
   `evaluate.py`, explicitly compare emulation accuracy and inference cost
   against running `pyretechnics` directly (the actual point of an ML
   emulator). Not yet done - next session's work.
4. **Registration + shadow-serving** - only once Phase 3 passes: register a
   real beta, add a `fire_weather_ml` branch to
   `api/models/versioning.py::validate_promotion_candidate`, and build
   `api/services/fire_weather_ml_shadow.py` mirroring
   `risk_fusion_glm_shadow.py`'s pattern.

Phases 3-4 are not implemented yet.
