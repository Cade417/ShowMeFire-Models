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

## Phase 3 - real fit + evaluate (done)

Fixed one more real bug on the way: `contract.assign_episodes` compared a
tz-naive `EPOCH` against the real panel's tz-AWARE `valid_time` (UTC,
`"...+00:00"`) and raised - Phase 1's synthetic tests never used tz-aware
timestamps, so this only surfaced against real data. Fixed by normalizing
tz-aware input to UTC-then-naive before comparing (regression test added).

**Real held-out result** (`training-data/reports/fire_weather_ml_offline_evaluation.json`,
5-block episode-blocked CV): candidate (all `FEATURE_COLUMNS`, including
KBDI/GDD) R²=0.942, MAE=0.0564 ch/h. Baseline (weather-only, no KBDI/GDD)
R²=0.962, MAE=0.0427 ch/h. **`beats_naive_weather_only_baseline` gate:
fail** - the candidate does not beat the baseline; it's slightly worse.

This is a real, honest, explainable finding, not a bug: the Rothermel
calculation's true causal inputs (fuel moisture, wind, terrain) are ALL
already in the baseline feature set - `fm1_pct`/`fm10_pct`/`fm100_pct` are
literally derived from the same `temp_c`/`rh_pct` the baseline sees, plus
the real observed 10-hr anchor. Both feature sets already contain
everything the physics calculation is a function of, so both hit
R²>=0.94, and KBDI/GDD (which the Rothermel formula never uses at all)
have no causal channel left to add value on **this specific target**. The
original design doc's "feature expansion beyond what Rothermel uses" value
proposition (see above) doesn't hold for pure spread-rate-emulation
accuracy - it would only matter for a target that actually depends on
drought/season memory (e.g. real fire behavior/occurrence), which this
model deliberately doesn't train against. Worth revisiting whether KBDI/GDD
belong in a pure-emulation model at all, or only in a future broader risk
score built on top of it - not resolved here, flagged for a future session.

**Real occurrence cross-check attempt** (`occurrence_crosscheck.py`):
correctly returns `available: False` - `fire_labels_20260808.csv` covers
2011-01-02 through 2020-12-31, the real historical panel covers 2025-07-14
through 2026-08-02. **Zero date overlap.** The advisory cross-check this
design promised genuinely cannot run with what's on disk today; the module
measures and reports that exact gap rather than fabricating a statistic.
Revisiting this needs either a re-exported, more recent `fire_labels` CSV,
or building the panel over 2011-2020 RTMA/station history instead (if that
history exists) - not attempted this session.

**Real emulation-cost result** (`emulation_cost.py`, 2000-row sample):
model scoring averages 1.73 microseconds/row; running the real Rothermel
physics (`rothermel_labels.py`) averages 56.6 microseconds/row - **a real,
measured 32.7x speedup**. This is the part of the original design's value
proposition that DOES hold up: whatever this model predicts, it predicts
it about 33x cheaper than running the actual physics, which matters at
scale (e.g. a statewide grid instead of 18 stations).

**Registration correctly refused**: `register_beta.py` ran against the real
fitted candidate (`fit_model.py` output,
`training-data/models/fire_weather_ml_shadow_candidate/`) and refused with
`failing gates ['beats_naive_weather_only_baseline']` - the safety gate
worked exactly as designed, blocking a candidate that doesn't clear its own
bar rather than registering it anyway.

New modules: `occurrence_crosscheck.py`, `emulation_cost.py`. `evaluate.py`
now actually computes both (previously permanently-`deferred` stubs) - the
occurrence gate's *status* stays `deferred` regardless of result (never
gates, per this model family's design), the cost gate's status reflects
whether it could be measured (`pass` here). 11 new tests.

## Phase 4 - unblock registration + shadow-serving (done)

**Decision** (asked of, and made by, the project owner): drop `kbdi`/
`gdd_accum` from the model's input features entirely rather than redesign
the comparison gate. `features.MODEL_FEATURE_COLUMNS` (weather/fuel-
moisture/terrain only) is now what the model actually trains/predicts on;
`FEATURE_COLUMNS` (the full panel schema, kbdi/gdd_accum included) stays as
computed panel data - still real, still available, just not fed to this
model, in case a future broader risk score wants it.

With no second feature set left to compare against, `evaluate.py`'s
`beats_naive_weather_only_baseline` gate (policy v1) was retired and
replaced with `achieves_high_emulation_accuracy` (policy v2): an absolute
`R^2 >= 0.90` bar, comfortably below the real observed ~0.96 so a
genuinely broken candidate still fails it, without being tied to a since-
removed comparison. **Real result on the actual panel: R²=0.9622,
MAE=0.0427 ch/h - identical to Phase 3's "baseline" numbers, confirming
the feature-set change was applied correctly.** All gates now pass.

**Real registration succeeded**: `fit_model.py` → `register_beta.py`
registered `fire_weather_ml` beta `0.0.1-beta.1` in this repo's own
training-side registry (`training-data/models/versions/`), `advisory_only:
true`, `production_eligible: false`, `prospective_shadow_required: true` -
the same advisory pattern `fire_risk_fusion` uses.

**Real shadow-serving wiring into `api/`** (on `api/`'s own
`ml-fire-weather-v1` branch, mirroring this repo's branch name - nothing
touches either repo's `main`): new `api/services/fire_weather_ml_shadow.py`
mirrors `risk_fusion_glm_shadow.py`'s exact shape (kill switch, persisted
state, immutable evidence, raw-bundle-via-`SMF_FIRE_WEATHER_ML_BUNDLE`
loading, never raises). Because `MODEL_FEATURE_COLUMNS` is exactly the set
of weather/fuel-moisture/terrain quantities `api/services/spread_rate.py`
already computes every run, the shadow scores the SAME grid the real
Rothermel calculation just ran on and directly compares against it - a
live, ongoing accuracy check beyond this offline evaluation. Also added: a
`fire_weather_ml` branch in `api/models/versioning.py::validate_promotion_candidate`
(mirroring `fire_risk_fusion`'s, for a future real promotion pipeline - not
currently exercised, matching how `fire_risk_fusion` itself is scored via
raw bundle rather than through the registry today), a
`/api/model/spatial/fire-weather-ml-shadow-diagnostics` route, and an
additive call site in `spread_rate.py`'s `generate_spread_rate()` (never
touches `grids` or anything derived from it - failure-isolated, logged and
swallowed on error). 13 new tests in `api/tests/test_fire_weather_ml_shadow.py`.

A real bug was found and fixed along the way: the shadow module's first
draft imported `aspect_degrees` from `services/spread_rate.py`, which pulled
in that module's entire heavy import chain (xarray, rtma_capture, etc.)
just to reuse a two-line trig function - fixed by defining it locally in
`fire_weather_ml_shadow.py` instead.

**Not yet done, deliberately deferred**: actually setting
`SMF_FIRE_WEATHER_ML_BUNDLE`/`FIRE_WEATHER_ML_SHADOW_ENABLED` on any real
server (dev or production) - this session only built and tested the
capability, following `risk_fusion`'s own precedent of landing the code
first and enabling it as a separate, deliberate step. The real occurrence
cross-check still can't run (fire_labels/panel date ranges still don't
overlap - unchanged from Phase 3, not addressed this phase).

## What this model actually is (clarified after a real gap in scope)

Important correction made mid-project: the original ask was an ML model to
predict **fire danger** (a continuous alternative to the bucketed public
category), not a spread-rate emulator specifically. What got built predicts
rate of spread - a real, physically-grounded fire *behavior* quantity, but
not by itself a *danger* score, and it was never mapped back to that
original goal. Spread rate is a legitimate basis for a danger signal
(faster spread = more dangerous), but the gap between "predicts fire
behavior" and "answers how dangerous" was never actually closed.

**The concrete bridge, built as a first step**: `api/services/fire_weather_ml_shadow.py`
now also computes the real public rule-based category (`core/fire_danger.py`,
unchanged) for the same cells, and reports the continuous ML "Fire Weather
Risk" (predicted spread rate) range found WITHIN each rule category - in
the evidence JSON, diagnostics state, manifest, and the rendered graphic.

**Real finding from running this against the full historical panel**: the
rule-based category is "Low" for **96.3%** of real hours (65,700 of
68,244) and never once reaches Elevated/Critical/Extreme across the full
13-month window - because its first check, `fuel_moisture >= 15%`,
unconditionally forces Low regardless of wind or humidity. Meanwhile the
continuous ML signal ranges from 0.0 to **6.49 ch/h** *within that single
"Low" bucket* - e.g. a real hour at CHOM7 (Dec 29 2025, fm=17.5%, RH=67%,
wind=14.9kts) predicted 6.49 ch/h and was called "Low," sitting right next
to the mean of the "Moderate" category (1.69 ch/h) - while a calm, humid
"Low" hour elsewhere predicted 0.0. The bucket compresses real, meaningful
variation the continuous score actually resolves.

**Follow-up correction**: the comparison graphic above was itself flagged
as "not actually different" - it was still framed around the old rule's
categories/spread-rate classes, not a standalone thing. Built in response:
a genuinely separate **ML Fire Weather Risk Score** product -
`api/services/fire_weather_ml_shadow.py::_render_risk_score_png`, its own
map, own continuous 0-100 colorbar, own manifest entry
(`fire_weather_ml_risk_score`), no discrete classes, no reference to the
rule-based category or the physics ROS_CLASS scale.

The 0-100 score itself is calibrated (`model_bundle.calibrate_risk_score`/
`risk_score_0_100`, a new required bundle asset,
`fire_weather_ml_risk_calibration.json`) against **this model's own real
prediction distribution**, not a borrowed physics scale - deliberately,
because that distribution is heavily right-skewed on real data (p50=0.07
ch/h, p90=1.08 ch/h): reusing a fixed scale built for more extreme fire
climates (like `spread_rate.py`'s own 0-150 ch/h classes) would collapse
almost everything into the bottom of the scale too, the exact same failure
mode being fixed. Percentile rank sidesteps that - it's evenly spread
across 0-100 by construction regardless of the raw distribution's shape.

Real registered candidate refit with this asset: beta `0.0.1-beta.3`.

**Not yet done**: this is real comparison + standalone-score infrastructure,
not a finished danger-score replacement decision. Real remaining work,
explicitly not attempted this session: deciding whether this continuous
score (or a combination with fireline intensity/flame length, both already
computed by `rothermel_labels.py` but not currently modeled as outputs)
should become an actual proposed alternative/supplement to the public
category - that's a real product decision, not something to default into.

## Phases

1. **Scaffolding** - done.
2. **Historical data build** - done. Real coverage: 18 stations, 68,244
   rows, 64,320 labeled, ~13 months.
3. **Fit + evaluate** - done. Found and explained why kbdi/gdd_accum didn't
   help pure emulation accuracy.
4. **Registration + shadow-serving** - done (see above). Registered beta
   `0.0.1-beta.1`; shadow-serving code built and tested in `api/` on its
   own branch, not yet enabled on any running server. Now also reports the
   continuous Fire Weather Risk signal against the real rule-based
   category - the concrete first step toward the original "danger" ask,
   not yet a finished replacement for it.

All four phases are complete for this v1 increment. Real future work:
enable the shadow on an actual server and let evidence accumulate; get
fresher `fire_labels` data (or a panel over 2011-2020) so the occurrence
cross-check can finally run; decide whether kbdi/gdd_accum earn a place in
a future broader risk score built on top of this pure spread-rate emulator.
