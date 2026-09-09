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

## Known environment limitation (discovered, not yet resolved)

`pyretechnics==2025.5.15` cannot currently be installed in this Windows dev
venv: it has no prebuilt wheel here and its Cython extension source targets
NumPy's old C-API (`_PyArray_Descr.subarray`), which NumPy 2.x's C-API
removed - a real ABI incompatibility discovered while setting up this model
family, not a`pip` or dependency-resolution problem. `pip install
--no-deps --no-build-isolation -r requirements-pyretechnics.txt` gets past
the numpy-version constraint and the missing-Cython error, but fails at the
final compile step with `C2039: 'subarray' is not a member of
'_PyArray_Descr'`. Production presumably installs a working prebuilt wheel
on Linux; that assumption should be checked before relying on it. Until this
is resolved (or run inside a Linux/WSL environment), `rothermel_labels.py`'s
pyretechnics-dependent tests are skipped (`unittest.skipUnless`), not failed -
see `tests/test_rothermel_labels.py`.

## Phases

1. **Scaffolding** (this document + the module map above) - done.
2. **Historical data build** - run the label generator across available
   historical RTMA + station fuel-moisture history to build a real
   `station_panel.csv`; measure real coverage before fitting anything.
3. **Fit + evaluate** - fit against the real panel, run the gates above,
   explicitly compare emulation accuracy and inference cost against running
   `pyretechnics` directly (the actual point of an ML emulator).
4. **Registration + shadow-serving** - only once Phase 3 passes: register a
   real beta, add a `fire_weather_ml` branch to
   `api/models/versioning.py::validate_promotion_candidate`, and build
   `api/services/fire_weather_ml_shadow.py` mirroring
   `risk_fusion_glm_shadow.py`'s pattern.

Phases 2-4 are not implemented yet - this increment is scaffolding only.
