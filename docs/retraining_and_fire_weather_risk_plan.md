# Biweekly Fuel-Moisture Retraining + Fire Weather Risk ML Model

## Context

Two related asks: (1) make the production fuel-moisture XGBoost model easy to retrain every two weeks in a way that's actually likely to improve it, with cleaner data management, and (2) go beyond the current rule-based fire-danger category (which leans almost entirely on fuel moisture) with a real ML "Fire Weather Risk" signal.

Research turned up a key fact that changes the shape of (2): **`model-training/risk_fusion/` already *is* a Fire Weather Risk ML model** — a Poisson GLM predicting fire likelihood per county-day from weather features (KBDI, VPD, wind, RH, precip, seasonal baseline), evaluated against real fire-occurrence labels (`fire_events`, deduped, satellite+citizen). It's just stuck permanently in shadow-only/advisory-only status and was never meant to reach users. The highest-leverage move is evolving that existing, already-validated pipeline rather than building a parallel model from scratch.

For (1), the current pipeline is a manual 8-phase shell script (`trainnewmodel.sh`) with fixed hyperparameters (no search), a fully destructive re-ingest every run (12,692 rows, full DB wipe/reload — not incremental), and a two-repo GitHub-release round-trip to promote. None of that is fundamentally broken, but there's no automation, no visibility into whether a retrain actually helped, and several hardcoded assumptions (e.g. station `ASLM7` hardcoded in `prepare_features.py`).

## 1. Fuel-Moisture Retraining — Redesign

**Goal:** running the pipeline every two weeks should be one command, should surface whether the new model is actually better, and should give the model more chances to improve (hyperparameter search) instead of just reprocessing the same fixed recipe on slightly more data.

**a) Single orchestrator, not a new pipeline.** Wrap the existing 8 phases (`ingest_obs.py` → `index_stations.py` → `create_snapshots.py`/`reset_snapshots.py` → `extract_hrrr.py` → `generate_training_set.py` → `prepare_features.py` → `train_model.py` → validation) in one `model-training/pipelines/retrain_fuel_moisture.py` that calls each phase's existing `main()`/function directly (not via subprocess-per-script), so failures are caught centrally and the whole thing is a single `python retrain_fuel_moisture.py` invocation. This is `trainnewmodel.sh` made robust, not replaced — no phase logic changes.
- Fix the hardcoded-path issue found in `check_directories()` (bare `data`/`models`/`plots`/`cache/hrrr` names ignoring `paths.py`/`SMF_DATA_ROOT`) while touching this file, so the orchestrator works regardless of working directory.
- Leave `ASLM7`-hardcoded preview plot in `prepare_features.py` alone (cosmetic, not a correctness issue) unless it turns out to block automation.

**b) Make "did this help?" visible.** This is the actual answer to "hard to get a model to improve" — right now nobody can see if a retrain is better than last time without manually comparing printed metrics. Add a tiny experiment log: `model-training/pipelines/experiment_log.py` appends one row per training run to `training-data/experiments/fuel_moisture.jsonl` — `{run_id, timestamp, git-ish feature list hash, training_samples, test_samples, mae, r2, beta_version}` — and after registering the new beta, print a diff against the last logged run and against whatever is currently `stable`'s recorded `performance` (already stored in `models/config.json`). No new dependency (no MLflow) — this is intentionally lightweight, matching the project's existing "flat JSON files as the source of truth" convention (`models/config.json`, evidence JSONs).

**c) Give the model room to actually improve: hyperparameter search.** `train_fuel_moisture_model` today fits one fixed `XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)` with a single chronological 80/20 split — no search, no cross-validation. Add a small rolling-origin CV search (3-4 chronological folds, not k-fold — this is time-series data) over a modest grid (`n_estimators`, `max_depth`, `learning_rate`, maybe `min_child_weight`), reusing `sklearn`'s `ParameterGrid` (already a dependency) rather than adding Optuna. Keep the final holdout evaluation exactly as-is for comparability with historical metrics.

**d) Promotion stays human-gated, but informed.** Don't automate `promote_model.py` — the project's whole safety culture (checksum gates, shadow evidence, manual promote) argues against auto-promoting a production-facing model. Instead, the orchestrator's final output should be a clear "beta MAE X vs stable MAE Y, N% {better|worse}" line so the human running `publish_release.py`/`promote_model.py` is making an informed call, not eyeballing raw numbers.

**e) Scheduling.** A biweekly OS-level scheduled task (cron on the training machine, or Windows Task Scheduler given this is a Windows box) invokes `retrain_fuel_moisture.py`, stopping after step (b)/(c) — it produces a beta + experiment log entry + comparison line, and stops. Publishing/promotion remains a manual, deliberate step. No new scheduler infrastructure needed (this doesn't touch `api/core/scheduler.py`, since it's a training-repo-side job, not a serving-side one).

## 2. Cleaner Data Management

**Don't build a feature store or adopt DVC** — 12,692 rows / single state is not a scale problem, and the project already has a working single-source-of-truth convention (`paths.py`, `SMF_DATA_ROOT`). The actual pain is "which of 5 separate scripts do I run to get new data," not "we lack versioning."

- **One sync entrypoint:** `model-training/scripts/sync_training_data.py` calls the existing incremental pullers in sequence — `pull_archives.sh` (already incremental/rsync-resumable), `backfill_synoptic.py`, `backfill_rtma_for_hrrr.py` (already has `--dry-run`/`--limit`) — and prints a one-line summary of what's new (file counts, date ranges). This replaces "read the README, remember 3 script names" with one command; none of the underlying pull logic changes.
- **Replace ad hoc backup-suffix files with one manifest.** Today there are files like `station_leads.before-20260801.csv` standing in for versioning. Add a single `training-data/manifest.json` (updated by `sync_training_data.py`) recording last-synced date ranges per source — enough to answer "how fresh is my data" without a real versioning system.
- Leave `ingest_obs.py`'s full-wipe-and-reload behavior as-is — with only 396 daily JSON files and 12,692 rows, a full reprocess is fast and simple, and making it incremental would add real complexity (partial-state bugs, dedup-on-append logic) for a dataset this small. Flag this explicitly as a non-goal rather than silently leaving it unaddressed.

## 3. Fire Weather Risk ML Model

**Revised after closer inspection (2026-08-14): this section originally assumed risk_fusion was an unpromotable GLM with no uncertainty and no serving path. That was wrong — the system is already substantially more built-out than the assumption behind the original 3(a)/3(b) below. Corrected understanding and the actual remaining work:**

**What already exists (verified by reading the code, not assumed):**
- A real, gate-passing Poisson GLM (monthly-baseline + fast-weather residual, `fit_glm.py`/`model_bundle.py`) fit on real fire-occurrence labels, registered as `fire_risk_fusion` beta `0.0.1-beta.2` in the training registry, all evaluation gates passing (`evaluate_risk_fusion.py`).
- **Calibrated uncertainty already implemented**: `fit_glm.fit_lambda_uncertainty`/`lambda_interval` (empirical residual-quantile half-widths, monthly regime buckets) — the same pattern the earlier draft of this plan proposed adding from scratch.
- **A full shadow-serving path already wired into live forecasts**: `api/services/risk_fusion_shadow.py` (Phase A, zero-label rule-Monte-Carlo) and `api/services/risk_fusion_glm_shadow.py` (Phase B, scores the real registered GLM), both called from `api/services/risk_fusion_hook.py` right after every `DailyForecast.py` run. Both are failure-isolated (never raise into the public forecast path), write immutable evidence files, and auto-disable after repeated failures — exactly the "start accumulating shadow evidence toward promotion" step this plan originally proposed as new work.
- Diagnostics already exposed at `/api/model/spatial/risk-fusion-glm-shadow-diagnostics`.

**~~a) Upgrade the algorithm to XGBoost count:poisson~~ — explicitly rejected, don't do this.** `fit_glm.py`'s own module docstring documents why: at the label volumes available so far (~40 independent synoptic episodes' worth of signal — the unit of independent information here is the *episode*, not the county-day row), a GBM's variance would dominate its bias advantage. The codebase already has a placeholder for a future "guard/GBM residual" stage (`guard_active_row_fraction` metadata field, mirroring V5's base+guard architecture) gated on more label data accumulating first — the team already made this call. Revisit only once episode count grows substantially, not as a standing task.

**a) [DONE 2026-08-14] Enable the existing shadow pipeline.** It was fully built but dormant (no env vars set). Set in `api/.env`:
```
RISK_FUSION_SHADOW_ENABLED=true
RISK_FUSION_GLM_SHADOW_ENABLED=true
SMF_RISK_FUSION_GLM_BUNDLE=<path to training-data/models/risk_fusion_shadow_candidate>
```
Verified: both phases report `enabled: True`, the registered bundle's `feature_module_sha256` matches the current `api/core/risk_fusion_features.py` exactly (no drift), and `load_bundle()` validates cleanly. This bundle predates uncertainty fitting (`uncertainty_available: False`) — re-running `model_bundle.fit()`/`register_risk_fusion_beta.py` on a future retrain would pick up `fit_lambda_uncertainty` and add real intervals to the shadow evidence.

**This only takes effect on whichever machine's `.env` gets these three lines.** The production server needs the same change made there separately — this session only touched the local dev checkout.

**b) [DONE 2026-08-16] Unlock KBDI properly.** `features.py`'s `keetch_byram_drought_index()` requires a real per-county mean-annual-precipitation climate normal and explicitly refuses a guessed value. Closed via `risk_fusion/build_precip_normals.py`, which pulls real 1991-2020 annual precipitation from NOAA NCEI's public Climate at a Glance County Time Series data service (endpoint verified by inspecting the tool's own network requests in a browser, not guessed) for all 115 Missouri counties in `county_reference.json`, and computes the 30-year mean per county. Output: `risk_fusion/county_precip_normals.json` (115/115 counties succeeded, all real). `build_county_days.py::add_stateful_features()` now uses the real normal when a county is present in that file, falling back to the old archive-derived proxy (clearly tagged) only for a county missing from it. Values show a sensible north-south gradient (~875mm in northwest MO to ~1275mm in the southeast bootheel), consistent with actual Missouri climatology — a real sanity check, not just "it ran without erroring."

Rerun `python -m risk_fusion.build_precip_normals` if `county_reference.json`'s county list ever changes (new counties, boundary source change) — it's idempotent and safe to rerun.

**c) Explicit non-goals:** no algorithm swap (see rejected item above) unless label volume grows substantially; no change to `labels.py`'s dedup/verification logic (already solid); no blending the GLM into the rule-based `calculate_fire_danger()` category — that stays a future phase gated on enough shadow evidence, not something to force now.

## Sequencing

1. **Retraining orchestrator + experiment log** (Section 1a/1b) — fastest win, immediately makes "did 2 more weeks of data help" answerable, no model-quality risk.
2. **Hyperparameter search** (Section 1c) — still fuel-moisture-only, moderate effort, directly targets "hard to get the model to improve."
3. **Data sync cleanup** (Section 2) — small, independent, can happen in parallel with 1-2.
4. **Risk-fusion algorithm upgrade + advisory exposure** (Section 3a/3b) — the biggest lift, reuses existing validated infrastructure, but needs its own shadow-accumulation runway (weeks) before it's trustworthy even as an advisory signal.
5. **KBDI climate normals** (Section 3c) — can start anytime, independent, but only pays off once (4) is underway.

## Verification

- Retraining orchestrator: run `retrain_fuel_moisture.py` end-to-end locally, confirm it produces a beta registration + an experiment-log JSONL row + a printed beta-vs-stable comparison, and confirm it's idempotent (safe to re-run without manual cleanup).
- Hyperparameter search: confirm the rolling-origin CV picks a configuration with holdout MAE ≤ the current fixed-hyperparameter baseline on the same data; log the chosen params in the experiment-log row for reproducibility.
- Data sync: run `sync_training_data.py` twice in a row and confirm the second run reports "nothing new" rather than re-downloading, and that `manifest.json` reflects the latest synced date range.
- Risk-fusion upgrade: run the boosted model through `evaluate_risk_fusion.py` and confirm it passes the same gates the GLM does (or document exactly which gate it fails and why), then compare its held-out fit quality against the existing GLM baseline before enabling shadow logging.
- Advisory exposure: hit the fire-danger endpoint locally, confirm `risk_ml_score`/`risk_ml_category` appears alongside the existing rule-based `category` without altering the existing field's value or shape (backward-compatible addition).
