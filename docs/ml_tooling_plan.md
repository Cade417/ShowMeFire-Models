# ML Tooling: Drift Monitoring, Calibrated Uncertainty, Live Explainability

## Context

ShowMeFire's ML stack (fuel-moisture XGBoost, ONNX spatial model, V4/V5 station models, risk-fusion GLM) already has a rigorous shadow-registry/promotion safety net (`api/models/versioning.py`, `promote()`, checksum-gated bundles, write-once prediction/observation logs). But three real gaps sit on top of that solid foundation:

1. **No drift monitoring** — the shadow pipeline checks accuracy at day14/day30 checkpoints, but nothing watches for feature or prediction distribution shift *between* checkpoints, so a silently-degrading data source (e.g. a station going stale) wouldn't be caught until a scheduled evaluation.
2. **Uncertainty is inconsistent** — only the V5 station model has calibrated prediction intervals (`v5_guard.py::fit_uncertainty`/`intervals`). The fuel-moisture XGBoost model (the one actually served to users via `DailyForecast.py`) and the risk-fusion GLM have no confidence bands at all.
3. **`/api/models/formulas` is hard-coded text** (`api/main.py:1361`) that can silently drift from what the live model actually does — there's no live, model-derived explainability anywhere, even though the diagnostics-router pattern (`api/routers/spatial_model.py`) already exists to hang a new endpoint off of.

All three reuse existing conventions (shadow bundle format, `services/<name>.py` + `diagnostics()` pattern, APScheduler job registration) rather than introducing new architecture. Goal: land all three without touching the production forecast path itself — they are additive, read-only-with-respect-to-predictions layers.

## 1. Drift Monitoring

**What it watches:** feature-distribution drift (input weather/RTMA features) and prediction-distribution drift, computed against a rolling reference window, for each model family that already has shadow evidence: fuel-moisture (`model_shadow.py` JSONL), V4/V5 (`v4_shadow.py`/`v5_shadow.py` prediction/observation JSON pairs), risk-fusion GLM shadow.

**New module:** `model-training/monitoring/drift.py`
- `compute_psi(reference: np.ndarray, current: np.ndarray, bins=10) -> float` — Population Stability Index per feature/prediction column. PSI is cheap, well-understood, and doesn't require a reference model.
- `evaluate_drift(model_type: str, evidence_root: Path, reference_window_days=30, current_window_days=7) -> dict` — loads shadow evidence via the *existing* readers (`evaluate_v5_shadow.py::load_evidence` for V5; append-only JSONL read for `model_shadow.py`; risk-fusion GLM shadow state), splits into reference vs. current windows by timestamp, computes PSI per feature + per prediction, and returns a report dict: `{model_type, generated_at, features: {name: psi}, prediction_psi, flags: [names where psi > threshold]}`. Thresholds: PSI < 0.1 stable, 0.1–0.25 moderate, > 0.25 flagged (standard industry cutoffs) — configurable via `DRIFT_PSI_WARN`/`DRIFT_PSI_ALERT` env vars.
- Writes each report as an immutable, timestamped JSON file under `evidence_root/drift/{model_type}_{run_id}.json` (same `open(path, "x")` write-once convention as `v5_shadow.py::record_predictions`) — a drift history becomes an append-only audit trail, consistent with the existing "predictions are immutable" invariant tested in `test_v5_safety.py`.

**Serving/surfacing:** `api/services/drift_monitor.py`
- Thin wrapper mirroring `services/model_shadow.py`'s `_state` + `diagnostics()` pattern: `_state = {}`, `def diagnostics(): return dict(_state)`, updated whenever `run_drift_check()` executes.
- `run_drift_check()` calls `drift.evaluate_drift(...)` for each shadow-tracked model type, updates `_state`, and — following the existing failure-isolated convention in `api/core/scheduler.py` — logs but never raises on failure.
- New route in `api/routers/spatial_model.py`: `GET /api/model/spatial/drift-diagnostics` → `services.drift_monitor.diagnostics()`. Same thin-passthrough shape as the five existing diagnostics routes.

**Scheduling:** register in `api/core/scheduler.py::start_scheduler_jobs`, alongside `verify_v5_shadow_observations`:
```python
scheduler.add_job(run_drift_check_job, 'cron', hour=4, minute=0, id='drift_check', max_instances=1, coalesce=True)
```
(after the V5 verification job at hour=3 and cache purge at hour=3:15, so drift check runs on freshly-verified evidence). Wrapped in `asyncio.to_thread(...)` + try/except-log, matching existing jobs.

**Alerting hook (minimal):** when `flags` is non-empty, log at `WARNING` level with model_type + feature name — no new notification channel in this pass; the existing admin log/report file browser already surfaces `WARNING`-level logs, so this is visible without new infrastructure. (Wiring into Discord/mobile push is a natural fast-follow, not in scope here.)

**Tests:** `api/tests/test_drift_monitor.py` mirroring `test_v5_safety.py` conventions — assert PSI computation correctness on synthetic distributions (identical dist → ~0, shifted dist → high), assert report files are write-once (`"x"` mode, second write for same run_id fails), assert circuit-breaker-style isolation (a failure in one model type's drift check doesn't block others).

## 2. Calibrated Uncertainty

Reuse the exact `v5_guard.py` pattern (`fit_uncertainty` → empirical quantile widths by regime, serialized to checksummed `uncertainty.json`, looked up at serve time) for the two models that currently lack it.

**a) Fuel-moisture XGBoost** (`model-training/pipelines/train_model.py`)
- Add `fit_uncertainty(residuals, regime_column, target=0.8, minimum_rows=300)` in a new `model-training/pipelines/fm_uncertainty.py` — same shape as `v5_guard.py::fit_uncertainty`: global + per-regime empirical error quantiles (regime = month or temp/RH bucket, reusing whatever bucketing V5 already uses for consistency).
- Persist as `uncertainty.json` alongside the trained model artifact, checksummed the same way other bundle assets are (`sha256` field in `models/config.json` metadata), so it flows through the *existing* `validate_promotion_candidate` checksum gate for free — no new gating code needed.
- Serving: in `api/forecast/DailyForecast.py::predict_fm_grid`, after `preds = FM_MODEL.predict(dmat)`, load the sibling `uncertainty.json` (via `load_active_model_path`'s directory) and apply the same `intervals()`-style lookup used in `v5_runtime.py` to produce `preds_lo`, `preds_hi` alongside the existing `preds_2d` grid. Add `fuel_moisture_p10`/`fuel_moisture_p90` (or similar) to the forecast response.
- **Important constraint discovered:** `train_model.py` currently registers via the *unmetadata'd* `model-training/models/versioning.py`, which doesn't satisfy `api/models/versioning.py`'s `REQUIRED_BETA_METADATA`. This uncertainty work is a natural point to also close that gap — pass `metadata={feature_schema_version, rule_spec_version, training_window, ...}` at registration time so the fuel-moisture beta path actually promotes cleanly under the stricter gate. Flag this explicitly rather than silently leaving it inconsistent.

**b) Risk-fusion GLM** (`model-training/risk_fusion/fit_glm.py`)
- `fit_regularized` doesn't give a covariance matrix (already noted as a design constraint in the file), so use the same empirical-residual approach rather than analytic Poisson CIs: after fitting, compute residuals on held-out folds, bucket by month/regime (reusing `MONTH_DUMMY_COLUMNS` buckets already in the file), and fit empirical quantiles the same way as (a).
- Since risk-fusion is `advisory_only` and currently shadow-only (no live prediction endpoint — only `risk_fusion_glm_shadow.py::diagnostics()`), surface the interval directly in the existing shadow diagnostics dict rather than adding a new public endpoint. This keeps the "advisory, not served" boundary intact.

**Tests:** extend `test_model_contracts.py`/add `test_fm_uncertainty.py` verifying: quantile widths are non-negative, coverage on a held-out set is close to the target (e.g. 80% target → actual coverage within a tolerance band on synthetic data), and that missing/insufficient-support regimes correctly fall back to the `"global"` width (mirrors `v5_guard.py` behavior).

## 3. Live-Derived Explainability

Replace the static text in `/api/models/formulas` with values actually derived from the live model, and add a per-prediction explanation surface.

**a) Fix the static/live drift itself**
- New `api/services/fm_explain.py`:
  - `global_importance() -> dict` — `FM_MODEL.get_score(importance_type="gain")` (native XGBoost Booster method, zero new dependencies) mapped to human-readable feature names, cached at import time next to `FM_MODEL` in `DailyForecast.py`.
  - `explain_prediction(feature_row: dict) -> dict` — per-prediction contribution breakdown. Use `xgb.Booster.predict(..., pred_contribs=True)` (SHAP-values-via-XGBoost's built-in TreeSHAP, **no new `shap` dependency needed** — this is a genuine simplification over adding the `shap` package) to get per-feature contribution for a single grid cell / station query.
- Update `GET /api/models/formulas` (`api/main.py:1361`) to merge the existing static prose (thresholds, category definitions — those genuinely are policy, not model-derived) with a new `"feature_importance"` field populated from `fm_explain.global_importance()`, and bump `"last_updated"`/`"version"` to reflect it's now partially live. This directly closes the drift risk instead of leaving the static doc in place.

**b) New per-prediction explainability endpoint**
- `api/routers/spatial_model.py`: new route `GET /api/model/spatial/fm-explain?lat=&lon=&...` (or reuse whatever query params the existing forecast endpoint takes) → builds the same feature row `predict_fm_grid` would, calls `fm_explain.explain_prediction(...)`, returns `{prediction, base_value, contributions: {feature: value}}`. Same thin-router-calls-service-module pattern as every other route in this file.
- This is the concrete building block for the "why is this rated Extreme" UI panel discussed earlier — out of scope for this plan (frontend), but this endpoint is what it would call.

**Dependencies:** none new — `get_score`/`pred_contribs=True` are native to the `xgboost==3.1.2` already pinned in both `api/requirements.txt` and `model-training/requirements.txt`. (Explicitly skip adding `shap`, since XGBoost's built-in contribution output covers the need without a new dependency.)

**Tests:** `api/tests/test_fm_explain.py` — assert `global_importance()` returns all `FEATURES` with non-negative scores; assert `explain_prediction()` contributions sum to `prediction - base_value` within floating-point tolerance (a property of TreeSHAP/`pred_contribs`, cheap to verify and catches wiring bugs).

## Sequencing

1. **Uncertainty first** (fuel-moisture, then GLM) — most self-contained, directly reuses `v5_guard.py` code shape, and the metadata-gap fix it forces is worth doing early.
2. **Explainability second** — small, no new dependencies, immediately fixes the static-doc drift risk.
3. **Drift monitoring third** — depends on nothing from 1/2, but benefits from being last since it's the most "new infrastructure" of the three (new scheduler job, new evidence directory) and should land on a stable base.

## Verification

- Unit tests per component as listed above, run via the existing `api/tests/` suite (`pytest api/tests/test_fm_uncertainty.py test_fm_explain.py test_drift_monitor.py`) and any new tests under `model-training/` following that project's existing test conventions.
- For uncertainty: manually trigger `train_fuel_moisture_model(channel="beta")`, confirm `uncertainty.json` is written and checksummed in `models/config.json`, then call the forecast endpoint locally and confirm `fuel_moisture_p10/p90` appear in the response.
- For explainability: hit `GET /api/models/formulas` locally and confirm `feature_importance` is populated and non-static across two different loaded model versions (swap beta/stable to confirm it actually changes).
- For drift: seed synthetic reference/current evidence files with a known shift, run `run_drift_check()` manually, confirm the PSI report flags the shifted feature and that the diagnostics endpoint (`/api/model/spatial/drift-diagnostics`) reflects it. Confirm the scheduled job appears in `api/core/scheduler.py` logs at startup without raising.
