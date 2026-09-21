# Model Lifecycle: Data → Train → Publish → Import → Promote

This is the canonical, model-type-agnostic walkthrough of how a ShowMeFire ML model goes from raw data to something live (or live-in-shadow) on the server. It replaces the model-specific fragments that used to live only in `README.md`, `pipelines/training_guide.md`, `docs/model_training_quickstart.md`, `api/models/README.md`, and (partially) `api/docs/beta_operations_runbook.md` — those files now point here for the full picture and keep only what's specific to them (see "What moved where" at the bottom).

If you already know which family you're working on, jump to its row in the **quick reference table**, find its **archetype**, and follow that archetype's walkthrough.

## 1. The five stages

Every model family goes through the same five stages, whether or not every stage is currently reachable for that family:

1. **Capture data** — pull/refresh whatever raw inputs the model trains on (HRRR/RTMA archives, Synoptic observations, fire-occurrence records, etc.), all on the `model-training` side.
2. **Train & evaluate** — fit the model and run its own offline evaluation script, producing a report with named pass/fail/deferred gates.
3. **Register as beta** — validate that report and write the model's assets into `model-training`'s own local registry (`models/config.json`, under `SMF_DATA_ROOT`) as a `beta` candidate. This never touches the live server.
4. **Publish & import** — push the beta candidate to a GitHub pre-release (`publish_release.py`), then pull it down on the server into *the server's own, independent* registry (`api/pipelines/import_model.py`), also landing in `beta` there. The training-side and server-side registries are never shared directly — versions on each side are numbered independently.
5. **Promote (or activate)** — a deliberate, gated step that moves a beta candidate to `stable` (what serving/shadow code actually loads) on the server. Can be reversed with rollback.

```
 model-training repo                                    api repo (server)
┌─────────────┐   ┌──────────┐   ┌──────────────┐        ┌──────────────┐   ┌─────────┐
│ capture data│──▶│  train + │──▶│ register beta│──publish/import──▶│ register beta│──▶│ promote │
│             │   │ evaluate │   │  (local reg.)│  (GitHub release) │ (server reg.)│   │ (stable)│
└─────────────┘   └──────────┘   └──────────────┘        └──────────────┘   └─────────┘
```

### Getting fresh data (before any of the above)

One command pulls everything the `fuel_moisture`/spatial pipelines need (archive zips, Synoptic station observations, RTMA):

```bash
python scripts/sync_training_data.py
```

`--status` reports freshness without pulling anything. `--skip-archives`/`--skip-synoptic`/`--skip-rtma` skip individual sources. By default the RTMA step only looks at HRRR runs from the last 21 days (`--rtma-days-back`) — pass `--rtma-full-history` only for a genuine one-time historical backfill, since scanning the entire cached HRRR history (including years-old risk-fusion panel runs) is slow and rarely what you want for a routine sync.

## 2. Per-family quick reference

| Family | Training entrypoint | Register script | Asset roles | CLI publish/import/promote | Website promote/activate | v1 boundary |
|---|---|---|---|---|---|---|
| `fuel_moisture` | `pipelines/retrain_fuel_moisture.py` | inline in `train_model.py` | single file | ✅ / ✅ / ✅ | ✅ | none — live production |
| `fire_danger` | `fire-danger-model/run_pipeline.py` (standalone, disconnected) | none (not wired to the registry) | single file | ✅ / ✅ / ✅ | ✅ | **dead weight** — no live code loads a registry `fire_danger` model; public forecasts use rule-based `core/fire_danger.py` instead |
| `fuel_moisture_spatial` | `spatial/train_station_sequence.py` + `spatial/run_ablation.py` | `spatial/register_spatial_candidate.py` | ONNX + checkpoint + static bundle + manifest + eval + smoke | ✅ / ✅ / ✅ | ✅ | none — live production |
| `fire_behavior_static` | `static_features/build_bundle.py` | (see spatial operator runbook) | static bundle + manifest | ✅ / ✅ / ✅ | ✅ | none — live production (Testbed spread-rate) |
| `fire_risk_fusion` | `risk_fusion/fit_risk_fusion.py` | `risk_fusion/register_risk_fusion_beta.py` | glm/guard/calibration/effort/county_cells/parity_vector/contract/evaluation | ✅ / ✅ / ✅ | ✅ | **hard**: `advisory_only` must be `True`, permanently, until a deliberate policy change |
| `fire_weather_ml` | `fire_weather_ml/fit_model.py` | `fire_weather_ml/register_beta.py` | model + metadata + risk_calibration + contract | ✅ / ✅ / ✅ | ✅ (via shadow migration) | **hard**: `advisory_only` |
| `fire_weather_index` | `fire_weather_index/calibrate.py` (weights are hand-set, only cutpoints are fit) | `fire_weather_index/register_beta.py` | factor_weights + category_thresholds | ✅ / ✅ / ✅ | ✅ (via shadow migration) | **hard**: `advisory_only` |
| `risk_fusion_glm` | `risk_fusion/fit_glm.py` | uploaded as a zip via the website (no GitHub-release path) | contract + shadow_manifest + climatology + residual + effort | – / – / ✅ | ✅ (via shadow migration) | **hard**: `advisory_only` (asserted, no public-facing path exists) |
| `v4` (`fuel_moisture_station_guarded`) | `spatial/fit_v4.py` | `spatial/register_v4.py` | model (`.pt`) + base_model + lead_guard + contract + calibration | (✅) / – / ✅ | ✅ (via shadow migration) | **hard**: `advisory_only` |
| `v5` (`fuel_moisture_station_summer_guarded`) | `spatial/fit_v5.py` | `spatial/register_v5_beta.py` | model + base_model + guard + uncertainty + contract | (✅) / – / ✅ | ✅ (via shadow migration) | **hard**: `advisory_only` |
| `fuel_moisture_station_hybrid` (V3 hybrid) | `spatial/hybrid_training.py` | `spatial/register_hybrid_station.py` | base_model + model + calibration + contract | – / – / – | – | not wired to promotion anywhere yet |
| `fuel_moisture_station_sequence` | `spatial/train_station_sequence.py` (pre-guard) | `spatial/register_station_candidate.py` | single checkpoint file | – / – / – | – | not wired to promotion anywhere yet |

"CLI publish/import/promote" = whether `pipelines/publish_release.py` (training side), `pipelines/import_model.py`, and `pipelines/promote_model.py` (server side) accept that model type today. A "–" doesn't mean impossible — it means that stage currently has no CLI/UI entrypoint for that family. `(✅)` for `v4`/`v5` publish specifically means "the CLI's argparse will accept it, but it's not practically useful" — see the note in Archetype C below.

`risk_fusion` **Phase A** (the live rule's own Monte Carlo) has no trained artifact at all and is intentionally outside this whole registry — it's reported alongside the others in diagnostics for monitoring, nothing more.

## 3. Walkthrough by archetype

Rather than one section per family (~12x duplication of the same shape), pick the archetype your family matches in the table above.

### Archetype A — single-file model (`fuel_moisture`, `fire_danger`)

```bash
# model-training repo
python pipelines/retrain_fuel_moisture.py        # data prep + train + register as beta, one command
python pipelines/publish_release.py --model fuel_moisture

# api repo (server)
python pipelines/import_model.py --model fuel_moisture --tag fuel_moisture-v<version> --repo <owner>/ShowMeFire-Models
python pipelines/promote_model.py --model fuel_moisture --version <version>
```

`retrain_fuel_moisture.py` runs a rolling-origin hyperparameter search by default (`--no-search` to skip it), prints a beta-vs-current-stable comparison (e.g. `beta mae=2.35 vs stable mae=3.00 (21.7% better)` — don't publish a run that reports **worse**), and appends one row per run to `<SMF_DATA_ROOT>/reports/experiments/fuel_moisture.jsonl` for tracking trend over time. `--skip-ingest`/`--skip-index`/`--skip-snapshots`/`--skip-extract` skip individual data-prep phases; `--extract-days-back N` (default 21) bounds how far back HRRR feature extraction looks — `--extract-full-history` removes that bound for a one-time full reprocess (slow).

`fuel_moisture`'s promotion gate additionally requires 30+ days of shadow evidence and ground-truth shadow accuracy (`scripts/finalize_shadow_validation.py` attaches it) before `promote_model.py` will let it through — see §5.

### Archetype B — multi-asset, promotable (`fuel_moisture_spatial`, `fire_behavior_static`, `fire_risk_fusion`)

```bash
# model-training repo — fuel_moisture_spatial example
python spatial/build_aligned_dataset.py
python spatial/train_station_sequence.py
python spatial/check_spatial_gate.py
python spatial/run_ablation.py --static-bundle data/static/bundles/static_features_2026.1.nc
python spatial/register_spatial_candidate.py --static-bundle <bundle> --sample <sample.npz>
python pipelines/publish_release.py --model fuel_moisture_spatial

# api repo (server)
python pipelines/import_model.py --model fuel_moisture_spatial --tag fuel_moisture_spatial-v<version> --repo <owner>/ShowMeFire-Models
python pipelines/promote_model.py --model fuel_moisture_spatial --version <version>
```

`fire_risk_fusion` follows the same shape (`risk_fusion/fit_risk_fusion.py` → `risk_fusion/register_risk_fusion_beta.py` → `publish_release.py --model fire_risk_fusion`), but **note the v1 boundary**: `validate_promotion_candidate()` hard-blocks promotion unless `metadata["advisory_only"] is True`. Importing and running `promote_model.py --model fire_risk_fusion` will not crash (a real bug this project fixed — see the changelog note in §6), but it will cleanly report `advisory_only` as a blocker unless that boundary is deliberately lifted as a separate policy decision.

**Website equivalent**: on `/admin/models`, pick the family, use the new **Import from GitHub Release** panel (tag + optional repo override + bump) instead of `import_model.py`, then click **Promote** on the resulting beta row instead of running `promote_model.py`.

### Archetype C — multi-asset, advisory-only v1 (`fire_weather_ml`, `fire_weather_index`, `risk_fusion_glm`, `v4`, `v5`)

These five all have a **hard, structural** `advisory_only` gate — nothing here can reach a public-facing serving path in v1, only shadow scoring. Two different mechanisms exist depending on when the family was migrated:

**`fire_weather_ml` / `fire_weather_index`** were built with a training-side publish path and can be imported like Archetype B:

```bash
# model-training repo
python -m fire_weather_ml.fit_model      # or fire_weather_index's own calibrate step
python -m fire_weather_ml.register_beta
python pipelines/publish_release.py --model fire_weather_ml

# api repo (server)
python pipelines/import_model.py --model fire_weather_ml --tag fire_weather_ml-v<version> --repo <owner>/ShowMeFire-Models
python pipelines/promote_model.py --model fire_weather_ml --version <version>
```

**`risk_fusion_glm` / `v4` / `v5`** have no *working* training-side GitHub-release path — they're server-only concepts. `risk_fusion_glm` doesn't correspond to any training-side model_type at all (`publish_release.py --model risk_fusion_glm` is rejected outright). `v4`/`v5` are subtler: their real training-side names (`fuel_moisture_station_guarded`, `fuel_moisture_station_summer_guarded`) *are* accepted by `publish_release.py` (they're real registered model types there), but `api/pipelines/import_model.py` has no matching `--model` choice for either name — so a release published under those names has nowhere to land on the server. Don't use `publish_release.py` for v4/v5 candidates; their bundles arrive as a **zip upload through the website** instead, not `import_model.py`:

1. On `/admin/models`, select the family (it appears under "Guarded Shadow" in the family dropdown).
2. Upload a zip of the candidate directory (must include the family's own required files — see `models/shadow_bundles.py::_asset_filenames_for` for the exact list per family). This **registers a beta candidate** in the unified registry automatically (dual-write — see §6).
3. Click **Activate** on the version you want live in shadow. This runs a real, gated `promote()`/`rollback()` against the registry (not just an ungated pointer flip like it used to be), then syncs `shadow_bundles.py`'s own active-version pointer so existing diagnostics stay consistent.

CLI equivalent for step 3 (once a beta is registered, e.g. via the website upload): `python pipelines/promote_model.py --model risk_fusion_glm` (or `v4`/`v5`) — but prefer the website for these three, since the CLI path doesn't sync `shadow_bundles.py`'s own pointer the way the website's `/activate` endpoint does.

## 4. Publishing & importing

- `pipelines/publish_release.py --model <type> [--version <ver>] [--repo <owner>/repo]`: wraps `gh release create`/`gh release upload`. Requires the `gh` CLI authenticated (`gh auth login`, or reuse the server's `GITHUB_TOKEN` as `GH_TOKEN`). `--model` accepts any registered type — the old hardcoded 3-item allowlist was removed; the actual list of importable/promotable types per stage is the columns in §2's table, not this script's argument parser.
- `api/pipelines/import_model.py --model <type> --tag <release-tag> [--repo ...] [--bump patch|minor|major]`: downloads the release, verifies every declared asset's checksum (plus model-type-specific smoke tests — an ONNX inference check for `fuel_moisture_spatial`, a NetCDF grid/channel check for `fire_behavior_static`), then registers a beta candidate in the server's own registry.
- **Website Import button** (`/admin/models` → pick a registry family → "Import from GitHub Release"): the same operation as `import_model.py`, without CLI/SSH access. Calls `POST /api/admin/models/{family}/import` with `{tag, repo?, bump?}`. The repo is checked against an allowlist (`SMF_ALLOWED_IMPORT_REPOS`, or just `SMF_GITHUB_REPO` if that's unset) — a client can't point it at an arbitrary repo. Only works for the families in `REGISTRY_MODEL_TYPES` (Archetype A/B); the Archetype-C server-only families (`risk_fusion_glm`/`v4`/`v5`) use the zip-upload panel instead, not this one.

## 5. Promoting & rolling back

CLI: `python pipelines/promote_model.py --model <type> [--version <ver>]`, or `--rollback` to reactivate a prior stable version.

Website: `/admin/models` → pick the family → click **Promote** (registry families) or **Activate** (the five migrated shadow families — same underlying `promote()`/`rollback()` call) on a version row; **Rollback to this** on an older stable row.

Every promotion runs `validate_promotion_candidate()` first. Gates vary by family (see `api/models/versioning.py`):

- `fuel_moisture` / `fire_danger`: full metadata contract (`feature_schema_version`, `training_window`, etc.), plus `fuel_moisture` alone additionally needs 30+ days of shadow evidence and ground-truth shadow accuracy.
- `fuel_moisture_spatial` / `fire_behavior_static`: required assets present, static-bundle grid/channel/checksum checks.
- `fire_risk_fusion` / `fire_weather_ml` / `fire_weather_index` / `risk_fusion_glm` / `v4` / `v5`: `advisory_only` must be `True` (hard, structural — see Archetype C), plus family-specific contract checks (rule-spec checksum, precipitation-contract version, feature-module checksum, etc., mirroring what each family's own shadow-scoring module already validates before scoring).

Every gate check and the artifact-copy step are content-addressed and checksum-verified — a corrupted or tampered artifact fails loudly rather than being silently served.

## 6. What changed recently (read before assuming an older doc is still accurate)

- **`publish_release.py`'s model-type allowlist was removed.** It used to hardcode `["fuel_moisture", "fire_danger", "fuel_moisture_spatial"]`; it now accepts any known model type (`models/model_types.py::KNOWN_MODEL_TYPES`).
- **Two real crash bugs were fixed** in both the training-side and server-side `models/versioning.py`: `promote()` and `rollback()` used to assume every candidate had a single top-level `file`, which is `None` for any multi-asset bundle whose roles don't include `model`/`checkpoint`/`static_bundle` (i.e. every Archetype-C family). Attempting to promote `fire_risk_fusion`, `fire_weather_ml`, or `fire_weather_index` used to crash with a `TypeError` before ever reaching the real `advisory_only` gate. Fixed — these now report the real gate blocker instead of crashing.
- **`fire_risk_fusion` promotion is now unblocked** (mechanically) — it's in `promote_model.py`'s choices and the website's `REGISTRY_MODEL_TYPES`. Its `advisory_only` hard gate still applies; lifting that boundary is a separate, deliberate policy decision, not something this fixed.
- **All five previously "guarded shadow" families** (`risk_fusion_glm`, `fire_weather_ml`, `fire_weather_index`, `v4`, `v5`) — which used to live entirely outside the registry, in a parallel fixed-directory/env-var mechanism (`api/models/shadow_bundles.py`) with no gate on activation at all — now **dual-write** into the unified registry on upload, and their website "Activate" button runs a real gated `promote()`/`rollback()` instead of an ungated pointer flip. `shadow_bundles.py` itself is not yet deleted (kept as a compatibility layer, including its own version-history listing and the operator env-var override, which always takes precedence over the registry) — that cleanup is deliberately deferred until a full deploy cycle confirms nothing still depends on the raw env var.
- **A shared training-side registration helper** (`model-training/models/register.py`) replaced ~6 near-duplicated `register_beta.py` scripts (`fire_weather_index`, `fire_weather_ml`, `fire_risk_fusion`, `v4`, `v5`, hybrid-station). Each now defines a small `ModelRegistrationSpec` (asset filenames, its own validate/build-candidate/build-performance functions) and calls the shared `register_beta()` driver — the register/write-back plumbing no longer needs re-implementing per model type. `fuel_moisture`'s inline registration and the two spatial candidates with genuinely bespoke asset resolution (`register_station_candidate.py`, `register_spatial_candidate.py`) were deliberately left as-is.
- **The website Import button is new** (`POST /api/admin/models/{family}/import`) — pulling a GitHub release into the server's beta registry no longer requires CLI/SSH access, for the families in §2's "CLI publish/import/promote" column that show ✅ for import. It's gated on `import_model.IMPORTABLE_MODEL_TYPES`, not `REGISTRY_MODEL_TYPES` — `fire_weather_ml`/`fire_weather_index` are importable even though they're "guarded shadow" families in the dropdown, since they do have a training-side GitHub-release path.
- **A real, pre-existing metadata gap was found and fixed**: `fire_weather_index`/`fire_weather_ml`/`fire_risk_fusion`'s `register_beta.py` scripts computed the exact promotion-gate contract (`advisory_only`, `model_family`, checksums, etc.) their model type needs, but only ever merged it into `performance` (`**build_metadata(...)`) — `models/register.py`'s shared driver had no `build_metadata` field at all, so that contract never reached `register_trained_model`'s `metadata=` argument, `publish_release.py`'s release payload, or the server's `validate_promotion_candidate()` gate. A real import of `fire_weather_index` failed promotion with `missing metadata: advisory_only, model_family` even though that data existed the whole time, just in the wrong field. Fixed: `ModelRegistrationSpec` now has a `build_metadata` field, wired to each script's existing `build_metadata()` function (for `fire_weather_ml`/`fire_risk_fusion`, which already had the right contract computed) or a new minimal one (for `fire_weather_index`, which didn't have one before). Any beta registered before this fix (e.g. `fire_weather_index` betas up through `0.0.1-beta.7`) needs re-registering to pick up real metadata — re-running `register_beta` is sufficient, no data recapture needed.
- **A real gap in the website's Import feature was found and fixed**: importing `fire_weather_index`/`fire_weather_ml` reported success but the version never appeared in the website's version list, and Activate would have 404'd on it. Root cause: `GET /{family}/versions` returned only `shadow_bundles.list_versions()` for every guarded-shadow family, but a GitHub-release import never touches `shadow_bundles` at all — it registers straight into the unified registry, which has no `shadow_bundle_version` metadata for an imported candidate (that field only ever gets set by the zip-upload path). Fixed: `/versions` now merges in registry-only entries for migrated families, and the Activate dispatch (`_registry_action_for_shadow_bundle`) now matches on either the zip-upload identifier (`metadata["shadow_bundle_version"]`) or the registry's own native version string, so both arrival paths (zip upload and GitHub import) work correctly for `fire_weather_index`/`fire_weather_ml` — the two families that support both.
- **`models/model_types.py::KNOWN_MODEL_TYPES` briefly listed `"v4"`/`"v5"`** — a bug caught during a post-implementation audit: those are `api/models/shadow_bundles.py`'s server-only synthetic names, not real training-side model types. The actual names (`fuel_moisture_station_guarded`/`fuel_moisture_station_summer_guarded`, confirmed against `spatial/register_v4.py`/`register_v5_beta.py`'s own `MODEL_TYPE` constants) are what's listed now. §2's table already reflected the correct real names in its `v4 (fuel_moisture_station_guarded)` / `v5 (...)` row labels — only the CLI allowlist itself had the wrong strings.

### Checking in on an already-shadow-scoring model without retraining it

Several families (`risk_fusion_glm`, `fire_weather_ml`, `fire_weather_index`, `v4`, `v5`) aren't retrained on a schedule — they're already scoring every live forecast in the background, advisory-only. Check their diagnostics occasionally rather than retraining reflexively:

```
GET /api/model/spatial/risk-fusion-glm-shadow-diagnostics   # or the equivalent for the other families
```

Look for `enabled: true` and a growing `runs`/`successful_runs` count. `auto_disabled: true` means something is failing repeatedly — check `last_error` in the same response. Retraining `risk_fusion_glm` specifically is a bigger, occasional undertaking (new labeled fire data, refit the GLM, re-run `risk_fusion/evaluate_risk_fusion.py`'s gates) — see that script and `risk_fusion/register_risk_fusion_beta.py`, and don't switch it to XGBoost/GBM without substantially more labeled data first (`fit_glm.py`'s module docstring explains why the GLM choice is deliberate). If the county list changes, refresh KBDI climate normals with `python -m risk_fusion.build_precip_normals` — safe to rerun any time, it just re-pulls real NOAA data per county.

## 7. Troubleshooting

Migrated from `api/docs/beta_operations_runbook.md` (which now covers only day-2 operational monitoring, not the mechanics below):

- **A release download or import fails partway** — `gh release download` requires `gh auth login` (or `GH_TOKEN` set to the server's `GITHUB_TOKEN`). Verify with `gh auth status` on whichever machine is running it.
- **Checksum mismatch on import** — the release's declared `sha256` for an asset doesn't match the downloaded file. Don't force past this; re-publish from the training side rather than hand-editing the release.
- **`promote_model.py` reports "Candidate is not promotable"** — it prints every blocker by name. For the five Archetype-C families this is almost always `advisory_only` (working as intended) or a stale contract/rule-spec checksum (retrain or re-mirror against the current live contract).
- **Website Activate/Promote returns 422** — same gate check as the CLI, surfaced as an HTTP error with the blocker list in the response detail.
- **Website Import returns 403** — the repo you specified (or the `SMF_GITHUB_REPO` default) isn't in `SMF_ALLOWED_IMPORT_REPOS`. This is a deliberate allowlist, not a bug — widen it explicitly if you need to import from a different repo.
- **`auto_rollback` kicked in unexpectedly** — `load_active_model_path(..., auto_rollback=True)` (used by live-serving code for `fuel_moisture`/`fuel_moisture_spatial`/`fire_behavior_static`) automatically reverts to the previous stable version if the current stable artifact is missing or its checksum no longer matches. Check what happened to the file on disk before assuming this is a bug in the rollback logic itself.
- **Shadow evidence / ground-truth shadow requirements for `fuel_moisture`** — on the server side specifically, the full sequence is: capture the current rollback baseline (`python scripts/capture_model_baseline.py`), generate features (`pipelines/generate_training_set.py && pipelines/prepare_features.py`), train a beta candidate (`pipelines/train_model.py`), leave production on stable while shadow records accumulate in `logs/model_shadow.jsonl` for 30+ days with an Elevated-or-higher sample, attach evidence (`scripts/finalize_shadow_validation.py`), then promote (`pipelines/promote_model.py --model fuel_moisture`) or roll back (`--rollback`). The daily validation pipeline also runs a seven-day post-promotion monitor that automatically rolls back material live metric regressions.
- **Observed Rothermel spread rate (`fire_behavior_static`) specifically** needs a post-promotion cache warm: `python -c "from services.rtma_capture import warmup_rtma_cache; print(warmup_rtma_cache(days=7))"`, and depends on 120–168 cached RTMA hours before `/api/testbed/spread-rate/status` reports anything other than `warming`.

## What moved where

- `model-training/README.md`'s training-pipeline walkthrough → summarized into §1–§3 here; the README now keeps only repo setup, data-sync instructions, and the Model Lab pointer.
- `model-training/pipelines/training_guide.md` → fully absorbed into Archetype A above; kept as a redirect stub.
- `model-training/docs/model_training_quickstart.md` → fully absorbed into Archetype A, §1, and §6 (the Fire Weather Risk / GLM check-in section); kept as a redirect stub.
- `api/models/README.md` → absorbed into §5's `fuel_moisture`-specific gate description and §7's troubleshooting entry; kept as a redirect stub.
- `api/docs/beta_operations_runbook.md` → trimmed to day-2 operational monitoring only (the daily verification workflow, evidence file locations, promotion *policy* as opposed to promotion *mechanics*); the mechanical publish/import/promote steps it used to duplicate now live here.
- `model-training/fire-danger-model/README.md` → **not absorbed** — that pipeline is fully disconnected from this registry (see §2's row for `fire_danger` vs. the standalone `fire-danger-model/` experiment, which are two different things sharing a confusingly similar name). Read it directly if you're working on that experiment specifically.
