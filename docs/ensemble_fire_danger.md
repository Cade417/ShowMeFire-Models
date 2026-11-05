# Ensemble Fire Danger (Day 1, BETA)

> **Retired sources (2026-10).** NOAA retired NAM, the NAM 3 km nest, SREF, HREF, HiResW and NAM MOS together. They are no longer members: the members track is now **HRRR + RRFS** (5 members with the time-lagged runs, `min_members` 4) and the ensprod track runs on **REFS** alone. The live run also drops those source keys from any registered bundle's `member_config.json` (`runtime.RETIRED_SOURCES`, env `SMF_ENSEMBLE_FD_RETIRED_SOURCES`), so a bundle fitted before the change still works until a retrained one is promoted. The calibration was fitted with the retired members, so run `retrain --rescore` and promote. Sections below that mention NAM, HiResW or HREF describe the design as built and the findings made while they existed.

Chance of **Moderate / Elevated (High) / Critical (Very High) / Extreme or greater** fire danger, plus a **categorical ensemble fire danger forecast**, from short-range convection-allowing guidance. The product is five graphics in the public forecast house style, beta-labelled and uploaded to the public CDN. It sits beside the operational rule-based map and does not replace it.

- Model type: `ensemble_fire_danger` (advisory-only, Archetype C in [model_lifecycle.md](model_lifecycle.md))
- Runtime: `api/services/ensemble_fire_danger/`, entrypoint `api/scripts/run_ensemble_fire_danger.py`
- Training: `model-training/ensemble_fire_danger/`
- Retraining steps: [ensemble_fire_danger_retraining_runbook.md](ensemble_fire_danger_retraining_runbook.md)

---

## 1. Why there are two tracks

The obvious design is "take the HREF / REFS members and run each through the fire danger rule". That is not possible with public data. This was confirmed live on 2026-09-22:

| Product | Where | What is actually published |
|---|---|---|
| **HREF** | NOMADS `com/href/prod/href.YYYYMMDD/ensprod/` (rolling ~2 days, no archive) | Ensemble products only: `mean`, `sprd`, `prob`, `avrg`, `pmmn`, `lpmm`, `eas`, `ffri`. 00z/12z, f01-f48. `prob` is "0/10" members. |
| **REFS** | `noaa-rrfs-ops-pds/refs.YYYYMMDD/HH/ensprod/` (since 2026-08-12) | Ensemble products only, same families. 00/06/12/18z. `prob` is "0/14" members. |
| Per-member surface grids | nowhere | Not published for either system. |

The mean/spread files do contain 2 m TMP, 2 m DPT and 10 m WIND. The `prob` files have only fixed thresholds: wind > 10.3 m/s, dewpoint thresholds, and `JFWPRB`, NCEP's own joint RH-and-wind fire-weather probability.

None of these include 10-hr fuel moisture, which the Show Me Fire rule requires. None apply the Show Me Fire thresholds.

So the product is built two ways and compared (your decision, 2026-09-22):

- **Track M, "members" (primary by default).** An HREF-style ensemble built from real deterministic runs, which are the same models HREF itself is made of, plus time-lagged runs. Each member goes through the operational fuel-moisture model and the operational rule, so each member has a genuine fire danger category.
- **Track E, "ensprod".** Synthetic members drawn from the REFS mean and spread (HREF was pooled in until it was retired). It has more members (40 draws) and uses NCEP's own ensemble statistics, but the draws are only as good as the Gaussian assumption.

The training pipeline evaluates both on the same days. It reports a `recommended_primary`, which is advice only; switching tracks is an explicit env change (§8).

## 2. Members (Track M)

The full definition, including URLs, cycles, maximum leads, wind factor and publish latency, lives in [`member_config.json`](../ensemble_fire_danger/member_config.json). It is a contract mirror, byte-identical in the api and training repos.

| id | model | cycle (target) | leads for the 16Z–03Z window | live source | archive |
|---|---|---|---|---|---|
| hrrr_0 (control) | HRRR 3 km | 12z | f04–f15 | AWS | AWS, years |
| hrrr_m6 | HRRR | 06z | f10–f21 | AWS | AWS |
| hrrr_m12 | HRRR | 00z | f16–f27 | AWS | AWS |
| rrfs_0 / rrfs_m6 | RRFS 13 km NA | 12z / 06z | same | `noaa-rrfs-ops-pds` | same, from 2026-08-12 |
| ~~arw_0, fv3_0, arw2_0~~ | HiResW (retired 2026-10) | | | | |

**Cycle resolution** (`members.resolve_plan`, identical live and in training):

1. Each member's target is anchor (12z) + `cycle_offset_hours`.
2. If that cycle isn't published yet (live: the `.idx` of the last needed lead is missing), step back one source cycle at a time, up to `max_cycle_age_hours` (30).
3. A cycle already claimed by another member of the same model is skipped. So when the 12z RRFS is late, rrfs_0 takes 06z and rrfs_m6 moves to 00z, and the two members never duplicate.
4. Every decision is recorded in the run's evidence file.

**Measured publish times** for the 12z f15, from real `Last-Modified` headers on 2026-09-22:

| Source | Published at (UTC) |
|---|---|
| HRRR | 13:10 |
| RRFS | 14:26 |
| REFS | 14:35 |

The ensemble step runs about 14:45–15:00Z, so HREF normally falls back to its 00z run. (The NAM 3 km nest was a member until 2026-10; it was removed ahead of NOAA retiring NAM. Its AWS copy was ~12 h late, which is why the config still supports a per-source `archive_url`.)

**Decoding details** that real files forced (`grib_idx.py`):

- **Packed U/V.** NAM and ARW-mem2 pack U+V into one multi-field GRIB message, with idx lines `624.1`/`624.2` at the same byte offset.
- **Precipitation windows** differ by model:
  - HRRR/RRFS/HiResW: 1 h bucket plus a run total.
  - NAM: 3 h resetting buckets.
  - ensprod: 1/3/6 h windows.

  `hourly_precip()` converts any of these to per-hour intervals.
- **HiResW 00z** 2.5 km runs lack instantaneous 2 m TMP at some leads. `t2m = (TMAX+TMIN)/2` is used and the run is tagged `t2m_is_tmax_tmin_proxy`, as `spatial/fv3hires_capture.py` already does.
- **eccodes on Windows** keeps any file it reads locked until the process exits, and segfaults on concurrent first use from threads. So:
  - ordinary messages are decoded from memory;
  - only multi-field messages use a temp file, and stale ones are swept;
  - all decoding is serialized behind one lock, while downloads stay parallel.

## 3. From member to category

Everything runs on one target grid: DailyForecast's HRRR `mo_bounds` subset, shape (196, 205). It is the same grid as the public map and `core/risk_fusion_reference/county_cells.json`. Other grids are regridded by barycentric (linear) interpolation. The Delaunay weights are cached per source grid in `cache/ensemble/regrid/`, so each grid is triangulated once rather than every run.

For each member, each hour from 16Z to 03Z (`tracks.py`):

1. **Wind for the rule** = |V10| × 0.8 (`HRRR_FIRE_WIND_REDUCTION_FACTOR`), in knots. Per-source `wind_factor` is in the config and is tunable later. **Wind for the FM model** is raw |V10| m/s, as in DailyForecast.
2. **Fuel moisture** is anchored to production:
   `FM_member = FM_operational + (FM_xgb(member weather) − FM_xgb(control weather))`, clipped to 1–40 %.
   - `FM_operational` is DailyForecast's own hourly FM: RAWS-initialized, or the spatial ONNX field when that is active. DailyForecast hands it off through `forecast_state.write()` (`data/forecast-state/forecast_state_YYYYMMDD_HHz.npz`).
   - The control is hrrr_0 for Track M and the ensemble-mean weather for Track E.
   - Each member therefore keeps the operational fuel state and only moves it by how its own weather would dry or wet fuels differently.
   - The XGBoost features are a byte-mirrored copy of `predict_fm_grid`'s, in `fm_features.py`. A parity test checks them against `predict_fm_grid`. The copy never calls the fuel-moisture shadow hook.
   - If the hand-off file is missing, members use raw XGBoost FM and evidence records `"fm_anchor": "none"`.
3. **Category**, in the same order as DailyForecast: the canonical rule (`rule_uncertainty.category_vectorized`, parity-checked against `calculate_fire_danger` at start-up), then seasonal green-up dampening (`core.seasonal_dampening_vectorized`, parity-tested cell for cell against `core.fire_danger.seasonal_dampening_adjustment`), then the snow mask forcing Low.
4. **Peak** = the maximum over the window, like the public map.

**Track E synthetic draws** (`core.synthetic_draws`):

- The REFS and HREF mean and spread are pooled as a mixture: variance = within-ensemble + between-ensemble.
- 40 draws, seed 20260922. Each draw uses one spatially coherent N(0,1) field per variable (Gaussian-smoothed white noise, σ = 3 cells, normalized by the analytic 1/(2√π σ)), held fixed through the day.
- The T/Td error correlation is −0.3. Td is capped at T, and wind is clipped at 0.
- A test caught an earlier per-draw standardization that under-dispersed the draws by 25%.

## 4. Probabilities and the categorical forecast

`core.probability_products`, per track:

- **Point probability** P(≥k) = weighted fraction of members whose peak ≥ k at the cell. A member missing at a cell drops out of that cell's denominator.
- **Neighborhood probability** (what the four maps show) = weighted fraction of members reaching ≥ k anywhere within **20 km** of the cell. This is the HREF/SPC "neighborhood maximum ensemble probability". A ~10-member 3 km ensemble has little skill at placing a crossing on an exact cell, but real skill at "near here". The maps say "within ~20 km (12 mi) of a point".
- Both are smoothed with σ = 1.5 cells (the public map's smoothing). Each then goes through the bundle's **isotonic calibration**: identity when uncalibrated, which the graphic states. Finally it is **forced monotone** across k, so P(≥Moderate) ≥ P(≥Elevated) ≥ ….
- **Categorical** = the highest k whose calibrated **point** P(≥k) ≥ τ_k. With τ_k = 0.5 (the default, before training) this is exactly the weighted ensemble median category. Training fits τ_k to maximize CSI within frequency bias 0.8–1.25 (§6).

## 5. Graphics

The five images per track come from `render.py`. They use the **same helpers as the public map**: `forecast/map_style.py`, moved verbatim out of DailyForecast.py, which re-imports them. That gives the same 2048×1152 at 144 dpi, Lambert projection, extent, county and state outlines, fonts, logo, title block and left colorbar.

| File | Content |
|---|---|
| `mo-forecast-ens-firedanger.png` | Categorical, exact production bins, colors and "Elevated/High", "Critical/Very High" labels |
| `mo-forecast-ens-prob-moderate.png` | P(≥ Moderate) within 20 km |
| `mo-forecast-ens-prob-elevated.png` | P(≥ Elevated / High) |
| `mo-forecast-ens-prob-critical.png` | P(≥ Critical / Very High) |
| `mo-forecast-ens-prob-extreme.png` | P(≥ Extreme) |

Each file also has a `.webp` copy. `FORECAST_IMAGE_SUFFIX` is honoured.

**Probability styling:**

- **9 filled bands** (10–20 … 90–100 %) in one hue per category, the category's own family: gold, orange, red, magenta-maroon.
- They are stepped in OKLCH and validated with the dataviz `validate_palette.js --ordinal` against the `#E8E8E8` background. All four pass: monotone lightness, ΔL ≥ 0.06 between steps, light end ≥ 2:1 contrast, single hue.
- A 10th filled 5–10 % band cannot pass: there isn't enough lightness range between a light end that reads on grey and black. So **5 % is a dashed outline** ("marginal signal"), as SPC-style graphics do.
- Every contour is labelled, so the value never depends on color alone. Fills are opaque, so the validated colors are what renders.
- If nothing reaches 5 %, a "Less than 5% chance statewide" box is shown.

## 6. Calibration and training

| Step | Module | Notes |
|---|---|---|
| Capture | `capture.py`, `scripts/capture_ensemble_members.py` | Uses the same `resolve_plan`, with `use_archive=True` (each source's `archive_url`, where it sets one) and `emulate_live_at = anchor + 3 h` plus per-source `latency_hours`. Without emulation, training would use a fresher member mix (12z HREF/HiResW) than live ever gets, and would be over-confident. |
| Labels | `labels.py` | "Observed" = the same rule on hourly **RTMA** (16Z–03Z), maxed. The FM comes from the same XGBoost model driven by RTMA T/RH/wind; precip is the HRRR control's 1 h forecast, because RTMA has no precip. Wind uses the same 0.8 factor. |
| Panel | `build_panel.py` | Per day, the same `tracks.py` code the live product runs, cached as `days/<day>.npz`. Then a long-format parquet over Missouri cells thinned every 4th row and column: raw point/neighborhood P, point/neighborhood observation, deterministic HRRR-control outcome, member count, JFWPRB. |
| Split | `contract.py` | Whole ISO weeks, every 4th week held out as test. Existing weeks never change sides as days are appended. |
| Fit | `fit.py` | Per track × flavour × k: isotonic regression (≤ 60 knots). Identity if < 20 events or < 20 non-events. τ_k chosen by max CSI with bias in [0.8, 1.25]. |
| Evaluate | `evaluate.py` | Brier raw / calibrated / deterministic, BSS vs climatology and vs deterministic, ECE plus reliability table, ROC AUC, categorical CSI and bias. Gates are listed below, and the report includes a Track M vs E comparison. |
| Register | `register_beta.py` | Refused if any gate is `fail`. Metadata: `model_family`, `advisory_only=True`, `core_module_sha256`, `recommended_primary`. |
| One command | `retrain.py` | Runs all of the above, prints the result against the previous beta, and logs to `reports/experiments/ensemble_fire_danger.jsonl`. |

**Gates** (per track):

- **`min_days`**: ≥ 10 test days, else deferred.
- **`events_k`**: ≥ 20 test neighborhood events, else deferred. Extreme will be deferred for a long time, which is expected.
- **`calibration_not_worse_k`**: calibrated Brier ≤ raw × 1.02. Evaluated only when events suffice.
- **`skill_vs_deterministic_k`**: BSS vs the HRRR-control rule > 0, for Moderate and Elevated.
- **`member_coverage`**: mean members/day ≥ 5 for Track M, or ≥ 1 source for Track E.

Deferred gates don't block a beta. A `fail` does.

**Known training-vs-live differences** (all deliberate, all small):

- **No FM anchoring historically.** The operational FM hand-off didn't exist, so training members *and* labels both use raw XGBoost FM.
- **No seasonal dampening in training**, because per-day GDD state isn't archived. It only demotes marginal cells in spring, identically for forecast and label.
- **No snow mask in training.**
- **Training days before 2026-10 include NAM/HiResW members that no longer exist.** Rescoring with the current config uses only HRRR + RRFS, so the panel and the live product now see the same member mix. HiResW and HREF were forward-only, so nothing more can be captured for them and the captured history stays as it is.

## 7. Outputs of one live run

| What | Where |
|---|---|
| Primary-track public images | `api/images/mo-forecast-ens-*.png` / `.webp` → R2 `yyyy/mm/dd/forecast/…` and `latest/…` via `scripts/upload_cdn.py::upload_forecast_files`. Gated by `uploadForecast` **and** `SMF_ENSEMBLE_FD_UPLOAD`. PNG only (the uploader labels every file `image/png`). |
| Both tracks, side by side | Testbed `BETA_ROOT/images/mo-forecast-ens-*-{members,ensprod}.png`. The manifest product key is `ensemble_fire_danger` (beta, not_for_operations). |
| Status | `status.json` key `ForecastEnsembleFireDanger` (beta: true). |
| Evidence | `data/model-shadow/ensemble-fire-danger/<run>.ensemble_fire_danger.json` holds the plan with the actual cycle per member, per-track summaries, county maxima, bundle, FM anchor and rule info. A matching `.npz` holds member peaks and raw probabilities, for future recalibration from live runs. |
| Diagnostics | `GET /api/model/spatial/ensemble-fire-danger-diagnostics` |

## 8. Operations

**Environment variables** (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `SMF_ENSEMBLE_FD_ENABLED` | true | false = the step does nothing |
| `SMF_ENSEMBLE_FD_PRIMARY` | members | Which track feeds the public images (`members` or `ensprod`) |
| `SMF_ENSEMBLE_FD_UPLOAD` | true | false = never upload (also off whenever `uploadForecast=false`) |
| `SMF_ENSEMBLE_FD_BUNDLE` | – | Raw bundle directory; overrides the registry (stable, then beta) |
| `SMF_ENSEMBLE_FM_MODEL` | registry stable `fuel_moisture` | XGBoost FM model file |
| `SMF_ENSEMBLE_FD_CACHE` | `/app/cache/ensemble` | Member-run and regrid cache |
| `SMF_ENSEMBLE_FD_CACHE_DAYS` | 10 | Days of member-run NetCDFs kept (~180 MB/day); older ones are pruned after each run |
| `SMF_FORECAST_STATE_DIR` | `data/forecast-state` | DailyForecast → ensemble FM hand-off |
| `SMF_ENSEMBLE_FD_IMAGES_DIR`, `SMF_ENSEMBLE_FD_STATUS_FILE`, `SMF_ENSEMBLE_FD_EVIDENCE_ROOT` | api defaults | Output redirects, for local testing |

**Scheduling and failure behaviour:**

- `scripts/forecasts.sh` runs `run_ensemble_fire_danger.py` right after DailyForecast, with `|| echo WARNING`. A failure never blocks or changes the public forecast.
- A missing optional member is just dropped. Under 5 members the run is marked **DEGRADED** on the graphic; under 3, Track M is skipped.
- If both tracks fail, the run exits 1 and records `last_error` in `runner-state.json`.
- A typical run takes about 7 minutes: fetching dominates, about 2 minutes per member with 6 in parallel.

**HREF retirement (done).** HREF was retired together with NAM, SREF and HiResW. The `href` ensprod member and the HiResW members are removed from `member_config.json`; Track E runs on REFS alone.

## 9. Known limitations

- **No ensemble fuel-moisture information exists.** Members only perturb FM through their weather (§3.2).
- **Time-lagged members are older forecasts**, weighted equally for now. `member_config.json` weights can be tuned once enough history exists; `fit.py` is the place to add weight fitting.
- **Short archives.** RRFS and REFS start on 2026-08-12, and HiResW/HREF are forward-only. Calibration will stay mostly identity or deferred for the higher categories until real Elevated+ days accumulate. The graphics say so ("uncalibrated raw member fraction").
- **Labels are analysed-weather categories, not fire occurrence.** That is the right target for a fire *weather* danger forecast, but it inherits RTMA's own errors, and the FM part is modelled, not observed.
