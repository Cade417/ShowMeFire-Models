# Model Training Quickstart

A short, practical guide to the day-to-day workflow: getting fresh data, retraining the fuel-moisture model, and checking in on the fire-weather risk model. For the full technical detail behind any of this, see the links at the bottom.

## What's in here

| Model | What it does | Status |
|---|---|---|
| **Fuel moisture (XGBoost)** | The production model — feeds the live forecast | You retrain this regularly |
| **Fire Weather Risk (GLM)** | Advisory "how likely is a fire today" signal, scored in shadow alongside every forecast | Already running in shadow — you mostly just check on it |

---

## 1. Get fresh data

One command pulls everything (archive zips, Synoptic station obs, RTMA):

```bash
cd model-training
python scripts/sync_training_data.py
```

Just want to know how fresh your data is, without pulling anything?

```bash
python scripts/sync_training_data.py --status
```

By default, RTMA backfill only looks at HRRR runs from the **last 21 days** (`--rtma-days-back`). Without that cutoff it would rescan your entire cached HRRR history — including years-old runs left over from building the risk-fusion historical panel — which is slow and pointless for a routine sync. Only pass `--rtma-full-history` if you genuinely need a one-time historical backfill.

Only need one source? `--skip-archives`, `--skip-synoptic`, `--skip-rtma` skip the others.

> Heads up: the RTMA step reads every cached HRRR file to figure out what's missing, so it can take a couple of minutes if your cache is large. That's normal, not a hang.

---

## 2. Retrain the fuel-moisture model

Do this roughly every two weeks, after syncing fresh data:

```bash
python pipelines/retrain_fuel_moisture.py
```

This single command:
- Runs all the data-prep steps (ingest → index → snapshots → HRRR extract → feature prep) in one process
- Searches a small grid of hyperparameters and picks the best one (skip with `--no-search` if you just want a quick run)
- Registers the result as a new **beta** version — it never overwrites what's currently live
- Prints how the new beta compares to the current stable model, so you can tell at a glance if it actually improved

Example output to look for at the end:

```
✅ Retrain pipeline completed in 0.3 min
   Registered beta version 0.0.1-beta.5
   beta mae=2.3500 vs stable mae=3.0000 (21.7% better)
```

If it says **worse**, don't publish it — something in the new data or search run made things regress; worth a second look before promoting.

**Useful flags:**
| Flag | When to use it |
|---|---|
| `--skip-ingest` | Data already ingested this run |
| `--skip-index`, `--skip-snapshots` | Skip a step you've already done |
| `--skip-extract` | Skip HRRR feature extraction entirely |
| `--full-retrain` | Force a complete reprocess of all cached HRRR data (rare — slow) |
| `--extract-days-back N` | How far back Phase 4 looks for unprocessed snapshots (default: 21 days) |
| `--extract-full-history` | Ignore the days-back cutoff and mine every unprocessed snapshot, however old |
| `--no-search` | Skip hyperparameter search for a faster, "just retrain" run |

By default, Phase 4 (HRRR feature extraction) only looks at snapshots from the **last 21 days** — same reasoning as the RTMA sync cutoff above. Without it, a snapshots table that's accumulated years of never-mined rows (e.g. from building the risk-fusion historical panel) gets fully rescanned every run, even though old snapshots with no matching observation don't add any training rows anyway — just wasted time (this is exactly what turned one real run into 44.8 minutes instead of a couple of minutes). Only reach for `--extract-full-history` when you actually mean to backfill the whole history again.

Every run is also logged to `<SMF_DATA_ROOT>/reports/experiments/fuel_moisture.jsonl` — one line per run, so you can see the trend over time, not just this run vs. last.

---

## 3. Publish and promote

Retraining only ever creates a beta. Getting it live is a separate, deliberate step — on purpose, so a bad retrain can't accidentally reach production:

```bash
python pipelines/publish_release.py --model fuel_moisture
```

Then on the server:

```bash
python pipelines/import_model.py --model fuel_moisture --tag fuel_moisture-v<version>
python pipelines/promote_model.py --model fuel_moisture --version <version>
```

(Full detail, including the GitHub pre-release step in between: [`README.md`](../README.md#getting-a-model-back-into-production).)

---

## 4. Check on the Fire Weather Risk (GLM) model

This one isn't something you retrain every two weeks — it's already scoring every live forecast in the background (shadow mode, advisory-only, never affects what users see). Check its diagnostics occasionally:

```
GET /api/model/spatial/risk-fusion-glm-shadow-diagnostics
```

Look for `enabled: true` and a growing `runs`/`successful_runs` count. If `auto_disabled: true` shows up, something's failing repeatedly — check `last_error` in the same response.

**Retraining this one** is a bigger, occasional undertaking (new labeled fire data, refit the GLM, re-run the evaluation gates) — see [`risk_fusion/evaluate_risk_fusion.py`](../risk_fusion/evaluate_risk_fusion.py) and [`risk_fusion/register_risk_fusion_beta.py`](../risk_fusion/register_risk_fusion_beta.py) if that day comes. Don't reach for XGBoost/GBM here without a lot more labeled data first — the current GLM choice is deliberate, not a placeholder (see `fit_glm.py`'s module docstring for why).

**Refreshing KBDI climate normals** (only needed if the county list changes, e.g. a new county added):

```bash
python -m risk_fusion.build_precip_normals
```

Safe to rerun any time — it just re-pulls real NOAA data per county and overwrites `risk_fusion/county_precip_normals.json`.

---

## When something looks wrong

- **Retrain crashed partway through** → the error message names which phase failed (e.g. `Phase 'Phase 5: Generate training set' failed: ...`). Fix that step, rerun with `--skip-*` for the phases that already succeeded.
- **Beta always looks worse than stable** → check `sync_training_data.py --status` first; a stale/incomplete data pull will do this.
- **Not sure a number is real** → the experiment log (`reports/experiments/fuel_moisture.jsonl`) has every run's MAE/R²/sample counts, so you can compare across time instead of trusting one run.

## Further reading

- [`training_guide.md`](../pipelines/training_guide.md) — full technical detail on both the legacy XGBoost and RTMA/spatial pipelines
- [`retraining_and_fire_weather_risk_plan.md`](retraining_and_fire_weather_risk_plan.md) — the design rationale behind this workflow, and what's still open
- [`README.md`](../README.md) — repo-wide setup and the spatial/static-bundle workflows this guide doesn't cover
