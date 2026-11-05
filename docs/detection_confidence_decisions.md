# Detection confidence: history, findings and decisions

Written 2026-10-27. Covers the detection-dataset work, what it showed, and where we landed. Companion docs:
`docs/detection_dataset.md` (how to run the pipeline) and `api/docs/detection_history_runbook.md` (FIRMS backfill, R2).

## 1. Goal
Build a dataset (and eventually a better model) for **detection confidence**: how likely a satellite detection
(VIIRS, MODIS, NGFS) is a real fire. Rule from earlier work: confidence means "is it a real fire of any kind".
Prescribed and agricultural burns are positives; only `not_a_fire` is a negative.

Inputs: the Show Me Fire API/DB (all detections, feedback, user reports), the NERIS Missouri outdoor-fire shapefile
(`NERIS_MO_OUTDOORFIRES_20260811.shp`), and a FIRMS archive backfill for longer detection history.

## 2. What was built
| Piece | Where |
|---|---|
| Dataset pipeline (labels, ~70 features, ML-derived columns, report) | `model-training/detection_dataset/` |
| Dashboard entry (one-click build, job log, experiment history) | `pipeline/registry.py`, `pipeline/status.py` |
| API-side export (read-only, checksummed, stdlib-only) | `api/scripts/export_detection_training_data.py` |
| FIRMS archive backfill to CSV (never touches the DB) | `api/services/firms_backfill.py`, `scripts/backfill_firms_history.py` |
| Ingest only NERIS-overlapping backfill rows, detections only (no incidents) | `scripts/ingest_firms_history_csv.py`, `cluster_incidents=False` |
| Upload to Cloudflare R2 + training-side pull (checksummed) | `api/services/detection_r2.py`, `detection_dataset/r2.py` |
| Optional daily R2 publish job (off by default) | `SMF_DETECTION_R2_PUBLISH=true` in `core/scheduler.py` |
| FIRMS vs NERIS overlap tool | `detection_dataset/overlap.py` |
| NERIS county coverage map | `detection_dataset/coverage.py` |

Verified in production by the user: R2 publish (snapshot `20261028T002843Z`, 36,415 detections, 47 feedback, 9 user
reports) and the FIRMS dry run (66,905 detections).

## 3. Decisions made (and why)
1. **Labels come from any source, not NERIS only**: approved feedback (w 1.0), admin tier (0.6), user reports (0.7),
   NERIS matches (0.8). Each source has its own dates; a detection is kept if any source speaks for it. No dates are
   hard-coded.
2. **Backfill goes to a CSV, not the DB.** Compare with NERIS first, then ingest only the overlapping detections.
   Detections only, no incidents ("that would just make things messy"). The full CSV is still used for training.
3. **Storage on the API side + R2** so training can pull without copying files by hand. Keep the R2 prefix off any
   public/CDN path (it contains moderation outcomes).
4. **A missing NERIS call is not a negative.** The earlier weak negatives (no dispatch nearby = 0, weight 0.25) were
   removed. Some areas do not report to NERIS, so absence proves nothing. Such detections are unlabeled, and kept only in
   counties with at least 10 NERIS calls (the "unlabeled" side of a positive-vs-unlabeled comparison). Only approved
   `not_a_fire` feedback produces a 0.
5. **NERIS county coverage map** added so it is visible where NERIS can and cannot be trusted (94 of 115 counties
   report; 21 do not; ~4,600 old "weak negatives" sat in silent counties).
6. **Burned-area labels (MCD64A1) dropped.** Poor at small burns and croplands, partly circular (it uses MODIS active
   fire as an input), and biased to large fires. Not downloaded.
7. **Corroboration is a feature, not a label.** Cross-sensor agreement measures "real heat", which v1 already tries to
   measure; as a label it would be circular.
8. **No ML for v2 yet.** ML needs a trustworthy label and we do not have one. v2 should be a transparent formula first.

## 4. Findings
- **0.999 AUC was a leak.** The strongest features existed only for detections in the DB (incident size, weather, etc.),
  and positives were the rows that had been ingested. Fixed with `PROVENANCE_ONLY`, CSV-authoritative merging and
  VIIRS/MODIS-only modelling.
- **No signal in the detection itself.** NERIS-positive vs unlabeled (reporting counties), detection-property features
  only: AUC 0.56 (earlier, with weak negatives: 0.51). Date-only 0.66; full model 0.65. All lift is location, season and
  neighbor density, likely NERIS reporting geography. Do not use `ml_p_fire_oof` / `ml_pu_score` as a fire probability.
- **Supervised model is skipped** (no real negatives): 2,008 positives, 0 negatives, 28,375 unlabeled.
- **v1 is a formula, not a model** (`api/services/detection_confidence.py`; no trained model file exists). Its score is
  compressed: nearly everything lands at 40-55 (VIIRS mean 46, middle half 44-48, std 4.9; MODIS 51; NGFS 47, std 7.6),
  so a "high confidence" cutoff is never reached. Day/night barely moves it. Sensors disagree on the same event
  (VIIRS/MODIS pair correlation 0.22 over 3,109 pairs). Repeat spots are steady (median within-spot std 3.8), but only
  because the score barely moves.
- **No cheap unbiased "fire or not" label exists for Missouri**: NERIS is biased, burned area and corroboration are
  circular, feedback is thin and positive-only.

## 5. Current state of the data
Latest build (both history CSVs + R2 export): 30,383 rows (2,008 positives, 28,375 unlabeled). Label sources: neris_pos
1,732, smf_feedback 173, smf_tier 92, user_report 11. 74,017 detections were excluded (no label source covers them).
Dataset is usable for analysis; it is not a basis for a production confidence model.

## 6. Plan agreed in principle (not yet built)
1. **v2 as a transparent formula**: rescale within each source so VIIRS/MODIS/NGFS share one scale and strong detections
   score clearly above weak ones; add corroboration by another sensor, repeat visits, recurring-source/exclusion-zone
   flags. Weights written down and easy to change.
2. **Shadow scoring**: a `detection_scores` table (detection id, model, version, score, timestamp) written at scoring
   time, v1 now and v2 once it exists. Nothing user-facing changes.
3. **Scoring-vs-labels report** in the training pipeline: AUC, precision at the top of the ranking and calibration per
   model, using feedback and user reports as they arrive. Label-free checks too: score spread and cross-sensor agreement.
4. **Review queue**: a small random sample of detections reviewed by an admin (fire / not a fire / can't tell), as the
   unbiased label source. Start with ~100 and see how many are decidable.
5. **Recurring-source / exclusion-zone detections as negatives** (persistent heat sources), as an extra label source.
6. **Promotion rule**: v2 replaces v1 only after beating it on enough labeled detections. Show the v1 vs v2 spread and
   cross-sensor numbers before anything touches production.
7. **ML later**, once there are enough reviewed positives and negatives over the same dates and v2 has been shadowed.
   Start small (calibrated logistic regression over the formula's inputs, or a shallow boosted model), ship only if it
   beats the formula.

## 6b. What was built from `wildland_fire_ml_confidence_plan.md` (2026-10-27; phases A-D, no activation)
Decision: build it into the detection flow, **make no decisions from it**; v2 only scores in shadow and its results are tracked.
- **A**: `docs/vegetation_fire_target_v1.md` (target, rubric, source mapping, feature availability).
- **B** (API): tables `detection_review_selection/outcome`, `detection_feature_snapshot`, `detection_score_history`;
  `services/detection_review.py` + `routers/detection_review.py` (`/api/admin/fires/detection-review/{next,stats,shadow-summary,<id>}`);
  admin page `/admin/fires/review-queue` (website; not browser-tested); export extended (`detection-training-v2`: review,
  selection, snapshots, scores, masks; reviewer pseudonymised).
- **C** (training): `detection_dataset/{review_labels,groups,shadow_report}.py`, `confidence_v2/{model,metrics,train}.py`
  (grouped CV, later-time holdout, protected subset, logistic + shallow XGB, v1 and prevalence baselines, gate: fits nothing
  without human-reviewed negatives). Dashboard target `detection_confidence_v2`.
- **D** (API): `services/detection_features_v2.py` (causal features, shared by serving and training), `services/detection_bundle.py`
  (pure-python JSON scoring), `services/detection_shadow.py` (scheduler hook, `SMF_DETECTION_SHADOW=true`, off by default;
  never writes `detection_confidence_pct`; VIIRS/MODIS only for v2).
- Deviation from the plan: the shallow XGBoost is a comparison only (no booster serving), the logistic bundle is the only
  servable candidate. Not built: phase E (activation).
- Real-data check (this export): the gate reports 0 human-reviewed negatives, so nothing is fitted; that is the intended behaviour.

## 7. Open items
- Not yet confirmed on prod: ingest of the NERIS-overlap CSV via `ingest_firms_history_csv.py` (back up
  `showmefire.db` first; a dry run was offered), enabling the scheduled R2 publish, deploying the updated code to the
  server container, checking the admin ungrouped-detections list for backfilled pending rows.
- The review queue and recurring-source labels are not built.
- Bias risk for any shadow evaluation: reviewers and reporters mostly see high-confidence detections, so log scores at
  detection time and rely on the random review queue as the unbiased check.
- Pre-existing, unrelated API test failures (stale NGFS fixture, `test_fires.py` background tasks, a state leak into
  ingest tests); not caused by this work.

## 8. Commands
    # export + build (training machine)
    python -m detection_dataset.build --pull --history-csv firms_history_*.csv
    # FIRMS backfill (server; writes a CSV only)
    python scripts/backfill_firms_history.py --start 2025-12-01 --end 2026-08-08
    # overlap with NERIS, then ingest only the overlap
    python -m detection_dataset.overlap --history firms_history_2025-12-01_2026-08-08.csv
    python scripts/ingest_firms_history_csv.py --csv firms_neris_overlap_<date>.csv --dry-run
    # R2 publish
    python scripts/publish_detection_training_data.py

Outputs land in `training-data/detection_dataset/` (dataset, `columns.json`, `neris_county_coverage_<date>.csv`) and
`training-data/reports/detection_dataset_<date>.md`.
