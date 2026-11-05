# Detection confidence dataset

Joins every SMF detection (viirs/modis/ngfs/goes) with Show Me Fire feedback and NERIS
outdoor-fire dispatches, adds engineered features and ML-generated columns.

## Run
1. Get the data into `SMF_DATA_ROOT/fire_labels/`, any one of:
   - **R2 (normal):** `python -m detection_dataset.build --pull` downloads the newest snapshot the API published
     (`api/scripts/publish_detection_training_data.py`, or the daily job). Needs the `R2_*` vars in `.env`.
   - **Zip:** `--import-bundle path\to\detection_training_export.zip` (made by `export_detection_training_data.py --zip`).
   - **Local DB:** `--refresh` / `--db path` exports from a SQLite file.
   A missing export is fetched automatically on the first run (R2 if configured, else the local DB).
   See `api/docs/detection_history_runbook.md` for the FIRMS backfill that lengthens detection history.
2. Training side: `python -m detection_dataset.build [--neris-radius-km 2] [--neris-path X.shp] [--no-ml]`,
   or press Train on "Detection confidence dataset" in the pipeline dashboard (port 8787).
3. Outputs in `SMF_DATA_ROOT/detection_dataset/`: dataset csv/parquet, `columns.json`
   (name, origin, group, `leaks_label`, `model_input`); report in `reports/detection_dataset_<date>.md`.

## Labels (nullable; null = no evidence)
Any real fire = 1; only `not_a_fire` = 0. Priority: approved incident feedback (w 1.0) >
admin-reviewed tier + cause (0.6) > NERIS match (0.8). A detection with no NERIS dispatch nearby is NOT a
negative: it stays unlabeled, because a missing report is not a verdict and some areas never report to NERIS.
Real negatives come only from approved `not_a_fire` feedback. NERIS is day-resolution, shifted UTC-5, filtered
to a Missouri bbox. `neris_*` and `label*` columns are never model inputs (`f_*` columns are).

## ML columns (all out-of-fold, spatial-block GroupKFold on ~25 km cells)
`ml_p_fire_oof`, `ml_p_burn_*`, `ml_pu_score` (NERIS-positive vs unlabeled), `ml_cluster_id/dist`,
`ml_anomaly_score`, `ml_embed_1..8`, `ml_neighbor_label_rate`. Supervised ones are skipped (with a reason
in the report) below 30 labels / 2 classes.

## Scope rule (any label source)
A detection is kept if ANY label source speaks for it: approved feedback, admin tier, a user report
(within 3 km, +-1 day), or a NERIS dispatch. Unmatched detections are also kept, unlabeled, when they fall on days
inside NERIS's date range in a dense month AND in a county that reports to NERIS (see below); they are the
"unlabeled" side of the positive-vs-unlabeled score. Sources have different dates, so none is required to overlap another. Per-source date
coverage and the number of excluded detections are in the report.

## NERIS county coverage
`detection_dataset/coverage.py` counts NERIS calls per county (a county "reports" at >= 10 calls in NERIS's date
range). The report lists reporting vs silent counties and the busiest of each, and
`neris_county_coverage_<date>.csv` has every county (calls, detections, matches, match rate). Detections in silent
counties are not compared against NERIS at all. Dataset columns: `neris_county_reporting`, `neris_unmatched_reporting`.

## Confound check
If positives and negatives come from different periods, a model can score well from the date alone. The report
includes `time_only_baseline` and `sensor_only_baseline` for the supervised label, plus `pu_time_only_baseline` /
`pu_sensor_only_baseline` for the NERIS positive-vs-unlabeled target (which is all that runs until real negatives exist),
and a warning when a date-only AUC > 0.8. Date features
(`TIME_CONFOUNDS` in derive.py) are excluded from model inputs, but proxies (detection density, location) can
remain, so a high AUC with a high baseline is not evidence of skill. Real negatives (approved `not_a_fire`
feedback) spread over the same dates as the positives are what fix this.

## Date handling (superseded: see scope rule above)
No dates are hard-coded. The build computes the window present in BOTH detections and NERIS
(`labels.overlap_window`) and only keeps/analyses detections on those days; the window and the number
excluded are in the report. `--start/--end` can only narrow it. If there is no overlap the build stops with
the two ranges. NERIS "coverage" months are relative (>= 25% of the median monthly
count). Features are computed before trimming so detections at the window edge keep their neighbours.

## Caveats
The local DB copy has no feedback and its detections do not overlap NERIS, so a real run needs a production
export whose detections cover the NERIS dates.
