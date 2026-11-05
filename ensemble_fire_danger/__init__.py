"""
ensemble_fire_danger - training side of the Day-1 ensemble fire danger
product (chance of >= Moderate / Elevated / Critical / Extreme, plus a
categorical ensemble forecast). Full design: docs/ensemble_fire_danger.md;
retraining steps: docs/ensemble_fire_danger_retraining_runbook.md.

What is "trained" here is deliberately small: the product's physics is the
operational fire danger rule applied to each ensemble member, unchanged.
Training fits only (a) a per-category isotonic calibration from raw
ensemble exceedance fractions to observed exceedance frequency, and (b) the
per-category probability threshold the categorical forecast uses - one set
per track (members / ensprod) and per probability flavour (point /
neighborhood). "Observed" = the same rule applied to hourly RTMA
analyses over the same peak window (labels.py).

Pipeline (each step is its own module, `python -m ensemble_fire_danger.<step>`;
retrain.py chains them):

  capture      fetch every member / ensprod run for each forecast day
               (same cycle-resolution rules as live, emulating live
               publish latency) into CACHE_ENSEMBLE_DIR
  build_panel  per day: both tracks' member peaks (tracks.py) + RTMA
               labels -> one .npz per day; then a thinned cell-day panel
  contract     blocked-by-week train/test split manifest
  fit          isotonic knots + categorical thresholds (train split)
  evaluate     Brier/BSS/reliability/ROC/CSI gates + Track M vs Track E
               comparison (test split)
  register_beta  bundle -> local registry as an advisory-only beta

Modules shared byte-for-byte with api/services/ensemble_fire_danger/
(contract mirrors, listed in api/core/contract_mirrors.json): core,
grib_idx, members, regrid, tracks, fm_features, member_config.json.
Never edit only one copy - edit, copy across, then refresh the recorded
checksums (see the runbook).
"""
