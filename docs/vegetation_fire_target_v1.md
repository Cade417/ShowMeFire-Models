# Target `veg-v1`: is this detection a vegetation fire?

Phase A of `wildland_fire_ml_confidence_plan.md`. This is the label definition, the review rubric, how every existing
evidence source maps to it, and which features are allowed at scoring time. Label-definition version: **`veg-v1`**.
Changing any rule below means a new version, never an edit in place.

## Definition
| Outcome | Target | Meaning |
|---|---|---|
| Vegetation fire | 1 | Grass, brush, crop, woodland or forest burning. Prescribed and agricultural burns count. A structure fire with burning vegetation counts when the detection can reasonably be tied to the vegetation. Being near vegetation is not enough. |
| Not vegetation fire | 0 | Routine industrial heat, reflection or sun glint, other artifact, or a fire with no vegetation involved. |
| Cannot determine | none | Evidence is missing, ambiguous or contradictory. Reported, never silently dropped. |
| Not reviewed | none | No adjudicated outcome. |

The score is about vegetation burning only. It does not say whether the fire is uncontrolled, dangerous, large or spreading.

## Review rubric
1. Reviewers see the detection's own measurements, nearby observations (7 days before to 1 day after), land cover, any known
   heat-source context, and imagery links. They never see v1 or candidate scores (anchoring).
2. Outcome plus an optional reason (`routine_industrial_heat`, `reflection_or_glint`, `structure_only_fire`,
   `other_non_vegetation`, `insufficient_evidence`, `conflicting_evidence`) and an evidence note or link.
3. Known heat-source masks are evidence, not answers. A fire can occur inside a mask; overlap alone never produces a 0.
4. A second reviewer double-checks a subset. Same answer -> `agree`; different -> `conflict`; a third opinion on a conflict
   -> `resolved`. Open conflicts are not labels.
5. Later evidence may settle an outcome, but the model only ever sees what was available at scoring time.

## Source mapping
| Source | Label | Provenance tag | Notes |
|---|---|---|---|
| Detection review (admin page) | 1 / 0 | `review` | Highest priority. Only settled outcomes. |
| Approved incident feedback `confirmed_fire`, `controlled_burn` | 1 | `feedback_vegetation_unspecified` | Feedback cannot say "vegetation", so this is a positive of unknown vegetation involvement. |
| Approved feedback `not_a_fire` | 0 | `feedback_vegetation_unspecified` | |
| Feedback `unsure` | none | | |
| Admin tier + cause category | 1 / 0 | `admin_tier_cause` | `wildfire`, `prescribed`, `agricultural`, `debris_burn` = 1; `not_a_fire` = 0. |
| NERIS match (2 km, +-1 day) | 1 | `neris_outdoor_unverified_type` | **The NERIS shapefile has only 7 columns (no incident type, cause or property use).** It was filtered to Outside Fires upstream and cannot be audited for non-vegetation fires from this file. Positive only. |
| No NERIS match | none | | Never a negative: some areas do not report, and absence of a call is not a verdict. |

**Conflicts.** When two sources disagree on a detection (for example a reviewer says not vegetation and NERIS matched),
the row is left unlabeled, `label_conflict = 1`, and it is counted in the report so it can be re-reviewed.

**Weights.** Source weights (reviewed and feedback = 1, NERIS swept over 0.25 / 0.5 / 1) are only sample weights for
sensitivity analysis. They are not probabilities that a label is right.

## Sampling streams
- **random**: uniform over all incoming VIIRS/MODIS/NGFS detections, masked and low-scoring included. One in five (stable
  hash of the event id) is **protected**: never used to fit, calibrate or choose a threshold. The only stream that supports
  unbiased performance claims.
- **enriched**: masked sources, extreme signals, low native confidence, mid-range v1. Artificial class balance; for training
  variety only.
- The first ~100 reviews are a workflow pilot (how often cases are decidable, how long they take), not evidence for production.

## Features at scoring time (schema `veg-conf-features-v1`)
Built by `api/services/detection_features_v2.py`. Counts only use detections that had ARRIVED (`first_seen_at`) by the scoring
time and were acquired no later than the detection (a 3 h look-ahead only for cross-sensor corroboration on an `updated`
score). An `initial` score is what was knowable 15 minutes after arrival; an `updated` score is recomputed once after 6 h.

| Used | Group |
|---|---|
| frp_log, bright_diff, bright_t7_norm, pixel_area_log, native_confidence, is_day, solar/satellite zenith, sensor one-hot | sensor |
| lc_cropland, lc_tree, lc_grass, lc_shrub, lc_developed, lc_water | land cover |
| in_exclusion_zone, in_recurring_source, recurring_source_km, recurring_source_detections | known heat sources (a source confirmed after scoring is not used) |
| n_{1,5}km_{24h,7d,365d}, distinct_days_1km_365d | prior detections |
| n_other_sensor_3h, max_other_frp_3h | cross-sensor corroboration |

Excluded on purpose: latitude/longitude (a model would memorise where NERIS reports), day of year and month, incident
grouping, `detection_confidence_pct`, weather scoring output, ingest lag, feedback and review state, NERIS indicators,
anything that appears after scoring. Missing values stay missing (median plus a missing-indicator in the model), never zero.

Not covered in v1: weather context (dryness is opportunity, not confirmation) and fuel model/canopy (DB-only, would mark
rows by origin). Add only if evaluation shows a need and with a provenance check.

## Known limits
- Historic FIRMS backfill rows have no arrival time, so only prospective snapshots count for evaluation; backfill rows can
  only support NERIS-augmented training with acquisition-time features.
- Exclusion zones and recurring sources have dates but no versions. Snapshots store which mask applied and when it was
  created/reviewed; deleting a zone is not retroactive.
- NGFS has no history and different fields: no v2 score is issued for it; v1 stays.
- This evaluates classification of detections the satellites made. It says nothing about fires the satellites missed.
