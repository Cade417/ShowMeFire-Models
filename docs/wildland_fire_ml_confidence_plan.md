# Machine learning confidence for wildland fire detections

Date: 2026-10-27
Status: Proposed implementation plan; no model or production change is implemented by this document.

## 1. What we are building

Build a machine learning model that estimates:

> Given this satellite detection and the information available when it arrived, how likely is it to represent an active vegetation fire?

Vegetation fire includes grass, brush, crops, woodland, and forest. Prescribed burns and agricultural burns count as positive examples. The score does not estimate whether a fire is uncontrolled, dangerous, large, or likely to spread. Those would require separate predictions.

The user-facing result is one confidence score. The model does not need to output an industrial-versus-wildland classification. Known industrial heat, land cover, satellite measurements, and observation history are evidence used to estimate the same target.

The eventual output is a calibrated probability between 0 and 1, optionally displayed as a percentage. For example, among representative detections assigned approximately 80%, approximately 80% should be verified vegetation fires. This is a population-level interpretation, not a guarantee about an individual detection.

Until evaluation demonstrates adequate calibration, describe the candidate output as an experimental model score rather than a validated percentage probability.

## 2. How this differs from the existing approach

The existing decisions document describes v1 as a formula with manually selected weights. Its proposed formula-based v2 would also have manually selected weights. Neither is machine learning.

This proposal uses reviewed outcomes to learn relationships between measurements and vegetation-fire outcomes. A trained model learns how much evidence to assign to its inputs. Human work is still required to define the target, review examples, prevent leakage, and evaluate results.

Retain v1 as the operational baseline. A transparent formula can remain an additional comparison if useful, but building a new formula is not a prerequisite for training a model.

The work already completed on exports, history backfills, feature construction, R2 delivery, and leakage investigation remains useful. This plan builds on it rather than requiring a new data platform.

## 3. What the existing experiments establish

According to detection_confidence_decisions.md, the latest documented dataset contains 2,008 positives, zero verified negatives, and 28,375 unlabeled detections. These counts are historical figures from that document, not a fresh dataset inspection.

The NERIS data was manually filtered to Outside Fires by the user. Treat it as an intentional source of positive evidence. Preserve the original incident category and matching details so that any cases outside the vegetation-fire definition can be identified without discarding the source wholesale.

A missing NERIS match must remain unlabeled. Missing reports can reflect coverage, timing, geolocation, matching limitations, or fires that were never reported.

The positive-versus-unlabeled experiments do not establish that satellite measurements lack useful fire signal. They test separation between NERIS-associated detections and a mixed group that can contain real fires. Their results cannot substitute for evaluation against verified vegetation and non-vegetation outcomes.

The previously reported ingestion-related leakage is a particularly important warning: a feature can look predictive because of how records entered the database. Future candidates must be checked for the same issue.

## 4. Define the training outcome

Use one binary outcome when evidence is sufficient:

| Review outcome | Training target | Meaning |
|---|---|---|
| Vegetation fire | 1 | Grass, brush, crop, woodland, or forest burning; prescribed and agricultural burns included |
| Not vegetation fire | 0 | Reviewed routine heat, artifact, or a fire with no vegetation involvement |
| Cannot determine | No binary target | Evidence is insufficient, ambiguous, or contradictory |
| Not reviewed | No binary target | No adjudicated outcome exists |

A mixed event involving a building and surrounding vegetation is positive when the detection can reasonably be associated with vegetation burning. Proximity to vegetation alone is insufficient.

Reviewers may record an optional reason such as routine industrial heat, reflection, or structure-only fire. These reasons support quality control and error analysis; they do not require a separate production classifier.

Keep raw evidence and adjudicated outcomes separate. Record label provenance, evidence, reviewer, review timestamp, and label-definition version. Do not treat existing source weights such as 0.8 as established probabilities that a label is correct. They are assumptions that require sensitivity analysis.

For conflicting evidence, retain the conflict and route it for review instead of silently choosing the most convenient source.

## 5. Use the existing heat-source masks

Known persistent heat-source masks are useful contextual inputs. Candidate features include mask overlap, distance to a known source, source category when reliable, and historical recurrence around that location.

A mask does not set confidence to zero. A vegetation fire can occur near a facility or within a broad mask. Strong evidence of changing spatial extent or nearby vegetation burning may outweigh routine-source context.

Masked detections are also useful candidates for manual negative-label collection. However, overlap alone must not generate an automatic negative. Review evidence that the observed activity corresponds to the usual source.

Save mask version and effective date. A mask created after an event must not silently appear in a replay of what the system knew before the event. If historical mask versions are unavailable, disclose that limitation and evaluate prospective snapshots.

Confirm that masked detections remain available somewhere in the incoming-data pipeline. If an existing filter discards them before storage, capture them for sampling and shadow scoring without changing the current alert behavior.

## 6. Review workflow and sampling

Build a small admin review queue with three required choices: vegetation fire, not vegetation fire, and cannot determine. Allow evidence links and a short note. Show acquisition time, sensor, location uncertainty or footprint where available, relevant imagery, and observation history.

Hide the current and candidate confidence scores during initial review to reduce anchoring. Record what evidence the reviewer used. Later evidence may establish the outcome even though the model must use only evidence available at scoring time.

Maintain two distinct sampling streams:

1. Representative evaluation sampling: randomly select from the complete incoming detection population, including masked and low-scoring detections. Keep a protected subset for final evaluation.
2. Training enrichment: deliberately select masked sources, unusual measurements, sensor disagreements, and ambiguous cases to obtain diverse positive and negative examples efficiently.

Store stream membership, sampling rules, and inclusion probabilities when sampling is unequal. Enriched training data will have an artificial class balance. Raw precision and calibration measured on it will not represent the operational detection stream.

The first approximately 100 reviews are a workflow pilot: establish how often cases are decidable, how long review takes, and which evidence is missing. Do not make production claims from that number alone.

Review a subset independently with a second reviewer and adjudicate disagreements. Track disagreement and unresolved rates by sensor and setting. Dropping unresolved cases without reporting them can make performance appear better than it is.

## 7. Features available to the model

Start with the features already available, then add costly data only if evaluation shows a need.

| Feature family | Candidate inputs | Important constraint |
|---|---|---|
| Satellite signal | Native confidence, brightness temperatures, FRP, relevant quality flags | Preserve source-specific meanings and units |
| Observation conditions | Sensor/product, day/night, scan geometry, pixel dimensions where available | Missing values must remain distinguishable from zero |
| Land context | Vegetation and land-cover fractions around the footprint, developed-land fraction | A point coordinate is not an exact fire perimeter |
| Persistent-source context | Mask overlap, source distance, historical recurrence | Masks provide evidence rather than hard labels |
| Recent detection context | Nearby observations, cross-sensor corroboration, recent spatial changes | Use only observations available by scoring time |
| Environmental context | Available weather and season | Dryness indicates opportunity, not confirmation |

Use appropriate footprint-aware neighborhoods rather than assuming a single universal radius for all sensors. Record the chosen distances and time windows as versioned feature settings.

Cross-sensor corroboration is useful but not independent proof. Two feeds may contain the same observation, and persistent industrial sources can be seen by multiple satellites. Deduplicate observations and preserve product lineage.

Absence of another detection is not automatically negative evidence. Clouds, missing coverage, resolution, acquisition schedules, and latency can explain absence. Model non-detection only if the observation opportunity and its limitations are known.

Exclude feedback outcomes, NERIS-match indicators, later incident associations, and moderation state from the initial satellite confidence inputs. These establish labels or occur after scoring. Evaluate any later evidence-assisted score as a separate mode with explicit semantics.

## 8. Capture information at scoring time

Distinguish acquisition time from arrival time. An observation acquired earlier may arrive after the initial alert, so filtering only by acquisition timestamp still allows leakage.

For every scoring run, save the feature values actually used, their schema version, the scoring time, and the relevant source availability timestamps. Include feature freshness and missingness where useful.

There can be two separately evaluated outputs:

- Initial confidence: computed when the detection is first processed.
- Updated confidence: recomputed when additional evidence arrives.

Store both as append-only score records. Do not overwrite the initial prediction and then evaluate the revised result as though it had been available initially.

## 9. First model and training procedure

Start with regularized logistic regression over a deliberately modest feature set. Apply numeric scaling, missing-value handling, categorical encoding, and selected transforms inside a fitted preprocessing pipeline. For example, highly skewed positive measurements may benefit from a documented transform.

Logistic regression is a useful baseline, not a promise that every relationship is linear or that its probabilities are automatically calibrated. Sensor-specific interactions or separate sensor models may be necessary if feature meanings differ substantially.

Then compare a shallow gradient-boosted tree model using the same legitimate inputs and evaluation splits. Limit complexity and tuning while labels are scarce. Neural networks are not a necessary starting point for this tabular dataset.

Compare reviewed-only training with training augmented by NERIS positives. This reveals whether extra labels improve generalization or mainly introduce reporting-source bias. Avoid allowing thousands of repeated detections from one fire or facility to dominate training; document event/site weighting and its implications.

Fit all preprocessing on training data only. Select features and hyperparameters without looking at the final test outcomes. Save random seeds, dataset manifests, checksums, feature definitions, model configuration, and software versions for reproducibility.

## 10. Evaluation splits and calibration

Rows are not independent. Nearby observations can describe the same fire, and repeated visits can describe the same facility.

Create event and persistent-site group identifiers. Keep related groups out of both sides of an evaluation split. Use a later-time holdout to test future performance, removing or explicitly accounting for groups spanning the boundary. Add a geographic holdout when data supports it.

Report familiar-site and unseen-site performance separately where possible. This distinguishes useful recognition of known recurring sources from generalization to new locations.

Separate model fitting, probability calibration, threshold selection, and final testing. Group-aware cross-validation can help use limited development data, but the final test set must remain untouched. Grouped splitters are described in the scikit-learn cross-validation guide [1].

Calibrate using independent, representative reviewed development data, or a justified sampling-weighted alternative. If class weighting or negative enrichment was used in fitting, do not assume the resulting probabilities reflect production prevalence. Evaluate a sigmoid calibrator first; more flexible calibration requires more data. Calibration must also respect group boundaries [2].

## 11. What success means

Choose an operational tradeoff before examining final test performance. The precise threshold is a product decision informed by measured false-alert cost and the cost of down-ranking a real fire; this document does not invent a required percentage.

Report:

- Precision at candidate alert thresholds: the share of flagged detections that are verified vegetation fires.
- Recall among reviewed satellite detections: the share of verified vegetation-fire detections above the threshold.
- Alert volume at each threshold, alongside both precision and recall.
- Precision-recall curves and average precision for ranking behavior.
- Brier score, log loss, and reliability plots for probability quality.
- Group-aware uncertainty intervals, event/site counts, unresolved counts, and sensor-specific results.

Standard precision and recall definitions are documented in [3]. Probability reliability and Brier/log-loss interpretation are discussed in [2]. No single metric is sufficient for promotion.

This evaluation measures classification of satellite detections. It cannot measure all fires missed by the satellites because undetected fires are absent from this dataset. Estimating overall fire-detection sensitivity would require an independent inventory and a different study.

Compare against v1 on identical reviewed cases. For probability metrics, v1 must first receive its own development-set calibration if its current numbers are arbitrary scores. Also compare a simple prevalence-only probability baseline. Wider score spread and greater sensor agreement are diagnostics, not promotion criteria.

## 12. Suggested storage contracts

These are proposed logical records, not verified descriptions of existing database tables. Reconcile them with the current schema before implementing migrations.

| Record | Minimum contents |
|---|---|
| Review selection | Detection ID, sampling stream, selection time, sampling version, inclusion probability where applicable |
| Review outcome | Detection ID, target version, outcome, reason, reviewer, evidence, review time, adjudication status |
| Label evidence | Source record ID, source type, event time, match distance/time, provenance, conflict state |
| Feature snapshot | Detection ID, score time, schema version, feature values, mask version, availability/freshness metadata |
| Score history | Detection ID, model/version, initial or updated mode, score, calibrated status, score time, snapshot reference |
| Model registry | Artifact checksum, training manifest, target/schema versions, calibration artifact, evaluation report, activation state |

Keep review and training artifacts protected consistently with the existing R2 moderation-data restrictions. Maintain corrections and version history rather than silently replacing evidence.

## 13. Shadow deployment and promotion

Run candidate inference alongside the existing system. Preserve normal user-facing behavior while collecting representative reviews and score history.

Test the serving path for preprocessing consistency, missing-data handling, supported sensors, artifact loading, and fallback behavior. A model trained only on VIIRS/MODIS must not silently issue calibrated NGFS probabilities. Retain the baseline for unsupported sources until evidence supports their inclusion.

Promote only after the candidate improves the chosen operational tradeoff on protected reviewed data, has acceptable calibration for intended use, and shows no material unexplained regression in sufficiently represented operating groups. Record evidence gaps where subgroup samples are small.

Version the model and threshold separately. Start rollout with an easy rollback to v1. Continue random review after activation and monitor calibration, alert volume, input missingness, sensor/product changes, and seasonal or geographic drift.

A detector or feed update can change the input distribution. Retraining should respond to new evidence and measured degradation; an automatic schedule alone does not establish that a new artifact is safe to promote.

## 14. Implementation sequence and deliverables

### Phase A: Target and data audit

Document the vegetation-fire definition and review rubric. Inspect existing feedback mappings, NERIS categories and matching logic, heat-source masks, and filter order. Identify which current features are available at initial scoring time.

Deliverable: versioned target specification, source mapping, and feature-availability inventory.

### Phase B: Review and snapshot collection

Implement the review queue, representative sampling, enrichment sampling, score history, and feature snapshots. Pilot approximately 100 reviews and revise the evidence display if cases cannot be resolved reliably.

Deliverable: an auditable review dataset with positive, negative, and unresolved examples, plus prospective score snapshots.

### Phase C: Reproducible baseline experiment

Create group identifiers and protected splits. Train logistic regression, compare reviewed-only and NERIS-augmented variants, then evaluate a small boosted model if justified. Fit calibration and choose candidate thresholds using development data.

Deliverable: reproducible artifacts and a report comparing candidates with v1 and a prevalence baseline, including limitations.

### Phase D: Shadow scoring

Integrate the candidate with explicit sensor support, missing-data behavior, versioning, and fallback. Collect independent reviews while comparing initial and updated scores separately.

Deliverable: a prospective evaluation report and a concrete promotion or continued-collection recommendation.

### Phase E: Controlled activation

Activate only the supported model/threshold combination after the evidence meets recorded promotion criteria. Continue review, monitoring, and rollback support.

Deliverable: versioned production confidence with documented meaning and measured performance.

## 15. Decisions to make during implementation

Routine implementation can proceed using the target agreed in this conversation. The following details should be resolved through inspection and the review pilot rather than assumed:

- Whether the remaining Outside Fires categories contain examples outside the vegetation-fire definition.
- Whether current feedback distinguishes vegetation fires from other real fires.
- Where masked detections are retained and how mask boundaries were created.
- Which review evidence is available often enough to establish reliable outcomes.
- How many independent negative sites and positive events the review process produces.
- Which sensors have sufficient training and calibration support.
- Which alert tradeoff is acceptable once measured results are available.

There is no defensible universal number of labels that guarantees readiness. Independent event/site diversity, negative-example coverage, label reliability, and uncertainty at the intended alert threshold matter more than a raw row count.

## References and provenance

Project context: detection_confidence_decisions.md, supplied by the user, plus the subsequent clarification that NERIS was manually filtered to Outside Fires and that the desired output is confidence in wildland/vegetation fire. Existing code and reported experiment results were not independently audited for this write-up.

[1] scikit-learn, Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html

[2] scikit-learn, Probability calibration: https://scikit-learn.org/stable/modules/calibration.html

[3] scikit-learn, Model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
