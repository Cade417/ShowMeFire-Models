"""Detection-confidence training dataset.

Joins every Show Me Fire satellite/NGFS detection with (a) public/admin
feedback on its incident and (b) NERIS outdoor-fire dispatches, engineers a
wide feature set, and adds ML-generated columns (out-of-fold fire
probability, burn-type probabilities, clusters, anomaly score, embeddings).

Stages: labels.py -> features.py -> derive.py, orchestrated by build.py
(`python -m detection_dataset.build`). Inputs come from the API's
scripts/export_detection_training_data.py (see paths.FIRE_LABELS_DIR).

Label rule: confidence means "is it a real fire of any kind" - only
not_a_fire is a negative; prescribed/ag burns are positives.
"""
