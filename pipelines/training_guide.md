# Show Me Fire Training Guide

This guide's content has moved to the [Model Lifecycle guide](../docs/model_lifecycle.md), which covers the same legacy-XGBoost and RTMA/spatial workflows alongside every other model family's train → publish → import → promote path in one place.

- Legacy XGBoost retrain workflow (`pipelines/retrain_fuel_moisture.py`) → Model Lifecycle guide, "Archetype A".
- RTMA/spatial pipeline, static terrain/fuel bundle → [Spatial Fuel-Moisture Operator Runbook](../docs/spatial_fuel_moisture_runbook.md) (full detail) and the Model Lifecycle guide's "Archetype B" (the publish/import/promote steps specifically).
- Data sync (`scripts/pull_archives.sh`, `backfill_synoptic.py`, `backfill_rtma_for_hrrr.py`) → Model Lifecycle guide, §1.
