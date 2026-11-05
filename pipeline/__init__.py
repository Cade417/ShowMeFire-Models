"""Unified data-acquisition + training orchestration layer.

This package does not reimplement any puller or training script - every
backfill/capture script under scripts/ and every per-family training script
keeps its own resumable manifest, retry logic, and CLI exactly as before.
What lives here is the piece that was previously duplicated between
scripts/sync_training_data.py and scripts/acquire_training_data.py: a single
declarative registry of what sources/models exist, a single way to turn
each one's manifest/registry state into a structured status dict, and a
single way to launch any of them as a subprocess job. The local dashboard
(dashboard/app.py) and both CLI entrypoints import from here instead of
each re-deriving their own copy.
"""
