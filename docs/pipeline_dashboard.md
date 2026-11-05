# Pipeline + dashboard

A local web dashboard over the existing data-acquisition and training
scripts. Nothing here replaces any puller or training script - it's a thin
layer that (1) dedupes the sequencing/status-reporting logic that used to
be copy-pasted between `scripts/sync_training_data.py` and
`scripts/acquire_training_data.py`, and (2) puts a browser UI on top so you
can see freshness/registry state and trigger runs without remembering which
script takes which flags.

## Running it

```
pip install -r requirements.txt -r requirements-dashboard.txt
python -m dashboard.app
```

Opens on `http://127.0.0.1:8787`, bound to localhost only.

## What it shows

- **Data sources** - one row per puller (R2 mirror restore, Synoptic, HRRR,
  RTMA, RRFS, FV3-HIRES, archive-zip rsync), with complete/pending/failed
  counts and the latest captured day/cycle, read straight from each
  puller's own manifest. "Sync" launches that puller's own CLI as a
  background job - it does exactly what running the script yourself would
  do.

### R2 mirror restore (no network)

`scripts/migration/` already pulled the *entire* R2 archive bucket down to
local disk once (see its own README) - that's a full mirror of the bucket's
key layout at `<SMF_DATA_ROOT>/r2_pull/data-v2/objects/<date>/<source>/
<sha256>/<name>`, going back further than most of the live pullers' own
retention (e.g. RRFS/FV3-HIRES source windows are short-lived upstream).
`pipeline/r2_restore.py` is the step that actually makes that data usable:
it copies anything still missing from that local mirror into the real
working cache dirs (`cache/hrrr`, `cache/rrfs`, `archive/raw_data`) under
the exact filenames the backfill scripts look for, verifying each file's
embedded sha256 as it goes. Zero network calls, zero R2 credentials needed
- this only reorganizes files already on disk. It runs first in the
sequence (`pipeline/registry.py`'s `SEQUENCED_STEPS`), so Synoptic/HRRR/RRFS
have less left to actually fetch from the internet afterward.

This only covers `hrrr`, `rrfs`, and `observations` (Synoptic raw_data) -
the three sources `api/services/archive_store.py` actually archives to R2.
RTMA and FV3-HIRES are not in that archive at all and stay internet-only.
- **Models** - one row per entry in `models/model_types.py::KNOWN_MODEL_TYPES`
  plus the detection-confidence model (see below), showing the current
  stable/beta version and performance metric from `models/config.json` and
  a training-run count from `reports/experiments/<model_type>.jsonl`.
  "Train" is only enabled for model types with a single-command retrain
  wired up in `pipeline/registry.py`'s `TRAIN_TARGETS`.
- **Job log** - every job launched from the dashboard (source sync or
  training), with its command and tailed stdout/stderr. Job logs persist as
  plain files under `data/reports/dashboard_jobs/<job_id>.log` and are
  re-listed from disk if the dashboard restarts.

## CLI equivalents

Nothing moved - the same commands as before still work:

```
python -m pipeline.run --status
python -m pipeline.run --dry-run
python scripts/sync_training_data.py
python scripts/acquire_training_data.py
```

`pipeline/run.py` is now the one place sequencing/status logic lives;
both of the above scripts call into it.

## Adding a new data source

Add one `DataSource(...)` entry to `pipeline/registry.py`'s `DATA_SOURCES`
list: key, label, the puller script's path, its manifest path, and the
manifest's top-level dict key (`"days"` / `"runs"` / `"analyses"` /
`"cycles"`, whatever that puller uses). If the new puller's window is
derived from another source instead of a standalone `--days-back` (like
RTMA depends on HRRR), don't add its key to `STEPS_SUPPORTING_DAYS_BACK`.

## Adding a new trainable model

Add one `TrainTarget(...)` entry to `pipeline/registry.py`'s
`TRAIN_TARGETS` list: `model_type` (must match a `KNOWN_MODEL_TYPES` entry),
a label, and either `script=` (a single Python script with a CLI) or
`module=` (for a `python -m package.module` style entrypoint, e.g.
`ensemble_fire_danger.retrain`).

Several model families (spatial v4/v5/hybrid/station-sequence) don't have a
single-command retrain - each is a multi-step fit -> calibrate -> evaluate
-> register pipeline with no fixed step order decided yet. Those are
intentionally left out of `TRAIN_TARGETS` rather than guessing; train them
via their existing per-step scripts as before, and only add a `TrainTarget`
once a single entrypoint script exists (or `TrainTarget.script` is pointed
at a wrapper script you've written that sequences the steps).

## Detection-confidence model (sibling `api/` repo)

`api/detection-confidence-model/retrain.py` lives in the separate `api/`
checkout, not here. Set `SMF_API_REPO_ROOT` in `.env` to that repo's path
(see `.env.example`) and the dashboard will pick it up automatically; leave
it unset if you don't have that repo checked out on this machine.
