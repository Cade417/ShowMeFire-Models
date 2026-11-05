# R2 + server -> local HDD migration

Pulls everything out of the Cloudflare R2 bucket (`cdn-showmefire`) and the
production VPS's own archive directory, verifies it on disk, and (as a
separate, manual, explicitly-confirmed step) deletes the verified copies
from R2 to free up Cloudflare storage/bandwidth. Nothing here is automatic -
each phase is its own command, and deletion never runs as a side effect of
a pull.

## Setup (one-time)

Add to `model-training/.env` (not committed - copy the R2 values from
`api/.env`):

```
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...

SMF_SSH_TARGET=user@server
SMF_REMOTE_ARCHIVE_DIR=/remote/path/to/api/data_archive_day/
```

Install the extra deps (kept out of `requirements.txt` on purpose - see its
own header comment about serving-only deps):

```powershell
cd M:\_Development\ShowMeFire\model-training
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-migration.txt
```

## Run

```powershell
# 1. List everything currently in the R2 bucket (read-only, writes inventory.json)
python scripts\migration\run_migration.py inventory

# 2. Pull it all to local disk (R2 objects + the server's SSH archive dir), verified as it goes
python scripts\migration\run_migration.py pull
python scripts\migration\run_migration.py pull --limit 50        # smoke test first
python scripts\migration\run_migration.py pull --skip-server     # R2 only
python scripts\migration\run_migration.py pull --skip-r2         # server only

# 3. Watch it happen - local dashboard at http://127.0.0.1:8765
python scripts\migration\run_migration.py dashboard
```

The dashboard is read-only: it polls `<SMF_DATA_ROOT>/migration/migration_status.json`
and never starts/stops anything itself. Run it in a separate terminal
alongside step 2, or any time after, to see the same state.

## Deleting from R2 (separate, manual, never automatic)

Only ever deletes a key that step 2 already downloaded and verified
(sha256 match for versioned archive objects; size+hash recorded for
everything else - see `r2_pull.py`'s docstring).

```powershell
python scripts\migration\r2_delete.py                 # dry run - reports what WOULD be deleted
python scripts\migration\r2_delete.py --confirm-delete # actually deletes
```

Every deletion is logged to `<SMF_DATA_ROOT>/migration/delete_audit.jsonl`
*before* the delete call fires, and already-deleted keys are skipped on a
re-run.

## Where things land

Everything is under `<SMF_DATA_ROOT>/migration/` and `<SMF_DATA_ROOT>/r2_pull/`
(currently `D:\_Development\ShowMeFire\training-data\...`):

| Path | What |
| --- | --- |
| `migration/inventory.json` | Full R2 bucket listing from step 1 |
| `migration/migration_status.json` | Live progress state the dashboard reads |
| `migration/verified.jsonl` | One line per object r2_pull.py has verified locally |
| `migration/delete_audit.jsonl` | One line per key deleted from R2, written before the delete call |
| `r2_pull/<bucket key path>` | The actual downloaded files, mirroring R2's own key layout |

Resumable throughout: re-running `pull` or `inventory` picks up where it
left off rather than starting over.
