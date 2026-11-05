# Ensemble Fire Danger: Retraining Runbook

Step-by-step commands for capturing data, (re)training, evaluating, publishing and activating the `ensemble_fire_danger` calibration. For the design and the reasoning behind it, see [ensemble_fire_danger.md](ensemble_fire_danger.md).

All training commands run from `model-training/` with its venv. **Always set UTF-8 mode first.** Herbie prints emoji and crashes with `UnicodeEncodeError` otherwise.

```powershell
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
```

Use the **server's stable fuel-moisture model** for training, so members and labels use the same FM model the live product uses:

```powershell
$env:SMF_ENSEMBLE_FM_MODEL = "<path to the stable fuel_moisture .json>"
```

`build_panel` otherwise falls back to this repo's registry: stable, then beta.

---

## 0. Ongoing capture (automatic)

The Claude scheduled task `smf-capture-rrfs-fv3hires` runs every 4 hours:

```powershell
.venv\Scripts\python.exe scripts\capture_ensemble_members.py --days-back 2
```

**HiResW and HREF are retired (2026-10), so nothing here is forward-only any more.** HRRR, RRFS and REFS all have archives and can be backfilled. Earlier, HiResW and HREF existed for only about 2 days on NOMADS, so a task that skipped 2+ days lost them for that period. Check it with:

```powershell
.venv\Scripts\python.exe -c "import json;m=json.load(open(r'..\data\cache\ensemble\capture_manifest.json'));print(sorted(m['days'])[-5:])"
```

(Adjust the path if `SMF_DATA_ROOT` points elsewhere. The manifest lives at `<SMF_DATA_ROOT>/cache/ensemble/capture_manifest.json`.)

## 1. First-time backfill

HRRR, RRFS and REFS can be backfilled back to 2026-08-12, the start of the RRFS/REFS operational archive:

```powershell
.venv\Scripts\python.exe scripts\capture_ensemble_members.py --start 2026-08-13 --end 2026-09-21 --workers 6
```

- It takes about 3–4 minutes per day and is resumable: completed days are skipped.
- **Exit code 4 with no output, or a `getaddrinfo`/DNS error, is a transient network drop.** Just re-run it.

## 2. Routine retrain (one command)

```powershell
.venv\Scripts\python.exe -m ensemble_fire_danger.retrain
```

This runs:

1. capture of the last 3 days;
2. scoring of new days (build_panel);
3. the week-blocked split;
4. fit;
5. evaluate;
6. register_beta, only if no gate failed.

It prints:

- `failed_gates`: must be empty to register;
- `deferred_gates`: normal while history is short;
- `track_comparison.recommended_primary`;
- headline Brier / BSS-vs-deterministic per track and category.

Useful flags:

| Flag | Use |
|---|---|
| `--skip-capture` | Work only with what's already captured |
| `--rescore` | Re-score every day. **Required after any change to** `tracks.py`, `core.py`, `fm_features.py`, `labels.py`, `member_config.json`, or the FM model |
| `--no-register` | Experiment without touching the registry |
| `--fm-model <path>` | Same as `SMF_ENSEMBLE_FM_MODEL` |

Individual steps, if you need them:

```powershell
.venv\Scripts\python.exe -m ensemble_fire_danger.build_panel [--rescore | --panel-only]
.venv\Scripts\python.exe -m ensemble_fire_danger.evaluate          # re-evaluate the current candidate
.venv\Scripts\python.exe -m ensemble_fire_danger.register_beta
```

### Reading the report

The report is written to `<SMF_DATA_ROOT>/reports/ensemble_fire_danger_offline_evaluation.json`.

- **`tracks.<track>.categories.<k>.neighborhood`** describes what the maps show:
  - `brier_cal` < `brier_raw`: calibration helps.
  - `bss_vs_det` > 0: the probabilities beat the deterministic HRRR-rule map.
  - `reliability_table`: forecast vs observed frequency per 10 % bin. On a well-calibrated product these match.
- **`categories.<k>.categorical`**: `csi`/`bias` for the categorical map, next to the deterministic control's `deterministic_csi`/`deterministic_bias`.
- **`track_comparison`**: mean calibrated neighborhood Brier over the categories where both tracks have enough events. Lower is better.

## 3. Publish and activate on the server

```powershell
.venv\Scripts\python.exe pipelines\publish_release.py --model ensemble_fire_danger
```

Then on the server, or via the website's `/admin/models`:

1. Pick `ensemble_fire_danger`.
2. Choose **Import from GitHub Release**.
3. Click **Promote**.

CLI equivalent:

```bash
python pipelines/import_model.py --model ensemble_fire_danger --tag ensemble_fire_danger-v<version> --repo <owner>/ShowMeFire-Models
python pipelines/promote_model.py --model ensemble_fire_danger --version <version>
```

**Promotion gate.** `validate_promotion_candidate` requires `advisory_only=True`, `model_family` and `core_module_sha256`. The checksum must match the server's `services/ensemble_fire_danger/core.py`, so a calibration fitted against a different numeric core is refused.

**The runtime also uses a beta.** Without a stable version, the runtime uses the registry **beta** too, because the product is beta-labelled anyway. The next run picks it up automatically, and the graphic's footnote switches from "uncalibrated raw member fraction" to "calibrated, model v…".

**Emergency override without the registry.** Copy the candidate directory to the server and set `SMF_ENSEMBLE_FD_BUNDLE=<dir>`.

## 4. Switching the primary track

When the report's `recommended_primary` consistently says `ensprod` and you agree, set `SMF_ENSEMBLE_FD_PRIMARY=ensprod` in the server `.env`. It is never switched automatically. The testbed always shows both tracks.

## 5. Changing members, weights or parameters

1. Edit `api/services/ensemble_fire_danger/member_config.json`: members, weights, `neighborhood_radius_km`, `synthetic`, latencies and so on.
2. Copy it and any other contract-mirror file you changed (`core.py`, `grib_idx.py`, `members.py`, `regrid.py`, `tracks.py`, `fm_features.py`) to `model-training/ensemble_fire_danger/`. **Both copies must stay byte-identical.**
3. Refresh the recorded checksums in `api/core/contract_mirrors.json` (the `ensemble_fire_danger_*` pairs), then verify:
   ```bash
   python scripts/verify_contract_mirrors.py --training-root ../model-training
   ```
   `api/tests/test_ensemble_fire_danger.py::ContractMirrorTests` also fails on drift.
4. `retrain --rescore`, then publish. The bundle carries its own `member_config.json`, and the runtime uses the bundle's config over the vendored default, so members and calibration can never disagree.

Changing `core.py` changes `core_module_sha256`. Registered calibrations fitted before the change will then be refused at promotion, so refit them.

## 6. Running the live product by hand

From `api/`:

```bash
python scripts/run_ensemble_fire_danger.py --no-upload                 # today
python scripts/run_ensemble_fire_danger.py --date 2026-09-22 --no-upload
```

For local testing, redirect every output away from the real files:

```bash
SMF_ENSEMBLE_FM_MODEL=<fm.json> SMF_ENSEMBLE_FD_IMAGES_DIR=/tmp/ens/images \
TESTBED_PRODUCTS_DIR=/tmp/ens/testbed TESTBED_IMAGES_DIR=/tmp/ens/testbed/images \
SMF_ENSEMBLE_FD_STATUS_FILE=/tmp/ens/status.json SMF_ENSEMBLE_FD_EVIDENCE_ROOT=/tmp/ens/evidence \
python scripts/run_ensemble_fire_danger.py --date 2026-09-22 --no-upload
```

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `UnicodeEncodeError ... '✅'` | Set `PYTHONUTF8=1` (see top) |
| Capture exits 4 with no output, or `getaddrinfo failed` | Transient network. Re-run; capture is resumable |
| A member is always `unavailable` | Check the plan's `reason` in the manifest or evidence. `predates archive` is normal for RRFS/REFS before 2026-08-12. `not published` for HiResW/HREF more than 2 days back is normal (forward-only) |
| `fm_anchor: none` in live evidence | DailyForecast didn't write `data/forecast-state/forecast_state_<date>_12z.npz`. Check the DailyForecast log for "Ensemble forecast-state hand-off failed" |
| Graphic says DEGRADED | Fewer than 5 real members that day; see the evidence plan |
| `county_cells grid ... != ensemble grid` warning | The target grid isn't (196, 205). County summaries are skipped. See `core/risk_fusion_county_reference.py` for how that grid is defined |
| Segfault or "no definitions found" from eccodes | Something decoded GRIB outside `grib_idx`'s lock. All decoding must go through `grib_idx._decode_block` |
| Promotion refused: `core_module_sha256 mismatch` | The server's `core.py` changed since the fit. `retrain --rescore` and re-publish |
