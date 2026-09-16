"""Create the V4 label dataset from the completed aligned dataset.

The existing aligned rows already identify the exact Synoptic observation used
for each fuel-moisture target.  Rejoining RH and wind on that station/timestamp
is equivalent to rebuilding the HRRR alignment, but is much faster and leaves
the V1-V3 dataset untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.observations import load_observations
from spatial.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Enrich aligned targets for V4 without rebuilding HRRR rows")
    parser.add_argument("--source", type=Path, default=paths.PRECIP_ALIGNED_DATASET)
    parser.add_argument("--output", type=Path, default=paths.V4_ALIGNED_DATASET)
    parser.add_argument("--manifest", type=Path, default=paths.ALIGNED_DIR / "v4_precipitation-v1_enrichment_manifest.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {args.output}; pass --force to replace it atomically")
    started = time.perf_counter()
    print(f"[1/5] Reading aligned rows from {args.source}...", flush=True)
    frame = pd.read_csv(args.source, parse_dates=["forecast_init_time", "valid_time", "target_time"])
    required_precip = {"hrrr_precip_accum_mm", "hrrr_precip_increment_mm", "precip_interval_hours", "precip_available"}
    if missing := required_precip.difference(frame.columns):
        raise RuntimeError(f"Source dataset lacks precipitation-v1 fields: {sorted(missing)}")
    print(f"[2/5] Loading {len(list(paths.ARCHIVE_RAW_DATA_DIR.glob('raw_data_*.json')))} Synoptic files...", flush=True)

    last_report = 0
    def progress(number, total, path):
        nonlocal last_report
        if number == 1 or number == total or number - last_report >= 25:
            print(f"      observations {number}/{total}: {path.name}", flush=True)
            last_report = number

    observations = load_observations(paths.ARCHIVE_RAW_DATA_DIR, progress=progress)
    labels = observations[["station_id", "time", "fuel_moisture", "obs_rh", "obs_wind_ms"]].rename(
        columns={"time": "target_time", "fuel_moisture": "source_target_fm",
                 "obs_rh": "source_target_rh", "obs_wind_ms": "source_target_wind_ms"})
    print(f"[3/5] Joining {len(frame):,} aligned rows to {len(labels):,} observations...", flush=True)
    frame["station_id"] = frame.station_id.astype(str)
    labels["station_id"] = labels.station_id.astype(str)
    enriched = frame.merge(labels, how="left", on=["station_id", "target_time"], validate="many_to_one")
    observed = enriched.target_mask.eq(1) & enriched.target_fm.notna()
    comparable = observed & enriched.source_target_fm.notna()
    mismatch = comparable & ~np.isclose(enriched.target_fm, enriched.source_target_fm, atol=1e-6)
    if mismatch.any():
        raise RuntimeError(f"Target provenance mismatch on {int(mismatch.sum())} rows; output was not written")
    missing_source = int((observed & enriched.source_target_fm.isna()).sum())
    if missing_source:
        raise RuntimeError(f"Missing source observation for {missing_source} scored rows; output was not written")
    for target_column, source_column in (
        ("target_rh", "source_target_rh"),
        ("target_wind_ms", "source_target_wind_ms"),
    ):
        if target_column in enriched:
            both = observed & enriched[target_column].notna() & enriched[source_column].notna()
            disagreement = both & ~np.isclose(enriched[target_column], enriched[source_column], atol=1e-6)
            if disagreement.any():
                raise RuntimeError(
                    f"Aligned {target_column} disagrees with its source observation on "
                    f"{int(disagreement.sum())} rows; output was not written"
                )
            enriched[target_column] = enriched[target_column].fillna(enriched[source_column])
        else:
            enriched[target_column] = enriched[source_column]
    enriched["target_match_age_minutes"] = (
        enriched.target_time - enriched.valid_time).abs().dt.total_seconds().div(60)
    enriched["target_rh_mask"] = (observed & enriched.target_rh.notna()).astype("int8")
    enriched["target_wind_mask"] = (observed & enriched.target_wind_ms.notna()).astype("int8")
    enriched = enriched.drop(columns=["source_target_fm", "source_target_rh", "source_target_wind_ms"])
    if (enriched.loc[observed, "target_match_age_minutes"] > 30.0001).any():
        raise RuntimeError("A scored target exceeds the 30-minute observation tolerance")
    print("[4/5] Validating seasonal and observed-weather coverage...", flush=True)
    summer = enriched.valid_time.dt.month.isin([6, 7, 8]) & observed
    report = {
        "source": str(args.source), "output": str(args.output),
        "source_sha256": sha256(args.source), "rows": int(len(enriched)),
        "observed_targets": int(observed.sum()), "summer_observed_targets": int(summer.sum()),
        "target_rh_coverage": float(enriched.loc[observed, "target_rh_mask"].mean()),
        "target_wind_coverage": float(enriched.loc[observed, "target_wind_mask"].mean()),
        "maximum_match_age_minutes": float(enriched.loc[observed, "target_match_age_minutes"].max()),
        "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
        "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
    }
    if report["summer_observed_targets"] < 1000:
        raise RuntimeError("Insufficient summer targets for V4")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    print(f"[5/5] Writing {args.output} atomically...", flush=True)
    enriched.to_csv(temporary, index=False)
    temporary.replace(args.output)
    report["output_sha256"] = sha256(args.output)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    manifest_tmp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(report, indent=2)); manifest_tmp.replace(args.manifest)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
