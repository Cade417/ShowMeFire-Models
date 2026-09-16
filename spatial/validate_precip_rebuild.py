"""Compare a versioned precipitation rebuild with the previous aligned data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

ROW_KEY = ["run_id", "station_id", "valid_time", "lead_hour"]
PRECIP_COLUMNS = {
    "hrrr_precip_mm", "hrrr_precip_accum_mm", "hrrr_precip_increment_mm",
    "precip_interval_hours", "precip_reset_flag", "precip_partial_window_flag",
    "precip_available",
}


def compare(previous_path: Path, rebuilt_path: Path) -> dict:
    previous = pd.read_csv(previous_path); rebuilt = pd.read_csv(rebuilt_path)
    for frame in (previous, rebuilt):
        frame["run_id"] = frame.run_id.astype(str)
    old = previous.set_index(ROW_KEY).sort_index(); new = rebuilt.set_index(ROW_KEY).sort_index()
    common_index = old.index.intersection(new.index)
    missing = old.index.difference(new.index); added = new.index.difference(old.index)
    comparable = sorted((set(old.columns) & set(new.columns)) - PRECIP_COLUMNS)
    mismatches = {}
    for column in comparable:
        left, right = old.loc[common_index, column], new.loc[common_index, column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            unequal = ~np.isclose(pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce"),
                                  rtol=1e-6, atol=1e-6, equal_nan=True)
        else:
            unequal = ~(left.fillna("<missing>").astype(str).to_numpy() == right.fillna("<missing>").astype(str).to_numpy())
        count = int(np.asarray(unequal).sum())
        if count: mismatches[column] = count
    nonzero = int((rebuilt.hrrr_precip_accum_mm.fillna(0) > 0).sum())
    report = {
        "previous": str(previous_path), "rebuilt": str(rebuilt_path),
        "previous_rows": len(previous), "rebuilt_rows": len(rebuilt), "common_rows": len(common_index),
        "missing_row_keys": len(missing), "added_row_keys": len(added),
        "nonprecipitation_mismatches": mismatches,
        "nonzero_precipitation_rows": nonzero,
    }
    report["pass"] = bool(missing.empty and added.empty and not mismatches and nonzero > 0)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, default=paths.ALIGNED_DIR / "station_leads.csv")
    parser.add_argument("--rebuilt", type=Path, default=paths.PRECIP_ALIGNED_DATASET)
    parser.add_argument("--output", type=Path, default=paths.REPORTS_DIR / "precipitation_rebuild_validation.json")
    args = parser.parse_args(); report = compare(args.previous, args.rebuilt)
    args.output.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__": main()
