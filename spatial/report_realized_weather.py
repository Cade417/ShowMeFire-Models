"""Held-out association report for HRRR error, realized RTMA, and FM change."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths


def _summary(values):
    values = np.asarray(values, "float64")
    return {"count": int(values.size), "mean": float(np.mean(values)) if values.size else None,
            "mae": float(np.mean(np.abs(values))) if values.size else None,
            "p90_abs": float(np.quantile(np.abs(values), .9)) if values.size else None}


def build_report(static_bundle: Path, output=None):
    files = sorted((paths.ALIGNED_DIR / "spatial_tensors").glob("spatial_*.npz")); cutoff = int(len(files) * .8)
    held_out = files[cutoff:]; by_lead = defaultdict(lambda: defaultdict(list)); transitions = {"rapid_rh": [], "other": []}
    with xr.open_dataset(static_bundle) as static:
        groups = {name: np.asarray(static[name].values) for name in ("nlcd_land_cover", "fbfm40", "fuel_vegetation_type", "canopy_cover_pct", "elevation_m") if name in static}
    grouped = defaultdict(lambda: defaultdict(list))
    for path in held_out:
        with np.load(path) as item:
            hrrr = item["hrrr_forecast"]; realized = item["realized_rtma_future"][[3 + index for index in range(12)]]
            metadata = json.loads(str(item["metadata"])); mask = (item["station_holdout_mask"] + item["region_holdout_mask"])[:, 0] > 0
            fm_change = item["target"][:, 0] - item["current_fm_state"][0]
            for index, lead in enumerate(metadata["hrrr_leads"]):
                for channel, name in enumerate(("temperature_c", "rh", "wind_speed")):
                    error = hrrr[index, channel] - realized[index, channel]
                    by_lead[int(lead)][name].extend(error[mask[index]].tolist())
                rapid = np.abs(realized[index, 1] - (realized[index - 1, 1] if index else item["antecedent_rtma"][-1, 1])) >= 20
                transitions["rapid_rh" if np.any(rapid & mask[index]) else "other"].extend(fm_change[index][mask[index]].tolist())
                divergence = np.sqrt(np.square(hrrr[index, :3] - realized[index, :3]).mean(axis=0))
                for group_name, values in groups.items():
                    for category in np.unique(values[mask[index]]):
                        selection = mask[index] & (values == category)
                        grouped[group_name][str(category)].extend(divergence[selection].tolist())
    report = {"title": "Realized-weather associations and forecast-error patterns",
        "interpretation": "These are held-out learned associations and forecast-error patterns, not proof of physical causation.",
        "held_out_files": len(held_out), "by_lead": {str(lead): {name: _summary(values) for name, values in channels.items()} for lead, channels in by_lead.items()},
        "fuel_moisture_change_composites": {name: _summary(values) for name, values in transitions.items()},
        "weather_divergence_by_static_group": {name: {category: _summary(values) for category, values in categories.items()} for name, categories in grouped.items()},
        "importance_contract": {"method": "Use frozen held-out grouped ablations/permutation; never training data.",
                                "groups": ["weather", "antecedent_history", "terrain", "land_cover", "fuels"],
                                "static_ablation_reports": "reports/spatial_<feature-set>_*.json"}}
    output = output or paths.REPORTS_DIR / "realized_weather_associations.json"
    output.write_text(json.dumps(report, indent=2)); print(output); return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--static-bundle", required=True, type=Path); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); build_report(args.static_bundle, args.output)
