"""
Build the calibration panel for ensemble_fire_danger.

Two stages, so recalibration experiments never have to re-score members:

1. per day (expensive, cached): for every captured forecast day with an
   HRRR control, score Track M (real members) and Track E (synthetic draws
   from ensprod) with the SAME tracks.py the live product runs, compute the
   RTMA-derived observed peak category (labels.py), and save one
   ENSEMBLE_FD_DAYS_DIR/<day>.npz with member peaks, weights, the
   deterministic HRRR-control peak, the observation, and ensprod extras
   (JFWPRB, spread).
2. panel (cheap): turn each day file into raw point / neighborhood
   exceedance probabilities (core.exceedance_probability + nan_gaussian,
   same radius/sigma as live) and matching observations, over Missouri
   cells thinned every THIN-th row/column (neighbouring 3 km cells are
   nearly duplicate samples), and write ENSEMBLE_FD_PANEL (parquet, long
   format: one row per day x track x k x cell).

Neighborhood verification uses the matching neighborhood observation
("did the category occur anywhere within radius of the cell") - the
standard way HREF/SPC-style neighborhood probabilities are verified, and
what the maps' "within ~20 km" wording promises.

Training-vs-live differences (documented in docs/ensemble_fire_danger.md):
no FM anchoring (the operational FM hand-off does not exist historically,
so members use raw XGBoost FM - as do the labels), no seasonal dampening,
no snow mask.

Usage:
    python -m ensemble_fire_danger.build_panel               # score new days + rebuild panel
    python -m ensemble_fire_danger.build_panel --rescore     # re-score every day (after a tracks/core change)
    python -m ensemble_fire_danger.build_panel --panel-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import maximum_filter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402

from . import capture, core, labels, members, tracks  # noqa: E402
from .regrid import Regridder, cell_size_km  # noqa: E402

logger = logging.getLogger(__name__)

THIN = 4
COUNTY_CELLS_API = REPO_ROOT / "risk_fusion" / "county_cells_api.json"


# --- shared helpers (also used by fit/evaluate) ------------------------------

def rule_categorize():
    from risk_fusion.rule_uncertainty import MISSING_CATEGORY, category_vectorized, check_parity
    from spatial.rule_contract import RULE_SPEC, category

    thresholds = RULE_SPEC["thresholds"]

    def scalar(fm, rh, wind, missing_category=None):
        value = category(fm, rh, wind)
        return missing_category if value is None else value

    check_parity(scalar, thresholds)
    return lambda fm, rh, wind: category_vectorized(fm, rh, wind, thresholds)


def load_fm_booster(path: Optional[str] = None):
    """XGBoost fuel-moisture model. For real calibration use the SAME file
    the server serves as stable fuel_moisture (pass --fm-model or set
    SMF_ENSEMBLE_FM_MODEL); falls back to this repo's registry stable, then beta."""
    import xgboost as xgb
    from models.versioning import load_active_model_path

    path = path or os.getenv("SMF_ENSEMBLE_FM_MODEL")
    if not path:
        for channel in ("stable", "beta"):
            try:
                path = str(load_active_model_path("fuel_moisture", channel=channel))
                break
            except FileNotFoundError:
                continue
    if not path:
        raise SystemExit("no fuel_moisture model: pass --fm-model or set SMF_ENSEMBLE_FM_MODEL")
    booster = xgb.Booster()
    booster.load_model(path)
    return booster, path


def missouri_cells(grid_shape) -> np.ndarray:
    """(n, 2) row/col of Missouri cells on the api grid (county_cells_api.json)."""
    data = json.loads(COUNTY_CELLS_API.read_text(encoding="utf-8"))
    if list(data["grid_shape"]) != list(grid_shape):
        raise RuntimeError(f"county_cells_api grid {data['grid_shape']} != ensemble grid {list(grid_shape)}")
    return np.array([[int(v) for v in key.split(",")] for key in data["cell_to_fips"]], dtype="int32")


class RegridderFactory:
    def __init__(self, dst_lat, dst_lon):
        self.dst_lat, self.dst_lon, self.cache = dst_lat, dst_lon, {}

    def __call__(self, src_lat, src_lon):
        key = (src_lat.shape, float(src_lat[0, 0]), float(src_lon[0, 0]), float(src_lat[-1, -1]))
        if key not in self.cache:
            self.cache[key] = Regridder(src_lat, src_lon, self.dst_lat, self.dst_lon,
                                        cache_dir=paths.CACHE_ENSEMBLE_DIR / "regrid")
        return self.cache[key]


# --- stage 1: per-day scoring ------------------------------------------------

def score_day(day: date, records: Dict[str, dict], config: dict, booster, feature_names, categorize) -> Optional[Path]:
    fetched = {mid: Path(r["path"]) for mid, r in records.items()
               if r.get("status") == "fetched" and Path(r.get("path", "")).exists()}
    if "hrrr_0" not in fetched:
        logger.info("%s: no HRRR control member - skipped", day)
        return None
    anchor = capture.anchor_for(day, config)
    window = config["window"]
    valid_times = [pd.Timestamp(anchor + timedelta(hours=lead), tz="UTC")
                   for lead in range(int(window["valid_lead_start"]), int(window["valid_lead_end"]) + 1)]

    with xr.open_dataset(fetched["hrrr_0"]) as ds:
        lat, lon = ds["latitude"].values, ds["longitude"].values
    rows, cols = members.api_bounds_slices(lat, lon)
    dst_lat, dst_lon = lat[rows, cols], lon[rows, cols]
    make_regridder = RegridderFactory(dst_lat, dst_lon)
    fm_clip = tuple(config.get("fm_clip", [1.0, 40.0]))
    member_cfg = {m["id"]: m for m in config["members"]}

    fields, weights, wind_factors = {}, {}, {}
    for mid, path in fetched.items():
        if mid not in member_cfg:
            continue
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        fields[mid] = tracks.regridded_member_fields(ds, make_regridder(ds["latitude"].values, ds["longitude"].values))
        weights[mid] = float(member_cfg[mid].get("weight", 1.0))
        wind_factors[mid] = float(config["sources"][member_cfg[mid]["source"]].get("wind_factor", 0.8))

    out: Dict[str, np.ndarray] = {"lat": dst_lat, "lon": dst_lon}
    track_m = tracks.track_members(fields, weights, wind_factors, "hrrr_0", booster, feature_names, valid_times,
                                   categorize, None, None, None, None, fm_clip)
    out["members__peaks"] = track_m["peaks"]
    out["members__weights"] = np.asarray(track_m["weights"], dtype="float32")
    out["members__ids"] = np.asarray(track_m["member_ids"])
    out["hrrr_control_peak"] = track_m["peaks"][track_m["member_ids"].index("hrrr_0")]

    ens_ids = [m["id"] for m in config.get("ensprod_members", []) if m["id"] in fetched]
    if ens_ids:
        ens_fields, ens_weights = {}, {}
        for mid in ens_ids:
            with xr.open_dataset(fetched[mid]) as ds:
                ds = ds.load()
            ens_fields[mid] = tracks.regridded_ensprod_fields(ds, make_regridder(ds["latitude"].values,
                                                                                  ds["longitude"].values))
            ens_weights[mid] = next(float(m.get("weight", 1.0)) for m in config["ensprod_members"] if m["id"] == mid)
        track_e = tracks.track_synthetic(ens_fields, ens_weights, config.get("synthetic", {}), booster,
                                         feature_names, valid_times, categorize, None, None, None, None, fm_clip)
        out["ensprod__peaks"] = track_e["peaks"]
        out["ensprod__weights"] = np.asarray(track_e["weights"], dtype="float32")
        out["ensprod__sources"] = np.asarray(track_e["sources"])
        for extra in ("jfwprb", "pwind_10p3", "spread_rh_proxy"):
            if extra in track_e:
                out[f"ensprod__{extra}"] = np.asarray(track_e[extra], dtype="float32")

    observed = labels.observed_peak(valid_times, make_regridder, booster, feature_names, categorize,
                                    fields["hrrr_0"]["precip_mm"])
    out["observed_peak"] = observed["peak"]
    out["observed_min_rh"] = observed["min_rh"].astype("float32")
    out["observed_max_wind_kts"] = observed["max_wind_kts"].astype("float32")

    paths.ENSEMBLE_FD_DAYS_DIR.mkdir(parents=True, exist_ok=True)
    target = paths.ENSEMBLE_FD_DAYS_DIR / f"{day.isoformat()}.npz"
    tmp = target.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(target)
    return target


# --- stage 2: panel ----------------------------------------------------------

def day_rows(day: str, data, config: dict, cells: np.ndarray) -> List[pd.DataFrame]:
    lat, lon = data["lat"], data["lon"]
    radius = float(config.get("neighborhood_radius_km", 20.0)) / max(cell_size_km(lat, lon), 0.1)
    sigma = float(config.get("smooth_sigma_cells", 1.5))
    footprint = core.disk_footprint(radius)
    thin = cells[(cells[:, 0] % THIN == 0) & (cells[:, 1] % THIN == 0)]
    r, c = thin[:, 0], thin[:, 1]
    obs = data["observed_peak"]
    control = data["hrrr_control_peak"]
    frames = []
    for track in ("members", "ensprod"):
        if f"{track}__peaks" not in data:
            continue
        peaks, weights = data[f"{track}__peaks"], data[f"{track}__weights"]
        member_count = int(peaks.shape[0])
        for k in core.CATEGORY_IDS:
            p_point = core.nan_gaussian(core.exceedance_probability(peaks, weights, k, 0.0), sigma)
            p_nbhd = core.nan_gaussian(core.exceedance_probability(peaks, weights, k, radius), sigma)
            obs_k = (obs >= k) & (obs != core.MISSING)
            obs_nbhd = maximum_filter(obs_k.astype("uint8"), footprint=footprint, mode="nearest") > 0
            det_k = (control >= k) & (control != core.MISSING)
            det_nbhd = maximum_filter(det_k.astype("uint8"), footprint=footprint, mode="nearest") > 0
            frame = pd.DataFrame({
                "day": day, "track": track, "k": k, "row": r, "col": c,
                "p_point": p_point[r, c].astype("float32"), "p_nbhd": p_nbhd[r, c].astype("float32"),
                "obs_point": obs_k[r, c], "obs_nbhd": obs_nbhd[r, c],
                "det_point": det_k[r, c], "det_nbhd": det_nbhd[r, c],
                "member_count": member_count,
            })
            if track == "ensprod" and "ensprod__jfwprb" in data:
                frame["jfwprb"] = data["ensprod__jfwprb"][r, c]
            frames.append(frame[np.isfinite(frame["p_point"]) & np.isfinite(frame["p_nbhd"])])
    return frames


def build_panel(config: Optional[dict] = None) -> pd.DataFrame:
    config = config or members.load_config()
    frames = []
    cells = None
    for path in sorted(paths.ENSEMBLE_FD_DAYS_DIR.glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path, allow_pickle=False) as data:
            data = dict(data)
        if cells is None:
            cells = missouri_cells(data["lat"].shape)
        frames.extend(day_rows(path.stem, data, config, cells))
    if not frames:
        raise SystemExit("no scored days - run capture + build_panel first")
    panel = pd.concat(frames, ignore_index=True)
    paths.ENSEMBLE_FD_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(paths.ENSEMBLE_FD_PANEL, index=False)
    return panel


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rescore", action="store_true", help="re-score days that already have a .npz")
    parser.add_argument("--panel-only", action="store_true", help="skip per-day scoring")
    parser.add_argument("--fm-model", help="XGBoost fuel-moisture model file (use the server's stable one)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    config = members.load_config()

    if not args.panel_only:
        booster, fm_path = load_fm_booster(args.fm_model)
        logger.info("FM model: %s", fm_path)
        categorize = rule_categorize()
        manifest = capture.read_manifest()
        for key, records in sorted(manifest["days"].items()):
            target = paths.ENSEMBLE_FD_DAYS_DIR / f"{key}.npz"
            if target.exists() and not args.rescore:
                continue
            try:
                result = score_day(date.fromisoformat(key), records, config, booster, list(booster.feature_names),
                                   categorize)
                logger.info("%s scored -> %s", key, result)
            except Exception as error:
                logger.exception("%s scoring failed: %s", key, error)
    panel = build_panel(config)
    summary = panel.groupby(["track", "k"]).agg(rows=("p_point", "size"), days=("day", "nunique"),
                                                events_point=("obs_point", "sum"), events_nbhd=("obs_nbhd", "sum"))
    print(summary.to_string())
    print(f"panel -> {paths.ENSEMBLE_FD_PANEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
