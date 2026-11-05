"""
Offline evaluation of an ensemble_fire_danger calibration on the TEST split,
with named pass / fail / deferred gates (same report shape as
fire_weather_index/evaluate.py and risk_fusion's evaluate, so
models/register.py and the lifecycle docs treat it the same way).

Per track x flavour (point / neighborhood) x category k, on test rows:
  brier_raw, brier_cal        Brier score of raw vs calibrated probability
  brier_det                   Brier of the deterministic HRRR-control rule
                              (0/1) - "what the public map already does"
  bss_vs_climo, bss_vs_det    skill scores of the calibrated probability
  reliability_ece             expected calibration error, 10 equal bins
  roc_auc                     discrimination (calibration-invariant)
Categorical (point, calibrated, tau_k): CSI and frequency bias per k for
">= k", vs the same for the deterministic control.

Gates (per track; the report's overall_pass needs no "fail"):
  min_days                 >= MIN_TEST_DAYS test days, else deferred
  events_<k>               >= fit.MIN_EVENTS test events, else deferred
                           (per k - Extreme is expected to stay deferred
                           for a long time; that is not a failure)
  calibration_not_worse_<k>  brier_cal <= brier_raw * (1 + TOL) on
                           neighborhood probabilities, when events suffice
  skill_vs_deterministic_<k> bss_vs_det > 0 on neighborhood probs for
                           k <= 2 (Moderate/Elevated) when events suffice
  member_coverage          mean members per day >= config min_members
                           (members track) / >= 1 source (ensprod)

Track comparison: mean neighborhood Brier (calibrated) over the k whose
events gate passed in BOTH tracks; the lower one is `recommended_primary`.
That is advice for SMF_ENSEMBLE_FD_PRIMARY, never applied automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402

from . import contract, fit as fit_mod, members  # noqa: E402

MODEL_FAMILY = "ensemble_fire_danger"
MIN_TEST_DAYS = 10
TOL = 0.02


def _brier(p, y):
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def _ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    table = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            gap = abs(float(p[mask].mean()) - float(y[mask].mean()))
            total += mask.mean() * gap
            table.append({"bin": b, "n": int(mask.sum()), "mean_forecast": round(float(p[mask].mean()), 4),
                          "observed_frequency": round(float(y[mask].mean()), 4)})
    return float(total), table


def _auc(p, y):
    if y.all() or not y.any():
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def _csi_bias(forecast, y):
    hits = int(np.sum(forecast & y))
    fa = int(np.sum(forecast & ~y))
    miss = int(np.sum(~forecast & y))
    csi = hits / (hits + fa + miss) if hits + fa + miss else None
    bias = (hits + fa) / (hits + miss) if hits + miss else None
    return csi, bias


def evaluate(panel: pd.DataFrame, split: Dict[str, str], fitted: dict, config: dict) -> dict:
    test = panel[panel["day"].map(split).eq("test")]
    calibration = fitted["calibration"]["tracks"]
    thresholds = fitted["categorical_thresholds"]["tracks"]
    report = {"model_family": MODEL_FAMILY, "advisory_only": True, "test_days": int(test["day"].nunique()),
              "train_days": int(panel["day"].map(split).eq("train").groupby(panel["day"]).first().sum()),
              "tracks": {}, "gates": []}
    comparable: Dict[str, Dict[int, float]] = {}
    for track, rows in test.groupby("track"):
        track_report = {"categories": {}}
        gates = []
        n_days = int(rows["day"].nunique())
        gates.append({"name": f"{track}:min_days", "status": "pass" if n_days >= MIN_TEST_DAYS else "deferred",
                      "value": n_days, "threshold": MIN_TEST_DAYS})
        mean_members = float(rows.groupby("day")["member_count"].first().mean())
        need = int(config.get("min_members", 5)) if track == "members" else 1
        gates.append({"name": f"{track}:member_coverage", "status": "pass" if mean_members >= need else "fail",
                      "value": round(mean_members, 2), "threshold": need})
        comparable[track] = {}
        for k, krows in rows.groupby("k"):
            k = int(k)
            entry = {}
            enough = int(krows["obs_nbhd"].sum()) >= fit_mod.MIN_EVENTS and n_days >= MIN_TEST_DAYS
            for flavour, (p_col, y_col) in fit_mod.FLAVOURS.items():
                p_raw = krows[p_col].to_numpy("float64")
                y = krows[y_col].to_numpy(bool)
                knots = (calibration.get(track, {}).get(flavour) or {}).get(str(k))
                p_cal = np.interp(p_raw, knots["x"], knots["y"]) if knots else p_raw
                det = krows["det_point" if flavour == "point" else "det_nbhd"].to_numpy(bool).astype(float)
                climo = float(y.mean()) if len(y) else float("nan")
                b_raw, b_cal, b_det = _brier(p_raw, y), _brier(p_cal, y), _brier(det, y)
                b_climo = _brier(np.full(len(y), climo), y)
                ece, table = _ece(p_cal, y)
                entry[flavour] = {
                    "rows": int(len(y)), "events": int(y.sum()), "base_rate": round(climo, 5),
                    "brier_raw": round(b_raw, 6), "brier_cal": round(b_cal, 6), "brier_det": round(b_det, 6),
                    "bss_vs_climo": round(1 - b_cal / b_climo, 4) if b_climo > 0 else None,
                    "bss_vs_det": round(1 - b_cal / b_det, 4) if b_det > 0 else None,
                    "reliability_ece": round(ece, 5), "reliability_table": table,
                    "roc_auc": _auc(p_cal, y), "calibrated": bool(knots),
                }
            # categorical (point) with tau_k
            knots = (calibration.get(track, {}).get("point") or {}).get(str(k))
            p_cal = np.interp(krows["p_point"].to_numpy("float64"), knots["x"], knots["y"]) if knots \
                else krows["p_point"].to_numpy("float64")
            tau = float(thresholds.get(track, {}).get(str(k), config.get("default_categorical_threshold", 0.5)))
            y = krows["obs_point"].to_numpy(bool)
            csi, bias = _csi_bias(p_cal >= tau, y)
            det_csi, det_bias = _csi_bias(krows["det_point"].to_numpy(bool), y)
            entry["categorical"] = {"threshold": tau, "csi": csi, "bias": bias,
                                    "deterministic_csi": det_csi, "deterministic_bias": det_bias}
            track_report["categories"][str(k)] = entry

            nb = entry["neighborhood"]
            gates.append({"name": f"{track}:events_{k}", "status": "pass" if enough else "deferred",
                          "value": nb["events"], "threshold": fit_mod.MIN_EVENTS})
            if enough:
                ok = nb["brier_cal"] <= nb["brier_raw"] * (1 + TOL)
                gates.append({"name": f"{track}:calibration_not_worse_{k}", "status": "pass" if ok else "fail",
                              "value": nb["brier_cal"], "threshold": round(nb["brier_raw"] * (1 + TOL), 6)})
                if k <= 2 and nb["bss_vs_det"] is not None:
                    gates.append({"name": f"{track}:skill_vs_deterministic_{k}",
                                  "status": "pass" if nb["bss_vs_det"] > 0 else "fail",
                                  "value": nb["bss_vs_det"], "threshold": 0.0})
                comparable[track][k] = nb["brier_cal"]
        track_report["gates"] = gates
        report["tracks"][track] = track_report
        report["gates"].extend(gates)

    shared = sorted(set.intersection(*(set(v) for v in comparable.values()))) if len(comparable) == 2 else []
    if shared:
        scores = {t: float(np.mean([comparable[t][k] for k in shared])) for t in comparable}
        report["track_comparison"] = {"categories_compared": shared, "mean_neighborhood_brier": scores,
                                      "recommended_primary": min(scores, key=scores.get)}
    else:
        report["track_comparison"] = {"categories_compared": [], "recommended_primary": None,
                                      "note": "not enough shared test events in both tracks yet"}
    if test.empty:
        report["gates"].append({"name": "test_split_nonempty", "status": "deferred", "value": 0,
                                "threshold": MIN_TEST_DAYS})
    report["overall_pass"] = not any(g["status"] == "fail" for g in report["gates"])
    report["split_sha256"] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    return report


def main(argv=None) -> int:
    from . import model_bundle

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-dir", type=Path, default=paths.ENSEMBLE_FD_CANDIDATE_DIR)
    args = parser.parse_args(argv)
    panel = pd.read_parquet(paths.ENSEMBLE_FD_PANEL)
    split = contract.read_manifest()["days"]
    fitted = model_bundle.load(args.candidate_dir)
    report = evaluate(panel, split, fitted, members.load_config())
    paths.ENSEMBLE_FD_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"overall_pass": report["overall_pass"],
                      "gates": {g["name"]: g["status"] for g in report["gates"]},
                      "track_comparison": report["track_comparison"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
