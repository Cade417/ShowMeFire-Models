"""
Offline evaluation of the seven-day outlook on TEST rows (held-out contiguous
periods AND held-out stations), same report shape as ensemble_fire_danger.evaluate
(named gates: pass / fail / deferred; overall_pass needs no "fail").

Weather and 10-hour fuel moisture are evaluated separately, by six-hour lead window
and by lead group (d1-3, d4-5, d6-7). For each variable: bias, MAE, p10-p90
interval coverage (raw and corrected) and MAE skill against persistence, a seasonal
baseline (train-row mean by valid month x six-hour bucket) and, for weather, NBM -
each on the rows where that baseline exists.

For each threshold (Elevated+, Critical+) and lead group: Brier (raw, calibrated,
climatology), expected calibration error, and a reliability table. Observed events
come from the worker's records; where observations cannot support a threshold the
gate is deferred, never passed. Weather verification does not give observed
fire-spread probabilities and nothing here claims it does.

Gates:
  min_test_days / min_test_stations / window_coverage   deferred until enough data
  interval_coverage_<v>_<g>     corrected coverage within COVERAGE_TOL of nominal
  beats_seasonal_<v>_<g>        corrected MAE < seasonal-baseline MAE
  beats_persistence_<v>_d1-3    corrected MAE < persistence MAE (days 1-3 only)
  events_<thr>_<g>              >= MIN_EVENTS test events, else deferred
  calibration_not_worse_<thr>_<g>  brier_cal <= brier_raw * (1 + TOL)
  critical_le_elevated          calibrated Critical never exceeds Elevated
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import contract, fit as fit_mod

MIN_TEST_DAYS = 10
MIN_TEST_STATIONS = 5
MIN_ROWS = 100
MIN_EVENTS = 20
COVERAGE_TOL = 0.10
TOL = 0.02


def _seasonal_table(train: pd.DataFrame, obs_col: str):
    d = train.dropna(subset=[obs_col])
    return d.groupby(["valid_month", "valid_bucket"])[obs_col].mean().to_dict(), float(d[obs_col].mean()) if len(d) else np.nan


def _seasonal_baseline(test: pd.DataFrame, table, overall) -> np.ndarray:
    keys = zip(test["valid_month"], test["valid_bucket"])
    return np.array([table.get(k, overall) for k in keys], dtype=float)


def _mae(a, b) -> float:
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


def _variable_block(test: pd.DataFrame, cols: Dict[str, str], params: dict | None,
                    seasonal: np.ndarray) -> dict:
    need = [cols["center"], cols["lo"], cols["hi"], cols["obs"]]
    d = test.dropna(subset=need)
    n = len(d)
    if n == 0:
        return {"n": 0}
    c, lo, hi, y = (d[cols[k]].to_numpy(float) for k in ("center", "lo", "hi", "obs"))
    lo2, c2, hi2 = fit_mod.apply_interval(params, c, lo, hi)
    out = {
        "n": int(n),
        "bias_raw": float(np.mean(c - y)), "bias_corrected": float(np.mean(c2 - y)),
        "mae_raw": _mae(c, y), "mae_corrected": _mae(c2, y),
        "coverage_raw": float(np.mean((y >= lo) & (y <= hi))),
        "coverage_corrected": float(np.mean((y >= lo2) & (y <= hi2))),
        "mean_width_raw": float(np.mean(hi - lo)), "mean_width_corrected": float(np.mean(hi2 - lo2)),
        "baselines": {},
    }
    base_arrays = {"seasonal": pd.Series(seasonal, index=test.index).loc[d.index].to_numpy(float)}
    for name in ("persist", "nbm"):
        col = cols.get(name)
        if col and col in d.columns:
            base_arrays["persistence" if name == "persist" else "nbm"] = d[col].to_numpy(float)
    for name, b in base_arrays.items():
        ok = np.isfinite(b)
        if ok.sum() < MIN_ROWS:
            continue
        mb, mf = _mae(b[ok], y[ok]), _mae(c2[ok], y[ok])
        out["baselines"][name] = {"n": int(ok.sum()), "mae_baseline": mb, "mae_corrected": mf,
                                  "skill": float(1 - mf / mb) if mb > 0 else float("nan")}
    return out


def _reliability(p, y, bins: int = 10) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        mp, my = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(mp - my)
        table.append({"bin": b, "n": int(m.sum()), "mean_agreement": mp, "observed_freq": my})
    return {"ece": float(ece), "table": table}


def _threshold_block(test: pd.DataFrame, group: str, thr: str, calibration: dict, p_cal: np.ndarray) -> dict:
    pcol, ycol = fit_mod.THRESHOLDS[thr]
    d = test.dropna(subset=[pcol, ycol])
    if d.empty:
        return {"n": 0, "events": 0}
    p, y = d[pcol].to_numpy(float), d[ycol].to_numpy(float)
    pc = pd.Series(p_cal, index=test.index).loc[d.index].to_numpy(float)
    base = float(y.mean())
    raw_rel, cal_rel = _reliability(p, y), _reliability(pc, y)
    return {
        "n": int(len(d)), "events": int(y.sum()), "base_rate": base,
        "brier_raw": float(np.mean((p - y) ** 2)), "brier_cal": float(np.mean((pc - y) ** 2)),
        "brier_climo": float(np.mean((base - y) ** 2)),
        "ece_raw": raw_rel["ece"], "ece_cal": cal_rel["ece"],
        "reliability_raw": raw_rel["table"], "reliability_cal": cal_rel["table"],
        "calibration_method": calibration.get(thr, {}).get(group, {}).get("method", "identity"),
    }


def _gate(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def evaluate(df: pd.DataFrame, side: pd.Series, fitted: dict) -> dict:
    train = df[(side == "train") & df["qc_ok"]]
    test = df[(side == "test") & df["qc_ok"]]
    gates: List[dict] = []
    report: dict = {
        "family": "fire_outlook", "test_rows": int(len(test)), "train_rows": int(len(train)),
        "test_days": int(test["issue_day"].nunique()), "test_stations": int(test["station_id"].nunique()),
        "coverage": {
            "issue_days": sorted(df["issue_day"].unique())[:1] + sorted(df["issue_day"].unique())[-1:],
            "stations": int(df["station_id"].nunique()),
            "months": sorted(int(m) for m in df["valid_month"].unique()),
            "windows_present_test": sorted(int(w) for w in test["window_idx"].unique()),
        },
        "variables": {}, "thresholds": {}, "per_window": {},
    }
    gates.append(_gate("min_test_days", "pass" if report["test_days"] >= MIN_TEST_DAYS else "deferred",
                       f"{report['test_days']} test issue days (need {MIN_TEST_DAYS})"))
    gates.append(_gate("min_test_stations", "pass" if report["test_stations"] >= MIN_TEST_STATIONS else "deferred",
                       f"{report['test_stations']} held-out stations (need {MIN_TEST_STATIONS})"))
    n_win = len(report["coverage"]["windows_present_test"])
    gates.append(_gate("window_coverage", "pass" if n_win == contract.N_WINDOWS else "deferred",
                       f"{n_win}/{contract.N_WINDOWS} windows have test rows"))

    for name in fitted["interval"]:
        cols = contract.variable_columns(name)
        table, overall = _seasonal_table(train, cols["obs"])
        seasonal = _seasonal_baseline(test, table, overall)
        by_group, by_window = {}, {}
        for g in contract.LEAD_GROUPS:
            m = (test["lead_group"] == g).to_numpy()
            blk = _variable_block(test[m], cols, fitted["interval"][name][g], seasonal[m])
            by_group[g] = blk
            if blk["n"] < MIN_ROWS:
                for gname in ("interval_coverage", "beats_seasonal"):
                    gates.append(_gate(f"{gname}_{name}_{g}", "deferred", f"{blk['n']} rows (need {MIN_ROWS})"))
                if g == "d1-3":
                    gates.append(_gate(f"beats_persistence_{name}_d1-3", "deferred", f"{blk['n']} rows"))
                continue
            dev = abs(blk["coverage_corrected"] - contract.INTERVAL_NOMINAL)
            gates.append(_gate(f"interval_coverage_{name}_{g}", "pass" if dev <= COVERAGE_TOL else "fail",
                               f"corrected coverage {blk['coverage_corrected']:.3f} vs nominal {contract.INTERVAL_NOMINAL}"))
            sb = blk["baselines"].get("seasonal")
            gates.append(_gate(f"beats_seasonal_{name}_{g}",
                               "deferred" if not sb else ("pass" if sb["mae_corrected"] < sb["mae_baseline"] else "fail"),
                               "no seasonal baseline rows" if not sb else
                               f"MAE {sb['mae_corrected']:.3f} vs seasonal {sb['mae_baseline']:.3f}"))
            if g == "d1-3":
                pb = blk["baselines"].get("persistence")
                gates.append(_gate("beats_persistence_%s_d1-3" % name,
                                   "deferred" if not pb else ("pass" if pb["mae_corrected"] < pb["mae_baseline"] else "fail"),
                                   "no persistence baseline rows" if not pb else
                                   f"MAE {pb['mae_corrected']:.3f} vs persistence {pb['mae_baseline']:.3f}"))
        for w in range(contract.N_WINDOWS):
            m = (test["window_idx"] == w).to_numpy()
            if m.any():
                params = fitted["interval"][name][contract.lead_group(w)]
                blk = _variable_block(test[m], cols, params, seasonal[m])
                by_window[str(w)] = {k: blk[k] for k in ("n", "bias_corrected", "mae_corrected", "coverage_corrected") if k in blk}
        report["variables"][name] = by_group
        report["per_window"][name] = by_window

    crit_ok = True
    for thr in fit_mod.THRESHOLDS:
        report["thresholds"][thr] = {}
    for g in contract.LEAD_GROUPS:
        m = (test["lead_group"] == g).to_numpy()
        sub = test[m]
        if sub.empty:
            e_cal = c_cal = np.array([])
        else:
            e_cal, c_cal = fit_mod.apply_agreement(fitted["agreement"], g, sub["agreement_elevated"], sub["agreement_critical"])
        crit_ok &= bool(np.all(c_cal <= e_cal + 1e-9))
        for thr, cal in (("elevated", e_cal), ("critical", c_cal)):
            blk = _threshold_block(sub, g, thr, fitted["agreement"], pd.Series(cal, index=sub.index) if len(sub) else np.array([]))
            report["thresholds"][thr][g] = blk
            if blk.get("events", 0) < MIN_EVENTS or blk.get("n", 0) - blk.get("events", 0) < MIN_EVENTS:
                gates.append(_gate(f"events_{thr}_{g}", "deferred",
                                   f"{blk.get('events', 0)} events of {blk.get('n', 0)} rows (need {MIN_EVENTS} each way)"))
                continue
            gates.append(_gate(f"events_{thr}_{g}", "pass", f"{blk['events']} events"))
            gates.append(_gate(f"calibration_not_worse_{thr}_{g}",
                               "pass" if blk["brier_cal"] <= blk["brier_raw"] * (1 + TOL) else "fail",
                               f"Brier cal {blk['brier_cal']:.4f} vs raw {blk['brier_raw']:.4f}"))
    gates.append(_gate("critical_le_elevated", "pass" if crit_ok else "fail",
                       "calibrated Critical agreement never exceeds Elevated" if crit_ok else "violation after calibration"))

    report["gates"] = gates
    report["overall_pass"] = not any(g["status"] == "fail" for g in gates)
    report["public_activation"] = ("Not implied. Plan gates still require the end-to-end pilot, "
                                   "a 30-day staff shadow pilot, and review of seasonal limits.")
    return report
