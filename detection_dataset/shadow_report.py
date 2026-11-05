"""How the shadow-scored models did against reviewed detections. Informational: nothing acts on this report.

    python -m detection_dataset.shadow_report [--export-dir DIR]

Uses only reviewed detections from the RANDOM stream (the one unbiased sample), scored by each model's INITIAL score (what
was known at first scoring), with group-resampled intervals. v1 is shown cross-calibrated so its probability metrics are
comparable; v2 updated scores are compared with initial scores on the same detections, never swapped in for them.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from detection_dataset import groups as grp, labels, review_labels
from detection_dataset.confidence_v2 import metrics as M
from detection_dataset.confidence_v2.train import cross_calibrate
from pipelines.experiment_log import append_experiment

MODEL_TYPE = "detection_confidence_v2_shadow"


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def build_report(export: dict, seed: int = 0) -> dict:
    out: dict = {"generated": datetime.now().isoformat(timespec="seconds"),
                 "note": "random-stream reviews only; informational, no decisions are made from this"}
    scores, sel, det = export.get("score_history"), export.get("review_selection"), export["detections"]
    final = review_labels.review_final(export)
    if scores is None or scores.empty or sel is None or sel.empty or final.empty:
        out["status"] = "nothing_to_report"
        out["message"] = "No score history or no reviewed random-stream detections in this export yet."
        return out
    settled = final[final["review_label"].notna()].set_index("event_id")["review_label"]
    random_ids = set(sel.loc[sel["stream"] == "random", "event_id"].astype(int))
    protected = set(sel.loc[(sel["stream"] == "random") & (sel["protected"] == 1), "event_id"].astype(int))
    pool = settled[settled.index.isin(random_ids)]
    out["reviewed_random_stream"], out["protected_subset"] = int(len(pool)), int(pool.index.isin(protected).sum())
    out["prevalence"] = round(float(pool.mean()), 4) if len(pool) else None

    d = det.assign(id=pd.to_numeric(det["id"], errors="coerce")).set_index("id")
    ev = d.loc[d.index.intersection(pool.index)]
    g = grp.assign_groups(ev.reset_index())["group_id"]
    gmap = dict(zip(ev.index, g.to_numpy()))
    out["models"] = {}
    for (name, version), table in scores[scores["mode"] == "initial"].groupby(["model_name", "model_version"]):
        s = table.sort_values("id").drop_duplicates("event_id", keep="last").set_index("event_id")["score"]
        ids = [e for e in pool.index if e in s.index]
        if not ids:
            continue
        y = pool.loc[ids].to_numpy(dtype=int)
        p = s.loc[ids].to_numpy(dtype=float)
        gg = np.array([gmap.get(e, e) for e in ids])
        entry = {"coverage": f"{len(ids)} of {len(pool)} reviewed detections scored",
                 "random_stream": M.evaluate(y, p, gg, seed=seed)}
        if name.startswith("v1"):
            cal = cross_calibrate(_logit(p), y, gg, seed)
            ok = ~np.isnan(cal)
            entry["v1_cross_calibrated"] = M.evaluate(y[ok], cal[ok], gg[ok], seed=seed) if ok.any() else {}
        prot = [i for i, e in enumerate(ids) if e in protected]
        entry["protected_only"] = M.evaluate(y[prot], p[prot], gg[prot], seed=seed) if prot else {"note": "no protected reviewed detections scored"}
        upd = scores[(scores["mode"] == "updated") & (scores["model_name"] == name) & (scores["model_version"] == version)]
        if len(upd):
            u = upd.sort_values("id").drop_duplicates("event_id", keep="last").set_index("event_id")["score"]
            both = [e for e in ids if e in u.index]
            if both:
                yb = pool.loc[both].to_numpy(dtype=int)
                entry["initial_vs_updated_same_detections"] = {
                    "n": len(both), "initial": M.evaluate(yb, s.loc[both].to_numpy(float), None), "updated": M.evaluate(yb, u.loc[both].to_numpy(float), None)}
        out["models"][f"{name}@{version}"] = entry
    if len(pool):
        out["prevalence_only"] = M.evaluate(pool.to_numpy(dtype=int), np.full(len(pool), pool.mean()), None)
    out["status"] = "reported"
    return out


def _fmt(m: dict) -> str:
    if not m:
        return "n/a"
    if "note" in m:
        return f"{m['note']} (n={m.get('n')})"
    ci = f" [{m['average_precision_ci95'][0]}-{m['average_precision_ci95'][1]}]" if "average_precision_ci95" in m else ""
    return f"AP {m.get('average_precision')}{ci}, AUC {m.get('auc')}, Brier {m.get('brier')} (n={m['n']}, pos={m['positives']}, groups={m.get('n_groups')})"


def markdown(r: dict) -> str:
    lines = [f"# Shadow scoring report ({r['generated']})", "", r["note"], ""]
    if r["status"] != "reported":
        return "\n".join(lines + [r["message"], ""]) + "\n"
    lines += [f"Reviewed random-stream detections: **{r['reviewed_random_stream']}** ({r['protected_subset']} protected). Prevalence {r['prevalence']}.", ""]
    for name, m in r["models"].items():
        lines += [f"## {name}", "", m["coverage"], "", f"- random stream: {_fmt(m['random_stream'])}"]
        if m.get("v1_cross_calibrated"):
            lines.append(f"- v1 cross-calibrated: {_fmt(m['v1_cross_calibrated'])}")
        lines.append(f"- protected only: {_fmt(m['protected_only'])}")
        iu = m.get("initial_vs_updated_same_detections")
        if iu:
            lines += [f"- initial vs updated on the same {iu['n']} detections:", f"  - initial: {_fmt(iu['initial'])}", f"  - updated: {_fmt(iu['updated'])}"]
        lines.append("")
    if r.get("prevalence_only"):
        lines += ["## Prevalence-only baseline", "", _fmt(r["prevalence_only"]), ""]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    report = build_report(labels.load_export(a.export_dir), a.seed)
    stamp = datetime.now().strftime("%Y%m%d")
    paths.REPORTS_DIR.joinpath(f"detection_shadow_report_{stamp}.md").write_text(markdown(report), encoding="utf-8")
    paths.DETECTION_CONFIDENCE_V2_DIR.mkdir(parents=True, exist_ok=True)
    (paths.DETECTION_CONFIDENCE_V2_DIR / f"shadow_report_{stamp}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    metrics = {"reviewed": report.get("reviewed_random_stream", 0)}
    append_experiment(MODEL_TYPE, metrics, params={"seed": a.seed})
    print(markdown(report))


if __name__ == "__main__":
    main()
