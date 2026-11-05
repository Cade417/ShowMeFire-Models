"""Build the detection-confidence dataset end to end.

    python -m detection_dataset.build [--neris-radius-km 2] [--neris-path X.shp]
                                      [--export-dir DIR] [--no-ml]

Writes detection_dataset_<date>.{parquet,csv}, columns.json (data dictionary)
and a markdown report. Also appends an experiment row so the pipeline
dashboard shows the run.
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
from detection_dataset import coverage, derive, features, labels, r2
from pipelines.experiment_log import append_experiment

MODEL_TYPE = "detection_dataset"
LABEL_COLS = ["label", "label_source", "sample_weight", "burn_type", "label_smf", "label_smf_source",
              "label_neris", "neris_in_coverage", "neris_county_reporting", "neris_unmatched_reporting", "incident_id_resolved"]


def build(radius_km: float = 2.0, neris_path: Path | None = None, export_dir: Path | None = None,
          run_ml: bool = True, start: str | None = None, end: str | None = None,
          refresh: bool = False, db: str | None = None, import_bundle: Path | None = None,
          pull: bool = False, history_csv: Path | None = None) -> dict:
    # Data lands in SMF_DATA_ROOT/fire_labels. Sources, in order: an explicit zip, R2 (--pull), the local API DB
    # (--refresh/--db). With none given, a missing export is fetched from R2 when configured, else from the local DB.
    have_export = (paths.FIRE_LABELS_DIR / "detection_training_manifest.json").exists()
    if import_bundle:
        labels.import_bundle(import_bundle)
    elif export_dir is None and pull:
        r2.pull_latest()
    elif export_dir is None and (refresh or db or not have_export):
        if not refresh and not db and r2.credentials_present():
            r2.pull_latest()
        else:
            labels.refresh_export(db)
    export = labels.load_export(export_dir)
    det = export["detections"].copy()
    if det.empty:
        raise SystemExit("detections.csv is empty - nothing to build")
    neris = labels.load_neris(neris_path)
    if history_csv:
        # FIRMS detections fetched by api/scripts/backfill_firms_history.py, most of them NOT in the database. The full file
        # is used, not just the NERIS-overlap rows, so the unmatched ones can serve as weak negatives.
        # The CSV is authoritative for what it holds: if only the NERIS-matched rows had been ingested, "in the database"
        # would itself predict the label. So a detection in both takes its sensor fields from the CSV and only the
        # fields the CSV cannot know (incident, review tier, cause, ...) from the database row.
        files = [history_csv] if isinstance(history_csv, (str, Path)) else list(history_csv)
        hist = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates("external_id")
        db_only = [c for c in det.columns if c not in hist.columns or c == "external_id"]
        in_db = hist["external_id"].isin(det["external_id"])
        hist = hist.merge(det[db_only], on="external_id", how="left")
        hist["id"] = hist["id"].where(in_db, -(pd.RangeIndex(len(hist)) + 1))  # negative id = not in the database
        det = pd.concat([det[~det["external_id"].isin(hist["external_id"])].assign(from_history_csv=0),
                         hist.assign(from_history_csv=1)], ignore_index=True)
        print(f"Merged {len(hist)} detections from {len(files)} history CSV(s) ({int(in_db.sum())} also in the database; "
              f"{len(hist) - int(in_db.sum())} only in the CSV)")

    det = labels.build_labels(det, export, neris, radius_km).reset_index(drop=True)
    # Features see every detection (so neighbour/cross-sensor context is complete); the dataset and
    # all ML then keep only detections some label source actually speaks for (approved feedback,
    # admin tier, a user report, or a NERIS dispatch/coverage day). Each source has its own dates,
    # so a detection needs ANY one of them, not all.
    county_cov = coverage.county_coverage(det, neris)  # before features/filtering, so it sees every detection
    X, feature_groups = features.build_features(det, neris)
    date_coverage = {
        "detections": labels.date_span(det["occurred_local_date"]),
        "neris": labels.date_span(neris["call_date"]),
        "smf_labels": labels.date_span(det.loc[det["label_smf"].notna(), "occurred_local_date"]),
        "user_reports": labels.date_span(det.loc[det["label_report"].notna(), "occurred_local_date"]),
    }
    # Keep what has a label, plus unmatched detections in counties where NERIS reports (the "unlabeled" side of the
    # positive-vs-unlabeled comparison). Unmatched detections elsewhere say nothing, so they are left out.
    keep_mask = det["label"].notna() | det["neris_unmatched_reporting"]
    if start:
        keep_mask &= det["occurred_local_date"] >= pd.Timestamp(start)
    if end:
        keep_mask &= det["occurred_local_date"] <= pd.Timestamp(end)
    inside = keep_mask.to_numpy()
    dropped = int((~inside).sum())
    if not inside.any():
        raise SystemExit("Nothing to build: no detection has a label from feedback, reports or NERIS. "
                         "Export detections that overlap at least one label source.")
    det, X = det[inside].reset_index(drop=True), X[inside].reset_index(drop=True)
    print(f"Kept {len(det)} detections ({int(det['label'].notna().sum())} labeled, {int(det['neris_unmatched_reporting'].sum())} unlabeled "
          f"in NERIS-reporting counties), dropped {dropped} with no label source covering them")
    grp = features.spatial_group(det)
    det["cv_block"] = grp

    ml, report = (derive.derive_all(det, X, grp, feature_groups) if run_ml else (pd.DataFrame(index=det.index), {}))

    keep = [c for c in det.columns if c not in ("footprint_geojson",)]
    final = pd.concat([det[keep], X.add_prefix("f_"), ml], axis=1)

    stamp = datetime.now().strftime("%Y%m%d")
    out_dir = paths.DETECTION_DATASET_DIR
    csv_path = out_dir / f"detection_dataset_{stamp}.csv"
    final.to_csv(csv_path, index=False)
    try:
        final.to_parquet(out_dir / f"detection_dataset_{stamp}.parquet", index=False)
    except ImportError:
        print("pyarrow not installed - wrote CSV only")

    county_cov.to_csv(out_dir / f"neris_county_coverage_{stamp}.csv", index=False)
    dictionary = _dictionary(final, det, feature_groups, ml)
    (out_dir / "columns.json").write_text(json.dumps(dictionary, indent=2), encoding="utf-8")
    sensitivity = _radius_sensitivity(det, neris, radius_km)
    summary = _summary(final, export, report, sensitivity, radius_km)
    summary["coverage"] = {k: (None if v is None else [str(v[0].date()), str(v[1].date())]) for k, v in date_coverage.items()}
    summary["rows_dropped_unlabeled"] = dropped
    summary["neris_county_coverage"] = coverage.summarize(county_cov)
    summary["neris_county_table"] = county_cov.head(15).to_dict("records")
    summary["neris_silent_busiest"] = county_cov[~county_cov["reporting"]].sort_values("detections", ascending=False).head(10).to_dict("records")
    paths.REPORTS_DIR.joinpath(f"detection_dataset_{stamp}.md").write_text(_markdown(summary), encoding="utf-8")

    metrics = {"rows": len(final), "labeled": int(final["label"].notna().sum()),
               "positives": int((final["label"] == 1).sum()), "negatives": int((final["label"] == 0).sum()),
               **{k: v for k, v in report.get("p_fire", {}).items() if isinstance(v, (int, float))}}
    append_experiment(MODEL_TYPE, metrics, params={"neris_radius_km": radius_km, "ml": run_ml, "csv": csv_path.name})
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {csv_path}")
    return summary


def _dictionary(final, det, feature_groups, ml) -> list:
    rows = []
    for c in final.columns:
        if c.startswith("f_"):
            origin, group, leak = "engineered", feature_groups.get(c[2:], "engineered"), False
        elif c in ml.columns:
            origin, group, leak = "ml", "ml", False
        elif c.startswith("neris_") or c == "label_neris":
            origin, group, leak = "label_source", "neris", True
        elif c in LABEL_COLS or c.startswith("label"):
            origin, group, leak = "label", "label", True
        else:
            origin, group, leak = "raw", "raw", False
        rows.append({"name": c, "origin": origin, "group": group, "leaks_label": leak,
                     "model_input": c.startswith("f_"), "null_frac": round(float(final[c].isna().mean()), 4)})
    return rows


def _radius_sensitivity(det, neris, base_radius) -> dict:
    out = {}
    for r in sorted({1.0, 2.0, 5.0, base_radius}):
        m = labels.match_neris(det, neris, r)
        out[str(r)] = int(m["neris_matched"].sum())
    return out


def _summary(final, export, report, sensitivity, radius_km) -> dict:
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rows": len(final), "neris_radius_km": radius_km,
        "detection_ranges": export["manifest"].get("detection_ranges", {}),
        "label_source_counts": final["label_source"].fillna("unlabeled").value_counts().to_dict(),
        "label_by_source": final.groupby(["source", final["label"].fillna(-1)]).size().rename("n").reset_index().to_dict("records"),
        "neris_radius_sensitivity_matched": sensitivity,
        "models": {k: v for k, v in report.items() if k != "feature_importance"},
        "top_features": dict(list((report.get("feature_importance") or {}).items())[:20]),
    }


def _markdown(s) -> str:
    lines = [f"# Detection dataset report ({s['generated']})", "", f"Rows: **{s['rows']}** - NERIS radius {s['neris_radius_km']} km", "",
             f"{s['rows_dropped_unlabeled']} detections were excluded because no label source covers them.", "",
             "## Date coverage by source", ""]
    lines += [f"- {k}: {'none' if not v else v[0] + ' to ' + v[1]}" for k, v in s["coverage"].items()]
    lines += [""]
    lines += [
             "## Detection coverage", ""]
    for src, r in s["detection_ranges"].items():
        lines.append(f"- {src}: {r['count']} rows, {r['min']} to {r['max']}")
    lines += ["", "## Labels by source", ""] + [f"- {k}: {v}" for k, v in s["label_source_counts"].items()]
    cv = s["neris_county_coverage"]
    lines += ["", "## Where NERIS reports (county coverage)", "",
              f"{cv['reporting']} of {cv['counties']} counties have at least {cv['min_calls_to_report']} NERIS calls; {cv['silent']} do not. "
              f"A detection with no nearby call only counts as unlabeled-for-comparison inside a reporting county; "
              f"nothing is ever treated as 'not a fire' because a call is missing.", "",
              f"- detections in reporting counties: {cv['detections_in_reporting']} ({cv['matched_in_reporting']} matched)",
              f"- detections in silent counties: {cv['detections_in_silent']} ({cv['matched_in_silent']} matched)", "",
              "Busiest NERIS counties:", ""]
    lines += [f"- {r['county']}: {r['neris_calls']} calls, {r['detections']} detections, {r['matched']} matched" for r in s["neris_county_table"]]
    lines += ["", "Silent counties with the most detections (NERIS says nothing about these):", ""]
    lines += [f"- {r['county']}: {r['neris_calls']} calls, {r['detections']} detections" for r in s["neris_silent_busiest"]]
    lines += ["", "## NERIS matched detections by radius (km)", ""] + [f"- {k}: {v}" for k, v in s["neris_radius_sensitivity_matched"].items()]
    lines += ["", "## ML outputs", "", "```json", json.dumps(s["models"], indent=2), "```", "", "## Top features", ""]
    lines += [f"- {k}: {v:.4f}" for k, v in s["top_features"].items()]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--neris-radius-km", type=float, default=2.0)
    p.add_argument("--neris-path", type=Path, default=None)
    p.add_argument("--export-dir", type=Path, default=None)
    p.add_argument("--no-ml", action="store_true")
    p.add_argument("--start", default=None, help="optionally narrow the overlap window (never widens it)")
    p.add_argument("--end", default=None)
    p.add_argument("--refresh", action="store_true", help="re-export from the API DB first (automatic if no export exists)")
    p.add_argument("--db", default=None, help="SQLite DB to export from (default: SMF_API_DB_PATH, else the api repo's local DB)")
    p.add_argument("--import-bundle", type=Path, default=None, help="detection_training_export.zip from the server; unpacked, verified, then used")
    p.add_argument("--pull", action="store_true", help="download the newest export published to R2 first")
    p.add_argument("--history-csv", type=Path, nargs="+", default=None,
                   help="one or more firms_history_*.csv from api/scripts/backfill_firms_history.py (detections mostly not in the database)")
    a = p.parse_args()
    build(a.neris_radius_km, a.neris_path, a.export_dir, not a.no_ml, a.start, a.end, a.refresh, a.db, a.import_bundle, a.pull,
          a.history_csv)


if __name__ == "__main__":
    main()
