"""Model Workshop facade: train → evaluate vs control → list experiments.

Station-sequence is the primary path (locked holdout + incumbent XGB control).
Legacy fuel-moisture XGBoost is a fast secondary path for quick iteration.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import paths
from model_lab.metrics import flatten_metric_block

WORKSHOP_DIR = paths.DATA_ROOT / "model_lab" / "workshop"
EXPERIMENTS_DIR = paths.MODELS_DIR / "experiments"
WORKSHOP_REPORTS_DIR = paths.REPORTS_DIR / "model_lab" / "workshop"
INDEX_PATH = WORKSHOP_DIR / "experiments_index.json"

STATION_DATASET = paths.ALIGNED_DIR / "station_leads.csv"
STATION_SPLIT = paths.REPORTS_DIR / "station_split_manifest.json"
LEGACY_DATASET = paths.TRAINING_DATA_DIR / "final_training_data.csv"
FD_SOURCE_DATASET = paths.DATA_ROOT / "data" / "fire_danger_training_data.csv"
FD_DIR = Path(__file__).resolve().parents[1] / "fire-danger-model"
FD_DATA_DIR = paths.DATA_ROOT / "fire-danger-model" / "data"
FD_MODELS_DIR = paths.DATA_ROOT / "fire-danger-model" / "models"


@dataclass
class ExperimentRecord:
    id: str
    family: str
    created_at: str
    checkpoint: str | None
    train_report: str | None
    eval_report: str | None
    status: str
    product: str = "fuel_moisture"
    rank: str = "candidate"
    notes: str = ""
    summary: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _ensure_dirs():
    WORKSHOP_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    WORKSHOP_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths.REPORTS_DIR.joinpath("experiments").mkdir(parents=True, exist_ok=True)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_index() -> list[dict]:
    _ensure_dirs()
    if not INDEX_PATH.exists():
        return []
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_index(rows: list[dict]) -> None:
    _ensure_dirs()
    temp = INDEX_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    temp.replace(INDEX_PATH)


def upsert_experiment(record: ExperimentRecord) -> ExperimentRecord:
    rows = _load_index()
    payload = record.to_dict()
    for i, row in enumerate(rows):
        if row.get("id") == record.id:
            rows[i] = payload
            break
    else:
        rows.insert(0, payload)
    _save_index(rows[:200])
    return record


RANKS = ("candidate", "challenger", "local_beta", "local_stable", "archived")


def _rank_index(rank: str) -> int:
    try:
        return RANKS.index(rank)
    except ValueError:
        return 0


def set_experiment_rank(exp_id: str, rank: str, *, register: bool = False) -> dict:
    """Move a local experiment on the ladder.

    This only mutates the training-data registry. It never imports to or
    promotes a model in the API checkout.
    """
    if rank not in RANKS:
        raise ValueError(f"Unknown local rank: {rank}")
    existing = get_experiment(exp_id)
    if not existing:
        raise KeyError(f"Unknown experiment: {exp_id}")
    checkpoint = Path(existing["checkpoint"]) if existing.get("checkpoint") else None
    if rank in ("local_beta", "local_stable") and (not checkpoint or not checkpoint.exists()):
        raise FileNotFoundError(f"Experiment artifact missing: {checkpoint}")

    registered_version = existing.get("summary", {}).get("registered_version")
    if register and rank in ("local_beta", "local_stable"):
        if existing.get("family") == "fuel_moisture_xgb":
            from models.versioning import register_trained_model
            report = {}
            if existing.get("eval_report") and Path(existing["eval_report"]).exists():
                report = json.loads(Path(existing["eval_report"]).read_text(encoding="utf-8"))
            candidate = report.get("candidate") or {}
            registered_version = register_trained_model(
                "fuel_moisture",
                source_path=checkpoint,
                performance={
                    "mae": candidate.get("mae"),
                    "r2_score": candidate.get("r2"),
                    "training_samples": candidate.get("samples"),
                    "workshop_id": exp_id,
                },
                channel="beta",
            )
        elif existing.get("family") == "fire_danger_xgb":
            from models.versioning import register_trained_model
            report = {}
            if existing.get("eval_report") and Path(existing["eval_report"]).exists():
                report = json.loads(Path(existing["eval_report"]).read_text(encoding="utf-8"))
            registered_version = register_trained_model(
                "fire_danger",
                source_path=checkpoint,
                performance=report.get("summary") or {},
                channel="beta",
            )
        else:
            raise ValueError("This model family is not registry-backed yet")

    summary = dict(existing.get("summary") or {})
    if registered_version:
        summary["registered_version"] = registered_version
    existing.update({
        "rank": rank,
        "status": f"rank_{rank}",
        "summary": summary,
    })
    upsert_experiment(ExperimentRecord(**{
        key: existing.get(key)
        for key in ExperimentRecord.__dataclass_fields__
    }))
    if rank == "local_stable" and registered_version:
        model_type = "fuel_moisture" if existing.get("family") == "fuel_moisture_xgb" else "fire_danger"
        from models.versioning import promote
        promote(model_type, version=registered_version)
    return existing


def promote_experiment(exp_id: str, *, register: bool = False) -> dict:
    existing = get_experiment(exp_id)
    if not existing:
        raise KeyError(exp_id)
    next_rank = RANKS[min(_rank_index(existing.get("rank", "candidate")) + 1, len(RANKS) - 1)]
    return set_experiment_rank(exp_id, next_rank, register=register or next_rank in ("local_beta", "local_stable"))


def demote_experiment(exp_id: str) -> dict:
    existing = get_experiment(exp_id)
    if not existing:
        raise KeyError(exp_id)
    current = _rank_index(existing.get("rank", "candidate"))
    next_rank = RANKS[max(current - 1, 0)]
    return set_experiment_rank(exp_id, next_rank, register=False)


def ladder_table(product: str = "fuel_moisture") -> pd.DataFrame:
    """Return a product-specific ranked view for the dashboard."""
    rows = []
    for record in list_experiments(200):
        family = record.get("family", "")
        record_product = record.get("product") or ("fire_danger" if family == "fire_danger_xgb" else "fuel_moisture")
        if record_product != product:
            continue
        summary = record.get("summary") or {}
        primary = summary.get("macro_f1") if product == "fire_danger" else (
            summary.get("candidate_mae") or summary.get("validation_mae")
        )
        rows.append({
            "id": record.get("id"),
            "family": family,
            "rank": record.get("rank", "candidate"),
            "status": record.get("status"),
            "primary_metric": primary,
            "candidate_mae": summary.get("candidate_mae"),
            "control_mae": summary.get("control_mae"),
            "delta_mae": summary.get("delta_mae"),
            "macro_f1": summary.get("macro_f1"),
            "daily_mae": summary.get("daily_mae"),
            "daily_macro_f1": summary.get("daily_macro_f1"),
            "pass": summary.get("pass"),
            "checkpoint": record.get("checkpoint"),
            "graphics": summary.get("graphics") or [],
            "created_at": record.get("created_at"),
            "notes": record.get("notes"),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    ascending = product != "fire_danger"
    return table.sort_values(
        ["rank", "primary_metric"],
        key=lambda col: col.map(_rank_index) if col.name == "rank" else col,
        ascending=[False, ascending],
    ).reset_index(drop=True)


def list_experiments(limit: int = 50) -> list[dict]:
    """Workshop index plus any orphan experiment checkpoints on disk."""
    indexed = {row["id"]: row for row in _load_index()}
    # Discover station checkpoints not yet indexed.
    if EXPERIMENTS_DIR.exists():
        for path in sorted(EXPERIMENTS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
            exp_id = path.stem
            if exp_id in indexed:
                continue
            indexed[exp_id] = ExperimentRecord(
                id=exp_id,
                family="station_sequence",
                created_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                checkpoint=str(path),
                train_report=str(paths.REPORTS_DIR / "experiments" / f"{exp_id}.json")
                if (paths.REPORTS_DIR / "experiments" / f"{exp_id}.json").exists() else None,
                eval_report=None,
                status="discovered",
            ).to_dict()
        for path in sorted(EXPERIMENTS_DIR.glob("legacy_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            exp_id = path.stem
            if exp_id in indexed:
                continue
            indexed[exp_id] = ExperimentRecord(
                id=exp_id,
                family="fuel_moisture_xgb",
                created_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                checkpoint=str(path),
                train_report=None,
                eval_report=None,
                status="discovered",
            ).to_dict()
    rows = sorted(indexed.values(), key=lambda r: r.get("created_at") or "", reverse=True)
    return rows[:limit]


def get_experiment(exp_id: str) -> dict | None:
    for row in list_experiments(200):
        if row.get("id") == exp_id:
            return row
    return None


def workshop_data_status() -> dict:
    fd_report_path = paths.DATA_ROOT / "reports" / "model_lab" / "fire_danger_dataset.json"
    fd_report = {}
    if fd_report_path.exists():
        try:
            fd_report = json.loads(fd_report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            fd_report = {}
    return {
        "station_dataset": str(STATION_DATASET),
        "station_dataset_exists": STATION_DATASET.exists(),
        "station_dataset_mb": round(STATION_DATASET.stat().st_size / 1e6, 1) if STATION_DATASET.exists() else None,
        "station_split": str(STATION_SPLIT),
        "station_split_exists": STATION_SPLIT.exists(),
        "legacy_dataset": str(LEGACY_DATASET),
        "legacy_dataset_exists": LEGACY_DATASET.exists(),
        "fd_dataset": str(FD_SOURCE_DATASET),
        "fd_dataset_exists": FD_SOURCE_DATASET.exists(),
        "fd_dataset_ready": bool(fd_report.get("ready_for_basic_training")),
        "fd_dataset_report": str(fd_report_path),
        "experiments_dir": str(EXPERIMENTS_DIR),
        "workshop_reports": str(WORKSHOP_REPORTS_DIR),
    }


def train_station_candidate(
    *,
    epochs: int = 20,
    fold: str = "fold-3",
    hidden_size: int = 16,
    num_layers: int = 1,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    patience: int = 5,
    dropout: float = 0.0,
    seed: int = 417,
    notes: str = "",
) -> dict:
    """Train a station-sequence experiment and register it in the workshop index."""
    from spatial.station_contract import load_or_create_manifest, sha256_file
    from spatial.train_station_sequence import load_frame, train_experiment
    import torch

    if not STATION_DATASET.exists():
        raise FileNotFoundError(f"Missing aligned dataset: {STATION_DATASET}")
    if num_layers == 1 and dropout:
        raise ValueError("dropout must be 0 for a one-layer GRU")

    _ensure_dirs()
    stamp = _stamp()
    exp_id = f"station_{stamp}"
    checkpoint_path = EXPERIMENTS_DIR / f"{exp_id}.pt"
    train_report_path = paths.REPORTS_DIR / "experiments" / f"{exp_id}.json"

    started = time.perf_counter()
    frame = load_frame(STATION_DATASET)
    manifest = load_or_create_manifest(STATION_SPLIT, frame, STATION_DATASET)
    fold_info = next(item for item in manifest["folds"] if item["name"] == fold)
    _, checkpoint, report = train_experiment(
        frame,
        manifest,
        fold_info["train_runs"],
        fold_info["validation_runs"],
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        seed=seed,
    )
    torch.save(checkpoint, checkpoint_path)
    report.update({
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fold": fold,
        "workshop_id": exp_id,
        "wall_seconds": time.perf_counter() - started,
    })
    train_report_path.parent.mkdir(parents=True, exist_ok=True)
    train_report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    val = report.get("validation_metrics") or {}
    summary = {
        "validation_mae": val.get("mae"),
        "best_epoch": report.get("best_epoch"),
        "device": report.get("device"),
        "runtime_seconds": report.get("runtime_seconds"),
    }
    upsert_experiment(ExperimentRecord(
        id=exp_id,
        family="station_sequence",
        created_at=report["created_at"],
        checkpoint=str(checkpoint_path),
        train_report=str(train_report_path),
        eval_report=None,
        status="trained",
        notes=notes,
        summary=summary,
    ))
    return {
        "id": exp_id,
        "checkpoint": str(checkpoint_path),
        "train_report": str(train_report_path),
        "summary": summary,
        "ok": True,
    }


def evaluate_station_vs_control(checkpoint: str | Path, *, notes: str = "") -> dict:
    """Score a station checkpoint on the locked holdout vs incumbent XGB control.

    Writes under reports/model_lab/workshop/ so the locked final_station_evaluation.json
    is never overwritten.
    """
    from spatial.evaluate_station_candidate import evaluate
    from spatial.station_contract import sha256_file

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not STATION_DATASET.exists() or not STATION_SPLIT.exists():
        raise FileNotFoundError("Station dataset or split manifest missing")

    _ensure_dirs()
    report = evaluate(checkpoint_path, STATION_DATASET, STATION_SPLIT)
    exp_id = checkpoint_path.stem
    out = WORKSHOP_REPORTS_DIR / f"{exp_id}_vs_control.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    candidate = report["metrics"]["candidate"]
    control = report["metrics"]["incumbent_control"]
    summary = {
        "pass": report["pass"],
        "candidate_mae": candidate.get("mae"),
        "control_mae": control.get("mae"),
        "delta_mae": (candidate.get("mae") - control.get("mae")) if candidate.get("mae") is not None and control.get("mae") is not None else None,
        "relative_improvement": (
            (control["mae"] - candidate["mae"]) / control["mae"]
            if control.get("mae") else None
        ),
        "candidate_bias": candidate.get("bias"),
        "control_bias": control.get("bias"),
        "checks": report["checks"],
        "samples": report.get("samples"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    existing = get_experiment(exp_id) or {}
    upsert_experiment(ExperimentRecord(
        id=exp_id,
        family=existing.get("family") or "station_sequence",
        created_at=existing.get("created_at") or datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint_path),
        train_report=existing.get("train_report"),
        eval_report=str(out),
        status="passed" if report["pass"] else "failed_gate",
        notes=notes or existing.get("notes") or "",
        summary={**(existing.get("summary") or {}), **summary},
    ))
    return {"id": exp_id, "eval_report": str(out), "summary": summary, "report": report, "ok": True}


def train_legacy_xgb_candidate(
    *,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    notes: str = "",
    register_beta: bool = False,
) -> dict:
    """Train a legacy point XGBoost FM model and compare to the previous registry beta if present."""
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score

    if not LEGACY_DATASET.exists():
        raise FileNotFoundError(f"Missing legacy training CSV: {LEGACY_DATASET}")

    _ensure_dirs()
    stamp = _stamp()
    exp_id = f"legacy_{stamp}"
    checkpoint_path = EXPERIMENTS_DIR / f"{exp_id}.json"
    eval_path = WORKSHOP_REPORTS_DIR / f"{exp_id}_holdout.json"

    df = pd.read_csv(LEGACY_DATASET, parse_dates=["obs_time"]).sort_values("obs_time")
    features_to_use = [
        "temp_c", "rel_humidity", "wind_speed_ms",
        "hour", "month", "emc_baseline",
        "temp_mean_3h", "rh_mean_3h", "temp_mean_6h", "rh_mean_6h",
    ]
    precip_features = [f for f in ("precip_1h", "precip_3h", "precip_6h", "precip_24h", "hours_since_rain") if f in df.columns]
    features_to_use.extend(precip_features)
    missing = [f for f in features_to_use if f not in df.columns]
    if missing:
        raise RuntimeError(f"Legacy dataset missing features: {missing}")

    X = df[features_to_use]
    y = df["target_fm"]
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        objective="reg:squarederror",
        random_state=417,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    candidate = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(np.mean((pred - y_test.to_numpy()) ** 2))),
        "bias": float(np.mean(pred - y_test.to_numpy())),
        "r2": float(r2_score(y_test, pred)),
        "samples": int(len(y_test)),
    }
    model.save_model(str(checkpoint_path))

    control = None
    control_source = None
    try:
        from models.versioning import load_active_model_path
        for channel in ("stable", "beta"):
            try:
                control_path = Path(load_active_model_path("fuel_moisture", channel=channel))
            except FileNotFoundError:
                continue
            if not control_path.exists():
                continue
            control_source = f"registry:{channel}:{control_path.name}"
            booster = xgb.Booster()
            booster.load_model(str(control_path))
            dmat = xgb.DMatrix(X_test, feature_names=features_to_use)
            cpred = booster.predict(dmat)
            control = {
                "mae": float(mean_absolute_error(y_test, cpred)),
                "rmse": float(np.sqrt(np.mean((cpred - y_test.to_numpy()) ** 2))),
                "bias": float(np.mean(cpred - y_test.to_numpy())),
                "r2": float(r2_score(y_test, cpred)),
                "samples": int(len(y_test)),
            }
            break
    except Exception:
        control = None

    if control is None:
        # Fall back: persistence-style control using emc_baseline when available.
        if "emc_baseline" in X_test.columns:
            baseline = X_test["emc_baseline"].to_numpy()
            control = {
                "mae": float(mean_absolute_error(y_test, baseline)),
                "rmse": float(np.sqrt(np.mean((baseline - y_test.to_numpy()) ** 2))),
                "bias": float(np.mean(baseline - y_test.to_numpy())),
                "r2": float(r2_score(y_test, baseline)),
                "samples": int(len(y_test)),
            }
            control_source = "emc_baseline"

    report = {
        "status": "legacy_holdout_evaluation",
        "family": "fuel_moisture_xgb",
        "checkpoint": str(checkpoint_path),
        "features": features_to_use,
        "candidate": candidate,
        "control": control,
        "control_source": control_source,
        "pass": bool(control and candidate["mae"] < control["mae"]),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "hparams": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
        },
    }
    if control:
        report["delta_mae"] = candidate["mae"] - control["mae"]
        report["relative_improvement"] = (control["mae"] - candidate["mae"]) / control["mae"] if control["mae"] else None
    eval_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    registered_version = None
    if register_beta:
        from models.versioning import register_trained_model
        registered_version = register_trained_model(
            "fuel_moisture",
            source_path=checkpoint_path,
            performance={"mae": round(candidate["mae"], 3), "r2_score": round(candidate["r2"], 3),
                          "training_samples": int(len(X_train)), "test_samples": int(len(X_test))},
            channel="beta",
        )

    summary = {
        "candidate_mae": candidate["mae"],
        "control_mae": None if not control else control["mae"],
        "delta_mae": report.get("delta_mae"),
        "relative_improvement": report.get("relative_improvement"),
        "pass": report["pass"],
        "control_source": control_source,
        "registered_version": registered_version,
    }
    upsert_experiment(ExperimentRecord(
        id=exp_id,
        family="fuel_moisture_xgb",
        created_at=report["evaluated_at"],
        checkpoint=str(checkpoint_path),
        train_report=None,
        eval_report=str(eval_path),
        status="registered_beta" if registered_version else ("beat_control" if report["pass"] else "trained"),
        notes=notes,
        summary=summary,
    ))
    return {"id": exp_id, "checkpoint": str(checkpoint_path), "eval_report": str(eval_path),
            "summary": summary, "report": report, "ok": True}


def train_fire_danger_candidate(
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    notes: str = "",
) -> dict:
    """Train and evaluate an advisory fire-danger XGBoost candidate.

    Fire danger uses categorical Macro F1 gates, not the FM MAE gates. The
    standalone FD package is intentionally isolated from the FM registry.
    """
    from model_lab.fd_three_class import train_three_class_candidate

    result = train_three_class_candidate(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
    )
    exp_id = Path(result["checkpoint"]).stem
    summary = {
        "class_mode": "three_class",
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "support_by_class": result["class_support"],
        "pass": result["ready_for_evaluation"],
    }
    upsert_experiment(ExperimentRecord(
        id=exp_id,
        family="fire_danger_xgb_three_class",
        product="fire_danger",
        created_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=result["checkpoint"],
        train_report=result["meta"],
        eval_report=result["eval_report"],
        status="passed" if summary["pass"] else "failed_gate",
        notes=notes,
        summary=summary,
    ))
    return {
        "id": exp_id,
        "checkpoint": result["checkpoint"],
        "meta": result["meta"],
        "eval_report": result["eval_report"],
        "summary": summary,
        "report": result,
        "ok": True,
    }

    # Legacy five-class implementation retained below for reference.
    import importlib
    import shutil
    import sys

    if not FD_SOURCE_DATASET.exists():
        raise FileNotFoundError(f"Missing fire-danger source CSV: {FD_SOURCE_DATASET}")
    _ensure_dirs()
    stamp = _stamp()
    exp_id = f"fd_{stamp}"
    checkpoint_path = EXPERIMENTS_DIR / f"{exp_id}.json"
    meta_path = EXPERIMENTS_DIR / f"{exp_id}_meta.json"
    eval_path = WORKSHOP_REPORTS_DIR / f"{exp_id}_evaluation.json"

    # The standalone package is a directory with config.py imports. Load it
    # without adding a second application or changing the live API artifact.
    old_path = list(sys.path)
    sys.path.insert(0, str(FD_DIR))
    try:
        fd_data_prep = importlib.import_module("data_prep")
        fd_train = importlib.import_module("train")
        fd_config = importlib.import_module("config")
        fd_evaluate = importlib.import_module("evaluate")
        fd_data_prep.prepare_dataset(str(FD_SOURCE_DATASET))
        fd_train.train_model(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
        shutil.copy2(fd_config.LATEST_MODEL_PATH, checkpoint_path)
        shutil.copy2(fd_config.LATEST_MODEL_META_PATH, meta_path)
        report = fd_evaluate.evaluate(
            model_path=str(checkpoint_path),
            model_meta_path=str(meta_path),
            test_data_path=str(fd_config.TEST_DATA_PATH),
        )
    finally:
        sys.path[:] = old_path

    # evaluate.py writes its own timestamped package report; save a stable
    # workshop copy for the ladder and inspection.
    eval_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    category = report.get("category") or {}
    gates = report.get("gates") or {}
    summary = {
        "macro_f1": category.get("macro_f1"),
        "weighted_f1": category.get("weighted_f1"),
        "baseline_macro_f1": (report.get("baseline") or {}).get("macro_f1"),
        "pass": (gates.get("overall") or {}).get("passed", False),
        "regression_mae": (report.get("regression") or {}).get("mae"),
        "support_by_class": category.get("support_by_class"),
    }
    upsert_experiment(ExperimentRecord(
        id=exp_id,
        family="fire_danger_xgb",
        product="fire_danger",
        created_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint_path),
        train_report=str(meta_path),
        eval_report=str(eval_path),
        status="passed" if summary["pass"] else "failed_gate",
        notes=notes,
        summary=summary,
    ))
    return {
        "id": exp_id,
        "checkpoint": str(checkpoint_path),
        "meta": str(meta_path),
        "eval_report": str(eval_path),
        "summary": summary,
        "report": report,
        "ok": True,
    }


def scorecard_from_eval(eval_path: str | Path | None = None, report: dict | None = None) -> pd.DataFrame:
    """Flatten candidate/control blocks into a small comparison table."""
    if report is None:
        if not eval_path:
            return pd.DataFrame()
        report = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    rows = []
    metrics = report.get("metrics") or {}
    if metrics:
        for name in ("candidate", "incumbent_control", "physics", "persistence"):
            block = metrics.get(name)
            if isinstance(block, dict):
                rows.append({"model": name, **flatten_metric_block(block)})
    elif report.get("candidate"):
        rows.append({"model": "candidate", **flatten_metric_block(report["candidate"])})
        if report.get("control"):
            rows.append({"model": "control", **flatten_metric_block(report["control"])})
    return pd.DataFrame(rows)
