"""ShowMeFire Model Lab — local Streamlit UI for registry, archive scoring, and eval browsing.

Run from model-training with SMF_DATA_ROOT set:

    streamlit run model_lab/app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import ARCHIVE_FORECASTS_DIR, ARCHIVE_RAW_DATA_DIR, DATA_ROOT, MODELS_DIR, REPORTS_DIR
from model_lab.data import (
    discover_forecast_dates,
    discover_forecast_series,
    forecast_to_dataframe,
    list_eval_reports,
    list_registry_models,
    list_series_for_date,
    list_shadow_candidates,
    list_xgb_artifacts,
    load_json,
    load_scored_series,
    resolve_series_path,
    score_date_range,
    series_file_prefix,
)
from model_lab.metrics import (
    attach_obs_fire_danger,
    calculate_metrics,
    fd_confusion,
    flatten_metric_block,
    metrics_by_hour,
    metrics_by_station,
    pairwise_series_delta,
)

st.set_page_config(page_title="ShowMeFire Model Lab", page_icon="🔥", layout="wide")

ACCENT = "#C45C26"
SERIES_COLORS = {
    "production": "#1F4E79",
    "beta": "#C45C26",
    "obs": "#2F6F4E",
}
_EXTRA_SERIES_PALETTE = ["#6B4C9A", "#2A9D8F", "#E9C46A", "#E76F51", "#264653", "#8AB17D", "#F4A261"]


def _series_color(name: str) -> str:
    if name in SERIES_COLORS:
        return SERIES_COLORS[name]
    # Stable-ish color from name.
    idx = sum(ord(ch) for ch in name) % len(_EXTRA_SERIES_PALETTE)
    return _EXTRA_SERIES_PALETTE[idx]


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def header():
    st.markdown(
        f"""
        <style>
          .block-container {{ padding-top: 1.2rem; }}
          h1, h2, h3 {{ font-family: "Segoe UI", "Helvetica Neue", sans-serif; }}
          div[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
          .smf-sub {{ color: #5c5c5c; margin-top: -0.6rem; margin-bottom: 1rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("ShowMeFire Model Lab")
    st.markdown(
        f'<p class="smf-sub">Train · test · compare · sync · data root <code>{DATA_ROOT}</code></p>',
        unsafe_allow_html=True,
    )
    st.info(
        "Start with **Train & test** to create a candidate. Use **Model ladder** for "
        "local ranking, **Compare archives** to compare forecast files against the "
        "same observations, and **Daily results** for the recent trend. "
        "Workshop models do not appear in archive comparisons until you generate a "
        "tagged forecast for them."
    )


def page_compare():
    st.subheader("Compare forecast series vs observations")
    mode = st.radio(
        "Compare mode",
        ["Archive forecast series", "Workshop experiment betas"],
        horizontal=True,
        help="Archives = day-by-day forecast files. Workshop = holdout scorecards from models you trained here.",
    )

    if mode == "Workshop experiment betas":
        _page_compare_workshop_betas()
        return

    dates = list(discover_forecast_dates())
    if not dates:
        st.warning(f"No forecast archives found under `{ARCHIVE_FORECASTS_DIR}`.")
        return

    available_series = list(discover_forecast_series())
    st.caption(
        "Available archive series: "
        + ", ".join(f"`{s}`" for s in available_series)
        + f" · tagged extras use `{series_file_prefix('beta_mytag')}_YYYYMMDD_HH.json`"
    )

    col_a, col_b, col_c = st.columns([1.2, 1.2, 1.4])
    with col_a:
        start = st.selectbox("Start date", dates, index=max(0, len(dates) - 14), format_func=lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}")
    with col_b:
        end = st.selectbox("End date", dates, index=len(dates) - 1, format_func=lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}")
    with col_c:
        default_series = [s for s in ("production", "beta") if s in available_series] or available_series[:2]
        series_choices = st.multiselect(
            "Series (pick several)",
            available_series,
            default=default_series,
        )

    selected = [d for d in dates if start <= d <= end]
    if not selected:
        st.info("No dates in that range.")
        return
    if not series_choices:
        st.info("Pick at least one series.")
        return

    availability = []
    for date_token in selected:
        paths = list_series_for_date(date_token, series_choices)
        row = {"date": date_token}
        for series in series_choices:
            row[series] = "yes" if paths.get(series) else "—"
        row["raw"] = "yes" if (ARCHIVE_RAW_DATA_DIR / f"raw_data_{date_token}.json").exists() else "?"
        availability.append(row)
    with st.expander("Archive coverage", expanded=False):
        st.dataframe(pd.DataFrame(availability), width="stretch", hide_index=True)

    if st.button("Score selected range", type="primary"):
        frames = []
        metric_tables = []
        progress = st.progress(0.0)
        for i, series in enumerate(series_choices):
            rows, metrics_df = score_date_range(selected, series)
            if not rows.empty:
                frames.append(rows)
            if not metrics_df.empty:
                metric_tables.append(metrics_df)
            progress.progress((i + 1) / len(series_choices))
        progress.empty()
        if not frames:
            st.error("No matched forecast/obs rows for that range. Check raw_data archives.")
            return
        all_rows = pd.concat(frames, ignore_index=True)
        metrics_df = pd.concat(metric_tables, ignore_index=True) if metric_tables else pd.DataFrame()
        st.session_state["compare_rows"] = all_rows
        st.session_state["compare_metrics"] = metrics_df
        st.session_state["compare_series"] = list(series_choices)

    all_rows = st.session_state.get("compare_rows")
    metrics_df = st.session_state.get("compare_metrics")
    series_choices = st.session_state.get("compare_series") or series_choices
    color_map = {name: _series_color(name) for name in series_choices}
    if all_rows is None or all_rows.empty:
        st.info("Choose a date range and click **Score selected range**.")
        return

    st.markdown("#### Headline fuel-moisture skill")
    headline_cols = st.columns(min(4, len(series_choices)))
    for idx, series in enumerate(series_choices):
        col = headline_cols[idx % len(headline_cols)]
        subset = all_rows[all_rows["series"] == series]
        m = calculate_metrics(subset)
        fm = m.get("Fuel Moisture (%)", {})
        with col:
            st.markdown(f"**{series}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", _fmt(fm.get("mae")))
            c2.metric("RMSE", _fmt(fm.get("rmse")))
            c3.metric("Bias", _fmt(fm.get("bias")))
            c4.metric("N", _fmt(fm.get("count"), 0))

    # Ranked summary across series
    rank_rows = []
    for series in series_choices:
        fm = calculate_metrics(all_rows[all_rows["series"] == series]).get("Fuel Moisture (%)", {})
        rank_rows.append({"series": series, "fm_mae": fm.get("mae"), "fm_bias": fm.get("bias"),
                          "fm_rmse": fm.get("rmse"), "n": fm.get("count")})
    rank_df = pd.DataFrame(rank_rows).sort_values("fm_mae", na_position="last")
    st.markdown("#### Series ranking (lower MAE better)")
    st.dataframe(rank_df, width="stretch", hide_index=True)

    if metrics_df is not None and not metrics_df.empty:
        st.markdown("#### Daily FM MAE")
        fig = px.line(
            metrics_df.dropna(subset=["fm_mae"]),
            x="date", y="fm_mae", color="series",
            markers=True, color_discrete_map=color_map,
            labels={"fm_mae": "FM MAE (%)", "date": "Date"},
        )
        fig.update_layout(height=360, legend_title_text="")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(metrics_df, width="stretch", hide_index=True)

    st.markdown("#### Predicted vs observed fuel moisture")
    scatter_cols = st.columns(min(3, len(series_choices)))
    for idx, series in enumerate(series_choices):
        subset = all_rows[all_rows["series"] == series].dropna(subset=["pred_fm", "obs_fm"])
        with scatter_cols[idx % len(scatter_cols)]:
            if subset.empty:
                st.write(f"{series}: no matched FM")
                continue
            lims = [
                float(min(subset["pred_fm"].min(), subset["obs_fm"].min())),
                float(max(subset["pred_fm"].max(), subset["obs_fm"].max())),
            ]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=subset["obs_fm"], y=subset["pred_fm"],
                mode="markers", marker=dict(size=5, opacity=0.45, color=color_map[series]),
                name=series, text=subset["stid"],
            ))
            fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="#888"), name="1:1"))
            fig.update_layout(title=series, xaxis_title="Observed FM %", yaxis_title="Predicted FM %",
                              height=380, showlegend=False)
            st.plotly_chart(fig, width="stretch")

    st.markdown("#### Station drill-down")
    stations = sorted(all_rows["stid"].dropna().unique())
    station = st.selectbox("Station", stations)
    station_df = all_rows[all_rows["stid"] == station].sort_values("timestamp")
    fig = go.Figure()
    obs = station_df.drop_duplicates(subset=["timestamp"])
    fig.add_trace(go.Scatter(
        x=obs["timestamp"], y=obs["obs_fm"], mode="lines+markers",
        name="Observed", line=dict(color=SERIES_COLORS["obs"], width=2),
    ))
    for series in series_choices:
        part = station_df[station_df["series"] == series]
        fig.add_trace(go.Scatter(
            x=part["timestamp"], y=part["pred_fm"], mode="lines+markers",
            name=series, line=dict(color=color_map[series]),
        ))
    fig.update_layout(height=400, yaxis_title="Fuel moisture %", legend_title_text="")
    st.plotly_chart(fig, width="stretch")

    if "production" in series_choices and len(series_choices) >= 2:
        st.markdown("#### Station MAE vs production")
        prod_st = metrics_by_station(all_rows[all_rows["series"] == "production"]).rename(
            columns={"mae": "production_mae", "bias": "production_bias", "count": "production_n"}
        )
        for series in series_choices:
            if series == "production":
                continue
            other = metrics_by_station(all_rows[all_rows["series"] == series]).rename(
                columns={"mae": "series_mae", "bias": "series_bias", "count": "series_n"}
            )
            if prod_st.empty or other.empty:
                continue
            joined = prod_st.merge(other, on="stid", how="outer")
            joined["delta_mae"] = joined["series_mae"] - joined["production_mae"]
            fig = px.scatter(
                joined.dropna(subset=["production_mae", "series_mae"]),
                x="production_mae", y="series_mae", hover_name="stid",
                labels={"production_mae": "Production FM MAE", "series_mae": f"{series} FM MAE"},
                title=f"{series} vs production",
            )
            max_v = float(joined[["production_mae", "series_mae"]].max().max())
            fig.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines",
                                     line=dict(dash="dash", color="#888"), name="Equal"))
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, width="stretch")

        st.markdown("#### Series disagreement vs production (matched times)")
        for series in series_choices:
            if series == "production":
                continue
            deltas = []
            for date_token in selected:
                left_path = resolve_series_path(date_token, "production")
                right_path = resolve_series_path(date_token, series)
                if not left_path or not right_path:
                    continue
                delta = pairwise_series_delta(
                    forecast_to_dataframe(load_json(left_path)),
                    forecast_to_dataframe(load_json(right_path)),
                    "pred_fm",
                )
                if not delta.empty:
                    delta["date"] = date_token
                    deltas.append(delta)
            if not deltas:
                st.write(f"{series}: no overlapping pairs with production.")
                continue
            delta_df = pd.concat(deltas, ignore_index=True)
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{series} mean |Δ FM|", _fmt(delta_df["abs_delta"].mean()))
            c2.metric("Max |Δ FM|", _fmt(delta_df["abs_delta"].max()))
            c3.metric("Pairs", _fmt(len(delta_df), 0))

    st.markdown("#### FM error by hour of day")
    hour_frames = []
    for series in series_choices:
        hdf = metrics_by_hour(all_rows[all_rows["series"] == series])
        if not hdf.empty:
            hdf = hdf.copy()
            hdf["series"] = series
            hour_frames.append(hdf)
    if hour_frames:
        hour_df = pd.concat(hour_frames, ignore_index=True)
        fig = px.line(hour_df, x="hour", y="mae", color="series", markers=True,
                      color_discrete_map=color_map, labels={"mae": "FM MAE"})
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Fire-danger confusion (vs obs-derived category)")
    fd_series = series_choices[0]
    fd_df = attach_obs_fire_danger(all_rows[all_rows["series"] == fd_series])
    matrix = fd_confusion(fd_df)
    if matrix.empty:
        st.write("Not enough fire-danger pairs.")
    else:
        st.caption(f"Series: {fd_series}")
        fig = px.imshow(matrix, text_auto=True, color_continuous_scale="YlOrRd",
                        labels=dict(x="Predicted", y="Observed", color="Count"))
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")


def _page_compare_workshop_betas():
    from model_lab.workshop import list_experiments

    st.caption(
        "This compares **workshop experiments** (holdout scorecards), not day-archive forecasts. "
        "Use this when you have several trained betas and want them side by side."
    )
    rows = list_experiments(100)
    usable = [r for r in rows if (r.get("summary") or {}).get("candidate_mae") is not None
              or (r.get("summary") or {}).get("validation_mae") is not None]
    if not usable:
        st.warning("No workshop experiments with scores yet. Train something in the Workshop tab first.")
        return

    labels = {
        r["id"]: (
            f"{r['id']} · {r.get('family')} · "
            f"MAE {_fmt((r.get('summary') or {}).get('candidate_mae') or (r.get('summary') or {}).get('validation_mae'))}"
        )
        for r in usable
    }
    picks = st.multiselect(
        "Experiments to compare",
        options=[r["id"] for r in usable],
        default=[r["id"] for r in usable[: min(4, len(usable))]],
        format_func=lambda i: labels.get(i, i),
    )
    if not picks:
        st.info("Pick two or more experiments.")
        return
    chosen = [r for r in usable if r["id"] in picks]
    table = pd.DataFrame([
        {
            "id": r.get("id"),
            "family": r.get("family"),
            "status": r.get("status"),
            "candidate_mae": (r.get("summary") or {}).get("candidate_mae")
            or (r.get("summary") or {}).get("validation_mae"),
            "control_mae": (r.get("summary") or {}).get("control_mae"),
            "delta_mae": (r.get("summary") or {}).get("delta_mae"),
            "rel_improvement": (r.get("summary") or {}).get("relative_improvement"),
            "pass": (r.get("summary") or {}).get("pass"),
            "notes": r.get("notes"),
        }
        for r in chosen
    ]).sort_values("candidate_mae", na_position="last")
    st.dataframe(table, width="stretch", hide_index=True)
    fig = px.bar(
        table.dropna(subset=["candidate_mae"]),
        x="id", y="candidate_mae", color="family",
        hover_data=["control_mae", "delta_mae", "pass"],
        title="Workshop candidate MAE (lower better)",
    )
    fig.update_layout(height=380, xaxis_title="")
    st.plotly_chart(fig, width="stretch")
    if table["control_mae"].notna().any():
        long = table.melt(
            id_vars=["id"],
            value_vars=[c for c in ("candidate_mae", "control_mae") if c in table.columns],
            var_name="which",
            value_name="mae",
        ).dropna(subset=["mae"])
        fig2 = px.bar(long, x="id", y="mae", color="which", barmode="group",
                      title="Candidate vs control MAE")
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, width="stretch")


def page_registry():
    st.subheader("Model registry & shadow candidates")
    registry = list_registry_models()
    shadows = list_shadow_candidates()

    active = [r for r in registry if r.source == "registry"]
    if active:
        rows = []
        for r in active:
            rows.append({
                "model_type": r.model_type,
                "channel": r.channel,
                "version": r.version or "—",
                "mae": r.performance.get("mae"),
                "rmse": r.performance.get("rmse"),
                "r2": r.performance.get("r2") or r.performance.get("r2_score"),
                "samples": r.performance.get("samples") or r.performance.get("test_samples"),
                "trained_at": r.trained_at,
                "artifact": str(r.path) if r.path else r.file,
                "exists": bool(r.path and r.path.exists()),
            })
        st.markdown("#### Active channels")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.warning(f"No registry entries in `{MODELS_DIR / 'config.json'}`.")

    history = [r for r in registry if r.source == "history"]
    if history:
        st.markdown("#### History")
        types = sorted({r.model_type for r in history})
        chosen = st.selectbox("Model type", types)
        hist_rows = []
        for r in history:
            if r.model_type != chosen:
                continue
            hist_rows.append({
                "version": r.version,
                "channel": r.channel,
                "mae": r.performance.get("mae"),
                "r2": r.performance.get("r2") or r.performance.get("r2_score"),
                "interval_coverage": r.performance.get("interval_coverage"),
                "trained_at": r.trained_at,
            })
        hdf = pd.DataFrame(hist_rows)
        st.dataframe(hdf, width="stretch", hide_index=True)
        if not hdf.empty and hdf["mae"].notna().any():
            fig = px.line(hdf.dropna(subset=["mae"]), x="version", y="mae", markers=True,
                          title=f"{chosen} — registered MAE trail")
            fig.update_layout(height=320)
            st.plotly_chart(fig, width="stretch")

    st.markdown("#### Shadow / candidate bundles")
    if not shadows:
        st.write("No shadow_candidate directories under models/.")
    else:
        shadow_rows = [{
            "name": s.model_type,
            "version": s.version,
            "mae": (s.performance or {}).get("mae"),
            "path": str(s.path),
        } for s in shadows]
        st.dataframe(pd.DataFrame(shadow_rows), width="stretch", hide_index=True)

    with st.expander("Raw config.json"):
        st.code((MODELS_DIR / "config.json").read_text(encoding="utf-8")[:8000]
                if (MODELS_DIR / "config.json").exists() else "missing", language="json")


def _extract_compare_blocks(report: dict) -> dict[str, dict]:
    """Find candidate/incumbent-style metric blocks inside common eval report shapes."""
    blocks = {}
    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(value, dict) and ("mae" in value or "regimes" in value or "by_lead" in value):
                blocks[key] = value
    evidence = report.get("evidence")
    if isinstance(evidence, dict):
        for key, value in evidence.items():
            if isinstance(value, dict) and "mae" in value:
                blocks[f"evidence.{key}"] = value
    # Flat report with top-level mae
    if "mae" in report and "candidate" not in blocks:
        blocks["report"] = report
    return blocks


def page_eval_reports():
    st.subheader("Offline evaluation reports")
    reports = list_eval_reports()
    if not reports:
        st.warning(f"No evaluation JSON files under `{REPORTS_DIR}`.")
        return

    labels = [f"{p.relative_to(REPORTS_DIR)}  ·  {pd.Timestamp(p.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')}"
              for p in reports]
    pick = st.selectbox("Report", range(len(reports)), format_func=lambda i: labels[i])
    path = reports[pick]
    report = load_json(path)
    st.caption(str(path))

    top = st.columns(4)
    top[0].metric("Pass", str(report.get("pass", report.get("status", "—"))))
    top[1].metric("Samples", _fmt(report.get("samples") or report.get("paired_rows"), 0))
    top[2].metric("Production eligible", str(report.get("production_eligible", "—")))
    top[3].metric("Shadow required", str(report.get("prospective_shadow_required", "—")))

    if isinstance(report.get("checks"), dict):
        st.markdown("#### Gate checks")
        checks = pd.DataFrame([{"check": k, "pass": v} for k, v in report["checks"].items()])
        st.dataframe(checks, width="stretch", hide_index=True)

    blocks = _extract_compare_blocks(report)
    if blocks:
        st.markdown("#### Model blocks")
        summary_rows = []
        for name, block in blocks.items():
            row = {"model": name, **flatten_metric_block(block)}
            summary_rows.append(row)
        summary = pd.DataFrame(summary_rows)
        st.dataframe(summary, width="stretch", hide_index=True)

        if "mae" in summary.columns and summary["mae"].notna().any():
            fig = px.bar(summary.dropna(subset=["mae"]), x="model", y="mae",
                         color="model", title="MAE by model block")
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, width="stretch")

        # Lead-hour curves when present
        lead_frames = []
        for name, block in blocks.items():
            by_lead = block.get("by_lead") or {}
            for lead, stats in by_lead.items():
                if isinstance(stats, dict) and stats.get("mae") is not None:
                    lead_frames.append({"model": name, "lead_hour": float(lead), "mae": stats["mae"],
                                        "bias": stats.get("bias"), "samples": stats.get("samples")})
        if lead_frames:
            lead_df = pd.DataFrame(lead_frames).sort_values("lead_hour")
            fig = px.line(lead_df, x="lead_hour", y="mae", color="model", markers=True,
                          title="MAE by lead hour")
            fig.update_layout(height=380)
            st.plotly_chart(fig, width="stretch")

        # Regime table for first block with regimes
        for name, block in blocks.items():
            regimes = block.get("regimes")
            if isinstance(regimes, dict):
                st.markdown(f"#### Regimes — {name}")
                rows = []
                for regime, stats in regimes.items():
                    if isinstance(stats, dict):
                        rows.append({"regime": regime, **flatten_metric_block(stats)})
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                break

        # Station MAE distribution for candidate if present
        for preferred in ("candidate", "incumbent_control"):
            block = blocks.get(preferred)
            if not block:
                continue
            by_station = block.get("by_station") or {}
            if by_station:
                st.markdown(f"#### Station MAE distribution — {preferred}")
                sdf = pd.DataFrame([
                    {"stid": k, **flatten_metric_block(v)}
                    for k, v in by_station.items() if isinstance(v, dict)
                ]).sort_values("mae")
                fig = px.histogram(sdf, x="mae", nbins=30, title=f"{preferred} station MAE")
                fig.update_layout(height=320)
                st.plotly_chart(fig, width="stretch")
                st.dataframe(sdf.head(40), width="stretch", hide_index=True)
                break

    with st.expander("Raw JSON (truncated)"):
        st.code(json.dumps(report, indent=2, default=str)[:12000], language="json")


def page_probe():
    st.subheader("Load & probe XGBoost fuel-moisture models")
    artifacts = list_xgb_artifacts()
    if not artifacts:
        st.warning("No fuel_moisture `.json` boosters found under the local registry.")
        return

    labels = [f"{a.model_type} · {a.channel} · {a.version or a.path.name}" for a in artifacts]
    pick = st.selectbox("Artifact", range(len(artifacts)), format_func=lambda i: labels[i])
    record = artifacts[pick]
    st.write(f"`{record.path}`")
    if record.performance:
        st.json(record.performance)

    if not record.path or not record.path.exists():
        st.error("Artifact file missing.")
        return

    try:
        import xgboost as xgb
    except ImportError:
        st.error("xgboost is not installed in this environment.")
        return

    if st.button("Load booster", type="primary"):
        booster = xgb.Booster()
        booster.load_model(str(record.path))
        st.session_state["probe_booster"] = booster
        st.session_state["probe_path"] = str(record.path)

    booster = st.session_state.get("probe_booster")
    if booster is None or st.session_state.get("probe_path") != str(record.path):
        st.info("Click **Load booster** to inspect this artifact.")
        return

    config = json.loads(booster.save_config())
    learner = config.get("learner", {})
    st.markdown("#### Booster summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Trees", booster.num_boosted_rounds())
    c2.metric("Objective", str(learner.get("objective") or learner.get("learner_model_param", {}).get("objective", "—")))
    feature_names = booster.feature_names
    c3.metric("Features", len(feature_names) if feature_names else "—")

    if feature_names:
        with st.expander("Feature names"):
            st.write(feature_names)

    try:
        score = booster.get_score(importance_type="gain")
        if score:
            imp = pd.DataFrame({"feature": list(score.keys()), "gain": list(score.values())})
            imp = imp.sort_values("gain", ascending=False)
            fig = px.bar(imp.head(30), x="gain", y="feature", orientation="h",
                         title="Feature importance (gain)")
            fig.update_layout(height=560, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(imp, width="stretch", hide_index=True)
        else:
            st.write("No importance scores available for this booster.")
    except Exception as exc:  # noqa: BLE001 — surface probe failures in UI
        st.warning(f"Could not read importance: {exc}")

    st.markdown("#### Quick predict")
    st.caption("Paste a JSON object of feature_name → value (same schema the booster expects).")
    sample = ""
    if feature_names:
        sample = json.dumps({name: 0.0 for name in feature_names[: min(12, len(feature_names))]}, indent=2)
        if len(feature_names) > 12:
            sample = json.dumps({name: 0.0 for name in feature_names}, indent=2)
    payload = st.text_area("Feature vector JSON", value=sample, height=220)
    if st.button("Predict"):
        try:
            values = json.loads(payload)
            if feature_names:
                row = [float(values.get(name, np.nan)) for name in feature_names]
                dmat = xgb.DMatrix(np.array([row], dtype=float), feature_names=feature_names)
            else:
                keys = sorted(values)
                dmat = xgb.DMatrix(np.array([[float(values[k]) for k in keys]], dtype=float))
            pred = float(booster.predict(dmat)[0])
            st.success(f"Predicted fuel moisture: **{pred:.3f}%**")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Predict failed: {exc}")


def page_single_day():
    """Fast single-day score without session range scoring."""
    st.subheader("Single-day quick look")
    dates = list(discover_forecast_dates())
    if not dates:
        st.warning("No forecast archives.")
        return
    date_token = st.selectbox("Date", list(reversed(dates)), format_func=lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}")
    available = list(discover_forecast_series())
    present = [s for s, p in list_series_for_date(date_token, available).items() if p]
    series = st.radio("Series", present or available, horizontal=True)
    merged, meta = load_scored_series(date_token, series)
    st.json(meta)
    if merged.empty:
        st.error("No matched rows.")
        return
    merged = attach_obs_fire_danger(merged)
    metrics = calculate_metrics(merged)
    cols = st.columns(4)
    fm = metrics["Fuel Moisture (%)"]
    cols[0].metric("FM MAE", _fmt(fm.get("mae")))
    cols[1].metric("FM Bias", _fmt(fm.get("bias")))
    cols[2].metric("FM RMSE", _fmt(fm.get("rmse")))
    cols[3].metric("N", _fmt(fm.get("count"), 0))
    st.json(metrics)
    fig = px.scatter(merged.dropna(subset=["pred_fm", "obs_fm"]), x="obs_fm", y="pred_fm",
                     hover_data=["stid", "timestamp"], opacity=0.5)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(metrics_by_station(merged), width="stretch", hide_index=True)


def page_data_sync():
    from datetime import date, timedelta

    from model_lab.cdn_sync import (
        CDN_BASE_URL,
        R2_ARCHIVE_PREFIX,
        discover_remote_archives,
        list_local_zips,
        local_sync_status,
        r2_credentials_present,
        sync_dates,
        unpack_local_zip,
    )

    st.subheader("CDN archive pull & unpack")
    st.caption(
        f"Public CDN: `{CDN_BASE_URL}/{R2_ARCHIVE_PREFIX}/YYYYMMDD.zip` · "
        f"local zips → `{DATA_ROOT / 'archive_zips'}` · "
        "oversized full-CONUS HRRR members are refused on unpack."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Local zips", len(list_local_zips()))
    c2.metric("R2 credentials", "yes" if r2_credentials_present() else "no (CDN HEAD/GET)")
    lookback = c3.number_input("Coverage lookback (days)", min_value=3, max_value=3650, value=30)

    if st.button("Refresh coverage table"):
        st.session_state.pop("sync_coverage", None)

    if "sync_coverage" not in st.session_state:
        with st.spinner("Probing CDN for recent dates…"):
            st.session_state["sync_coverage"] = local_sync_status(lookback_days=int(lookback))
    coverage = st.session_state["sync_coverage"]
    st.dataframe(coverage, width="stretch", hide_index=True)

    st.markdown("#### Pull selected dates")
    st.info(
        "This downloads archive bundles. It does not automatically rebuild the "
        "Fire Danger training CSV; after downloading, the raw observations still "
        "need to be prepared into labeled training rows."
    )
    end = date.today()
    start = end - timedelta(days=int(lookback) - 1)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        start_in = st.date_input("Start", value=start)
    with col_b:
        end_in = st.date_input("End", value=end)
    with col_c:
        unpack_after = st.checkbox("Unpack after download", value=True)

    prefer_r2 = st.checkbox("Prefer R2 API when credentials exist", value=False)
    force = st.checkbox("Force re-download even if local zip exists", value=False)

    remote = []
    if st.button("List remote archives in range"):
        with st.spinner("Listing…"):
            remote = discover_remote_archives(start=start_in, end=end_in, prefer_r2=prefer_r2 and r2_credentials_present())
            st.session_state["remote_archives"] = [
                {"date": item.date, "mb": round((item.size_bytes or 0) / 1e6, 1), "source": item.source, "url": item.public_url}
                for item in remote
            ]
    remote_rows = st.session_state.get("remote_archives")
    if remote_rows:
        st.dataframe(pd.DataFrame(remote_rows), width="stretch", hide_index=True)
        available_dates = [row["date"] for row in remote_rows]
    else:
        available_dates = [
            row["date"] for _, row in coverage.iterrows() if bool(row.get("on_cdn"))
        ]

    default_dates = [d for d in available_dates if d >= (date.today() - timedelta(days=3)).strftime("%Y%m%d")]
    pick = st.multiselect(
        "Dates to pull",
        options=available_dates,
        default=default_dates[-5:] if default_dates else [],
    )
    st.warning(
        "Each day is often ~400–500 MB because HRRR grids are bundled. "
        "Forecasts/obs still extract; oversized .nc is skipped."
    )

    if st.button("Download + unpack selected", type="primary", disabled=not pick):
        progress = st.progress(0.0, text="Starting…")
        status = st.empty()

        def _cb(date_token, written, total):
            if total:
                progress.progress(min(written / total, 1.0), text=f"{date_token}: {written/1e6:.1f}/{total/1e6:.1f} MB")
            else:
                status.write(f"{date_token}: {written/1e6:.1f} MB")

        results = sync_dates(
            pick,
            unpack=unpack_after,
            force_download=force,
            use_r2=prefer_r2 and r2_credentials_present(),
            progress=_cb,
        )
        progress.empty()
        st.session_state.pop("sync_coverage", None)
        st.dataframe(pd.DataFrame([r.to_dict() for r in results]), width="stretch", hide_index=True)
        errors = [r for r in results if r.error]
        if errors:
            st.error(f"{len(errors)} date(s) failed.")
        else:
            st.success("Sync finished. Compare tabs will see new forecasts after refresh.")

    st.markdown("#### Unpack local zips only")
    if st.button("Unpack all archive_zips"):
        rows = []
        for path in list_local_zips():
            stats = unpack_local_zip(path)
            rows.append({"zip": path.name, **{k: v for k, v in stats.items() if k not in ("unrecognized_names", "oversized_names")}})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("#### Build and validate Fire Danger dataset")
    st.caption(
        "Uses the archived Synoptic station observations to create a separate "
        "versioned FD training CSV. This does not overwrite Fuel Moisture data."
    )
    if st.button("Build FD dataset from archived station data", type="primary"):
        from model_lab.dataset_builder import build_fire_danger_dataset

        with st.spinner("Parsing station observations and validating coverage…"):
            try:
                report = build_fire_danger_dataset()
                st.session_state["fd_dataset_report"] = report
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
    if st.session_state.get("fd_dataset_report"):
        report = st.session_state["fd_dataset_report"]
        st.json(report)
        if report.get("ready_for_basic_training"):
            st.success("FD dataset has enough support for the staged three-class model.")
            if not report.get("ready_for_training"):
                st.warning("Five-class training remains blocked until Critical and Extreme support improves.")
        else:
            st.warning(
                "FD dataset was built, but its coverage/class support is not "
                "ready for three-class training yet."
            )

    st.markdown("#### Pull station history directly from Synoptic")
    st.caption(
        "This uses the API checkout's rolling-history backfill and requires "
        "`SYNOPTIC_API_TOKEN` in the API environment. It writes only to the "
        "local training-data archive."
    )
    from datetime import date, timedelta

    synoptic_end = st.date_input(
        "Synoptic history end date",
        value=date.today() - timedelta(days=1),
        key="synoptic_backfill_end",
    )
    synoptic_start = st.date_input(
        "Synoptic history start date",
        value=synoptic_end - timedelta(days=365),
        key="synoptic_backfill_start",
    )
    if st.button("Pull Synoptic station history", key="pull_synoptic_history"):
        if synoptic_start > synoptic_end:
            st.error("Start date must be on or before the end date.")
        else:
            api_script = REPO_ROOT.parent / "api" / "scripts" / "backfill_synoptic.py"
            command = [
                sys.executable,
                str(api_script),
                "--start",
                synoptic_start.isoformat(),
                "--end",
                synoptic_end.isoformat(),
                "--archive-dir",
                str(ARCHIVE_RAW_DATA_DIR),
            ]
            env = os.environ.copy()
            env["SMF_DATA_ROOT"] = str(DATA_ROOT)
            with st.spinner("Pulling station history from Synoptic…"):
                result = subprocess.run(
                    command,
                    cwd=str(api_script.parent.parent),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                    check=False,
                )
            if result.returncode == 0:
                st.success("Synoptic backfill completed. Rebuild the FD dataset above.")
            else:
                st.error("Synoptic backfill failed. Check the token and API access.")
            if result.stdout:
                st.code(result.stdout[-8000:])
            if result.stderr:
                st.code(result.stderr[-8000:], language="text")

    st.markdown("#### Pull matching RTMA + HRRR weather history")
    st.caption(
        "This downloads Missouri-cropped HRRR F04–F15 runs and the RTMA analyses "
        "needed to validate them. It can take a long time and requires network access."
    )
    if st.button("Pull RTMA + HRRR for this date range", key="pull_rtma_hrrr_history"):
        model_training_root = REPO_ROOT
        common_dates = ["--start", synoptic_start.isoformat(), "--end", synoptic_end.isoformat()]
        env = os.environ.copy()
        env["SMF_DATA_ROOT"] = str(DATA_ROOT)
        # HRRR workers emit status symbols; force UTF-8 on Windows so a
        # successful fetch is not recorded as failed by a console encoding error.
        env["PYTHONIOENCODING"] = "utf-8"
        hrrr_command = [
            sys.executable,
            str(model_training_root / "scripts" / "backfill_hrrr.py"),
            *common_dates,
            "--init-hour",
            "12",
            "--workers",
            "2",
        ]
        rtma_command = [
            sys.executable,
            str(model_training_root / "scripts" / "backfill_rtma_for_hrrr.py"),
            *common_dates,
            "--workers",
            "2",
        ]
        with st.spinner("Pulling HRRR history…"):
            hrrr_result = subprocess.run(
                hrrr_command,
                cwd=str(model_training_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=86400,
                check=False,
            )
        st.code(hrrr_result.stdout[-8000:] or hrrr_result.stderr[-8000:], language="text")
        if hrrr_result.returncode != 0:
            st.error("HRRR backfill failed; RTMA was not started.")
        else:
            with st.spinner("Pulling matching RTMA analyses…"):
                rtma_result = subprocess.run(
                    rtma_command,
                    cwd=str(model_training_root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=86400,
                    check=False,
                )
            st.code(rtma_result.stdout[-8000:] or rtma_result.stderr[-8000:], language="text")
            if rtma_result.returncode == 0:
                st.success("HRRR and RTMA backfill completed.")
            else:
                st.error("RTMA backfill failed; inspect the output above.")


def page_schedule():
    from model_lab.schedule import (
        LAB_JOBS,
        PRODUCTION_RUNTIME,
        apply_schedule,
        load_schedule_config,
        read_history,
        run_job,
        save_schedule_config,
        scheduler_status,
    )

    st.subheader("Schedule & model runtime")
    st.markdown("#### Production runtime (reference)")
    st.caption("What the live API host runs — not executed by this lab.")
    st.dataframe(pd.DataFrame(PRODUCTION_RUNTIME), width="stretch", hide_index=True)

    st.markdown("#### Local lab jobs")
    cfg = load_schedule_config()
    status_rows = scheduler_status()
    st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True)

    st.markdown("##### Edit schedule")
    edited = {"timezone": cfg.get("timezone") or "America/Chicago", "jobs": {}}
    for spec in LAB_JOBS:
        job_cfg = cfg.get("jobs", {}).get(spec.id, {})
        with st.expander(f"{spec.name} (`{spec.id}`)", expanded=False):
            st.write(spec.description)
            enabled = st.checkbox("Enabled", value=bool(job_cfg.get("enabled")), key=f"en_{spec.id}")
            c1, c2 = st.columns(2)
            hour = c1.number_input("Hour (CT)", min_value=0, max_value=23,
                                   value=int(job_cfg.get("hour", spec.default_hour)), key=f"h_{spec.id}")
            minute = c2.number_input("Minute", min_value=0, max_value=59,
                                     value=int(job_cfg.get("minute", spec.default_minute)), key=f"m_{spec.id}")
            params = dict(job_cfg.get("params") or spec.params)
            if "days" in params:
                params["days"] = st.number_input("Lookback days", min_value=1, max_value=60,
                                                 value=int(params.get("days", 7)), key=f"days_{spec.id}")
            if "unpack" in params:
                params["unpack"] = st.checkbox("Unpack after pull", value=bool(params.get("unpack", True)),
                                               key=f"unp_{spec.id}")
            edited["jobs"][spec.id] = {
                "enabled": enabled,
                "hour": int(hour),
                "minute": int(minute),
                "params": params,
            }
            if st.button("Run now", key=f"run_{spec.id}"):
                with st.spinner(f"Running {spec.id}…"):
                    entry = run_job(spec.id, params=params)
                if entry.get("ok"):
                    st.success(f"Finished `{spec.id}`")
                else:
                    st.error(entry.get("error") or "Job failed")
                st.json(entry.get("result") or entry)

    if st.button("Save & apply schedule", type="primary"):
        save_schedule_config(edited)
        apply_schedule(edited)
        st.success("Schedule saved. Enabled jobs are registered on the background scheduler.")
        st.rerun()

    st.markdown("#### Recent job history")
    history = read_history(40)
    if not history:
        st.write("No runs yet.")
    else:
        summary = pd.DataFrame([
            {
                "started_at": row.get("started_at"),
                "job_id": row.get("job_id"),
                "ok": row.get("ok"),
                "finished_at": row.get("finished_at"),
                "error": row.get("error"),
            }
            for row in history
        ])
        st.dataframe(summary, width="stretch", hide_index=True)
        with st.expander("Latest result detail"):
            st.json(history[0])


def _render_scorecard(summary: dict, scorecard_df: pd.DataFrame, checks: dict | None = None):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidate MAE", _fmt(summary.get("candidate_mae")))
    c2.metric("Control MAE", _fmt(summary.get("control_mae")))
    delta = summary.get("delta_mae")
    c3.metric("Δ MAE (cand − ctrl)", _fmt(delta))
    rel = summary.get("relative_improvement")
    c4.metric("Rel. improvement", "—" if rel is None else f"{rel * 100:.1f}%")
    if summary.get("pass") is True:
        st.success("Gate / beat-control: PASS")
    elif summary.get("pass") is False:
        st.error("Gate / beat-control: FAIL — candidate did not beat the control under current rules.")
    if checks:
        st.dataframe(pd.DataFrame([{"check": k, "pass": v} for k, v in checks.items()]),
                     width="stretch", hide_index=True)
    if scorecard_df is not None and not scorecard_df.empty:
        st.dataframe(scorecard_df, width="stretch", hide_index=True)
        if "mae" in scorecard_df.columns:
            fig = px.bar(scorecard_df.dropna(subset=["mae"]), x="model", y="mae", color="model",
                         title="MAE comparison")
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig, width="stretch")


def page_workshop():
    from model_lab.workshop import (
        evaluate_station_vs_control,
        get_experiment,
        list_experiments,
        scorecard_from_eval,
        train_legacy_xgb_candidate,
        train_station_candidate,
        workshop_data_status,
    )

    st.subheader("Model workshop")
    st.caption(
        "Train a candidate, test it on the same locked data as the control, and see whether it beats production-style baselines. "
        "Station sequence is the real gated path; legacy XGB is the fast loop."
    )

    status = workshop_data_status()
    with st.expander("Data readiness", expanded=not status["station_dataset_exists"]):
        st.json(status)
        if not status["station_dataset_exists"]:
            st.warning("Station aligned dataset missing — use Data sync / training pipelines first.")
        if not status["legacy_dataset_exists"]:
            st.warning("Legacy `final_training_data.csv` missing — legacy XGB train will fail.")

    mode = st.radio(
        "Workflow",
        ["Train station sequence", "Train legacy XGB (fast)", "Evaluate existing checkpoint", "Experiment shelf"],
        horizontal=True,
    )

    if mode == "Train station sequence":
        st.markdown("#### Train station-sequence candidate")
        c1, c2, c3 = st.columns(3)
        epochs = c1.number_input("Epochs", min_value=1, max_value=80, value=15)
        fold = c2.selectbox("Fold", ["fold-3", "fold-2", "fold-1"])
        hidden = c3.selectbox("Hidden size", [16, 32, 64], index=0)
        c4, c5, c6 = st.columns(3)
        lr = c4.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
        layers = c5.selectbox("GRU layers", [1, 2], index=0)
        patience = c6.number_input("Patience", min_value=1, max_value=20, value=5)
        notes = st.text_input("Notes", placeholder="why this run?")
        auto_eval = st.checkbox("Evaluate vs control after training", value=True)
        st.info("Typical train is a few minutes on GPU, longer on CPU. Eval refits an XGB control on the locked holdout.")
        if st.button("Train candidate", type="primary", disabled=not status["station_dataset_exists"]):
            with st.spinner("Training station-sequence model…"):
                try:
                    result = train_station_candidate(
                        epochs=int(epochs), fold=fold, hidden_size=int(hidden),
                        num_layers=int(layers), learning_rate=float(lr), patience=int(patience),
                        notes=notes,
                    )
                    st.session_state["workshop_last_train"] = result
                    st.success(f"Trained `{result['id']}` · val MAE {_fmt((result.get('summary') or {}).get('validation_mae'))}")
                    st.json(result["summary"])
                    if auto_eval:
                        with st.spinner("Evaluating vs incumbent control on locked holdout…"):
                            ev = evaluate_station_vs_control(result["checkpoint"], notes=notes)
                            st.session_state["workshop_last_eval"] = ev
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)

        if st.session_state.get("workshop_last_eval"):
            ev = st.session_state["workshop_last_eval"]
            st.markdown("#### Latest scorecard")
            _render_scorecard(ev["summary"], scorecard_from_eval(report=ev.get("report")),
                              checks=(ev.get("report") or {}).get("checks"))

    elif mode == "Train legacy XGB (fast)":
        st.markdown("#### Train legacy fuel-moisture XGBoost")
        c1, c2, c3 = st.columns(3)
        trees = c1.number_input("Trees", min_value=50, max_value=1000, value=200, step=50)
        lr = c2.number_input("Learning rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        depth = c3.number_input("Max depth", min_value=2, max_value=12, value=5)
        notes = st.text_input("Notes", key="legacy_notes")
        register_beta = st.checkbox("Also register as local registry beta", value=False)
        st.caption("Compares holdout skill to registry stable/beta if present, else EMC baseline.")
        if st.button("Train legacy model", type="primary", disabled=not status["legacy_dataset_exists"]):
            with st.spinner("Training XGBoost…"):
                try:
                    result = train_legacy_xgb_candidate(
                        n_estimators=int(trees), learning_rate=float(lr), max_depth=int(depth),
                        notes=notes, register_beta=register_beta,
                    )
                    st.session_state["workshop_last_legacy"] = result
                    st.success(f"Trained `{result['id']}`")
                    _render_scorecard(result["summary"], scorecard_from_eval(report=result.get("report")))
                    st.caption(f"Control source: {result['summary'].get('control_source')}")
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)

    elif mode == "Evaluate existing checkpoint":
        st.markdown("#### Evaluate a checkpoint vs control")
        experiments = [e for e in list_experiments(100) if e.get("family") == "station_sequence" and e.get("checkpoint")]
        if not experiments:
            st.warning("No station checkpoints in experiments/. Train one first.")
        else:
            labels = [f"{e['id']} · {e.get('status')}" for e in experiments]
            pick = st.selectbox("Checkpoint", range(len(experiments)), format_func=lambda i: labels[i])
            chosen = experiments[pick]
            st.write(chosen.get("checkpoint"))
            if st.button("Run holdout evaluation", type="primary"):
                with st.spinner("Evaluating…"):
                    try:
                        ev = evaluate_station_vs_control(chosen["checkpoint"])
                        st.session_state["workshop_last_eval"] = ev
                        _render_scorecard(ev["summary"], scorecard_from_eval(report=ev.get("report")),
                                          checks=(ev.get("report") or {}).get("checks"))
                    except Exception as exc:  # noqa: BLE001
                        st.exception(exc)
            if st.session_state.get("workshop_last_eval"):
                ev = st.session_state["workshop_last_eval"]
                if ev.get("id") == chosen["id"]:
                    st.markdown("#### Scorecard")
                    _render_scorecard(ev["summary"], scorecard_from_eval(report=ev.get("report")),
                                      checks=(ev.get("report") or {}).get("checks"))

    else:
        st.markdown("#### Experiment shelf")
        rows = list_experiments(100)
        if not rows:
            st.info("No experiments yet. Train something above.")
        else:
            table = pd.DataFrame([
                {
                    "id": r.get("id"),
                    "family": r.get("family"),
                    "status": r.get("status"),
                    "candidate_mae": (r.get("summary") or {}).get("candidate_mae") or (r.get("summary") or {}).get("validation_mae"),
                    "control_mae": (r.get("summary") or {}).get("control_mae"),
                    "delta_mae": (r.get("summary") or {}).get("delta_mae"),
                    "pass": (r.get("summary") or {}).get("pass"),
                    "created_at": r.get("created_at"),
                    "notes": r.get("notes"),
                }
                for r in rows
            ])
            st.dataframe(table, width="stretch", hide_index=True)
            ids = [r["id"] for r in rows]
            selected = st.selectbox("Inspect", ids)
            detail = get_experiment(selected)
            if detail:
                st.json(detail)
                if detail.get("eval_report") and Path(detail["eval_report"]).exists():
                    report = json.loads(Path(detail["eval_report"]).read_text(encoding="utf-8"))
                    _render_scorecard(detail.get("summary") or {}, scorecard_from_eval(report=report),
                                      checks=report.get("checks"))


def _rank_label(rank: str) -> str:
    return {
        "candidate": "Candidate",
        "challenger": "Challenger",
        "local_beta": "Local Beta",
        "local_stable": "Local Stable",
        "archived": "Archived",
    }.get(rank, rank)


def _render_unified_ladder(product: str):
    from model_lab.workshop import ladder_table, promote_experiment, demote_experiment

    st.subheader("Model ladder")
    st.caption("Local ranks only. These buttons do not change the live API.")
    table = ladder_table(product)
    if table.empty:
        st.info("No experiments for this product yet. Train one below.")
        return table

    display = table.copy()
    display["rank"] = display["rank"].map(_rank_label)
    display["metric"] = display["macro_f1"] if product == "fire_danger" else display["candidate_mae"]
    display = display[["id", "family", "rank", "metric", "control_mae", "delta_mae", "daily_mae", "daily_macro_f1", "pass", "status"]]
    display = display.rename(columns={
        "metric": "Macro F1" if product == "fire_danger" else "MAE",
        "control_mae": "Control MAE",
        "delta_mae": "Δ MAE",
        "daily_mae": "Daily MAE",
        "daily_macro_f1": "Daily Macro F1",
    })
    st.dataframe(display, width="stretch", hide_index=True)

    ids = table["id"].tolist()
    selected = st.selectbox("Selected model", ids, format_func=lambda value: f"{value} · {_rank_label(table.loc[table.id == value, 'rank'].iloc[0])}")
    action_a, action_b = st.columns(2)
    with action_a:
        if st.button("Move up local ladder", key=f"up_{product}"):
            try:
                promote_experiment(selected)
                st.success("Moved up locally.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with action_b:
        if st.button("Move down / archive", key=f"down_{product}"):
            try:
                demote_experiment(selected)
                st.success("Moved down locally.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    st.session_state[f"unified_selected_{product}"] = selected
    return table


def _render_unified_train(product: str):
    from model_lab.workshop import (
        evaluate_station_vs_control,
        scorecard_from_eval,
        train_fire_danger_candidate,
        train_legacy_xgb_candidate,
        train_station_candidate,
        workshop_data_status,
    )

    st.subheader("Create and train")
    status = workshop_data_status()
    with st.container(border=True):
        st.markdown("**Data readiness**")
        if product == "fuel_moisture":
            st.write(
                f"Legacy dataset: `{status['legacy_dataset']}` · "
                f"{'ready' if status['legacy_dataset_exists'] else 'missing'}"
            )
        else:
            st.write(
                f"Fire-danger dataset: `{status['fd_dataset']}` · "
                f"{'ready for 3-class training' if status['fd_dataset_ready'] else 'needs more class support'}"
            )
            if status["fd_dataset_exists"] and not status["fd_dataset_ready"]:
                st.warning(
                    "The dataset exists, but the validation report does not show "
                    "at least 25 rows for Low, Moderate, and Elevated+."
                )
        ready = status["legacy_dataset_exists"] if product == "fuel_moisture" else status["fd_dataset_ready"]
        if not ready:
            st.warning("Use the Data tab to download/unpack data before training.")
    if product == "fuel_moisture":
        path = st.radio("Fuel-moisture model path", ["Fast XGBoost", "Station sequence"], horizontal=True)
        if path == "Fast XGBoost":
            c1, c2, c3 = st.columns(3)
            trees = c1.number_input("Trees", 20, 1000, 50, step=10, key="unified_trees")
            lr = c2.number_input("Learning rate", 0.01, 0.5, 0.11, step=0.01, key="unified_lr")
            depth = c3.number_input("Max depth", 2, 12, 5, key="unified_depth")
            notes = st.text_input("Run notes", key="unified_fm_notes")
            if st.button("Train fuel-moisture candidate", type="primary",
                         disabled=not status["legacy_dataset_exists"]):
                with st.spinner("Training fuel-moisture XGBoost…"):
                    try:
                        result = train_legacy_xgb_candidate(
                            n_estimators=int(trees), learning_rate=float(lr),
                            max_depth=int(depth), notes=notes,
                        )
                        st.session_state["unified_last_result"] = result
                        st.success(f"Created {result['id']}")
                    except Exception as exc:
                        st.exception(exc)
        else:
            c1, c2, c3 = st.columns(3)
            epochs = c1.number_input("Epochs", 1, 80, 20, key="unified_epochs")
            hidden = c2.selectbox("Hidden size", [16, 32, 64], key="unified_hidden")
            fold = c3.selectbox("Fold", ["fold-3", "fold-2", "fold-1"], key="unified_fold")
            if st.button("Train station-sequence candidate", type="primary",
                         disabled=not status["station_dataset_exists"]):
                with st.spinner("Training station sequence…"):
                    try:
                        result = train_station_candidate(
                            epochs=int(epochs), hidden_size=int(hidden), fold=fold,
                        )
                        evaluate = evaluate_station_vs_control(result["checkpoint"])
                        result["evaluation"] = evaluate
                        st.session_state["unified_last_result"] = result
                        st.success(f"Created {result['id']}")
                    except Exception as exc:
                        st.exception(exc)
    else:
        st.caption("Fire danger uses Macro F1 and class-support gates, not fuel-moisture MAE.")
        c1, c2, c3 = st.columns(3)
        trees = c1.number_input("Trees", 50, 1000, 300, step=50, key="unified_fd_trees")
        lr = c2.number_input("Learning rate", 0.01, 0.5, 0.05, step=0.01, key="unified_fd_lr")
        depth = c3.number_input("Max depth", 2, 12, 5, key="unified_fd_depth")
        notes = st.text_input("Run notes", key="unified_fd_notes")
        if st.button("Train fire-danger candidate", type="primary",
                     disabled=not status["fd_dataset_exists"]):
            with st.spinner("Preparing and training fire-danger XGBoost…"):
                try:
                    result = train_fire_danger_candidate(
                        n_estimators=int(trees), learning_rate=float(lr),
                        max_depth=int(depth), notes=notes,
                    )
                    st.session_state["unified_last_result"] = result
                    st.success(f"Created {result['id']}")
                except Exception as exc:
                    st.exception(exc)

    result = st.session_state.get("unified_last_result")
    if result:
        st.markdown("#### Latest test result")
        summary = result.get("summary") or {}
        if product == "fire_danger":
            st.metric("Macro F1", _fmt(summary.get("macro_f1")))
            st.metric("Gate", "PASS" if summary.get("pass") else "FAIL")
        else:
            report = result.get("evaluation") or result.get("report")
            if report:
                st.dataframe(scorecard_from_eval(report=report), width="stretch", hide_index=True)
                if report.get("checks"):
                    st.dataframe(pd.DataFrame([
                        {"check": key, "pass": value}
                        for key, value in report["checks"].items()
                    ]), width="stretch", hide_index=True)


def _render_unified_daily(product: str):
    from model_lab.daily import cache_daily_scores, daily_summary

    st.subheader("Daily results")
    days = st.slider("Recent days", 3, 30, 14, key=f"daily_days_{product}")
    if st.button("Refresh daily scores", key=f"refresh_daily_{product}"):
        st.session_state[f"daily_{product}"] = cache_daily_scores(days=days, force=True)
    scores = st.session_state.get(f"daily_{product}")
    if scores is None:
        with st.spinner("Scoring archive series against observations…"):
            scores = cache_daily_scores(days=days)
            st.session_state[f"daily_{product}"] = scores
    if scores.empty:
        st.info("No matched daily archive data yet.")
        return
    metric = "fm_mae" if product == "fuel_moisture" else "fd_macro_f1"
    summary = daily_summary(product=product, days=days)
    st.dataframe(summary, width="stretch", hide_index=True)
    chart = px.line(
        scores.dropna(subset=[metric]),
        x="date", y=metric, color="series", markers=True,
        color_discrete_map={series: _series_color(series) for series in scores["series"].unique()},
        labels={metric: "FM MAE (lower better)" if product == "fuel_moisture" else "FD Macro F1 (higher better)"},
    )
    st.plotly_chart(chart, width="stretch")


def _render_unified_graphics(product: str):
    from model_lab.beta_forecast import run_beta_forecast
    from model_lab.workshop import list_experiments

    st.subheader("Create local beta forecast graphics")
    st.warning(
        "Local only: CDN upload is disabled. A forecast needs both an FM model "
        "and an FD model; selecting Fire Danger does not remove the FM initialization step."
    )
    experiments = list_experiments(200)
    fm_candidates = [
        row for row in list_experiments(200)
        if row.get("checkpoint")
        and row.get("family") in ("fuel_moisture_xgb", "station_sequence")
    ]
    fd_candidates = [
        row for row in experiments
        if row.get("checkpoint") and row.get("family") in (
            "fire_danger_xgb",
            "fire_danger_xgb_three_class",
        )
    ]
    if not fm_candidates:
        st.error(
            "No fuel-moisture artifact is available. Train an FM candidate first; "
            "the forecast cannot fall back to an API stable model that is not registered."
        )
        return
    fm_labels = [f"{row['id']} · {row.get('family')}" for row in fm_candidates]
    fm_pick = st.selectbox(
        "Fuel-moisture model (required)",
        range(len(fm_candidates)),
        format_func=lambda index: fm_labels[index],
        key=f"graphics_fm_pick_{product}",
    )
    selected_fm = fm_candidates[fm_pick]

    selected_fd = None
    if fd_candidates:
        fd_labels = [f"{row['id']} · {row.get('family')}" for row in fd_candidates]
        fd_pick = st.selectbox(
            "Fire-danger model (optional for FM maps; required for FD maps)",
            range(len(fd_candidates)),
            format_func=lambda index: fd_labels[index],
            key=f"graphics_fd_pick_{product}",
        )
        selected_fd = fd_candidates[fd_pick]
    elif product == "fire_danger":
        st.error("No fire-danger artifact is available. Train an FD candidate first.")
        return
    else:
        st.info("No local FD candidate exists; FM maps will use the API fire-danger copy.")

    tag = st.text_input(
        "Forecast tag",
        value=f"{product}_{selected_fm['id'][-8:]}",
        key=f"graphics_tag_{product}",
    )
    if st.button("Generate local beta graphics", type="primary", key=f"graphics_run_{product}"):
        try:
            kwargs = {
                "tag": tag,
                "fm_model_path": selected_fm["checkpoint"],
            }
            if selected_fd:
                kwargs["fd_model_path"] = selected_fd["checkpoint"]
                kwargs["fd_meta_path"] = selected_fd.get("train_report")
            with st.spinner("Running beta forecast and generating maps…"):
                result = run_beta_forecast(**kwargs)
            st.session_state["unified_graphics_result"] = result
            if result["ok"]:
                st.success("Local beta forecast completed.")
            else:
                st.error("Forecast process failed.")
        except Exception as exc:
            st.exception(exc)
    result = st.session_state.get("unified_graphics_result")
    if result:
        st.json({key: value for key, value in result.items() if key not in ("stdout", "stderr")})
        for image in result.get("maps", []):
            st.image(image, caption=Path(image).name, width="stretch")
        if result.get("stderr"):
            with st.expander("Forecast logs"):
                st.code(result["stderr"])


def _render_unified_live(product: str):
    from model_lab.live_actions import import_release, promote_api, shadow_diagnostics, upload_maps

    st.error("LIVE ACTIONS: these controls can change the API registry, public forecasts, or CDN.")
    token = st.text_input("Type LIVE to unlock", type="password", key=f"live_token_{product}")
    action = st.selectbox("LIVE operation", ["API import beta", "API promote beta → stable", "Read shadow diagnostics", "Upload maps to CDN"], key=f"live_action_{product}")
    model_type = "fuel_moisture" if product == "fuel_moisture" else "fire_danger"
    if action == "API import beta":
        tag = st.text_input("GitHub release tag", key=f"live_release_tag_{product}")
        repo = st.text_input("GitHub repository", key=f"live_repo_{product}")
        if st.button("LIVE: Import beta", key=f"live_import_{product}"):
            st.json(import_release(model_type, tag, repo, token))
    elif action == "API promote beta → stable":
        version = st.text_input("Version (blank = current API beta)", key=f"live_version_{product}")
        if st.button("LIVE: Promote to stable", key=f"live_promote_{product}"):
            st.json(promote_api(model_type, token, version or None))
    elif action == "Read shadow diagnostics":
        if st.button("LIVE: Read diagnostics", key=f"live_diag_{product}"):
            st.json(shadow_diagnostics(token))
    else:
        files = st.text_area("Local map paths, one per line", key=f"live_files_{product}").splitlines()
        keys = st.text_area("CDN keys, one per line", key=f"live_keys_{product}").splitlines()
        if st.button("LIVE: Upload maps", key=f"live_upload_{product}"):
            st.json(upload_maps(files, keys, token))


def _render_experiments(product: str, *, key_suffix: str = ""):
    from model_lab.workshop import get_experiment, list_experiments

    rows = [
        row for row in list_experiments(200)
        if row.get("product", "fuel_moisture") == product
    ]
    st.subheader("Saved experiments")
    st.caption("Every training run stays here so you can compare it and try again.")
    if not rows:
        st.info("No experiments yet. Create one from the Train tab.")
        return

    summary_rows = []
    for row in rows:
        summary = row.get("summary") or {}
        summary_rows.append({
            "id": row.get("id"),
            "model": row.get("family"),
            "status": row.get("status"),
            "MAE": summary.get("candidate_mae") or summary.get("validation_mae"),
            "control MAE": summary.get("control_mae"),
            "Macro F1": summary.get("macro_f1"),
            "pass": summary.get("pass"),
            "created": row.get("created_at"),
        })
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    ids = [row["id"] for row in rows]
    selected_id = st.selectbox(
        "Inspect experiment",
        ids,
        key=f"experiment_inspect_{product}{key_suffix}",
    )
    detail = get_experiment(selected_id)
    if not detail:
        return
    summary = detail.get("summary") or {}
    if product == "fire_danger":
        cols = st.columns(4)
        cols[0].metric("Macro F1", _fmt(summary.get("macro_f1")))
        cols[1].metric("Baseline F1", _fmt(summary.get("baseline_macro_f1")))
        cols[2].metric("Gate", "PASS" if summary.get("pass") else "FAIL")
        cols[3].metric("Artifact", "available" if detail.get("checkpoint") else "missing")
        support = summary.get("support_by_class")
        if support:
            st.dataframe(
                pd.DataFrame([{"class": key, "support": value} for key, value in support.items()]),
                width="stretch",
                hide_index=True,
            )
    else:
        _render_scorecard(summary, pd.DataFrame(), checks=summary.get("checks"))
    st.caption(f"Artifact: `{detail.get('checkpoint')}`")
    if detail.get("eval_report") and Path(detail["eval_report"]).exists():
        with st.expander("Evaluation report JSON"):
            st.json(load_json(Path(detail["eval_report"])))


def _render_test_compare(product: str):
    st.subheader("Test and compare on the same holdout")
    st.info(
        "This is the decision screen: compare each candidate with the control "
        "using the same locked test rows. Do not judge a model by its map before "
        "it passes here."
    )
    _render_experiments(product, key_suffix="_test")

    st.divider()
    st.subheader("Optional: compare generated forecast archives")
    st.caption(
        "Use this only after a candidate passes holdout testing and has a tagged "
        "forecast. It compares production/beta forecast files against observations."
    )
    page_compare()


def unified_dashboard():
    header()
    product_label = st.radio("Product", ["Fuel Moisture", "Fire Danger"], horizontal=True)
    product = "fuel_moisture" if product_label == "Fuel Moisture" else "fire_danger"

    tabs = st.tabs([
        "1 · Data",
        "2 · Train",
        "3 · Test & compare",
        "4 · Experiments",
        "5 · Forecast map (optional)",
    ])

    with tabs[0]:
        st.header("Download and prepare data")
        st.caption("Pull archived observations and weather data before training.")
        page_data_sync()

    with tabs[1]:
        st.header("Train a candidate")
        st.caption(
            "Choose a model path, train on the prepared dataset, and save the run "
            "for testing. Training never changes production."
        )
        _render_unified_train(product)

    with tabs[2]:
        _render_test_compare(product)

    with tabs[3]:
        st.header("Experiments")
        st.caption("Review saved runs, metrics, gates, and artifact paths.")
        _render_experiments(product)

    with tabs[4]:
        st.header("Generate an optional forecast map")
        st.caption(
            "Select a saved model to run it against the cached HRRR data. This "
            "does not upload to the CDN or change production."
        )
        _render_unified_graphics(product)


def main():
    unified_dashboard()


if __name__ == "__main__":
    main()
