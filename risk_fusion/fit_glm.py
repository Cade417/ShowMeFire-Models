"""
L2-regularized Poisson GLM with real weather covariates for
fire_risk_fusion, fit and evaluated over the risk_fusion_contract.py
episode-block folds.

Kept to <=12 features per the project plan's Stage-A design: at the
label volumes available so far (~40 independent synoptic episodes'
worth of signal), a GBM's variance would dominate its bias advantage,
and a small, interpretable GLM is the honest first model - see
risk_fusion/labels.py and effort.py for how the target/offset are built.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_FEATURES = [
    "temp_max_c", "rh_mean", "rh_min_afternoon", "wind_kts_max", "wind_kts_p90",
    "vpd_kpa_max", "precip_24h_mm", "kbdi", "gdd_accum_since_mar1",
    "valid_doy_sin", "valid_doy_cos", "is_weekend",
]

# offset + pure calendar seasonality (2 harmonics), no weather at all. Kept
# available for comparison, but MONTH_DUMMY_COLUMNS below is the standing
# default: a smooth harmonic can't represent "March and April behave
# differently" without adding more harmonics, and per-month levels are
# what was explicitly asked for ("parameters change... month to month").
CLIMATOLOGY_FEATURES = ["valid_doy_sin", "valid_doy_cos"]

# Discrete per-calendar-month baseline: one level per month, not a smooth
# curve. January is the implicit reference level (folded into the
# intercept); month_2..month_12 are each month's deviation from January.
# This is the standing seasonal-baseline architecture for fire_risk_fusion:
# it produces a usable rate with NO weather input at all (the whole point
# of a baseline that "gives a picture without weather"), refined by
# FAST_WEATHER_FEATURES as a residual on top - see fit_residual_glm.
MONTH_DUMMY_COLUMNS = [f"month_{m}" for m in range(2, 13)]


def add_month_dummies(panel: pd.DataFrame, date_column: str = "valid_local_date") -> pd.DataFrame:
    """Adds month_2..month_12 indicator columns (0/1) to a copy of panel. January has no column - it's the reference level."""
    panel = panel.copy()
    month = pd.to_datetime(panel[date_column]).dt.month
    for m in range(2, 13):
        panel[f"month_{m}"] = (month == m).astype("float64")
    return panel


def monthly_baseline_rate_multipliers(fit: Dict) -> pd.Series:
    """
    exp(coefficient) for each month relative to January (the reference
    level) - "March's baseline rate is 2.1x January's", not a raw log-scale
    number. Only meaningful for a fit whose feature_columns is
    MONTH_DUMMY_COLUMNS (or a subset).
    """
    coefficients = pd.Series(fit["result"].params[1:], index=fit["feature_columns"])
    multipliers = {1: 1.0}
    for month_column, coefficient in coefficients.items():
        month_number = int(month_column.split("_")[1])
        # Coefficients are on the STANDARDIZED dummy scale (fit_glm_with_covariates
        # standardizes every feature column, dummies included) - undo that
        # to get back to a per-unit-indicator (0/1) effect before exponentiating.
        std = fit["stds"][month_column]
        multipliers[month_number] = float(np.exp(coefficient / std))
    return pd.Series(multipliers).sort_index()

# Genuinely day-to-day (synoptic) variables only - explicitly excludes
# kbdi and gdd_accum_since_mar1 (both slow accumulators that move on a
# seasonal timescale, highly correlated with day-of-year itself) and
# temp_max_c (strongly seasonal - hot in summer, cold in winter). Fitting
# these alongside doy in one ridge-penalized model lets the penalty split
# credit between overlapping slow-moving signals, diluting whatever fast
# signal is actually there - see fit_residual_glm, which avoids that by
# fitting this list as a correction on top of an already-fitted seasonal
# baseline instead of a competitor to it.
FAST_WEATHER_FEATURES = [
    "rh_mean", "rh_min_afternoon", "wind_kts_max", "wind_kts_p90",
    "vpd_kpa_max", "precip_24h_mm", "is_weekend",
]

# Cross-validated on the real 2014-2020 panel via out-of-fold log_score
# (crossfit_compare_residual across an alpha grid): the original alpha=1.0
# guess was confirmed too conservative - out-of-fold performance improves
# monotonically as alpha shrinks all the way to ~1e-4, but down there the
# fitted multiplier distribution becomes implausibly skewed (median drifts
# to ~0.68, max 5.76x) - overfitting to rare-event noise that log_score
# alone doesn't penalize enough to show up as worse. alpha=0.1 is the
# smallest value that keeps the multiplier distribution physically sane
# AND whose extreme-day ceiling (1.97x) independently matches the raw,
# model-free historical finding (1.96x for the driest+windiest quarter of
# days, see risk_fusion_evidence discussion) - two different methods
# landing on the same answer is the actual justification, not just "the
# out-of-fold score is a bit better here."
FAST_WEATHER_RESIDUAL_ALPHA = 0.1


def filter_valid_rows(panel: pd.DataFrame, feature_columns: List[str] = DEFAULT_FEATURES) -> pd.DataFrame:
    """
    Drops rows where kbdi is not yet valid (each county's 90-day KBDI
    spin-up window - see risk_fusion/features.py) rather than fitting on
    a biased-low KBDI value. A ~3.7% row loss on the real panel, not
    worth a more complex imputation scheme for this first model.
    """
    if "kbdi_valid" in panel.columns and "kbdi" in feature_columns:
        panel = panel[panel["kbdi_valid"]].copy()
    return panel


def _standardize(train: pd.DataFrame, test: Optional[pd.DataFrame], feature_columns: List[str]):
    means = train[feature_columns].mean()
    stds = train[feature_columns].std().replace(0, 1.0)
    train_scaled = (train[feature_columns] - means) / stds
    test_scaled = (test[feature_columns] - means) / stds if test is not None else None
    return train_scaled, test_scaled, means, stds


def fit_glm_with_covariates(
    train_panel: pd.DataFrame,
    feature_columns: List[str] = DEFAULT_FEATURES,
    alpha: float = 1.0,
    offset_column: str = "log_effort",
) -> Dict:
    """
    Fits a ridge-penalized (L1_wt=0) Poisson GLM with the given offset
    (log_effort by default - see fit_residual_glm for fitting on top of
    another model's fitted rate instead). Returns {"result", "means",
    "stds", "feature_columns", "offset_column"} - means/stds are needed
    to standardize any future data the same way before predicting.
    """
    train_scaled, _, means, stds = _standardize(train_panel, None, feature_columns)
    exog = sm.add_constant(train_scaled.to_numpy())
    model = sm.GLM(
        train_panel["event_count"].to_numpy(), exog,
        offset=train_panel[offset_column].to_numpy(), family=sm.families.Poisson(),
    )
    result = model.fit_regularized(alpha=alpha, L1_wt=0.0)
    return {"result": result, "means": means, "stds": stds,
            "feature_columns": feature_columns, "offset_column": offset_column}


def linear_predictor(fit: Dict, panel: pd.DataFrame) -> np.ndarray:
    """log(lambda) for a fit_glm_with_covariates() result - the offset used, PLUS this model's own fitted terms."""
    scaled = (panel[fit["feature_columns"]] - fit["means"]) / fit["stds"]
    exog = sm.add_constant(scaled.to_numpy(), has_constant="add")
    return exog @ fit["result"].params + panel[fit["offset_column"]].to_numpy()


def predict(fit: Dict, panel: pd.DataFrame) -> np.ndarray:
    """Applies a fit_glm_with_covariates() result to (possibly new) rows."""
    return np.exp(linear_predictor(fit, panel))


def fit_residual_glm(
    train_panel: pd.DataFrame,
    base_fit: Dict,
    feature_columns: List[str],
    alpha: float = 1.0,
) -> Dict:
    """
    Fits `feature_columns` as a CORRECTION on top of base_fit's own
    fitted rate (base_fit's full linear_predictor becomes this model's
    offset), rather than competing with base_fit's features for the same
    ridge-penalized budget. Use this to test whether a fast-moving
    feature set (e.g. FAST_WEATHER_FEATURES) explains anything BEYOND an
    already-fitted seasonal baseline (e.g. a CLIMATOLOGY_FEATURES fit).
    """
    train_panel = train_panel.copy()
    train_panel["_base_eta"] = linear_predictor(base_fit, train_panel)
    return fit_glm_with_covariates(train_panel, feature_columns, alpha=alpha, offset_column="_base_eta")


def predict_residual(residual_fit: Dict, base_fit: Dict, panel: pd.DataFrame) -> np.ndarray:
    """Applies a fit_residual_glm() result: recomputes base_fit's eta on `panel`, then adds the residual model's own terms."""
    panel = panel.copy()
    panel["_base_eta"] = linear_predictor(base_fit, panel)
    return predict(residual_fit, panel)


def fit_offset_only(train_panel: pd.DataFrame) -> Dict:
    """The Stage-0 baseline: intercept + offset(log_effort), no covariates."""
    exog = np.ones((len(train_panel), 1))
    model = sm.GLM(train_panel["event_count"].to_numpy(), exog,
                   offset=train_panel["log_effort"].to_numpy(), family=sm.families.Poisson())
    result = model.fit()
    return {"result": result}


def predict_offset_only(fit: Dict, panel: pd.DataFrame) -> np.ndarray:
    exog = np.ones((len(panel), 1))
    eta = exog @ fit["result"].params + panel["log_effort"].to_numpy()
    return np.exp(eta)


def fit_effort_exponent(
    train_panel: pd.DataFrame,
    offset_column: str = "log_effort",
    feature_columns: Optional[List[str]] = None,
) -> Dict:
    """
    How steeply offset_column actually enters the mean model, instead of
    assuming it's a pure offset (coefficient exactly 1): fits an
    UNPENALIZED Poisson GLM with feature_columns (the seasonal baseline
    by default) plus offset_column as a free covariate - fit_regularized()
    has no covariance matrix, so this needs sm.GLM(...).fit() directly,
    not the ridge machinery elsewhere in this module.

    Real finding on the 2014-2020 panel: freeing log_effort gives
    coefficient ~1.05 with a 95% CI excluding 1.0 - real, given 254k
    rows, though the deviation is modest. model_bundle.fit() uses this
    function's coefficient as a data-driven scaling factor on log_effort
    instead of assuming 1 - see risk_fusion/evaluate_risk_fusion.py's
    offset_identifiability gate for how the estimate itself is checked
    for being well-identified.

    Returns {"coefficient", "ci_low", "ci_high"} for offset_column's term.
    """
    feature_columns = list(feature_columns) if feature_columns is not None else list(MONTH_DUMMY_COLUMNS)
    if any(column not in train_panel.columns for column in feature_columns):
        train_panel = add_month_dummies(train_panel)

    columns = feature_columns + [offset_column]
    exog = sm.add_constant(train_panel[columns].to_numpy())
    result = sm.GLM(train_panel["event_count"].to_numpy(), exog, family=sm.families.Poisson()).fit()

    coefficient_index = len(columns)  # const, then feature_columns, then offset_column last
    coefficient = float(result.params[coefficient_index])
    ci_low, ci_high = (float(v) for v in result.conf_int()[coefficient_index])
    return {"coefficient": coefficient, "ci_low": ci_low, "ci_high": ci_high}


def crossfit_compare(
    panel: pd.DataFrame,
    feature_columns: List[str] = DEFAULT_FEATURES,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Per-block: fit the offset-only baseline, the pure-seasonality
    ("climatological_doy") baseline, and the full covariate GLM - all on
    the other 4 blocks - then predict on the held-out block. Returns one
    row per input row with all three models' out-of-fold predictions
    alongside the actual counts, for risk_fusion_evidence's metrics.

    climatological_doy is the baseline that matters most for interpreting
    the covariates model's improvement: beating offset_only could just
    mean "the model learned Missouri's fire season peaks in spring" (doy
    alone can do that); beating climatological_doy specifically means the
    model is using actual day-to-day weather variation, not just calendar
    position.
    """
    from risk_fusion.risk_fusion_contract import crossfit_indices

    panel = filter_valid_rows(panel, feature_columns).reset_index(drop=True)
    rows = []
    for train_idx, test_idx in crossfit_indices(panel):
        train_panel = panel.loc[train_idx]
        test_panel = panel.loc[test_idx]

        offset_fit = fit_offset_only(train_panel)
        climatology_fit = fit_glm_with_covariates(train_panel, CLIMATOLOGY_FEATURES, alpha=0.01)
        covariate_fit = fit_glm_with_covariates(train_panel, feature_columns, alpha=alpha)

        out = test_panel[["county_fips", "valid_local_date", "event_count", "episode_id", "block"]].copy()
        out["lam_offset_only"] = predict_offset_only(offset_fit, test_panel)
        out["lam_climatological_doy"] = predict(climatology_fit, test_panel)
        out["lam_covariates"] = predict(covariate_fit, test_panel)
        rows.append(out)

    return pd.concat(rows, ignore_index=True)


def fit_lambda_uncertainty(
    crossfit_result: pd.DataFrame,
    prediction_column: str = "lam_climatology_plus_weather",
    target: float = 0.8,
    minimum_rows: int = 300,
) -> Dict:
    """
    Empirical residual-quantile half-widths around the fitted rate (lam),
    global + per-calendar-month, from crossfit_compare_residual()'s
    out-of-fold rows.

    fit_regularized()'s RegularizedResults has no covariance matrix (see
    _serialize_glm_fit's docstring in model_bundle.py), so there's no
    analytic Poisson CI to fall back on - this uses the same empirical-
    residual-quantile approach as the fuel-moisture model's
    api/models/fm_uncertainty.py, reimplemented here rather than imported
    (model-training and api are independent repos - see rule_uncertainty.py
    for why cross-repo imports aren't used). Bucketed by calendar month,
    reusing the same month granularity as MONTH_DUMMY_COLUMNS.
    """
    month = pd.to_datetime(crossfit_result["valid_local_date"]).dt.month.astype(str)
    residual = np.abs(
        crossfit_result["event_count"].to_numpy(dtype=float)
        - crossfit_result[prediction_column].to_numpy(dtype=float)
    )
    result = {
        "target_coverage": target,
        "global": float(np.quantile(residual, target)),
        "global_support": int(len(residual)),
        "regimes": {},
    }
    frame = pd.DataFrame({"month": month, "residual": residual})
    for value, group in frame.groupby("month"):
        if len(group) >= minimum_rows:
            result["regimes"][value] = {
                "half_width": float(np.quantile(group["residual"], target)),
                "support": int(len(group)),
            }
    return result


def lambda_interval(lam, month, uncertainty: Dict) -> np.ndarray:
    """Column-stacked (lo, lam, hi) - same lookup contract as api/models/fm_uncertainty.py::intervals."""
    lam = np.asarray(lam, dtype=float)
    month = np.asarray([str(value) for value in np.broadcast_to(month, lam.shape)])
    widths = np.asarray([
        uncertainty.get("regimes", {}).get(value, {}).get("half_width", uncertainty["global"])
        for value in month
    ])
    return np.column_stack((np.maximum(lam - widths, 0.0), lam, lam + widths))


def crossfit_compare_residual(
    panel: pd.DataFrame,
    fast_features: List[str] = FAST_WEATHER_FEATURES,
    climatology_features: List[str] = MONTH_DUMMY_COLUMNS,
    alpha: float = FAST_WEATHER_RESIDUAL_ALPHA,
    offset_column: str = "log_effort",
) -> pd.DataFrame:
    """
    Isolates day-to-day weather from season by fitting them sequentially
    rather than jointly: the seasonal baseline is fit first, then
    fast_features are fit as a residual correction on top of the
    baseline's own fitted rate (fit_residual_glm), so the ridge penalty
    never has to split credit between a slow seasonal signal and a fast
    weather one - they aren't competing for the same budget. Returns
    lam_offset_only (reporting bias alone), lam_climatological_doy (the
    seasonal baseline), and lam_climatology_plus_weather (baseline +
    residual) for every row, out-of-fold.

    climatology_features defaults to MONTH_DUMMY_COLUMNS (discrete
    per-month levels) - the standing seasonal-baseline architecture for
    fire_risk_fusion. Pass CLIMATOLOGY_FEATURES explicitly to compare
    against the older smooth-harmonic baseline instead.

    offset_column controls what the CANDIDATE (climatology + residual)
    fits use as their offset - defaults to raw log_effort, but pass a
    pre-scaled column (see fit_effort_exponent/model_bundle.fit) to
    evaluate the candidate as it will actually be deployed. lam_offset_only
    always uses the raw log_effort regardless of this argument - it's
    meant to represent the naive "exposure alone, coefficient exactly 1"
    null baseline, not the corrected candidate.
    """
    from risk_fusion.risk_fusion_contract import crossfit_indices

    if any(column not in panel.columns for column in climatology_features):
        panel = add_month_dummies(panel)

    all_features = list(dict.fromkeys(climatology_features + fast_features))
    panel = filter_valid_rows(panel, all_features).reset_index(drop=True)
    rows = []
    for train_idx, test_idx in crossfit_indices(panel):
        train_panel = panel.loc[train_idx]
        test_panel = panel.loc[test_idx]

        offset_fit = fit_offset_only(train_panel)
        climatology_fit = fit_glm_with_covariates(train_panel, climatology_features, alpha=0.01, offset_column=offset_column)
        residual_fit = fit_residual_glm(train_panel, climatology_fit, fast_features, alpha=alpha)

        out = test_panel[["county_fips", "valid_local_date", "event_count", "episode_id", "block"]].copy()
        out["lam_offset_only"] = predict_offset_only(offset_fit, test_panel)
        out["lam_climatological_doy"] = predict(climatology_fit, test_panel)
        out["lam_climatology_plus_weather"] = predict_residual(residual_fit, climatology_fit, test_panel)
        rows.append(out)

    return pd.concat(rows, ignore_index=True)
