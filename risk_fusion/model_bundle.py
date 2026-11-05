"""
The shared definition of "the fitted fire_risk_fusion model" (GLM-only
v1: monthly-baseline + fast-weather residual, per fit_glm.py) - fit,
score, and persist, all in one place so score_live.py's quick-look
scoring and fit_risk_fusion.py's registered-bundle path can't drift into
two different models answering to the same name.

No load()/reconstruct-a-working-predictor-from-disk function here yet:
nothing in this increment needs to reload a frozen bundle (score_live.py
fits fresh each run; register_risk_fusion_beta.py only checksums the
saved files). Add one when a real consumer needs it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from risk_fusion import effort, fit_glm
from risk_fusion.build_county_days import _load_county_reference
from risk_fusion.risk_fusion_contract import add_split_columns

TRAINING_WINDOW_START = "2014-09-01"
TRAINING_WINDOW_END = "2020-12-31"

BUNDLE_ASSET_FILENAMES = {
    "climatology": "glm_climatology.json",
    "residual": "glm_residual.json",
    "effort": "effort.json",
    "county_reference": "county_reference.json",
}

# Optional - not in BUNDLE_ASSET_FILENAMES (which score_glm_for_forecast's
# required-role check in api/services/risk_fusion_glm_shadow.py enforces).
# A bundle saved before uncertainty fitting existed simply won't have this
# file; consumers must treat its absence as "no interval available", never
# as an invalid bundle.
UNCERTAINTY_ASSET_FILENAME = "glm_uncertainty.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_panel(panel_path: Path) -> pd.DataFrame:
    """
    Reads a labeled county-day panel CSV, restricted to the window with
    confirmed label coverage (the CSV itself is not pre-filtered - every
    caller must apply this, it isn't baked into the file). county_fips is
    normalized to a zero-padded string: the CSV round-trips it as int64,
    but every county-reference lookup (county_reference.json,
    county_cells.json) keys on the string form.
    """
    panel = pd.read_csv(panel_path)
    panel["valid_local_date"] = pd.to_datetime(panel["valid_local_date"])
    panel = panel[(panel["valid_local_date"] >= TRAINING_WINDOW_START)
                  & (panel["valid_local_date"] <= TRAINING_WINDOW_END)].copy()
    panel["valid_local_date"] = panel["valid_local_date"].dt.strftime("%Y-%m-%d")
    panel["county_fips"] = panel["county_fips"].astype(str).str.zfill(5)
    return panel


def fit(train_panel: pd.DataFrame) -> Dict:
    """
    Fits the reporting-rate table (for projecting the effort offset onto
    new, unlabeled rows - e.g. a live forward day), the effort exponent
    (how steeply log_effort actually enters the mean model - see
    fit_glm.fit_effort_exponent; a fixed offset assumes exactly 1, but on
    the real panel that assumption doesn't hold, so this estimates it
    instead), and the monthly-baseline + weather-residual GLM pair fit
    against the SCALED offset. train_panel must already carry a
    log_effort column (see effort.py).
    """
    train_panel = fit_glm.add_month_dummies(train_panel)

    county_reference = (
        train_panel.drop_duplicates("county_fips")
        .set_index("county_fips")[["burnable_area_km2"]]
        .to_dict(orient="index")
    )
    primary_counts = train_panel.loc[train_panel["event_count"] > 0, ["county_fips", "event_count"]]
    n_days_observed = train_panel["valid_local_date"].nunique()
    rate_table = effort.fit_reporting_rate(primary_counts, county_reference, n_days_observed)

    effort_exponent = fit_glm.fit_effort_exponent(train_panel)["coefficient"]
    train_panel = train_panel.copy()
    train_panel["log_effort_scaled"] = effort_exponent * train_panel["log_effort"]

    climatology_fit = fit_glm.fit_glm_with_covariates(
        train_panel, fit_glm.MONTH_DUMMY_COLUMNS, alpha=0.01, offset_column="log_effort_scaled")
    residual_fit = fit_glm.fit_residual_glm(
        train_panel, climatology_fit, fit_glm.FAST_WEATHER_FEATURES, alpha=fit_glm.FAST_WEATHER_RESIDUAL_ALPHA)

    # Out-of-fold residuals (same offset the fitted pair above actually
    # uses) to calibrate an empirical interval around lam - see
    # fit_glm.fit_lambda_uncertainty's docstring for why this is empirical
    # rather than an analytic Poisson CI. crossfit_compare_residual needs
    # episode/block split columns that this fit()'s caller has no reason
    # to have added already (a single direct fit, unlike fit_glm.py's own
    # crossfit_compare* evaluation entry points) - add them on a copy here
    # rather than pushing that requirement onto every caller of fit().
    oof = fit_glm.crossfit_compare_residual(
        add_split_columns(train_panel), fast_features=fit_glm.FAST_WEATHER_FEATURES,
        climatology_features=fit_glm.MONTH_DUMMY_COLUMNS,
        alpha=fit_glm.FAST_WEATHER_RESIDUAL_ALPHA, offset_column="log_effort_scaled")
    uncertainty_fit = fit_glm.fit_lambda_uncertainty(oof)

    return {"rate_table": rate_table, "county_reference": county_reference, "effort_exponent": effort_exponent,
            "climatology_fit": climatology_fit, "residual_fit": residual_fit, "uncertainty_fit": uncertainty_fit}


def score(target_panel: pd.DataFrame, bundle: Dict) -> pd.DataFrame:
    """
    Expected fire count (lam) and P(>=1 fire) for every county-day row in
    target_panel. Uses source_coverage_fraction=1.0 (the default) - a
    live/forward prediction isn't a training row, so the historical
    reporting-completeness discount doesn't apply. Applies the SAME
    effort_exponent scaling fit() used, so the offset means the same
    thing at score time as it did at fit time.
    """
    target_panel = target_panel.copy()
    target_panel["county_fips"] = target_panel["county_fips"].astype(str).str.zfill(5)
    target_panel = fit_glm.add_month_dummies(target_panel)

    target_panel["log_effort"] = effort.log_effort_offset_batch(
        target_panel["county_fips"], bundle["rate_table"], bundle["county_reference"])
    target_panel["log_effort_scaled"] = bundle["effort_exponent"] * target_panel["log_effort"]
    target_panel["lam"] = fit_glm.predict_residual(
        bundle["residual_fit"], bundle["climatology_fit"], target_panel)
    target_panel["p_ge1_fire"] = 1.0 - np.exp(-target_panel["lam"])
    if bundle.get("uncertainty_fit") is not None:
        month = pd.to_datetime(target_panel["valid_local_date"]).dt.month
        bounds = fit_glm.lambda_interval(target_panel["lam"].to_numpy(), month.to_numpy(), bundle["uncertainty_fit"])
        target_panel["lam_lo"], target_panel["lam_hi"] = bounds[:, 0], bounds[:, 2]

    names = _load_county_reference()
    target_panel["county_name"] = target_panel["county_fips"].map(
        lambda fips: names.get(fips, {}).get("name", fips))
    return target_panel.sort_values("lam", ascending=False).reset_index(drop=True)


def _serialize_glm_fit(glm_fit: Dict) -> Dict:
    """A fit_glm_with_covariates()/fit_residual_glm() result, as plain JSON-able types.

    No `bse` (standard errors): fit_regularized()'s RegularizedResults has
    no covariance matrix (ridge-penalized fits don't have one in the
    classical sense), so there's nothing honest to serialize there.
    """
    return {
        "feature_columns": list(glm_fit["feature_columns"]),
        "offset_column": glm_fit["offset_column"],
        "params": [float(v) for v in glm_fit["result"].params],
        "means": {k: float(v) for k, v in glm_fit["means"].items()},
        "stds": {k: float(v) for k, v in glm_fit["stds"].items()},
    }


def save(bundle: Dict, directory: Path) -> None:
    """Persists a fit() bundle to directory as one JSON file per asset role (see BUNDLE_ASSET_FILENAMES)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / BUNDLE_ASSET_FILENAMES["climatology"]).write_text(
        json.dumps(_serialize_glm_fit(bundle["climatology_fit"]), indent=2))
    (directory / BUNDLE_ASSET_FILENAMES["residual"]).write_text(
        json.dumps(_serialize_glm_fit(bundle["residual_fit"]), indent=2))

    rate_table = bundle["rate_table"]
    effort_payload = {
        "rate_table": rate_table.reset_index().to_dict(orient="records"),
        "state_mean_rate": rate_table.attrs.get("state_mean_rate"),
        "shrinkage_k": rate_table.attrs.get("shrinkage_k"),
        "n_days_observed": rate_table.attrs.get("n_days_observed"),
        "effort_exponent": bundle["effort_exponent"],
    }
    (directory / BUNDLE_ASSET_FILENAMES["effort"]).write_text(json.dumps(effort_payload, indent=2))
    (directory / BUNDLE_ASSET_FILENAMES["county_reference"]).write_text(
        json.dumps(bundle["county_reference"], indent=2))
    if bundle.get("uncertainty_fit") is not None:
        (directory / UNCERTAINTY_ASSET_FILENAME).write_text(json.dumps(bundle["uncertainty_fit"], indent=2))
