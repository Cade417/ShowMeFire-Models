"""
The log-effort offset for the county-day count model:

    log E_cd = log(burnable_area_km2_c) + log(reporting_rate_shrunk_c)
             + log(source_coverage_fraction_d)

This is the mechanism that turns reporting bias from an invisible
artifact of negative sampling into something estimated and diagnosable -
see the project plan's problem-formulation section. A county that has
historically reported few fires gets a low offset for reasons the model
can name (small burnable area, or a genuinely low observed rate), rather
than the model quietly learning "this county never burns" from an
uninformative absence.

reporting_rate_shrunk uses empirical-Bayes shrinkage toward the
statewide rate: a Gamma(k, k/state_mean_rate) prior on each county's
per-km2-per-day event rate, updated by that county's observed events and
exposure (burnable_area_km2 x days_observed). Standard Poisson-Gamma
conjugacy gives a closed form - no MCMC needed. k is the shrinkage
strength in "equivalent state-mean events": a county with much less
exposure than k pulls hard toward the statewide rate; a county with much
more barely moves from its own raw rate.

source_coverage_fraction is NOT implemented yet: doing it properly needs
day-level reporting-coverage tracking per official_source_system, which
the Phase-1 label export does not populate yet (a known gap - see the
project plan's list of columns the export still needs to add). Until
then this module takes it as an explicit parameter and documents the
simplification; every day in the label's observed window is currently
treated as fully covered (fraction 1.0).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_SHRINKAGE_K = 5.0


def county_event_totals(primary_counts: pd.DataFrame) -> pd.Series:
    """Total observed events per county over the whole label window. primary_counts is sparse (only county-days with >=1 event)."""
    if primary_counts.empty:
        return pd.Series(dtype="float64")
    return primary_counts.groupby("county_fips")["event_count"].sum()


def fit_reporting_rate(
    primary_counts: pd.DataFrame,
    county_reference: Dict[str, Dict],
    n_days_observed: int,
    k: float = DEFAULT_SHRINKAGE_K,
) -> pd.DataFrame:
    """
    Empirical-Bayes-shrunk per-county reporting rate (events per km2 per day).

    county_reference: {fips: {"burnable_area_km2": float, ...}} - e.g.
    risk_fusion.county_geometry.build_county_table() keyed by fips.

    Returns a DataFrame indexed by county_fips with columns:
    events, exposure_km2_days, raw_rate, reporting_rate_shrunk.
    Every county in county_reference gets a row, including counties with
    zero observed events (a genuine zero, not a missing one) - their
    posterior rate is pulled toward the statewide mean rather than
    estimated as exactly zero, which would make their log-effort offset
    -inf.
    """
    if n_days_observed <= 0:
        raise ValueError("n_days_observed must be positive")

    totals = county_event_totals(primary_counts)
    rows = []
    for fips, county in county_reference.items():
        area = float(county["burnable_area_km2"])
        events = float(totals.get(fips, 0.0))
        exposure = area * n_days_observed
        rows.append({"county_fips": fips, "events": events, "exposure_km2_days": exposure})
    frame = pd.DataFrame(rows).set_index("county_fips")

    total_events = frame["events"].sum()
    total_exposure = frame["exposure_km2_days"].sum()
    if total_exposure <= 0:
        raise ValueError("total exposure is zero - check county_reference burnable_area_km2 values")
    state_mean_rate = total_events / total_exposure

    frame["raw_rate"] = frame["events"] / frame["exposure_km2_days"].replace(0, np.nan)
    # Poisson-Gamma conjugate posterior mean: prior Gamma(k, k/state_mean_rate)
    # has mean state_mean_rate; posterior mean given (events, exposure) is
    # (k + events) / (k/state_mean_rate + exposure).
    beta_prior = k / state_mean_rate
    frame["reporting_rate_shrunk"] = (k + frame["events"]) / (beta_prior + frame["exposure_km2_days"])
    frame.attrs["state_mean_rate"] = state_mean_rate
    frame.attrs["shrinkage_k"] = k
    frame.attrs["n_days_observed"] = n_days_observed
    return frame


def log_effort_offset(
    county_fips: str,
    reporting_rate_table: pd.DataFrame,
    county_reference: Dict[str, Dict],
    source_coverage_fraction: float = 1.0,
) -> float:
    """
    log E_cd for one county-day. source_coverage_fraction defaults to 1.0 -
    see module docstring for why day-level coverage isn't tracked yet.
    """
    if county_fips not in reporting_rate_table.index:
        raise KeyError(f"{county_fips} not present in the reporting-rate table")
    if source_coverage_fraction <= 0:
        raise ValueError("source_coverage_fraction must be positive (a day with zero coverage should be dropped, not zeroed)")

    burnable_area_km2 = float(county_reference[county_fips]["burnable_area_km2"])
    reporting_rate = float(reporting_rate_table.loc[county_fips, "reporting_rate_shrunk"])
    return float(np.log(burnable_area_km2) + np.log(reporting_rate) + np.log(source_coverage_fraction))


def log_effort_offset_batch(
    county_fips_series: pd.Series,
    reporting_rate_table: pd.DataFrame,
    county_reference: Dict[str, Dict],
    source_coverage_fraction: Optional[pd.Series] = None,
) -> pd.Series:
    """Vectorized log_effort_offset for a column of county_fips values (e.g. a full county-day panel)."""
    area = county_fips_series.map(lambda fips: float(county_reference[fips]["burnable_area_km2"]))
    rate = county_fips_series.map(lambda fips: float(reporting_rate_table.loc[fips, "reporting_rate_shrunk"]))
    coverage = source_coverage_fraction if source_coverage_fraction is not None else pd.Series(1.0, index=county_fips_series.index)
    if (coverage <= 0).any():
        raise ValueError("source_coverage_fraction must be positive everywhere (drop zero-coverage days instead of zeroing them)")
    return np.log(area) + np.log(rate) + np.log(coverage)
