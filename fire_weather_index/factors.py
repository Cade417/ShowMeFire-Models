"""
Continuous fire-weather factor ramps and the weighted score they combine into.

No if/elif branching anywhere in here - every input passes through a
clipped LINEAR ramp (0 = no danger contribution, 1 = maximal), and the
factors combine via a weighted average. Where a ramp's anchor points come
from an already-vetted physical threshold in this project (RH/wind from
api/core/fire_danger_rules.json's thresholds, the green-up/curing GDD
anchors from api/core/fire_danger.py's seasonal_dampening_adjustment), this
reuses that same vetted number as a ramp ENDPOINT rather than a branch
condition - the physical calibration behind those numbers is real and worth
keeping, only the branching structure is being removed. Anchors with no
prior vetted precedent in this project (VPD, KBDI, precip relief) are
explicitly marked FIRST_PASS below and need calibration.py's monotonicity
check run against real data before being trusted operationally.

Every raw input here can come from any blend of HRRR/RRFS/FV3-HIRES - this
module only deals in already-aggregated county-day values (rh_mean,
wind_kts_max, etc.), not raw grids. See build_county_days.py for the
blending step that produces those values.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_COUNTY_FUEL_TYPE_PATH = Path(__file__).resolve().parent / "county_fuel_type.json"
_county_fuel_type_cache: Optional[Dict[str, dict]] = None

# --- Ramp anchors -----------------------------------------------------------
# Reused from api/core/fire_danger_rules.json (the live rule's own vetted
# thresholds) - see this module's docstring for why reusing the anchor is
# legitimate even though the branching structure isn't.
RH_BENIGN = 45.0   # moderate_rh: at/above this, RH contributes ~no danger
RH_EXTREME = 20.0  # extreme_rh: at/below this, RH is at max danger contribution
WIND_BENIGN = 10.0   # moderate_wind
WIND_EXTREME = 25.0  # extreme_wind

# Reused from api/core/fire_danger.py's seasonal_dampening_adjustment - same
# anchors, inverted here (this is a CURING/danger factor, that code computes
# a GREEN/protective factor) since a fully cured fuel bed is the higher-risk
# state this project already treats as such.
CURE_GREEN_CEILING_C = 200.0   # gdd_accum_since_mar1 at/below this: fully green, cure_factor=0
CURE_CURED_FLOOR_C = 1200.0    # at/above this: fully cured, cure_factor=1

# FIRST_PASS anchors: no prior vetted threshold exists anywhere in this
# project for these three - physically reasonable starting points, but
# calibrate.py's monotonicity-against-real-fire-occurrence check is what
# actually justifies them, not this docstring.
VPD_BENIGN_KPA = 1.0     # FIRST_PASS
VPD_EXTREME_KPA = 4.0    # FIRST_PASS
KBDI_BENIGN_MM = 0.0     # FIRST_PASS
KBDI_EXTREME_MM = 150.0  # FIRST_PASS (roughly 6in on the traditional 0-8in KBDI scale)
PRECIP_RELIEF_FULL_MM = 10.0  # FIRST_PASS: 24h precip at/above this fully suppresses the score

# Fuel-type-aware version of the anchor above: fine grass fuels wet through
# (and dry back out) with much less rain than heavy timber litter/duff, so
# the SAME 10mm that fully relieves a grass county barely dents a timber
# county's danger. FIRST_PASS same as the flat constant it replaces -
# needs calibration once there's a real labeled panel to check it against.
# "shrub" keeps the original flat value as the middle-ground default; also
# used as the fallback for "nonburnable" counties (LANDFIRE's own
# "non-burnable" class is dominated by active cropland in Missouri - see
# scripts/build_county_fuel_type.py - not literally fuel-free: field edges,
# pasture, and fallow ground still carry real fine-fuel fire risk) and
# "unknown" (no fuel-type data for this county at all).
PRECIP_RELIEF_FULL_MM_BY_FUEL_GROUP = {
    "grass": 5.0,
    "shrub": PRECIP_RELIEF_FULL_MM,
    "timber": 18.0,
    "nonburnable": PRECIP_RELIEF_FULL_MM,
    "unknown": PRECIP_RELIEF_FULL_MM,
}

# Absolute ceiling anchor, NOT a percentile: the peak raw (pre-rescale)
# weighted-average score computed for eastern Missouri (county centroids
# east of -91.0 longitude - see scripts/find_ceiling_anchor_score.py) on
# 2025-03-14, a real historic extreme fire-weather day in that region.
# compute_score() divides by this before its final clip, so that day's
# worst county effectively becomes the score's practical 1.0/"maxed out"
# reference point instead of a never-quite-reached theoretical ceiling -
# this is a real observed anchor, not a FIRST_PASS guess like the ramp
# anchors above. Re-derive via scripts/find_ceiling_anchor_score.py if the
# upstream factor math (ramps/weights) ever changes, since this value is
# only valid against the score formula that produced it.
RAW_SCORE_CEILING = 0.6076826763957318  # 29221 Washington, 2025-03-14 (see scripts/find_ceiling_anchor_score.py output)



def _load_county_fuel_type() -> Dict[str, dict]:
    global _county_fuel_type_cache
    if _county_fuel_type_cache is None:
        try:
            _county_fuel_type_cache = json.loads(_COUNTY_FUEL_TYPE_PATH.read_text(encoding="utf-8"))["counties"]
        except FileNotFoundError:
            _county_fuel_type_cache = {}  # not built yet in this environment - every county falls back to "unknown"
    return _county_fuel_type_cache


def fuel_group_for_county(county_fips: Optional[str]) -> str:
    """"grass"/"shrub"/"timber"/"nonburnable" from
    scripts/build_county_fuel_type.py's output, or "unknown" if that county
    (or the file itself) isn't available - never a fabricated guess."""
    if county_fips is None:
        return "unknown"
    return _load_county_fuel_type().get(str(county_fips), {}).get("fuel_group", "unknown")

# Initial weights - a first-pass, physically-motivated starting point (RH
# and wind get the most weight, matching their central role in the live
# rule; VPD/KBDI/cure are secondary; precip is a small subtractive relief
# term), NOT fit against any label. Revisit once calibrate.py has enough
# history to say whether any factor should be re-weighted.
FACTOR_WEIGHTS = {
    "rh": 0.25,
    "wind": 0.25,
    "vpd": 0.20,
    "kbdi": 0.15,
    "cure": 0.10,
    "precip_relief": 0.05,  # subtracted, not added - see compute_score
}


def _ramp(value: Optional[float], benign: float, extreme: float) -> Optional[float]:
    """Linear ramp from `benign` (-> 0.0) to `extreme` (-> 1.0), clipped to [0, 1]. None in, None out."""
    if value is None or not np.isfinite(value):
        return None
    fraction = (value - benign) / (extreme - benign)
    return float(np.clip(fraction, 0.0, 1.0))


def rh_factor(rh_min_afternoon: Optional[float]) -> Optional[float]:
    return _ramp(rh_min_afternoon, RH_BENIGN, RH_EXTREME)


def wind_factor(wind_kts_max: Optional[float]) -> Optional[float]:
    return _ramp(wind_kts_max, WIND_BENIGN, WIND_EXTREME)


def vpd_factor(vpd_kpa_max: Optional[float]) -> Optional[float]:
    return _ramp(vpd_kpa_max, VPD_BENIGN_KPA, VPD_EXTREME_KPA)


def kbdi_factor(kbdi: Optional[float]) -> Optional[float]:
    return _ramp(kbdi, KBDI_BENIGN_MM, KBDI_EXTREME_MM)


def cure_factor(gdd_accum_since_mar1: Optional[float]) -> Optional[float]:
    return _ramp(gdd_accum_since_mar1, CURE_GREEN_CEILING_C, CURE_CURED_FLOOR_C)


def precip_relief_factor(precip_mm: Optional[float], fuel_group: str = "unknown") -> Optional[float]:
    """0 = no rain, no relief; 1 = at/above this fuel group's full-relief
    threshold. `precip_mm` should already be the COMBINED total (current
    forecast-window precip + real antecedent precip from prior days, see
    build_county_days.py::add_antecedent_precip) - this function itself
    doesn't know or care where the number came from."""
    threshold = PRECIP_RELIEF_FULL_MM_BY_FUEL_GROUP.get(fuel_group, PRECIP_RELIEF_FULL_MM)
    return _ramp(precip_mm, 0.0, threshold)


def compute_factors(row: Dict) -> Dict[str, Optional[float]]:
    """row: one county-day record with rh_min_afternoon, wind_kts_max,
    vpd_kpa_max, kbdi, gdd_accum_since_mar1, precip_24h_mm, and optionally
    antecedent_precip_mm (real rain from prior days - see
    build_county_days.py::add_antecedent_precip; treated as 0 additional
    rain, not "unknown", when absent - a relief factor that can't confirm
    extra rain happened should NOT grant relief for it, so 0 is the honest
    default here, unlike most other factors' None-on-missing convention)
    and county_fips (for the fuel-type-aware precip relief threshold -
    "unknown" fuel group, i.e. today's original flat threshold, if absent).
    Returns each factor in [0, 1], or None if the underlying input was
    missing/non-finite - never silently substituted, same convention as
    gust_available elsewhere in this project."""
    precip_total = row.get("precip_24h_mm")
    if precip_total is not None and np.isfinite(precip_total):
        precip_total = precip_total + (row.get("antecedent_precip_mm") or 0.0)
    return {
        "rh": rh_factor(row.get("rh_min_afternoon")),
        "wind": wind_factor(row.get("wind_kts_max")),
        "vpd": vpd_factor(row.get("vpd_kpa_max")),
        "kbdi": kbdi_factor(row.get("kbdi")) if row.get("kbdi_valid") else None,
        "cure": cure_factor(row.get("gdd_accum_since_mar1")),
        "precip_relief": precip_relief_factor(precip_total, fuel_group_for_county(row.get("county_fips"))),
    }


def compute_score(factors: Dict[str, Optional[float]], weights: Dict[str, float] = None) -> Optional[float]:
    """Weighted average of available factors (precip_relief subtracted, not
    added), renormalized over whichever weights actually had a value this
    row - a missing factor changes the denominator, it never gets treated
    as 0 (which would silently understate danger) or imputed. Returns None
    only if every factor is missing."""
    weights = weights or FACTOR_WEIGHTS
    numerator = 0.0
    denominator = 0.0
    for name, weight in weights.items():
        value = factors.get(name)
        if value is None:
            continue
        signed_weight = -weight if name == "precip_relief" else weight
        numerator += signed_weight * value
        denominator += weight
    if denominator == 0.0:
        return None
    raw_score = numerator / denominator
    # Rescale against the real-event ceiling anchor (see RAW_SCORE_CEILING's
    # docstring) rather than clipping the raw weighted average directly -
    # 1.0 now means "at/beyond the 2025-03-14 eastern-MO extreme," not a
    # theoretical every-factor-maxed value that real weather rarely reaches.
    return float(np.clip(raw_score / RAW_SCORE_CEILING, 0.0, 1.0))
