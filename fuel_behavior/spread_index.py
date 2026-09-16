"""
A relative fire spread/intensity index - fuel model x slope x wind, in the
functional spirit of Rothermel's surface spread model and NFDRS's Spread
Component, without requiring Rothermel's full parameter set (fuel bed
packing ratio, characteristic SAV, reaction intensity, etc.).

This is a deliberate simplification, not a calibrated physical model:

1. No real Missouri fuel/terrain data exists yet to calibrate or validate
   against - training-data/static/bundles/ and .../static/source/ are
   both empty; the download_sources.py/build_bundle.py pipeline has never
   actually been run. Every example in this module's tests is therefore
   illustrative (grass vs. timber litter, flat vs. steep, calm vs. windy),
   not a claim about any specific Missouri location.
2. fuel_models.py's spread_base/intensity_base are a coarse 1-4 ordinal
   simplification of Scott & Burgan's fuel model characteristics, not
   their published numeric fuel-load/SAV/depth parameters.
3. Live fuel moisture, moisture of extinction, and canopy-height/crown-
   ratio-based wind reduction (vs. this module's flat canopy-cover-based
   approximation) are all real Rothermel inputs this module does not
   model at all.

Treat this as a relative ranking tool (this cell vs. that cell, under
comparable weather) rather than an absolute rate-of-spread or fireline-
intensity prediction. Revisit the simplifications above before using it
for anything beyond relative comparison.
"""
from __future__ import annotations

import math
from typing import Optional

from fuel_behavior.fuel_models import lookup

SLOPE_CAP_DEGREES = 45.0  # tan^2 diverges past this; caps extrapolation into terrain this index isn't meant for
SLOPE_K = 6.0  # scales the slope term's contribution, chosen so a 30 degree slope roughly doubles the base spread rate

CANOPY_MAX_REDUCTION = 0.85  # at 100% canopy, only 15% of open wind reaches surface fuels
WIND_K = 0.06
WIND_POWER = 1.5  # Rothermel's own wind term is also a power law with an exponent in this range


def slope_factor(slope_degrees: float, cap_degrees: float = SLOPE_CAP_DEGREES, k: float = SLOPE_K) -> float:
    """Relative upslope spread multiplier - tan(slope)^2, Rothermel's own slope term's functional form, without a packing ratio."""
    slope = min(max(float(slope_degrees), 0.0), cap_degrees)
    return 1.0 + k * math.tan(math.radians(slope)) ** 2


def wind_reduction_factor(canopy_cover_pct: float, max_reduction: float = CANOPY_MAX_REDUCTION) -> float:
    """Fraction of open (10m) wind reaching surface fuels - linear falloff with canopy cover, no canopy-height/crown-ratio inputs."""
    canopy = min(max(float(canopy_cover_pct), 0.0), 100.0)
    return 1.0 - max_reduction * (canopy / 100.0)


def wind_factor(wind_kts: float, canopy_cover_pct: float, k: float = WIND_K, power: float = WIND_POWER) -> float:
    """Relative wind multiplier on top of the sheltered (canopy-reduced) wind speed."""
    effective_wind = max(float(wind_kts), 0.0) * wind_reduction_factor(canopy_cover_pct)
    return 1.0 + k * effective_wind ** power


def _environmental_multiplier(slope_degrees: float, wind_kts: float, canopy_cover_pct: float) -> float:
    return slope_factor(slope_degrees) * wind_factor(wind_kts, canopy_cover_pct)


def relative_spread_index(
    fuel_model_code: str, slope_degrees: float, wind_kts: float, canopy_cover_pct: float,
) -> Optional[float]:
    """Relative spread-rate ranking for one cell. None for an unrecognized fuel model code; 0.0 for a non-burnable one."""
    model = lookup(fuel_model_code)
    if model is None:
        return None
    if model.spread_base == 0.0:
        return 0.0
    return model.spread_base * _environmental_multiplier(slope_degrees, wind_kts, canopy_cover_pct)


def relative_intensity_index(
    fuel_model_code: str, slope_degrees: float, wind_kts: float, canopy_cover_pct: float,
) -> Optional[float]:
    """Relative fireline-intensity ranking for one cell. None for an unrecognized fuel model code; 0.0 for a non-burnable one."""
    model = lookup(fuel_model_code)
    if model is None:
        return None
    if model.intensity_base == 0.0:
        return 0.0
    return model.intensity_base * _environmental_multiplier(slope_degrees, wind_kts, canopy_cover_pct)
