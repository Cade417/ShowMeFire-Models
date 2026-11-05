"""
Computes the training LABEL for fire_weather_ml: a physical fire-behavior
proxy (rate of spread, fireline intensity, flame length), not a fire-
occurrence report.

This is an independent implementation of the same physics call path
api/services/spread_rate.py::_compute_cell_ros already runs in production
(same `pyretechnics` library, same Rothermel surface-fire model, same unit
conversions) - re-derived here rather than imported, per this repo's
model-training-never-imports-api boundary. It operates on one row at a time
(a station/hour observation) rather than a grid, since fire_weather_ml's
training panel is row-based (see panel.py), not gridded.

test_rothermel_labels.py's cross-check test feeds the same sample inputs
into both this module and api/services/spread_rate.py's logic and asserts
matching output - the concrete proof this is really the same physics, not
an approximation of it. That test is skipped (not failed) if `pyretechnics`
cannot be imported - see requirements-pyretechnics.txt for why this
package currently fails to build from source in this Windows dev
environment (a real, discovered incompatibility between its Cython
extension and NumPy 2.x's C API, unrelated to fire_weather_ml's own code).

Field-name caveat: `max_fireline_intensity`/`max_flame_length` below are
pyretechnics' documented output keys for calc_surface_fire_behavior_max.
They have not been runtime-verified against an actually-importable
pyretechnics in this environment (see above) - confirm these key names the
first time this module runs somewhere pyretechnics installs successfully
(e.g. the Linux/API deployment environment), rather than trusting docs
alone.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

M_PER_MIN_TO_CH_PER_H = 60.0 / 20.1168
FT_PER_M = 3.28084
MS_TO_FT_PER_MIN = FT_PER_M * 60.0
WIND_10M_TO_20FT = 0.9  # open-wind reduction from 10m to the 20ft standard height Rothermel's wind term expects


class RothermelLabel(NamedTuple):
    ros_ch_per_h: float
    spread_direction_deg: float
    fireline_intensity_kw_per_m: Optional[float]
    flame_length_m: Optional[float]


def _percent_to_fraction(value: float) -> float:
    return float(np.clip(value / 100.0, 0.0, 1.5))


def wind_from_degrees(u_ms: float, v_ms: float) -> float:
    """Meteorological wind-FROM direction in degrees clockwise from north."""
    return (math.degrees(math.atan2(-u_ms, -v_ms)) + 360.0) % 360.0


def spread_direction_degrees(direction_vector: tuple[float, float, float]) -> float:
    x, y, _z = direction_vector
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def compute_row_label(
    fuel_model_code: int,
    fm1_pct: float,
    fm10_pct: float,
    fm100_pct: float,
    live_herbaceous_pct: float,
    live_woody_pct: float,
    wind_ms: float,
    wind_from_deg: float,
    slope_deg: float,
    aspect_deg: float,
    canopy_cover_pct: float,
    canopy_height_m: float,
) -> Optional[RothermelLabel]:
    """
    One row's Rothermel surface head-fire behavior, or None if the fuel
    model code is unrecognized/non-burnable or the calculation degenerates
    (zero base spread rate) - never silently zero-fills a physically
    meaningless row.
    """
    if fuel_model_code < 1 or fuel_model_code > 999:
        return None

    from pyretechnics.fuel_models import fuel_model_exists, get_fuel_model, moisturize
    from pyretechnics.surface_fire import (
        calc_midflame_wind_speed,
        calc_surface_fire_behavior_max,
        calc_surface_fire_behavior_no_wind_no_slope,
    )

    if not fuel_model_exists(int(fuel_model_code)):
        return None
    fuel_model = get_fuel_model(int(fuel_model_code))
    if not fuel_model.get("burnable", True):
        return None

    moisture = (
        _percent_to_fraction(fm1_pct),
        _percent_to_fraction(fm10_pct),
        _percent_to_fraction(fm100_pct),
        _percent_to_fraction(live_herbaceous_pct),
        _percent_to_fraction(live_woody_pct),
        _percent_to_fraction(live_woody_pct),
    )
    moisturized = moisturize(fuel_model, moisture)
    surface_min = calc_surface_fire_behavior_no_wind_no_slope(moisturized)
    if surface_min["base_spread_rate"] <= 0.0:
        return None

    wind_20ft_ft_min = max(0.0, float(wind_ms)) * MS_TO_FT_PER_MIN * WIND_10M_TO_20FT
    bed_depth_ft = float(moisturized["delta"])
    canopy_cover = float(np.clip(canopy_cover_pct / 100.0, 0.0, 1.0))
    canopy_height_ft = max(0.0, float(canopy_height_m) * FT_PER_M)
    midflame = calc_midflame_wind_speed(wind_20ft_ft_min, bed_depth_ft, canopy_height_ft, canopy_cover)
    slope_fraction = math.tan(math.radians(max(0.0, float(slope_deg))))

    behavior = calc_surface_fire_behavior_max(
        surface_min, midflame, float(wind_from_deg), slope_fraction, float(aspect_deg),
    )
    ros_ch_h = float(behavior["max_spread_rate"]) * M_PER_MIN_TO_CH_PER_H
    direction = spread_direction_degrees(behavior["max_spread_direction"])
    intensity = behavior.get("max_fireline_intensity")
    flame_length = behavior.get("max_flame_length")
    return RothermelLabel(
        ros_ch_per_h=ros_ch_h,
        spread_direction_deg=direction,
        fireline_intensity_kw_per_m=(float(intensity) if intensity is not None else None),
        flame_length_m=(float(flame_length) if flame_length is not None else None),
    )


LABEL_INPUT_COLUMNS = (
    "fuel_model_code", "fm1_pct", "fm10_pct", "fm100_pct",
    "live_herbaceous_pct", "live_woody_pct", "wind_ms", "wind_from_deg",
    "slope_deg", "aspect_deg", "canopy_cover_pct", "canopy_height_m",
)


def compute_labels_for_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Appends ros_ch_per_h/spread_direction_deg/fireline_intensity_kw_per_m/
    flame_length_m columns to a copy of `panel`, one row at a time (this is
    a physics calculation, not a vectorizable-in-numpy one - pyretechnics'
    own API operates per-cell, same as api/services/spread_rate.py does).
    Rows the Rothermel calculation can't produce a label for (unrecognized/
    non-burnable fuel, degenerate inputs) get NaN labels, not a dropped row
    - the caller decides whether to drop them.
    """
    missing = [c for c in LABEL_INPUT_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing required label-input columns: {missing}")

    labeled = panel.copy()
    ros, direction, intensity, flame_length = [], [], [], []
    for row in panel[list(LABEL_INPUT_COLUMNS)].itertuples(index=False):
        result = compute_row_label(*row)
        if result is None:
            ros.append(np.nan)
            direction.append(np.nan)
            intensity.append(np.nan)
            flame_length.append(np.nan)
        else:
            ros.append(result.ros_ch_per_h)
            direction.append(result.spread_direction_deg)
            intensity.append(result.fireline_intensity_kw_per_m)
            flame_length.append(result.flame_length_m)

    labeled["ros_ch_per_h"] = ros
    labeled["spread_direction_deg"] = direction
    labeled["fireline_intensity_kw_per_m"] = intensity
    labeled["flame_length_m"] = flame_length
    return labeled
