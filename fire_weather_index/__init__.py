"""fire_weather_index: a numeric fire-weather danger score, not another if/then rule.

Separate, advisory-only model line, sibling to risk_fusion and
fire_weather_ml. See docs/fire_weather_index_plan.md (or the project memory
this was designed from) for the full rationale. Short version:

- The live rule (api/core/fire_danger.py::calculate_fire_danger) is an
  if/elif cascade over fuel-moisture/RH/wind thresholds. This package
  computes the SAME Low/Moderate/Elevated/Critical/Extreme spectrum from a
  continuous, weighted combination of numeric factors instead - every input
  passes through a smooth ramp (see factors.py), never a branch, and the
  category boundary is a calibrated score percentile, never a physical
  if/then threshold.
- Unlike risk_fusion (a Poisson GLM fit to historical fire-OCCURRENCE
  counts), this model is not fit to any label at all. Historical
  fire-occurrence data (risk_fusion.labels) is used only in calibrate.py,
  to validate that the score is monotonically associated with real fire
  outcomes and to set the category cutpoints - never as a regression target.
- Inputs blend three weather sources: HRRR (full history, required
  backbone), RRFS (real but ~1 month deep, growing), and FV3-HIRES (NOMADS
  rolling window only, no 2m instantaneous temperature). Availability is
  tracked explicitly per source (rrfs_available, fv3hires_available) -
  never assumed.
"""
