"""
fire_weather_ml: a third, independent model family (alongside the
fuel_moisture XGBoost lineage and risk_fusion's county-day GLM).

Unlike risk_fusion (fit against real fire-occurrence reports - a noisy proxy,
since ignition also depends on human/lightning source availability, not just
weather), this model is fit against a physical fire-behavior proxy: the same
Rothermel surface-fire calculation api/services/spread_rate.py already runs
live in production (rate of spread, fireline intensity, flame length), via
the same `pyretechnics` library, computed offline over historical weather
and fuel-moisture data instead of only the latest hour.

Real fire-occurrence dates and any observed-fire-behavior records are used
only as a secondary, advisory cross-check in evaluate.py (does the model
rank days/places consistent with where fires actually happened) - never as
the training label and never as a promotion gate. See
docs/fire_weather_ml_plan.md for the full design rationale.

Shadow-only / advisory-only for this whole model family until an explicit
future phase accumulates a prospective evaluation record - see
register_beta.py.
"""
