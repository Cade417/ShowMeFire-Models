"""Single source of truth for every model type this repo can train/register.

Previously each of publish_release.py's --model choices, and every
register_beta.py script's own hardcoded model_type string, had to agree by
convention alone. This list exists so "every known model type" is
maintained in exactly one place.

Note: the server-side guarded-shadow families are named "v4"/"v5" in
api/models/shadow_bundles.py, but that's a server-only synthetic name for
the fixed-directory mechanism - the actual model_type string these
candidates are registered under on THIS side (see spatial/register_v4.py,
spatial/register_v5_beta.py) is fuel_moisture_station_guarded /
fuel_moisture_station_summer_guarded. Do not "fix" this list to say
"v4"/"v5" - that was a real bug, caught by cross-checking every
register_trained_model() call site's actual model_type argument.
"""

KNOWN_MODEL_TYPES = [
    "fuel_moisture",
    "fire_danger",
    "fuel_moisture_spatial",
    "fire_behavior_static",
    "fire_risk_fusion",
    "fire_weather_ml",
    "fire_weather_index",
    "fuel_moisture_station_guarded",         # server-side "v4"
    "fuel_moisture_station_summer_guarded",  # server-side "v5"
    "fuel_moisture_station_hybrid",
    "fuel_moisture_station_sequence",
]
