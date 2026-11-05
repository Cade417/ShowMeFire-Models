"""
Seven-day Show Me Fire outlook: historical evaluation and (versioned) calibration.

The outlook itself (28 six-hour windows, 31 GEFS members) is computed by the
`fire-outlook` worker in the api repo. Per the plan, training and historical
evaluation live here. The worker exports verification records (one row per
issue time x window x station); this package turns them into:

    contract   record schema, lead groups, and the contiguous period + station split
    fit        lead-dependent bias / interval-width correction for 10-hour fuel
               moisture and weather, and isotonic calibration of scenario agreement
    evaluate   per-window / per-lead-group metrics vs persistence, seasonal and NBM
               baselines, with pass / fail / deferred gates
    retrain    one command: load -> split -> fit -> evaluate -> candidate bundle

Nothing here changes what the worker publishes. Calibration is a candidate bundle
for a later versioned release; the public label stays "scenario agreement" until a
bundle passes review and the staff-only shadow pilot.
"""
MODEL_FAMILY = "fire_outlook"
