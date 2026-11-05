"""
Measures the actual point of an ML emulator: is scoring the trained
XGBoost model genuinely cheaper than running the real Rothermel physics
(`rothermel_labels.compute_labels_for_panel`) row-by-row? An emulator with
equal accuracy but no speed/scale advantage over the physics calculation
itself wouldn't be worth serving (see evaluate.py's module docstring).

Timed on a random sample of the panel (not the whole thing, for a quick
per-call comparison) - `sample_size` rows, same rows for both timings so
the comparison is apples-to-apples.
"""
from __future__ import annotations

import time
from typing import Dict

import pandas as pd

from fire_weather_ml import model_bundle, rothermel_labels

DEFAULT_SAMPLE_SIZE = 2000


def measure_emulation_speedup(
    panel: pd.DataFrame, bundle: Dict, sample_size: int = DEFAULT_SAMPLE_SIZE, seed: int = 0,
) -> Dict:
    missing_label_inputs = [c for c in rothermel_labels.LABEL_INPUT_COLUMNS if c not in panel.columns]
    if missing_label_inputs:
        return {"available": False, "reason": f"panel is missing label-input columns needed for the physics timing: {missing_label_inputs}"}

    sample = panel.sample(n=min(sample_size, len(panel)), random_state=seed)

    start = time.perf_counter()
    model_bundle.score(sample, bundle)
    model_seconds = time.perf_counter() - start

    start = time.perf_counter()
    rothermel_labels.compute_labels_for_panel(sample)
    physics_seconds = time.perf_counter() - start

    return {
        "available": True,
        "sample_rows": int(len(sample)),
        "model_seconds": model_seconds,
        "physics_seconds": physics_seconds,
        "model_seconds_per_row": model_seconds / len(sample),
        "physics_seconds_per_row": physics_seconds / len(sample),
        "speedup_factor": (physics_seconds / model_seconds) if model_seconds > 0 else None,
    }
