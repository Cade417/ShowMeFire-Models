import unittest

import numpy as np
import pandas as pd

from fire_weather_ml import evaluate, model_bundle
from fire_weather_ml.features import FEATURE_COLUMNS


def _synthetic_panel(n_rows: int = 600, seed: int = 0, label_is_predictable: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {column: rng.uniform(0, 30, size=n_rows) for column in FEATURE_COLUMNS}
    frame = pd.DataFrame(data)
    frame["station_id"] = rng.choice(["S1", "S2", "S3"], size=n_rows)
    frame["valid_time"] = pd.date_range("2021-01-01", periods=n_rows, freq="h").astype(str)
    if label_is_predictable:
        # A strong, low-noise relationship using only MODEL_FEATURE_COLUMNS
        # (no kbdi/gdd_accum) - the model should comfortably clear the
        # absolute accuracy bar on this.
        frame[model_bundle.DEFAULT_LABEL_COLUMN] = (
            frame["wind_ms"] * 1.2 - frame["fm10_pct"] * 0.9 + rng.normal(0, 0.3, size=n_rows)
        ).clip(lower=0.0)
    else:
        # Pure noise, unrelated to any feature - no model should clear the
        # accuracy bar on this.
        frame[model_bundle.DEFAULT_LABEL_COLUMN] = rng.uniform(0, 10, size=n_rows)
    return frame


class BuildReportTests(unittest.TestCase):
    def test_report_has_every_expected_gate_with_no_silent_omission(self):
        report = evaluate.build_report(_synthetic_panel())
        gate_names = {gate["name"] for gate in report["gates"]}
        self.assertEqual(gate_names, {
            "sufficient_test_rows", "achieves_high_emulation_accuracy",
            "fire_occurrence_ranking_advisory", "emulation_cost_advantage_documented",
        })
        for gate in report["gates"]:
            self.assertIn(gate["status"], ("pass", "fail", "deferred", "not_applicable"))

    def test_advisory_gate_never_reports_pass_or_fail(self):
        report = evaluate.build_report(_synthetic_panel())
        advisory = next(g for g in report["gates"] if g["name"] == "fire_occurrence_ranking_advisory")
        self.assertEqual(advisory["status"], "deferred")

    def test_a_strong_low_noise_signal_clears_the_accuracy_bar(self):
        report = evaluate.build_report(_synthetic_panel(n_rows=800, label_is_predictable=True))
        gate = next(g for g in report["gates"] if g["name"] == "achieves_high_emulation_accuracy")
        self.assertEqual(gate["status"], "pass")

    def test_pure_noise_fails_the_accuracy_bar(self):
        report = evaluate.build_report(_synthetic_panel(n_rows=800, label_is_predictable=False))
        gate = next(g for g in report["gates"] if g["name"] == "achieves_high_emulation_accuracy")
        self.assertEqual(gate["status"], "fail")

    def test_sufficient_test_rows_fails_on_a_tiny_panel(self):
        report = evaluate.build_report(_synthetic_panel(n_rows=50))
        rows_gate = next(g for g in report["gates"] if g["name"] == "sufficient_test_rows")
        self.assertEqual(rows_gate["status"], "fail")

    def test_overall_pass_is_false_when_any_checkable_gate_fails(self):
        report = evaluate.build_report(_synthetic_panel(n_rows=50))
        self.assertFalse(report["overall_pass"])


if __name__ == "__main__":
    unittest.main()
