import unittest

import numpy as np
import pandas as pd

from spatial.v5_evidence import POLICY_VERSION, evaluate_policy, paired_block_bootstrap


def evidence(days=40, rows_per_day=40, improvement=.1):
    rng = np.random.default_rng(11); rows = days * rows_per_day
    actual = rng.normal(9, 2, rows); incumbent = actual + rng.normal(.2, 2, rows)
    candidate = actual + (incumbent - actual) * (1 - improvement)
    actual_category = np.where(actual <= 6, 3, np.where(actual <= 10, 2, 1))
    candidate_category = np.where(candidate <= 6, 3, np.where(candidate <= 10, 2, 1))
    incumbent_category = np.where(incumbent <= 6, 3, np.where(incumbent <= 10, 2, 1))
    width = np.quantile(np.abs(actual - candidate), .8)
    return pd.DataFrame({"bootstrap_block": np.repeat(pd.date_range("2025-06-01", periods=days).astype(str), rows_per_day),
        "actual_fm": actual, "candidate_fm": candidate, "incumbent_fm": incumbent,
        "p10": candidate-width, "p50": candidate, "p90": candidate+width,
        "actual_category": actual_category, "candidate_category": candidate_category,
        "incumbent_category": incumbent_category, "summer": True,
        "critical": actual <= 6, "rain_event": np.arange(rows) % 3 == 0})


class V5EvidenceTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_blocked(self):
        frame = evidence()
        first = paired_block_bootstrap(frame, samples=500); second = paired_block_bootstrap(frame, samples=500)
        self.assertEqual(first, second); self.assertEqual(first["blocks"], 40)
        self.assertLess(first["delta"], 0)

    def test_policy_has_no_five_percent_gate(self):
        report = evaluate_policy(evidence(improvement=.02))
        self.assertEqual(report["policy_version"], POLICY_VERSION)
        self.assertNotIn("mae_5pct_better", report["checks"])
        self.assertIn("mae_probability", report["checks"])

    def test_quantile_crossing_fails(self):
        frame = evidence(); frame.loc[0, "p10"] = frame.loc[0, "p90"] + 1
        report = evaluate_policy(frame)
        self.assertFalse(report["checks"]["interval_ordering"])

    def test_prospective_missing_critical_is_not_fabricated_failure(self):
        frame = evidence(); frame["actual_fm"] += 10; frame["critical"] = False
        report = evaluate_policy(frame, minimum_days=30)
        self.assertTrue(report["checks"]["critical_support"])
        self.assertTrue(report["checks"]["critical_mae_no_worse"])


if __name__ == "__main__": unittest.main()
