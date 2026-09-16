import unittest

import numpy as np

from spatial.rule_contract import RULE_SPEC
from risk_fusion import rule_uncertainty as ru


def _reference_adapter(fm, rh, wind, missing_category=None):
    """spatial/rule_contract.py::category() returns None for missing, not
    a caller-supplied sentinel - adapt it to check_parity's expected shape."""
    from spatial.rule_contract import category
    result = category(fm, rh, wind)
    return missing_category if result is None else result


class TrainingSideRuleUncertaintyParityTests(unittest.TestCase):
    """
    Confirms the vendored risk_fusion/rule_uncertainty.py mirror (which
    must be byte-identical to api/services/rule_uncertainty.py - see
    contract_mirrors.json) also agrees with THIS repo's own canonical
    rule implementation, spatial/rule_contract.py::category().
    """

    def setUp(self):
        self.thresholds = RULE_SPEC["thresholds"]

    def test_matches_training_side_rule_contract(self):
        ru.check_parity(_reference_adapter, self.thresholds, n_samples=1000, seed=42)

    def test_sample_category_probabilities_runs_end_to_end(self):
        result = ru.sample_category_probabilities(
            fm=np.array([5.0, 20.0]), rh=np.array([15.0, 50.0]), wind_kts=np.array([30.0, 5.0]),
            fm_sigma=np.array([2.0, 2.0]), rh_sigma=np.array([5.0, 5.0]), wind_sigma_log=np.array([0.3, 0.3]),
            thresholds=self.thresholds, n_draws=200, seed=1,
        )
        totals = sum(result[f"category_probability_{label}"] for label in ru.CATEGORY_LABELS)
        np.testing.assert_allclose(totals, 1.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
