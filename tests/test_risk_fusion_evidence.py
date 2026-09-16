import unittest

import numpy as np

from risk_fusion import risk_fusion_evidence as evidence


class LogScoreTests(unittest.TestCase):
    def test_accurate_predictions_score_lower_than_inaccurate_ones(self):
        rng = np.random.default_rng(0)
        lam_true = rng.uniform(0.01, 0.5, size=1000)
        y = rng.poisson(lam_true)
        accurate_score = evidence.log_score(y, lam_true)
        inaccurate_score = evidence.log_score(y, np.full_like(lam_true, lam_true.mean() * 5))
        self.assertLess(accurate_score, inaccurate_score)

    def test_handles_zero_counts_without_nan(self):
        y = np.array([0.0, 0.0, 0.0])
        lam = np.array([0.01, 0.05, 0.1])
        score = evidence.log_score(y, lam)
        self.assertFalse(np.isnan(score))


class PoissonDevianceTests(unittest.TestCase):
    def test_perfect_fit_is_near_zero(self):
        lam = np.array([0.1, 0.5, 1.0, 2.0])
        deviance = evidence.poisson_deviance(lam, lam)
        self.assertAlmostEqual(deviance, 0.0, places=6)

    def test_handles_zero_counts_without_nan_or_inf(self):
        y = np.array([0.0, 0.0])
        lam = np.array([0.1, 0.2])
        deviance = evidence.poisson_deviance(y, lam)
        self.assertTrue(np.isfinite(deviance))

    def test_worse_fit_has_higher_deviance(self):
        y = np.array([5.0, 5.0, 5.0])
        good_lam = np.array([5.0, 5.0, 5.0])
        bad_lam = np.array([0.5, 0.5, 0.5])
        self.assertLess(evidence.poisson_deviance(y, good_lam), evidence.poisson_deviance(y, bad_lam))


class PairedBlockBootstrapTests(unittest.TestCase):
    def test_clearly_better_candidate_gets_high_probability(self):
        rng = np.random.default_rng(1)
        n = 2000
        block_ids = rng.integers(0, 20, size=n)  # 20 distinct episode blocks
        lam_true = rng.uniform(0.01, 0.3, size=n)
        y = rng.poisson(lam_true)
        result = evidence.paired_block_bootstrap(
            y, lam_candidate=lam_true, lam_baseline=np.full(n, lam_true.mean() * 3),
            block_ids=block_ids, samples=500,
        )
        self.assertGreater(result["probability_candidate_better"], 0.9)
        self.assertLess(result["point_estimate_delta"], 0.0)

    def test_clearly_worse_candidate_gets_low_probability(self):
        rng = np.random.default_rng(2)
        n = 2000
        block_ids = rng.integers(0, 20, size=n)
        lam_true = rng.uniform(0.01, 0.3, size=n)
        y = rng.poisson(lam_true)
        result = evidence.paired_block_bootstrap(
            y, lam_candidate=np.full(n, lam_true.mean() * 5), lam_baseline=lam_true,
            block_ids=block_ids, samples=500,
        )
        self.assertLess(result["probability_candidate_better"], 0.1)

    def test_requires_at_least_two_blocks(self):
        y = np.array([1.0, 2.0])
        lam = np.array([1.0, 2.0])
        block_ids = np.array([0, 0])
        with self.assertRaises(ValueError):
            evidence.paired_block_bootstrap(y, lam, lam, block_ids, samples=10)

    def test_identical_models_give_roughly_even_odds(self):
        rng = np.random.default_rng(3)
        n = 1000
        block_ids = rng.integers(0, 15, size=n)
        lam = rng.uniform(0.01, 0.3, size=n)
        y = rng.poisson(lam)
        result = evidence.paired_block_bootstrap(y, lam, lam, block_ids, samples=500)
        self.assertAlmostEqual(result["point_estimate_delta"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
