"""Tests for the fire_outlook training pipeline (records, split, fit, evaluate)."""
import unittest
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from fire_outlook import contract, evaluate, fit


def synthetic_records(n_days=90, n_stations=40, seed=0):
    """Forecast fuel moisture is biased +3 and over-confident (spread too narrow); agreement is over-confident."""
    rng = np.random.default_rng(seed)
    rows = []
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    for d in range(n_days):
        for hour in (0, 12):
            issue = start + timedelta(days=d, hours=hour)
            for w in range(contract.N_WINDOWS):
                for s in range(n_stations):
                    truth_fm = 10 + 4 * np.sin(d / 9) + rng.normal(0, 2 + 0.2 * w)
                    p50 = truth_fm + 3 + rng.normal(0, 1)
                    elev = float(np.clip(rng.beta(0.7, 2.0), 0, 1))
                    crit = float(elev * rng.random())
                    rows.append({
                        "issue_time": issue, "window_idx": w, "station_id": f"S{s:03d}", "n_members": 31,
                        "agreement_elevated": elev, "agreement_critical": crit,
                        "obs_elevated": bool(rng.random() < 0.4 * elev + 0.02),
                        "obs_critical": bool(rng.random() < 0.3 * crit + 0.01),
                        "qc_ok": True,
                        "fm10_p10": p50 - 0.5, "fm10_p50": p50, "fm10_p90": p50 + 0.5, "fm10_obs": truth_fm,
                        "fm10_init_obs": 10 + 4 * np.sin(d / 9) + rng.normal(0, 1),
                        "t2m_c_mean": 15 + rng.normal(0, 1), "t2m_c_p10": 13.5, "t2m_c_p90": 16.5,
                        "t2m_c_obs": 15 + rng.normal(0, 1.5),
                    })
    return pd.DataFrame(rows)


class ContractTests(unittest.TestCase):
    def test_lead_groups_cover_all_windows(self):
        groups = [contract.lead_group(w) for w in range(contract.N_WINDOWS)]
        self.assertEqual(groups.count("d1-3"), 12)
        self.assertEqual(groups.count("d4-5"), 8)
        self.assertEqual(groups.count("d6-7"), 8)

    def test_missing_required_column_raises(self):
        with self.assertRaises(ValueError):
            contract.prepare(pd.DataFrame({"issue_time": []}))

    def test_prepare_drops_bad_rows_and_counts_them(self):
        df = synthetic_records(n_days=1, n_stations=2).head(20).copy()
        df.loc[0, "window_idx"] = 99
        df.loc[1, "agreement_critical"] = 0.9
        df.loc[1, "agreement_elevated"] = 0.2
        dup = df.iloc[[5]]
        out, issues = contract.prepare(pd.concat([df, dup], ignore_index=True))
        self.assertEqual(issues["dropped_bad_window_idx"], 1)
        self.assertEqual(issues["critical_exceeds_elevated_rows"], 1)
        self.assertEqual(issues["dropped_duplicates"], 1)
        self.assertEqual(len(out), 18)  # 21 in - bad window - duplicate - inverted agreement

    def test_split_is_contiguous_purged_and_station_disjoint(self):
        df, _ = contract.prepare(synthetic_records(n_days=90, n_stations=40))
        side, manifest = contract.assign_split(df)
        train, test = df[side == "train"], df[side == "test"]
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertFalse(set(train["station_id"]) & set(test["station_id"]))
        self.assertFalse(set(train["issue_day"]) & set(test["issue_day"]))
        tdays = sorted(date.fromisoformat(d) for d in test["issue_day"].unique())
        for d in train["issue_day"].unique():
            gap = min(abs((date.fromisoformat(d) - t).days) for t in tdays)
            self.assertGreater(gap, contract.PURGE_DAYS)

    def test_split_stable_when_days_are_appended(self):
        days = [(date(2026, 3, 1) + timedelta(days=i)).isoformat() for i in range(70)]
        first = contract.build_day_split(days)
        more = contract.build_day_split(days + [(date(2026, 3, 1) + timedelta(days=i)).isoformat() for i in range(70, 100)])
        for d in days[:40]:
            self.assertEqual(first[d], more[d])


class FitEvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, _ = contract.prepare(synthetic_records())
        cls.side, cls.split = contract.assign_split(cls.df)
        cls.fitted = fit.fit(cls.df, cls.side)
        cls.report = evaluate.evaluate(cls.df, cls.side, cls.fitted)

    def test_fm_bias_recovered_and_intervals_widened(self):
        p = self.fitted["interval"]["fm10"]["d1-3"]
        self.assertEqual(p["method"], "bias_scale")
        self.assertAlmostEqual(p["bias"], -3.0, delta=0.5)
        self.assertGreater(p["scale"], 2.0)

    def test_correction_improves_test_error_and_coverage(self):
        blk = self.report["variables"]["fm10"]["d1-3"]
        self.assertLess(blk["mae_corrected"], blk["mae_raw"])
        self.assertLess(abs(blk["coverage_corrected"] - 0.8), abs(blk["coverage_raw"] - 0.8))

    def test_small_groups_stay_identity(self):
        small = self.df.head(50)
        side = pd.Series("train", index=small.index)
        out = fit.fit(small, side)
        self.assertEqual(out["interval"]["fm10"]["d1-3"]["method"], "identity")
        self.assertEqual(out["agreement"]["elevated"]["d1-3"]["method"], "identity")

    def test_agreement_calibration_reduces_overconfidence(self):
        blk = self.report["thresholds"]["elevated"]["d1-3"]
        self.assertEqual(blk["calibration_method"], "isotonic")
        self.assertLess(blk["brier_cal"], blk["brier_raw"])

    def test_critical_never_exceeds_elevated_after_calibration(self):
        e, c = fit.apply_agreement(self.fitted["agreement"], "d1-3", [0.1, 0.5, 0.9], [0.9, 0.9, 0.9])
        self.assertTrue(np.all(c <= e + 1e-12))
        self.assertEqual(next(g for g in self.report["gates"] if g["name"] == "critical_le_elevated")["status"], "pass")

    def test_missing_weather_baselines_are_skipped_not_zeroed(self):
        base = self.report["variables"]["t2m_c"]["d1-3"]["baselines"]
        self.assertNotIn("nbm", base)
        self.assertIn("seasonal", base)

    def test_gates_are_explicit_and_overall_pass_has_no_fail(self):
        names = {g["name"] for g in self.report["gates"]}
        self.assertIn("min_test_days", names)
        self.assertIn("interval_coverage_fm10_d1-3", names)
        self.assertEqual(self.report["overall_pass"], not any(g["status"] == "fail" for g in self.report["gates"]))

    def test_sparse_events_defer_rather_than_pass(self):
        df = self.df.copy()
        df["obs_critical"] = False
        fitted = fit.fit(df, self.side)
        rep = evaluate.evaluate(df, self.side, fitted)
        gate = next(g for g in rep["gates"] if g["name"] == "events_critical_d1-3")
        self.assertEqual(gate["status"], "deferred")

    def test_no_training_rows_leak_into_test(self):
        train_stations = set(self.df[self.side == "train"]["station_id"])
        test_stations = set(self.df[self.side == "test"]["station_id"])
        self.assertFalse(train_stations & test_stations)


if __name__ == "__main__":
    unittest.main()
