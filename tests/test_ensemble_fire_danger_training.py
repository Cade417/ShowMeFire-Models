"""Tests for the ensemble_fire_danger training pipeline (split, fit, evaluate, bundle)."""
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_fire_danger import contract, evaluate, fit, members, model_bundle


def synthetic_panel(n_days=70, cells=400, seed=0, tracks=("members", "ensprod")):
    """Raw ensemble fraction p is OVER-confident: true event rate is 0.5*p + 0.02."""
    rng = np.random.default_rng(seed)
    frames = []
    start = date(2026, 6, 1)
    for d in range(n_days):
        day = (start + timedelta(days=d)).isoformat()
        for track in tracks:
            for k in (1, 2, 3, 4):
                p = np.clip(rng.beta(0.6, 2.0 + k, cells), 0, 1)
                truth = rng.random(cells) < (0.5 * p + 0.02)
                det = rng.random(cells) < p
                frames.append(pd.DataFrame({
                    "day": day, "track": track, "k": k, "row": np.arange(cells), "col": 0,
                    "p_point": p.astype("float32"), "p_nbhd": np.clip(p * 1.2, 0, 1).astype("float32"),
                    "obs_point": truth, "obs_nbhd": truth | (rng.random(cells) < 0.05),
                    "det_point": det, "det_nbhd": det, "member_count": 10 if track == "members" else 40,
                }))
    return pd.concat(frames, ignore_index=True)


class SplitTests(unittest.TestCase):
    def test_whole_weeks_and_stable_as_days_are_appended(self):
        days = [(date(2026, 8, 1) + timedelta(days=i)).isoformat() for i in range(60)]
        split = contract.build_split(days)
        by_week = {}
        for day, side in split.items():
            by_week.setdefault(contract.week_key(day), set()).add(side)
        self.assertTrue(all(len(sides) == 1 for sides in by_week.values()))
        self.assertTrue(0 < sum(v == "test" for v in split.values()) < len(days))
        more = contract.build_split(days + [(date(2026, 8, 1) + timedelta(days=i)).isoformat() for i in range(60, 80)])
        self.assertTrue(all(more[d] == split[d] for d in days))


class FitTests(unittest.TestCase):
    def setUp(self):
        self.panel = synthetic_panel()
        self.split = contract.build_split(self.panel["day"].unique())

    def test_isotonic_learns_overconfidence_and_is_monotone(self):
        fitted = fit.fit(self.panel, self.split)
        knots = fitted["calibration"]["tracks"]["members"]["point"]["1"]
        self.assertTrue(np.all(np.diff(knots["y"]) >= -1e-12))
        self.assertLess(np.interp(0.8, knots["x"], knots["y"]), 0.6)  # truth ~0.42 at p=0.8
        self.assertGreater(np.interp(0.8, knots["x"], knots["y"]), 0.25)

    def test_identity_when_too_few_events(self):
        tiny = self.panel[self.panel["day"].isin(sorted(self.panel["day"].unique())[:2])].copy()
        tiny["obs_point"] = False
        tiny["obs_nbhd"] = False
        fitted = fit.fit(tiny, contract.build_split(tiny["day"].unique()))
        self.assertEqual(fitted["calibration"]["tracks"]["members"]["point"], {})
        self.assertEqual(fitted["categorical_thresholds"]["tracks"]["members"]["1"], 0.5)

    def test_threshold_respects_bias_bounds(self):
        rng = np.random.default_rng(1)
        p = rng.random(5000)
        y = rng.random(5000) < p
        tau = fit.best_threshold(p, y, 0.5)
        self.assertEqual(tau["source"], "fitted")
        self.assertTrue(fit.BIAS_LO <= tau["bias"] <= fit.BIAS_HI)


class EvaluateTests(unittest.TestCase):
    def test_gates_and_track_comparison(self):
        panel = synthetic_panel()
        split = contract.build_split(panel["day"].unique())
        fitted = fit.fit(panel, split)
        report = evaluate.evaluate(panel, split, fitted, members.load_config())
        names = {g["name"]: g["status"] for g in report["gates"]}
        self.assertEqual(names["members:calibration_not_worse_1"], "pass")
        self.assertIn(report["track_comparison"]["recommended_primary"], ("members", "ensprod"))
        nb = report["tracks"]["members"]["categories"]["1"]["neighborhood"]
        self.assertLessEqual(nb["brier_cal"], nb["brier_raw"])

    def test_small_data_is_deferred_not_failed(self):
        panel = synthetic_panel(n_days=6)
        split = contract.build_split(panel["day"].unique())
        report = evaluate.evaluate(panel, split, fit.fit(panel, split), members.load_config())
        statuses = {g["name"]: g["status"] for g in report["gates"]}
        self.assertTrue(any(v == "deferred" for v in statuses.values()))
        self.assertTrue(report["overall_pass"])


class BundleTests(unittest.TestCase):
    def test_roundtrip_and_core_checksum_matches_api(self):
        panel = synthetic_panel(n_days=14, tracks=("members",))
        split = contract.build_split(panel["day"].unique())
        fitted = fit.fit(panel, split)
        with tempfile.TemporaryDirectory() as tmp:
            model_bundle.save(Path(tmp), fitted, evaluation={"ok": True})
            model_bundle.set_version(Path(tmp), "0.0.1-beta.1")
            loaded = model_bundle.load(Path(tmp))
            for name in model_bundle.BUNDLE_ASSET_FILENAMES.values():
                self.assertTrue((Path(tmp) / name).exists())
        self.assertEqual(loaded["calibration"]["version"], "0.0.1-beta.1")
        self.assertEqual(loaded["calibration"]["core_module_sha256"], model_bundle.core_module_sha256())
        api_core = Path(__file__).resolve().parents[2] / "api" / "services" / "ensemble_fire_danger" / "core.py"
        if api_core.exists():
            import hashlib
            self.assertEqual(hashlib.sha256(api_core.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                             model_bundle.core_module_sha256())


if __name__ == "__main__":
    unittest.main()
