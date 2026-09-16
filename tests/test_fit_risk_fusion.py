import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from risk_fusion import fit_risk_fusion, model_bundle


def _synthetic_panel_csv(path: Path, n=3000, seed=0):
    """Same shape as _synthetic_train_panel in test_model_bundle.py, written to a real CSV like the real panel."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", "2019-12-31", freq="D")
    date_choices = rng.choice(dates, size=n)
    counties = rng.choice([29001, 29003, 29005], size=n)  # int, like the real CSV round-trips it
    area_by_county = {29001: 1000.0, 29003: 500.0, 29005: 2000.0}

    frame = pd.DataFrame({
        "county_fips": counties,
        "valid_local_date": pd.Series(date_choices).dt.strftime("%Y-%m-%d"),
        "burnable_area_km2": pd.Series(counties).map(area_by_county).to_numpy(),
        "rh_mean": rng.uniform(20, 90, size=n),
        "rh_min_afternoon": rng.uniform(10, 80, size=n),
        "wind_kts_max": rng.uniform(2, 30, size=n),
        "wind_kts_p90": rng.uniform(2, 25, size=n),
        "vpd_kpa_max": rng.uniform(0, 4, size=n),
        "precip_24h_mm": rng.exponential(2.0, size=n),
        "is_weekend": rng.integers(0, 2, size=n).astype(bool),
        "log_effort": rng.uniform(-6.0, -3.0, size=n),
    })
    month = pd.to_datetime(frame["valid_local_date"]).dt.month
    seasonal = np.where(month.isin([3, 4, 5]), 1.0, -0.5)
    lam_true = np.exp(frame["log_effort"] + seasonal - 0.02 * frame["rh_min_afternoon"])
    frame["event_count"] = rng.poisson(lam_true)
    frame.to_csv(path, index=False)


class BuildContractTests(unittest.TestCase):
    def test_returns_expected_top_level_keys(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path = Path(root) / "panel.csv"
            _synthetic_panel_csv(csv_path)
            panel = model_bundle.load_training_panel(csv_path)
        contract = fit_risk_fusion.build_contract(panel)
        self.assertEqual(contract["model_family"], "glm")
        self.assertTrue(contract["advisory_only"])
        self.assertIn("feature_module_sha256", contract)
        self.assertIn("offset_definition_sha256", contract)
        self.assertIn("split_manifest", contract)

    def test_split_manifest_reflects_the_passed_panel(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path = Path(root) / "panel.csv"
            _synthetic_panel_csv(csv_path, n=1000)
            panel = model_bundle.load_training_panel(csv_path)
        contract = fit_risk_fusion.build_contract(panel)
        self.assertEqual(contract["split_manifest"]["total_rows"], len(panel))


class MainEndToEndTests(unittest.TestCase):
    def test_writes_full_bundle_plus_contract(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path = Path(root) / "panel.csv"
            _synthetic_panel_csv(csv_path)
            output_dir = Path(root) / "candidate"

            with patch.object(fit_risk_fusion, "TRAINING_PANEL_PATH", csv_path), \
                 patch.object(sys, "argv", ["fit_risk_fusion.py", "--output-dir", str(output_dir)]):
                fit_risk_fusion.main()

            written = {p.name for p in output_dir.iterdir()}
        expected = set(model_bundle.BUNDLE_ASSET_FILENAMES.values()) | {"contract.json"}
        self.assertEqual(written, expected)

    def test_contract_json_is_valid_and_advisory_only(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path = Path(root) / "panel.csv"
            _synthetic_panel_csv(csv_path)
            output_dir = Path(root) / "candidate"

            with patch.object(fit_risk_fusion, "TRAINING_PANEL_PATH", csv_path), \
                 patch.object(sys, "argv", ["fit_risk_fusion.py", "--output-dir", str(output_dir)]):
                fit_risk_fusion.main()

            contract = json.loads((output_dir / "contract.json").read_text())
        self.assertTrue(contract["advisory_only"])
        self.assertEqual(contract["model_family"], "glm")


if __name__ == "__main__":
    unittest.main()
