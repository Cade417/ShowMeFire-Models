import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import versioning
from risk_fusion import model_bundle, register_risk_fusion_beta as register
from risk_fusion.risk_fusion_evidence import POLICY_VERSION, policy_sha256


def _valid_report(**overrides):
    report = {
        "policy_version": POLICY_VERSION,
        "policy_sha256": policy_sha256(),
        "advisory_only": True,
        "model_family": "glm",
        "row_count": 1000,
        "scores": {"offset_only": 0.2, "climatological_monthly": 0.15, "candidate": 0.14},
        "gates": [
            {"name": "log_score_lower_than_offset_only", "status": "pass"},
            {"name": "log_score_lower_than_rule_only", "status": "not_applicable"},
            {"name": "minimum_prospective_days", "status": "deferred"},
            {"name": "no_unverified_labels", "status": "assumed_from_label_pipeline"},
        ],
    }
    report.update(overrides)
    return report


def _write_candidate_bundle(directory: Path, contract_overrides=None):
    directory.mkdir(parents=True, exist_ok=True)
    for filename in model_bundle.BUNDLE_ASSET_FILENAMES.values():
        (directory / filename).write_text("{}")
    contract = {"advisory_only": True, "model_family": "glm", "count_family": "poisson",
                "feature_module_sha256": "abc123", "offset_definition_sha256": "def456"}
    if contract_overrides:
        contract.update(contract_overrides)
    (directory / "contract.json").write_text(json.dumps(contract))
    return contract


class ValidateBetaRegistrationTests(unittest.TestCase):
    def test_refuses_on_wrong_policy_version(self):
        with tempfile.TemporaryDirectory() as root:
            candidate_dir = Path(root)
            _write_candidate_bundle(candidate_dir)
            with self.assertRaises(RuntimeError):
                register.validate_beta_registration(candidate_dir, _valid_report(policy_version="stale-v0"))

    def test_refuses_on_stale_policy_sha256(self):
        with tempfile.TemporaryDirectory() as root:
            candidate_dir = Path(root)
            _write_candidate_bundle(candidate_dir)
            with self.assertRaises(RuntimeError):
                register.validate_beta_registration(candidate_dir, _valid_report(policy_sha256="stale-hash"))

    def test_refuses_when_not_advisory_only(self):
        with tempfile.TemporaryDirectory() as root:
            candidate_dir = Path(root)
            _write_candidate_bundle(candidate_dir)
            with self.assertRaises(RuntimeError):
                register.validate_beta_registration(candidate_dir, _valid_report(advisory_only=False))

    def test_refuses_when_not_glm(self):
        with tempfile.TemporaryDirectory() as root:
            candidate_dir = Path(root)
            _write_candidate_bundle(candidate_dir)
            with self.assertRaises(RuntimeError):
                register.validate_beta_registration(candidate_dir, _valid_report(model_family="gbm"))

    def test_refuses_on_any_failing_gate(self):
        with tempfile.TemporaryDirectory() as root:
            candidate_dir = Path(root)
            _write_candidate_bundle(candidate_dir)
            report = _valid_report()
            report["gates"].append({"name": "calibration_slope", "status": "fail"})
            with self.assertRaises(RuntimeError):
                register.validate_beta_registration(candidate_dir, report)

    def test_refuses_when_contract_missing(self):
        with tempfile.TemporaryDirectory() as root:
            candidate_dir = Path(root)
            candidate_dir.mkdir(exist_ok=True)
            with self.assertRaises(RuntimeError):
                register.validate_beta_registration(candidate_dir, _valid_report())

    def test_refuses_when_contract_not_advisory_only(self):
        with tempfile.TemporaryDirectory() as root:
            candidate_dir = Path(root)
            _write_candidate_bundle(candidate_dir, contract_overrides={"advisory_only": False})
            with self.assertRaises(RuntimeError):
                register.validate_beta_registration(candidate_dir, _valid_report())

    def test_refuses_when_a_bundle_asset_is_missing(self):
        with tempfile.TemporaryDirectory() as root:
            candidate_dir = Path(root)
            _write_candidate_bundle(candidate_dir)
            (candidate_dir / model_bundle.BUNDLE_ASSET_FILENAMES["effort"]).unlink()
            with self.assertRaises(RuntimeError):
                register.validate_beta_registration(candidate_dir, _valid_report())

    def test_accepts_a_valid_report_and_bundle_and_returns_the_contract(self):
        with tempfile.TemporaryDirectory() as root:
            candidate_dir = Path(root)
            contract = _write_candidate_bundle(candidate_dir)
            result = register.validate_beta_registration(candidate_dir, _valid_report())
        self.assertEqual(result, contract)


class BuildMetadataTests(unittest.TestCase):
    def test_returns_every_required_field_populated(self):
        with tempfile.TemporaryDirectory() as root:
            panel_path = Path(root) / "panel.csv"
            panel_path.write_text("county_fips,event_count\n29001,0\n")
            contract = {"count_family": "poisson", "model_family": "glm",
                        "offset_definition_sha256": "abc", "feature_module_sha256": "def"}
            with patch.object(register, "TRAINING_PANEL_PATH", panel_path):
                metadata = register.build_metadata(_valid_report(), contract)

        for field in register.REQUIRED_RISK_FUSION_METADATA_FIELDS:
            self.assertIsNotNone(metadata.get(field), msg=f"{field} was not populated")

    def test_raises_if_a_required_field_cannot_be_populated(self):
        with tempfile.TemporaryDirectory() as root:
            panel_path = Path(root) / "panel.csv"
            panel_path.write_text("county_fips,event_count\n29001,0\n")
            incomplete_contract = {}  # count_family/model_family/etc. all missing
            with patch.object(register, "TRAINING_PANEL_PATH", panel_path):
                with self.assertRaises(RuntimeError):
                    register.build_metadata(_valid_report(), incomplete_contract)


class MainEndToEndTests(unittest.TestCase):
    def test_registers_a_beta_version_without_touching_the_real_registry(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            candidate_dir = root_path / "candidate"
            _write_candidate_bundle(candidate_dir)

            evaluation_path = root_path / "evaluation.json"
            evaluation_path.write_text(json.dumps(_valid_report()))

            panel_path = root_path / "panel.csv"
            panel_path.write_text("county_fips,event_count\n29001,0\n")

            fake_models_dir = root_path / "models"
            fake_versions_dir = fake_models_dir / "versions"
            fake_config_path = fake_models_dir / "config.json"

            with patch.object(versioning, "MODELS_DIR", fake_models_dir), \
                 patch.object(versioning, "VERSIONS_DIR", fake_versions_dir), \
                 patch.object(versioning, "CONFIG_PATH", fake_config_path), \
                 patch.object(versioning, "API_DIR", root_path), \
                 patch.object(register, "TRAINING_PANEL_PATH", panel_path), \
                 patch.object(sys, "argv", ["register_risk_fusion_beta.py",
                                            "--candidate-dir", str(candidate_dir),
                                            "--evaluation", str(evaluation_path)]):
                register.main()

            self.assertTrue(fake_config_path.exists())
            registered = json.loads(fake_config_path.read_text())
            self.assertIn("fire_risk_fusion", registered)
            self.assertIsNotNone(registered["fire_risk_fusion"]["beta"])
            self.assertTrue(registered["fire_risk_fusion"]["beta"]["performance"]["advisory_only"])


if __name__ == "__main__":
    unittest.main()
