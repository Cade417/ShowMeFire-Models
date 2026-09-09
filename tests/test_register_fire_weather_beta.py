import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import versioning
from fire_weather_ml import model_bundle, register_beta as register


def _valid_report(**overrides):
    report = {
        "advisory_only": True,
        "model_family": "xgboost_regressor",
        "row_count": 1000,
        "scores": {"candidate_mae": 1.0, "candidate_r2": 0.6, "baseline_mae": 1.5, "baseline_r2": 0.4},
        "gates": [
            {"name": "sufficient_test_rows", "status": "pass"},
            {"name": "beats_naive_weather_only_baseline", "status": "pass"},
            {"name": "fire_occurrence_ranking_advisory", "status": "deferred"},
            {"name": "emulation_cost_advantage_documented", "status": "deferred"},
        ],
    }
    report.update(overrides)
    return report


def _write_candidate_bundle(directory: Path, contract_overrides=None):
    directory.mkdir(parents=True, exist_ok=True)
    for filename in model_bundle.BUNDLE_ASSET_FILENAMES.values():
        (directory / filename).write_text("{}")
    contract = {
        "advisory_only": True, "model_family": "xgboost_regressor",
        "feature_module_sha256": "abc123", "label_module_sha256": "def456",
        "label_column": "ros_ch_per_h",
        "split_manifest": {"manifest_sha256": "ghi789"},
    }
    if contract_overrides:
        contract.update(contract_overrides)
    (directory / "contract.json").write_text(json.dumps(contract))
    return contract


class ValidateBetaRegistrationTests(unittest.TestCase):
    def test_refuses_when_not_advisory_only(self):
        with tempfile.TemporaryDirectory() as root:
            candidate_dir = Path(root)
            _write_candidate_bundle(candidate_dir)
            with self.assertRaises(RuntimeError):
                register.validate_beta_registration(candidate_dir, _valid_report(advisory_only=False))

    def test_refuses_on_any_failing_gate(self):
        with tempfile.TemporaryDirectory() as root:
            candidate_dir = Path(root)
            _write_candidate_bundle(candidate_dir)
            report = _valid_report()
            report["gates"].append({"name": "beats_naive_weather_only_baseline", "status": "fail"})
            with self.assertRaises(RuntimeError):
                register.validate_beta_registration(candidate_dir, report)

    def test_allows_deferred_and_not_applicable_gates(self):
        with tempfile.TemporaryDirectory() as root:
            candidate_dir = Path(root)
            _write_candidate_bundle(candidate_dir)
            result = register.validate_beta_registration(candidate_dir, _valid_report())
        self.assertTrue(result["advisory_only"])

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
            (candidate_dir / model_bundle.BUNDLE_ASSET_FILENAMES["model"]).unlink()
            with self.assertRaises(RuntimeError):
                register.validate_beta_registration(candidate_dir, _valid_report())


class BuildMetadataTests(unittest.TestCase):
    def test_returns_every_required_field_populated(self):
        contract = {
            "feature_module_sha256": "abc", "label_module_sha256": "def",
            "label_column": "ros_ch_per_h", "model_family": "xgboost_regressor",
            "split_manifest": {"manifest_sha256": "ghi"},
        }
        metadata = register.build_metadata(_valid_report(), contract)
        for field in register.REQUIRED_METADATA_FIELDS:
            self.assertIsNotNone(metadata.get(field), msg=f"{field} was not populated")

    def test_raises_if_a_required_field_cannot_be_populated(self):
        with self.assertRaises(RuntimeError):
            register.build_metadata(_valid_report(), {})


class MainEndToEndTests(unittest.TestCase):
    def test_registers_a_beta_version_without_touching_the_real_registry(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            candidate_dir = root_path / "candidate"
            _write_candidate_bundle(candidate_dir)

            evaluation_path = root_path / "evaluation.json"
            evaluation_path.write_text(json.dumps(_valid_report()))

            fake_models_dir = root_path / "models"
            fake_versions_dir = fake_models_dir / "versions"
            fake_config_path = fake_models_dir / "config.json"

            with patch.object(versioning, "MODELS_DIR", fake_models_dir), \
                 patch.object(versioning, "VERSIONS_DIR", fake_versions_dir), \
                 patch.object(versioning, "CONFIG_PATH", fake_config_path), \
                 patch.object(versioning, "API_DIR", root_path), \
                 patch.object(sys, "argv", ["register_beta.py",
                                            "--candidate-dir", str(candidate_dir),
                                            "--evaluation", str(evaluation_path)]):
                register.main()

            self.assertTrue(fake_config_path.exists())
            registered = json.loads(fake_config_path.read_text())
            self.assertIn("fire_weather_ml", registered)
            self.assertIsNotNone(registered["fire_weather_ml"]["beta"])
            self.assertTrue(registered["fire_weather_ml"]["beta"]["performance"]["advisory_only"])


if __name__ == "__main__":
    unittest.main()
