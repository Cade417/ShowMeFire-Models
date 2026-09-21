import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import versioning
from models.register import ModelRegistrationSpec, register_beta


class RegisterBetaTests(unittest.TestCase):
    def _isolated_registry(self, root_path):
        return (
            patch.object(versioning, "MODELS_DIR", root_path / "models"),
            patch.object(versioning, "VERSIONS_DIR", root_path / "models" / "versions"),
            patch.object(versioning, "CONFIG_PATH", root_path / "models" / "config.json"),
            patch.object(versioning, "API_DIR", root_path),
        )

    def test_registers_a_beta_candidate_and_writes_back_the_version(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            candidate_dir = root_path / "candidate"
            candidate_dir.mkdir()

            report_path = root_path / "report.json"
            report_path.write_text(json.dumps({"overall_pass": True, "gates": []}))

            build_calls = []

            def build_candidate(directory):
                build_calls.append(directory)
                (directory / "weights.json").write_text("{}")

            spec = ModelRegistrationSpec(
                model_type="fake_model",
                asset_filenames={"weights": "weights.json"},
                validate_report=lambda candidate_dir, report: report,
                build_candidate=build_candidate,
                build_performance=lambda report, context: {"overall_pass": report["overall_pass"]},
            )

            patches = self._isolated_registry(root_path)
            with patches[0], patches[1], patches[2], patches[3]:
                version = register_beta(spec, report_path=report_path, candidate_dir=candidate_dir)

                config = json.loads(versioning.CONFIG_PATH.read_text())

            self.assertEqual(build_calls, [candidate_dir])
            self.assertIn("fake_model", config)
            self.assertEqual(config["fake_model"]["beta"]["version"], version)
            self.assertEqual(config["fake_model"]["beta"]["performance"], {"overall_pass": True})
            self.assertIn("weights", config["fake_model"]["beta"]["assets"])

            version_file = candidate_dir / "registered_version.json"
            self.assertTrue(version_file.exists())
            self.assertEqual(json.loads(version_file.read_text()),
                              {"model_type": "fake_model", "version": version})

    def test_skips_write_back_when_disabled(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            candidate_dir = root_path / "candidate"
            candidate_dir.mkdir()
            (candidate_dir / "weights.json").write_text("{}")

            report_path = root_path / "report.json"
            report_path.write_text(json.dumps({"overall_pass": True}))

            spec = ModelRegistrationSpec(
                model_type="fake_model_no_writeback",
                asset_filenames={"weights": "weights.json"},
                validate_report=lambda candidate_dir, report: report,
                build_candidate=lambda directory: None,
                build_performance=lambda report, context: {},
                write_back_registered_version=False,
            )

            patches = self._isolated_registry(root_path)
            with patches[0], patches[1], patches[2], patches[3]:
                register_beta(spec, report_path=report_path, candidate_dir=candidate_dir)

            self.assertFalse((candidate_dir / "registered_version.json").exists())

    def test_validate_report_can_refuse_registration(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            candidate_dir = root_path / "candidate"
            candidate_dir.mkdir()

            report_path = root_path / "report.json"
            report_path.write_text(json.dumps({"overall_pass": False}))

            def refuse(candidate_dir, report):
                raise RuntimeError("refused")

            spec = ModelRegistrationSpec(
                model_type="fake_model_refused",
                asset_filenames={"weights": "weights.json"},
                validate_report=refuse,
                build_candidate=lambda directory: None,
                build_performance=lambda report, context: {},
            )

            patches = self._isolated_registry(root_path)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaises(RuntimeError):
                    register_beta(spec, report_path=report_path, candidate_dir=candidate_dir)

    def test_build_metadata_is_stored_as_a_real_metadata_field_not_just_merged_into_performance(self):
        # Regression test for a real bug found in production: several
        # model types (fire_weather_index/fire_weather_ml/fire_risk_fusion)
        # computed the promotion-gate contract but only ever merged it into
        # `performance` - register_beta() had no build_metadata field at
        # all, so it never reached register_trained_model's metadata=
        # argument, publish_release.py's release payload, or the server's
        # validate_promotion_candidate() gate. An imported beta ended up
        # with metadata={}, failing promotion with
        # "missing metadata: advisory_only, model_family" even though the
        # data existed the whole time - just in the wrong field.
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            candidate_dir = root_path / "candidate"
            candidate_dir.mkdir()

            report_path = root_path / "report.json"
            report_path.write_text(json.dumps({"overall_pass": True, "gates": []}))

            spec = ModelRegistrationSpec(
                model_type="fake_model_with_metadata",
                asset_filenames={"weights": "weights.json"},
                validate_report=lambda candidate_dir, report: report,
                build_candidate=lambda directory: (directory / "weights.json").write_text("{}"),
                build_performance=lambda report, context: {"overall_pass": report["overall_pass"]},
                build_metadata=lambda report, context: {"model_family": "fake", "advisory_only": True},
            )

            patches = self._isolated_registry(root_path)
            with patches[0], patches[1], patches[2], patches[3]:
                register_beta(spec, report_path=report_path, candidate_dir=candidate_dir)
                config = json.loads(versioning.CONFIG_PATH.read_text())

            beta = config["fake_model_with_metadata"]["beta"]
            self.assertEqual(beta["metadata"], {"model_family": "fake", "advisory_only": True})
            self.assertEqual(beta["performance"], {"overall_pass": True})

    def test_build_metadata_is_optional(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            candidate_dir = root_path / "candidate"
            candidate_dir.mkdir()

            report_path = root_path / "report.json"
            report_path.write_text(json.dumps({"overall_pass": True}))

            spec = ModelRegistrationSpec(
                model_type="fake_model_no_metadata",
                asset_filenames={"weights": "weights.json"},
                validate_report=lambda candidate_dir, report: report,
                build_candidate=lambda directory: (directory / "weights.json").write_text("{}"),
                build_performance=lambda report, context: {"overall_pass": report["overall_pass"]},
            )

            patches = self._isolated_registry(root_path)
            with patches[0], patches[1], patches[2], patches[3]:
                register_beta(spec, report_path=report_path, candidate_dir=candidate_dir)
                config = json.loads(versioning.CONFIG_PATH.read_text())

            self.assertNotIn("metadata", config["fake_model_no_metadata"]["beta"])

    def test_raises_a_clear_error_when_the_report_is_missing(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            spec = ModelRegistrationSpec(
                model_type="fake_model_missing_report",
                asset_filenames={},
                validate_report=lambda candidate_dir, report: report,
                build_candidate=lambda directory: None,
                build_performance=lambda report, context: {},
            )
            with self.assertRaises(SystemExit):
                register_beta(spec, report_path=root_path / "does_not_exist.json", candidate_dir=root_path)


if __name__ == "__main__":
    unittest.main()
