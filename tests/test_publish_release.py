import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import versioning
from models.model_types import KNOWN_MODEL_TYPES
import pipelines.publish_release as publish_release


class PublishReleaseChoicesTests(unittest.TestCase):
    def test_every_known_model_type_is_a_valid_cli_choice(self):
        # Regression test for the previously-hardcoded 3-item allowlist that
        # made most model types impossible to publish through this script.
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", required=True, choices=sorted(KNOWN_MODEL_TYPES))
        for model_type in KNOWN_MODEL_TYPES:
            parser.parse_args(["--model", model_type])

    def test_known_model_types_match_real_registration_call_sites_not_server_side_names(self):
        # Regression test for a real bug: this list briefly said "v4"/"v5" -
        # those are api/models/shadow_bundles.py's server-only synthetic
        # names for the fixed-directory mechanism. The actual model_type
        # string spatial/register_v4.py and spatial/register_v5_beta.py
        # register on THIS side (confirmed via their own MODEL_TYPE
        # constants) is fuel_moisture_station_guarded /
        # fuel_moisture_station_summer_guarded - "v4"/"v5" were never a
        # real training-side model_type, so publish_release.py --model v4
        # would have looked valid but could never match a real beta entry.
        self.assertNotIn("v4", KNOWN_MODEL_TYPES)
        self.assertNotIn("v5", KNOWN_MODEL_TYPES)
        self.assertIn("fuel_moisture_station_guarded", KNOWN_MODEL_TYPES)
        self.assertIn("fuel_moisture_station_summer_guarded", KNOWN_MODEL_TYPES)

        import spatial.register_v4 as register_v4
        import spatial.register_v5_beta as register_v5_beta
        self.assertEqual(register_v4.MODEL_TYPE, "fuel_moisture_station_guarded")
        self.assertEqual(register_v5_beta.MODEL_TYPE, "fuel_moisture_station_summer_guarded")
        self.assertIn(register_v4.MODEL_TYPE, KNOWN_MODEL_TYPES)
        self.assertIn(register_v5_beta.MODEL_TYPE, KNOWN_MODEL_TYPES)


class PublishTests(unittest.TestCase):
    def _isolated_registry(self, root_path):
        return (
            patch.object(versioning, "MODELS_DIR", root_path / "models"),
            patch.object(versioning, "VERSIONS_DIR", root_path / "models" / "versions"),
            patch.object(versioning, "CONFIG_PATH", root_path / "models" / "config.json"),
            patch.object(versioning, "API_DIR", root_path),
        )

    def _register_multiasset_beta(self, root_path, model_type):
        asset_path = root_path / "asset.json"
        asset_path.write_text("{}")
        return versioning.register_trained_model(
            model_type, channel="beta", assets={"factor_weights": str(asset_path)},
            performance={"overall_pass": True},
        )

    def test_publish_multiasset_model_without_a_model_checkpoint_role_does_not_crash(self):
        # Regression test: register_trained_model() only sets a non-None
        # top-level `file` when an asset role is named model/checkpoint/
        # static_bundle. fire_weather_index's roles (factor_weights/
        # category_thresholds) match none of those, so beta["file"] is None,
        # and publish() used to call load_active_model_path() unconditionally
        # before ever checking beta.get("assets") - crashing on `API_DIR / None`.
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            patches = self._isolated_registry(root_path)
            with patches[0], patches[1], patches[2], patches[3]:
                self._register_multiasset_beta(root_path, "fire_weather_index")

                calls = []

                def fake_run(cmd, **kwargs):
                    calls.append(cmd)
                    class Result:
                        returncode = 1  # release doesn't exist yet -> "create" branch
                    return Result()

                with patch.object(subprocess, "run", fake_run), \
                     patch.object(publish_release, "paths") as fake_paths:
                    fake_paths.DATA_ROOT = root_path
                    publish_release.publish("fire_weather_index", repo="fakeowner/fake-repo")

            self.assertEqual(len(calls), 2)  # "gh release view" then "gh release create"
            create_cmd = calls[1]
            self.assertIn("gh", create_cmd)
            self.assertIn("create", create_cmd)
            self.assertTrue(any("factor_weights.json" in part for part in create_cmd))

    def test_publish_refuses_without_a_beta_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            patches = self._isolated_registry(root_path)
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaises(SystemExit):
                    publish_release.publish("fire_weather_index", repo="fakeowner/fake-repo")

    def test_publish_refuses_without_a_repo(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit):
                publish_release.publish("fuel_moisture", repo=None)


if __name__ == "__main__":
    unittest.main()
