import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_environment as ce


class ParsePinnedRequirementsTests(unittest.TestCase):
    def test_parses_simple_pin(self):
        with _temp_requirements("pandas==2.3.3\n") as path:
            self.assertEqual(ce._parse_pinned_requirements(path), {"pandas": "2.3.3"})

    def test_ignores_comments_and_loose_floors(self):
        with _temp_requirements("# a comment\npandas>=2.0\nxgboost==3.1.2\n") as path:
            self.assertEqual(ce._parse_pinned_requirements(path), {"xgboost": "3.1.2"})

    def test_excludes_pins_whose_platform_marker_does_not_match(self):
        # Mirrors requirements.txt's real eccodeslib line - the exact bug
        # this parser must not repeat (naive string-splitting flagged this
        # as "missing" on Windows even though the marker correctly excludes it there).
        with _temp_requirements('eccodeslib==2.44.1.8; platform_system != "definitely_not_a_real_platform"\n') as path:
            self.assertEqual(ce._parse_pinned_requirements(path), {"eccodeslib": "2.44.1.8"})

    def test_includes_pins_whose_platform_marker_matches(self):
        import platform
        with _temp_requirements(f'pandas==2.3.3; platform_system == "{platform.system()}"\n') as path:
            self.assertEqual(ce._parse_pinned_requirements(path), {"pandas": "2.3.3"})

    def test_missing_file_returns_empty(self):
        self.assertEqual(ce._parse_pinned_requirements(Path("/nonexistent/requirements.txt")), {})


class CheckPackageVersionsTests(unittest.TestCase):
    def test_installed_version_matching_pin_passes(self):
        with _temp_requirements("pandas==2.3.3\n") as path:
            with patch("importlib.metadata.version", return_value="2.3.3"):
                results = ce.check_package_versions(path)
        self.assertEqual(results[0].level, "pass")

    def test_installed_version_mismatch_warns_not_fails(self):
        with _temp_requirements("pandas==2.3.3\n") as path:
            with patch("importlib.metadata.version", return_value="2.0.0"):
                results = ce.check_package_versions(path)
        self.assertEqual(results[0].level, "warn")

    def test_not_installed_fails(self):
        import importlib.metadata
        with _temp_requirements("pandas==2.3.3\n") as path:
            with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
                results = ce.check_package_versions(path)
        self.assertEqual(results[0].level, "fail")


class CheckDataRootTests(unittest.TestCase):
    def test_unset_env_var_warns(self):
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("SMF_DATA_ROOT", None)
            result = ce.check_data_root()
        self.assertEqual(result.level, "warn")
        self.assertIn("SMF_DATA_ROOT is not set", result.detail)

    def test_nonexistent_path_fails(self):
        with patch.dict("os.environ", {"SMF_DATA_ROOT": "/definitely/not/a/real/path/xyz"}):
            result = ce.check_data_root()
        self.assertEqual(result.level, "fail")

    def test_existing_writable_path_passes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"SMF_DATA_ROOT": tmp}):
                result = ce.check_data_root()
        self.assertEqual(result.level, "pass")


class CheckDiskSpaceTests(unittest.TestCase):
    def test_below_threshold_warns(self):
        class FakeUsage:
            free = 5 * 1024 ** 3  # 5 GB
        with patch("shutil.disk_usage", return_value=FakeUsage()):
            result = ce.check_disk_space(min_free_gb=20.0)
        self.assertEqual(result.level, "warn")

    def test_above_threshold_passes(self):
        class FakeUsage:
            free = 500 * 1024 ** 3  # 500 GB
        with patch("shutil.disk_usage", return_value=FakeUsage()):
            result = ce.check_disk_space(min_free_gb=20.0)
        self.assertEqual(result.level, "pass")


class CheckOptionalTokenTests(unittest.TestCase):
    def test_set_passes(self):
        with patch.dict("os.environ", {"SOME_TOKEN": "value"}):
            result = ce.check_optional_token("SOME_TOKEN", "testing")
        self.assertEqual(result.level, "pass")

    def test_unset_warns(self):
        import os
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("SOME_TOKEN", None)
            result = ce.check_optional_token("SOME_TOKEN", "testing")
        self.assertEqual(result.level, "warn")


class SummarizeTests(unittest.TestCase):
    def test_exit_code_zero_when_no_failures(self):
        results = [ce.CheckResult("a", "pass", ""), ce.CheckResult("b", "warn", "")]
        self.assertEqual(ce.summarize(results), 0)

    def test_exit_code_one_when_any_failure(self):
        results = [ce.CheckResult("a", "pass", ""), ce.CheckResult("b", "fail", "")]
        self.assertEqual(ce.summarize(results), 1)


def _temp_requirements(content: str):
    import tempfile

    class _Ctx:
        def __enter__(self):
            self.tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
            self.tmp.write(content)
            self.tmp.close()
            return Path(self.tmp.name)

        def __exit__(self, *exc_info):
            Path(self.tmp.name).unlink(missing_ok=True)

    return _Ctx()


if __name__ == "__main__":
    unittest.main()
