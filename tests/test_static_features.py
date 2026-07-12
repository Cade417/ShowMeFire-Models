import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from models import versioning
from static_features.build_bundle import _encode_categories
from static_features.download_sources import fetch
from static_features.schema import (CATEGORICAL_CHANNELS, CONTINUOUS_CHANNELS, SCHEMA_VERSION,
                                    grid_fingerprint, sha256_file, validate_bundle)


class StaticFeatureTests(unittest.TestCase):
    def test_http_download_resumes_partial_file(self):
        class Response:
            status_code = 206
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def raise_for_status(self): pass
            def iter_content(self, _): yield b"new"
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "source.tif"; target.with_suffix(".tif.partial").write_bytes(b"old")
            with patch("static_features.download_sources.requests.get", return_value=Response()) as request:
                fetch("https://official.example/source.tif", target)
            self.assertEqual(target.read_bytes(), b"oldnew")
            self.assertEqual(request.call_args.kwargs["headers"]["Range"], "bytes=3-")

    def test_category_encoding_reserves_zero_for_unknown(self):
        encoded, mapping = _encode_categories(np.array([[11, 21], [np.nan, -9999]], dtype=float))
        self.assertEqual(encoded[1, 0], 0); self.assertEqual(encoded[1, 1], 0)
        self.assertEqual(set(np.unique(encoded)), {0, 1, 2}); self.assertEqual(mapping["0"], 0)

    def test_bundle_contract_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle = root / "static_features_test.nc"; manifest_path = bundle.with_suffix(".json")
            x = np.arange(256, dtype=float); y = np.arange(256, dtype=float); crs = "LOCAL_CS[\"test\"]"; fingerprint = grid_fingerprint(x, y, crs)
            variables = {name: (("y", "x"), np.ones((256, 256), dtype="float32")) for name in CONTINUOUS_CHANNELS}
            variables.update({name: (("y", "x"), np.ones((256, 256), dtype="int16")) for name in CATEGORICAL_CHANNELS})
            xr.Dataset(variables, coords={"x": x, "y": y}, attrs={"crs_wkt": crs, "grid_fingerprint": fingerprint}).to_netcdf(bundle)
            manifest = {"schema_version": SCHEMA_VERSION, "sha256": sha256_file(bundle), "grid_fingerprint": fingerprint}
            manifest_path.write_text(json.dumps(manifest)); self.assertEqual(validate_bundle(bundle)["sha256"], manifest["sha256"])
            bundle.write_bytes(bundle.read_bytes() + b"corrupt")
            with self.assertRaises(ValueError): validate_bundle(bundle)

    def test_multi_asset_registry_keeps_legacy_file_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); models = root / "models"; versions = models / "versions"; config = models / "config.json"
            model = root / "candidate.onnx"; bundle = root / "static.nc"; model.write_bytes(b"model"); bundle.write_bytes(b"bundle")
            with patch.multiple(versioning, MODELS_DIR=models, VERSIONS_DIR=versions, CONFIG_PATH=config, API_DIR=root):
                assigned = versioning.register_trained_model("fuel_moisture_spatial", performance={"mae": 1}, assets={"model": model, "static_bundle": bundle})
                entry = versioning.get_model_entry("fuel_moisture_spatial")["beta"]
                self.assertTrue(entry["file"].endswith("_model.onnx")); self.assertEqual(set(entry["assets"]), {"model", "static_bundle"})
                self.assertEqual(versioning.load_active_assets("fuel_moisture_spatial", "beta")["model"]["path"].read_bytes(), b"model")
                versioning.promote("fuel_moisture_spatial", assigned)
                stable = versioning.load_active_assets("fuel_moisture_spatial", "stable")
                self.assertEqual(stable["static_bundle"]["path"].read_bytes(), b"bundle")


if __name__ == "__main__": unittest.main()
