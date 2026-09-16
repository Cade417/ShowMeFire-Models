import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from pipelines import unpack_archive_zip as uaz


def _write_zip(path: Path, members: dict) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for arcname, data in members.items():
            compression = zipfile.ZIP_STORED if arcname.endswith(".nc") else zipfile.ZIP_DEFLATED
            zf.writestr(arcname, data, compress_type=compression)
    return path


def _destinations_under(root: Path):
    # DESTINATIONS binds each dest_dir at import time, so a test must swap
    # the whole list rather than patch paths.* after the fact.
    return [
        (lambda name: name.startswith("rtma_") and name.endswith(".nc"), root / "cache_rtma"),
        (lambda name: name.endswith(".nc"), root / "cache_hrrr"),
        (lambda name: name.startswith("raw_data_") and name.endswith(".json"), root / "raw_data"),
        (lambda name: name.startswith("station_forecasts_") and name.endswith(".json"), root / "forecasts"),
    ]


class OversizedNcGuardTests(unittest.TestCase):
    def test_refuses_an_uncropped_full_conus_sized_nc_member(self):
        # Regression test for a real incident: the server's own cache/hrrr
        # archive stores the full-CONUS HRRR grid (~488MB) under the exact
        # same filename this repo's fetch_hrrr() uses for its Missouri-cropped
        # (~25MB) cache file. Unpacking a server zip must never let that
        # oversized member overwrite the correctly-cropped cache.
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            zip_path = _write_zip(root_path / "20260327.zip", {
                "hrrr/hrrr_20260327_12z_f04-15.nc": b"x" * (uaz.MAX_CROPPED_NC_BYTES + 1),
            })
            with patch.object(uaz, "DESTINATIONS", _destinations_under(root_path)):
                extracted, skipped, unrecognized, oversized = uaz.unpack_zip(zip_path)

            self.assertEqual(extracted, 0)
            self.assertEqual(len(oversized), 1)
            self.assertFalse((root_path / "cache_hrrr" / "hrrr_20260327_12z_f04-15.nc").exists())

    def test_extracts_a_correctly_cropped_nc_member(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            zip_path = _write_zip(root_path / "20260327.zip", {
                "hrrr/hrrr_20260327_12z_f04-15.nc": b"x" * 1024,
            })
            with patch.object(uaz, "DESTINATIONS", _destinations_under(root_path)):
                extracted, skipped, unrecognized, oversized = uaz.unpack_zip(zip_path)

            self.assertEqual(extracted, 1)
            self.assertEqual(oversized, [])
            self.assertTrue((root_path / "cache_hrrr" / "hrrr_20260327_12z_f04-15.nc").exists())

    def test_forecasts_and_raw_data_members_are_unaffected_by_the_size_guard(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            zip_path = _write_zip(root_path / "20260806.zip", {
                "forecasts/station_forecasts_20260806_12.json": b"{}",
                "raw_data/raw_data_20260806.json": b"{}",
                "hrrr/hrrr_20260806_12z_f04-15.nc": b"x" * (uaz.MAX_CROPPED_NC_BYTES + 1),
            })
            with patch.object(uaz, "DESTINATIONS", _destinations_under(root_path)):
                extracted, skipped, unrecognized, oversized = uaz.unpack_zip(zip_path)

            self.assertEqual(extracted, 2)
            self.assertEqual(len(oversized), 1)
            self.assertTrue((root_path / "forecasts" / "station_forecasts_20260806_12.json").exists())
            self.assertTrue((root_path / "raw_data" / "raw_data_20260806.json").exists())


if __name__ == "__main__":
    unittest.main()
