import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection_dataset import r2


def _bundle(tmp_path, detections_rows=2):
    """A zip laid out the way api/services/detection_r2.py publishes it."""
    src = tmp_path / "src"
    src.mkdir()
    files = {}
    for key, name in (("detections", "detections.csv"), ("incidents", "incidents.csv"),
                      ("feedback", "feedback.csv"), ("user_reports", "user_reports.csv")):
        (src / name).write_text("id\n" + "".join(f"{i}\n" for i in range(detections_rows if key == "detections" else 0)))
        files[key] = {"filename": name, "rows": detections_rows if key == "detections" else 0,
                      "sha256": hashlib.sha256((src / name).read_bytes()).hexdigest(), "missing_columns": []}
    manifest = {"schema_version": "detection-training-v1", "files": files,
                "detection_ranges": {"viirs": {"min": "2026-01-01T00:00:00Z", "max": "2026-01-02T00:00:00Z", "count": detections_rows}}}
    (src / "detection_training_manifest.json").write_text(json.dumps(manifest))
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in src.iterdir():
            z.write(f, f.name)
    return zip_path, manifest


class FakeS3:
    def __init__(self, zip_path, manifest, sha=None):
        self.zip_path = zip_path
        self.pointer = {"snapshot": "20261027T000000Z", "zip_key": "training/detections/snapshots/x/detection_training_export.zip",
                        "zip_sha256": sha or hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                        "rows": {"detections": 2}, "detection_ranges": manifest["detection_ranges"]}

    def get_object(self, Bucket, Key):
        assert Key == "training/detections/latest.json"
        return {"Body": type("B", (), {"read": lambda s: json.dumps(self.pointer).encode()})()}

    def download_file(self, bucket, key, dest):
        assert key == self.pointer["zip_key"]
        shutil.copy(self.zip_path, dest)


def test_pull_latest_verifies_and_unpacks(tmp_path, monkeypatch):
    monkeypatch.delenv("R2_TRAINING_BUCKET", raising=False)
    zip_path, manifest = _bundle(tmp_path)
    target = tmp_path / "fire_labels"
    out = r2.pull_latest(target, client=FakeS3(zip_path, manifest))
    assert out == target / "detection_training_manifest.json" and (target / "detections.csv").exists()


def test_pull_latest_rejects_a_zip_that_does_not_match_the_pointer(tmp_path):
    zip_path, manifest = _bundle(tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        r2.pull_latest(tmp_path / "fire_labels", client=FakeS3(zip_path, manifest, sha="0" * 64))
    assert not (tmp_path / "fire_labels").exists()


def test_missing_pointer_gives_a_publish_hint():
    class Empty:
        def get_object(self, Bucket, Key):
            err = Exception("nope")
            err.response = {"Error": {"Code": "NoSuchKey"}}
            raise err
    with pytest.raises(FileNotFoundError, match="publish_detection_training_data"):
        r2.latest_pointer(Empty())
