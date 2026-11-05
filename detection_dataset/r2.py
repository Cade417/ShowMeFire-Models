"""Download the latest detection training export from Cloudflare R2.

Reads the layout api/services/detection_r2.py publishes (keep the two in sync):
    <prefix>/latest.json -> {"snapshot", "zip_key", "zip_sha256", "zip_bytes", "rows", "detection_ranges", ...}
The zip's sha256 is checked against the pointer here, and every file inside against its manifest by
labels.import_bundle, before anything lands in SMF_DATA_ROOT/fire_labels.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def credentials_present() -> bool:
    return all(os.getenv(k) for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"))


def bucket() -> str:
    return os.getenv("R2_TRAINING_BUCKET") or os.getenv("R2_ARCHIVE_BUCKET") or os.getenv("R2_BUCKET", "cdn-showmefire")


def prefix() -> str:
    return os.getenv("R2_TRAINING_PREFIX", "training/detections").strip("/")


def make_client():
    import boto3
    from botocore.config import Config
    if not credentials_present():
        raise RuntimeError("R2 credentials are not configured (R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY)")
    return boto3.client(
        service_name="s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"), region_name="auto")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_pointer(client=None) -> dict:
    client = client or make_client()
    try:
        body = client.get_object(Bucket=bucket(), Key=f"{prefix()}/latest.json")["Body"].read()
    except Exception as exc:  # noqa: BLE001 - botocore's ClientError
        if getattr(exc, "response", {}).get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
            raise FileNotFoundError(f"nothing published yet at r2://{bucket()}/{prefix()}/latest.json - "
                                    "run api/scripts/publish_detection_training_data.py") from None
        raise
    return json.loads(body)


def pull_latest(directory: Path | None = None, client=None) -> Path:
    """Download + verify the newest snapshot and unpack it into fire_labels. Returns the manifest path."""
    from detection_dataset import labels
    client = client or make_client()
    pointer = latest_pointer(client)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "detection_training_export.zip"
        client.download_file(bucket(), pointer["zip_key"], str(zip_path))
        if _sha256(zip_path) != pointer["zip_sha256"]:
            raise ValueError(f"checksum mismatch for {pointer['zip_key']} - re-publish and retry")
        ranges = pointer.get("detection_ranges", {})
        print(f"Pulled snapshot {pointer['snapshot']}: {pointer['rows'].get('detections')} detections, "
              + ", ".join(f"{k} {v['min'][:10]}..{v['max'][:10]}" for k, v in ranges.items()))
        return labels.import_bundle(zip_path, directory)
