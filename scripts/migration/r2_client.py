"""Shared R2 client construction for the migration tool.

Mirrors api/services/archive_bundler.py's _r2_client() / api/services/
archive_store.py's client() so credentials and bucket/prefix resolution
stay in one place instead of being rebuilt per script, and so this tool
reads the exact same env vars the production archive code already uses.
"""
import os


def r2_credentials_present():
    return all(os.getenv(k) for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"))


def r2_client():
    import boto3
    from botocore.config import Config

    if not r2_credentials_present():
        raise RuntimeError(
            "R2 credentials are not configured (R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY)"
        )
    return boto3.client(
        service_name="s3",
        endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def r2_bucket():
    return os.getenv("R2_ARCHIVE_BUCKET") or os.getenv("R2_BUCKET", "cdn-showmefire")


def r2_data_prefix():
    return os.getenv("R2_DATA_PREFIX", "data-v2").strip("/")
