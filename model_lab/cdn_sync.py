"""Pull daily archive zips from the ShowMeFire CDN / R2 and unpack them locally.

Primary path uses the public CDN URL (no credentials):

    https://cdn.showmefire.org/data-archive/YYYYMMDD.zip

Optional R2 listing/download when R2_* env vars are set (same keys as the API).

Unpack reuses pipelines.unpack_archive_zip so oversized full-CONUS HRRR
members are refused and only Missouri-safe / JSON members land under
SMF_DATA_ROOT.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from paths import ARCHIVE_ZIPS_DIR, DATA_ROOT
from pipelines.unpack_archive_zip import unpack_zip

CDN_BASE_URL = os.getenv("CDN_BASE_URL", "https://cdn.showmefire.org").rstrip("/")
R2_ARCHIVE_PREFIX = os.getenv("SMF_R2_ARCHIVE_PREFIX", "data-archive")
R2_BUCKET = os.getenv("R2_BUCKET", "cdn-showmefire")
DATE_RE = re.compile(r"^(20\d{6})\.zip$")
DEFAULT_TIMEOUT_SEC = 60
# Cloudflare blocks the default Python-urllib User-Agent with 403.
USER_AGENT = os.getenv("SMF_CDN_USER_AGENT", "ShowMeFire-ModelLab/0.1")


def _cdn_request(url: str, method: str = "GET", headers: dict | None = None) -> Request:
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    return Request(url, method=method, headers=merged)

@dataclass
class RemoteArchive:
    date: str
    key: str
    size_bytes: int | None = None
    source: str = "cdn"  # cdn | r2
    etag: str | None = None
    last_modified: str | None = None

    @property
    def filename(self) -> str:
        return f"{self.date}.zip"

    @property
    def public_url(self) -> str:
        return f"{CDN_BASE_URL}/{self.key}"


@dataclass
class SyncResult:
    date: str
    downloaded: bool
    unpacked: bool
    local_path: str | None
    extracted: int = 0
    skipped: int = 0
    unrecognized: int = 0
    oversized: int = 0
    bytes_written: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def r2_credentials_present() -> bool:
    return all([
        os.getenv("R2_ACCESS_KEY_ID"),
        os.getenv("R2_SECRET_ACCESS_KEY"),
        os.getenv("R2_ACCOUNT_ID"),
    ])


def _r2_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        service_name="s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def list_local_zips() -> list[Path]:
    ARCHIVE_ZIPS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(ARCHIVE_ZIPS_DIR.glob("*.zip"))


def local_zip_for_date(date_token: str) -> Path | None:
    path = ARCHIVE_ZIPS_DIR / f"{date_token}.zip"
    return path if path.exists() else None


def _head_cdn(date_token: str) -> RemoteArchive | None:
    key = f"{R2_ARCHIVE_PREFIX}/{date_token}.zip"
    url = f"{CDN_BASE_URL}/{key}"
    try:
        with urlopen(_cdn_request(url, method="HEAD"), timeout=15) as response:
            length = response.headers.get("Content-Length")
            return RemoteArchive(
                date=date_token,
                key=key,
                size_bytes=int(length) if length and length.isdigit() else None,
                source="cdn",
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
            )
    except HTTPError as exc:
        if exc.code == 404:
            return None
        if exc.code in (403, 405):
            # Some edges reject HEAD; probe with a 1-byte range GET instead.
            try:
                with urlopen(
                    _cdn_request(url, method="GET", headers={"Range": "bytes=0-0"}),
                    timeout=15,
                ) as response:
                    total = None
                    content_range = response.headers.get("Content-Range") or ""
                    if "/" in content_range:
                        tail = content_range.rsplit("/", 1)[-1]
                        if tail.isdigit():
                            total = int(tail)
                    length = response.headers.get("Content-Length")
                    if total is None and length and length.isdigit():
                        total = int(length)
                    return RemoteArchive(
                        date=date_token,
                        key=key,
                        size_bytes=total,
                        source="cdn",
                        etag=response.headers.get("ETag"),
                        last_modified=response.headers.get("Last-Modified"),
                    )
            except HTTPError as inner:
                if inner.code == 404:
                    return None
                raise
        raise
    except URLError:
        return None

def probe_cdn_dates(start: date, end: date, max_workers: int = 8) -> list[RemoteArchive]:
    """HEAD-probe each day in [start, end] on the public CDN."""
    if end < start:
        start, end = end, start
    tokens = []
    cursor = start
    while cursor <= end:
        tokens.append(cursor.strftime("%Y%m%d"))
        cursor += timedelta(days=1)

    found: list[RemoteArchive] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_head_cdn, token): token for token in tokens}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                found.append(result)
    return sorted(found, key=lambda item: item.date)


def list_r2_archives(prefix: str | None = None) -> list[RemoteArchive]:
    if not r2_credentials_present():
        raise RuntimeError("R2 credentials not configured (R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_ACCOUNT_ID)")
    client = _r2_client()
    prefix = prefix or f"{R2_ARCHIVE_PREFIX}/"
    paginator = client.get_paginator("list_objects_v2")
    archives: list[RemoteArchive] = []
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            name = Path(key).name
            match = DATE_RE.match(name)
            if not match:
                continue
            archives.append(RemoteArchive(
                date=match.group(1),
                key=key,
                size_bytes=int(obj.get("Size") or 0),
                source="r2",
                etag=obj.get("ETag"),
                last_modified=obj["LastModified"].isoformat() if obj.get("LastModified") else None,
            ))
    return sorted(archives, key=lambda item: item.date)


def discover_remote_archives(
    start: date | None = None,
    end: date | None = None,
    prefer_r2: bool = False,
) -> list[RemoteArchive]:
    end = end or date.today()
    start = start or (end - timedelta(days=14))
    if prefer_r2 and r2_credentials_present():
        archives = list_r2_archives()
        return [item for item in archives if start.strftime("%Y%m%d") <= item.date <= end.strftime("%Y%m%d")]
    return probe_cdn_dates(start, end)


def download_archive(
    archive: RemoteArchive,
    *,
    force: bool = False,
    progress: Callable[[int, int | None], None] | None = None,
    use_r2: bool = False,
) -> Path:
    """Download one daily zip into ARCHIVE_ZIPS_DIR. Skips if local size matches."""
    ARCHIVE_ZIPS_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_ZIPS_DIR / archive.filename
    expected = archive.size_bytes
    if dest.exists() and not force:
        if expected is None or dest.stat().st_size == expected:
            if progress:
                progress(dest.stat().st_size, expected or dest.stat().st_size)
            return dest

    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()

    if use_r2 and r2_credentials_present():
        client = _r2_client()

        def _cb(bytes_transferred):
            if progress:
                progress(bytes_transferred, expected)

        client.download_file(R2_BUCKET, archive.key, str(tmp), Callback=_cb)
    else:
        with urlopen(_cdn_request(archive.public_url, method="GET"), timeout=DEFAULT_TIMEOUT_SEC) as response, open(tmp, "wb") as handle:
            total = expected
            if total is None:
                length = response.headers.get("Content-Length")
                total = int(length) if length and length.isdigit() else None
            written = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if progress:
                    progress(written, total)

    tmp.replace(dest)
    return dest


def unpack_local_zip(zip_path: Path) -> dict:
    extracted, skipped, unrecognized, oversized = unpack_zip(zip_path)
    return {
        "extracted": extracted,
        "skipped": skipped,
        "unrecognized": len(unrecognized),
        "oversized": len(oversized),
        "unrecognized_names": unrecognized[:20],
        "oversized_names": [name for name, _ in oversized[:20]],
    }


def sync_dates(
    dates: list[str],
    *,
    unpack: bool = True,
    force_download: bool = False,
    use_r2: bool = False,
    progress: Callable[[str, int, int | None], None] | None = None,
) -> list[SyncResult]:
    """Download + optionally unpack a list of YYYYMMDD archive dates."""
    results: list[SyncResult] = []
    for date_token in dates:
        archive = RemoteArchive(date=date_token, key=f"{R2_ARCHIVE_PREFIX}/{date_token}.zip", source="r2" if use_r2 else "cdn")
        # Prefer size from CDN HEAD when available (skip re-download).
        try:
            probed = _head_cdn(date_token)
            if probed is not None:
                archive = probed
        except Exception:
            pass

        result = SyncResult(date=date_token, downloaded=False, unpacked=False, local_path=None)
        try:
            def _prog(written, total, _date=date_token):
                if progress:
                    progress(_date, written, total)

            path = download_archive(archive, force=force_download, progress=_prog, use_r2=use_r2)
            result.downloaded = True
            result.local_path = str(path)
            result.bytes_written = path.stat().st_size if path.exists() else 0
            if unpack:
                stats = unpack_local_zip(path)
                result.unpacked = True
                result.extracted = stats["extracted"]
                result.skipped = stats["skipped"]
                result.unrecognized = stats["unrecognized"]
                result.oversized = stats["oversized"]
        except Exception as exc:  # noqa: BLE001 — surface per-date failures to the UI
            result.error = str(exc)
        results.append(result)

    # Invalidate cached date discovery used by the compare tabs.
    try:
        from model_lab.data import discover_forecast_dates, discover_forecast_series
        discover_forecast_dates.cache_clear()
        discover_forecast_series.cache_clear()
    except Exception:
        pass
    return results


def sync_missing_since(days: int = 7, unpack: bool = True) -> list[SyncResult]:
    end = date.today()
    start = end - timedelta(days=max(1, days) - 1)
    remote = discover_remote_archives(start=start, end=end, prefer_r2=r2_credentials_present())
    missing = [item.date for item in remote if local_zip_for_date(item.date) is None]
    if not missing:
        # Still unpack any local zips that might not have been extracted yet.
        results = []
        for item in remote:
            path = local_zip_for_date(item.date)
            if path is None:
                continue
            result = SyncResult(date=item.date, downloaded=False, unpacked=False, local_path=str(path),
                                bytes_written=path.stat().st_size)
            if unpack:
                stats = unpack_local_zip(path)
                result.unpacked = True
                result.extracted = stats["extracted"]
                result.skipped = stats["skipped"]
                result.unrecognized = stats["unrecognized"]
                result.oversized = stats["oversized"]
            results.append(result)
        return results
    return sync_dates(missing, unpack=unpack, use_r2=r2_credentials_present())


def local_sync_status(lookback_days: int = 30):
    """Return a DataFrame describing local vs remote coverage for recent days."""
    import pandas as pd

    end = date.today()
    start = end - timedelta(days=lookback_days - 1)
    try:
        remote = {item.date: item for item in discover_remote_archives(start=start, end=end, prefer_r2=False)}
    except Exception:
        remote = {}
    rows = []
    cursor = start
    while cursor <= end:
        token = cursor.strftime("%Y%m%d")
        local = local_zip_for_date(token)
        remote_item = remote.get(token)
        rows.append({
            "date": token,
            "on_cdn": remote_item is not None,
            "cdn_mb": round((remote_item.size_bytes or 0) / 1e6, 1) if remote_item else None,
            "local_zip": local is not None,
            "local_mb": round(local.stat().st_size / 1e6, 1) if local else None,
            "data_root": str(DATA_ROOT),
        })
        cursor += timedelta(days=1)
    return pd.DataFrame(rows)