"""Download and register the fire-behavior static source rasters.

The script uses the LANDFIRE Product Service for the three LANDFIRE layers
and The National Map API for 3DEP DEM tiles.  It deliberately requires an
explicit LANDFIRE release so a later provider release cannot silently change
an experiment.

Example (PowerShell):

    python static_features/download_fire_behavior_assets.py `
      --email analyst@example.org `
      --landfire-release LF2025
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable
from urllib.parse import urljoin

import numpy as np
import requests
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths


DEFAULT_BBOX = (-96.8, 34.8, -88.1, 41.8)  # west, south, east, north
LFPS_BASE = "https://lfps.usgs.gov"
LFPS_SUBMIT = f"{LFPS_BASE}/api/job/submit"
LFPS_STATUS = f"{LFPS_BASE}/api/job/status"
TNM_PRODUCTS = "https://tnmaccess.nationalmap.gov/api/v1/products"
TNM_DATASET = "National Elevation Dataset (NED) 1/3 arc-second"
TNM_TILE_RE = re.compile(r"\b([ns]\d{2}[ew]\d{3})\b", re.IGNORECASE)
PRODUCTS = ("dem", "fbfm40", "canopy_cover", "canopy_height")
UNITS = {"dem": "m", "fbfm40": "code", "canopy_cover": "percent", "canopy_height": "m"}


def retrying_session() -> requests.Session:
    """Create an HTTP session resilient to transient DNS/server failures."""
    retry = Retry(
        total=8,
        connect=8,
        read=4,
        status=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    """Parse west,south,east,north and reject malformed or inverted extents."""
    try:
        bbox = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north") from exc
    if len(bbox) != 4:
        raise argparse.ArgumentTypeError("bbox must contain four comma-separated numbers")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise argparse.ArgumentTypeError("bbox coordinates are invalid or inverted")
    return bbox  # type: ignore[return-value]


def landfire_layers(release: str) -> tuple[str, str, str]:
    """Return LFPS layer names in the requested output order."""
    match = re.fullmatch(r"LF\d{4}", release.upper())
    if not match:
        raise ValueError("LANDFIRE release must look like LF2025")
    release = release.upper()
    return (f"{release}_FBFM40", f"{release}_CC", f"{release}_CH")


def _json_url(value: Any) -> str | None:
    """Find a download URL in the varying LFPS status response shapes."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and (
                key.lower() in {"url", "downloadurl", "download_url", "fileurl", "file_url"}
                or item.lower().startswith(("http://", "https://"))
            ):
                if item.lower().startswith(("http://", "https://")):
                    return item
                if "download" in key.lower() and item.strip():
                    return item
            found = _json_url(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _json_url(item)
            if found:
                return found
    elif isinstance(value, str) and value.lower().startswith(("http://", "https://")):
        return value
    return None


def submit_landfire(
    session: requests.Session,
    email: str,
    release: str,
    bbox: tuple[float, float, float, float],
    timeout: int,
    output_resolution: int | None = None,
) -> tuple[str, str]:
    layers = landfire_layers(release)
    west, south, east, north = bbox
    form = {
        "Email": email,
        "Layer_List": ";".join(layers),
        "Area_of_Interest": f"{west} {south} {east} {north}",
        "Include_Layer_List_XML_File": "true",
    }
    if output_resolution is not None:
        if output_resolution <= 30:
            raise ValueError("output resolution must be greater than 30 metres")
        form["Resample_Resolution"] = str(output_resolution)
    # LFPS documents GET and POST, but the live endpoint currently returns
    # HTTP 500 for the equivalent POST form submission. Query parameters are
    # accepted by GET and are also easier to reproduce from the provider docs.
    response = session.get(
        LFPS_SUBMIT,
        params=form,
        timeout=timeout,
    )
    if not response.ok:
        raise requests.HTTPError(
            f"LFPS submit failed with HTTP {response.status_code}: {response.text[:1000]}",
            response=response,
        )
    payload = response.json()
    job_id = payload.get("jobId") or payload.get("jobID") or payload.get("JobId")
    if not job_id:
        raise RuntimeError(f"LFPS did not return a job ID: {payload}")
    return str(job_id), ";".join(layers)


def wait_for_landfire(
    session: requests.Session,
    job_id: str,
    timeout: int,
    poll_seconds: int,
    max_wait_seconds: int,
) -> str:
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        response = session.get(LFPS_STATUS, params={"JobId": job_id}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get("status", "")).lower()
        if status in {"succeeded", "success", "complete", "completed"}:
            url = _json_url(payload)
            if not url:
                raise RuntimeError(f"LFPS job succeeded without a download URL: {payload}")
            return urljoin(LFPS_BASE, url)
        if status in {"failed", "canceled", "cancelled"}:
            raise RuntimeError(f"LFPS job {job_id} {status}: {payload}")
        time.sleep(poll_seconds)
    raise TimeoutError(f"LFPS job {job_id} did not finish within {max_wait_seconds} seconds")


def download_file(session: requests.Session, url: str, target: Path, timeout: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with session.get(url, headers=headers, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        append = existing > 0 and response.status_code == 206
        if existing and not append:
            existing = 0
        with partial.open("ab" if append else "wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
    partial.replace(target)


def discover_dem_urls(
    session: requests.Session,
    bbox: tuple[float, float, float, float],
    timeout: int,
) -> list[str]:
    response = session.get(
        TNM_PRODUCTS,
        params={
            "datasets": TNM_DATASET,
            "bbox": ",".join(str(value) for value in bbox),
            "outputFormat": "JSON",
            "max": 1000,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    urls = select_latest_dem_urls(response.json().get("items", []))
    if not urls:
        raise RuntimeError("3DEP API returned no DEM tiles for the requested bbox")
    return urls


def select_latest_dem_urls(items: Iterable[dict[str, Any]]) -> list[str]:
    """Keep the newest dated 3DEP record for each geographic tile.

    The TNM API returns historical revisions as separate records. Downloading
    every record would repeatedly fetch the same tile and can multiply the
    requested storage by an order of magnitude.
    """
    selected: dict[str, tuple[str, str]] = {}
    fallback: set[str] = set()
    for item in items:
        url = item.get("downloadURL")
        if not url:
            continue
        title = str(item.get("title") or "")
        match = TNM_TILE_RE.search(title)
        if not match:
            fallback.add(url)
            continue
        tile = match.group(1).lower()
        dates = re.findall(r"\b\d{8}\b", title)
        date = max(dates) if dates else ""
        current = selected.get(tile)
        if current is None or date > current[0]:
            selected[tile] = (date, url)
    return sorted(fallback | {url for _, url in selected.values()})


def _single_tif(directory: Path, label: str) -> Path:
    candidates = sorted(path for path in directory.rglob("*") if path.suffix.lower() in {".tif", ".tiff"})
    if len(candidates) != 1:
        raise RuntimeError(f"{label} archive must contain exactly one GeoTIFF; found {len(candidates)}")
    return candidates[0]


def extract_landfire(
    archive: Path,
    output_dir: Path,
    release: str,
) -> dict[str, Path]:
    """Extract the three requested bands, preserving their raw numeric values."""
    layers = landfire_layers(release)
    with TemporaryDirectory(prefix="landfire-") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
        tifs = sorted(path for path in extracted.rglob("*") if path.suffix.lower() in {".tif", ".tiff"})
        if len(tifs) != 1:
            raise RuntimeError(f"LANDFIRE archive must contain one multiband GeoTIFF; found {len(tifs)}")
        source = tifs[0]
        with rasterio.open(source) as src:
            if src.count != len(layers):
                raise RuntimeError(f"LANDFIRE GeoTIFF has {src.count} bands; expected {len(layers)}")
            descriptions = [description or "" for description in src.descriptions]
            outputs: dict[str, Path] = {}
            for index, layer in enumerate(layers, start=1):
                band = index
                matching = [i + 1 for i, description in enumerate(descriptions) if layer.lower() in description.lower()]
                if matching:
                    band = matching[0]
                target = output_dir / {
                    layers[0]: "fbfm40.tif",
                    layers[1]: "canopy_cover.tif",
                    layers[2]: "canopy_height.tif",
                }[layer]
                profile = src.profile.copy()
                profile.update(count=1, compress="deflate", tiled=True)
                with rasterio.open(target, "w", **profile) as dst:
                    dst.write(src.read(band), 1)
                    dst.set_band_description(1, layer)
                product = (
                    "fbfm40" if layer.endswith("_FBFM40")
                    else "canopy_cover" if layer.endswith("_CC")
                    else "canopy_height"
                )
                outputs[product] = target
        return outputs


def mosaic_dem(
    session: requests.Session,
    urls: Iterable[str],
    output: Path,
    temporary: Path,
    timeout: int,
    workers: int = 6,
) -> list[str]:
    urls = list(urls)
    tile_cache = paths.CACHE_DIR / "3dep"
    tile_cache.mkdir(parents=True, exist_ok=True)

    def fetch_tile(item: tuple[int, str]) -> tuple[int, Path]:
        index, url = item
        tile_match = TNM_TILE_RE.search(url)
        cache_name = (
            f"{tile_match.group(1).lower()}_{Path(url.split('?')[0]).name}"
            if tile_match else f"dem_{index:04d}_{Path(url.split('?')[0]).name}"
        )
        downloaded = tile_cache / cache_name
        if downloaded.is_file() and downloaded.stat().st_size > 0:
            return index, downloaded
        # Use an independent session per worker; requests sessions are not
        # guaranteed to be thread-safe.
        with retrying_session() as worker_session:
            download_file(worker_session, url, downloaded, timeout)
        return index, downloaded

    downloaded_tiles: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_tile, item) for item in enumerate(urls)]
        for completed, future in enumerate(as_completed(futures), start=1):
            index, downloaded = future.result()
            downloaded_tiles[index] = downloaded
            print(f"Downloaded DEM tile {completed}/{len(urls)}", flush=True)

    tile_paths: list[Path] = []
    for index, url in enumerate(urls):
        suffix = ".zip" if ".zip" in url.lower().split("?")[0] else ".tif"
        downloaded = downloaded_tiles[index]
        if suffix == ".zip":
            with TemporaryDirectory(prefix="dem-tile-") as extracted:
                with zipfile.ZipFile(downloaded) as archive:
                    archive.extractall(extracted)
                tile = _single_tif(Path(extracted), "3DEP")
                persistent = temporary / f"tile_{index:04d}.tif"
                shutil.copy2(tile, persistent)
                tile_paths.append(persistent)
        else:
            tile_paths.append(downloaded)
    mosaic_partial = output.with_suffix(".tif.partial")
    mosaic_partial.unlink(missing_ok=True)
    sources = [rasterio.open(tile) for tile in tile_paths]
    try:
        with sources[0] as first:
            left = min(source.bounds.left for source in sources)
            bottom = min(source.bounds.bottom for source in sources)
            right = max(source.bounds.right for source in sources)
            top = max(source.bounds.top for source in sources)
            res_x, res_y = first.res
            width = int(np.ceil((right - left) / res_x))
            height = int(np.ceil((top - bottom) / res_y))
            transform = from_origin(left, top, res_x, res_y)
            profile = first.profile.copy()
            profile.update(
                height=height,
                width=width,
                transform=transform,
                count=1,
                compress="deflate",
                BIGTIFF="YES",
                tiled=True,
            )
            # dst_path makes rasterio.merge process bounded windows instead
            # of materializing the complete 31+ GiB mosaic.
            merge(
                sources,
                bounds=(left, bottom, right, top),
                res=(res_x, res_y),
                nodata=first.nodata,
                dst_path=mosaic_partial,
                dst_kwds=profile,
                mem_limit=256,
            )
    finally:
        for source in sources[1:]:
            source.close()
    valid_fraction = raster_valid_fraction(mosaic_partial)
    if valid_fraction < 0.80:
        mosaic_partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"DEM mosaic covers only {valid_fraction:.1%} of its tile extent; "
            "refusing to replace the existing DEM"
        )
    mosaic_partial.replace(output)
    return list(urls)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raster_metadata(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        if src.crs is None or src.count != 1 or src.nodata is None:
            raise ValueError(f"{path} must be single-band with CRS and nodata")
        return {
            "crs": src.crs.to_string(),
            "bounds": list(src.bounds),
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }


def raster_valid_fraction(path: Path) -> float:
    """Estimate full-raster coverage without loading the raster into RAM."""
    with rasterio.open(path) as src:
        sample = src.read(
            1,
            out_shape=(1, min(src.height, 256), min(src.width, 256)),
            masked=True,
        )
        return float(np.ma.count(sample) / sample.size)


def register(
    files: dict[str, Path],
    urls: dict[str, Any],
    release: str,
    bbox: tuple[float, float, float, float],
) -> Path:
    manifest_path = paths.STATIC_SOURCE_DIR / "source_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        manifest = {}
    products = dict(manifest.get("products") or {})
    for product, path in files.items():
        metadata = raster_metadata(path)
        if product == "canopy_cover":
            with rasterio.open(path) as src:
                values = src.read(1, masked=True)
                if values.count() and (float(values.min()) < 0 or float(values.max()) > 100):
                    raise ValueError("canopy cover must contain values in the range 0..100")
        products[product] = {
            "path": str(path),
            "url": urls[product],
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "units": UNITS[product],
            "raster": metadata,
            "source_release": release if product != "dem" else "USGS 3DEP 1/3 arc-second",
        }
    result = {
        "release": release,
        "bbox": bbox,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "updated_products": list(files),
        "products": products,
        "official_sources": {
            "dem": "USGS 3DEP/The National Map",
            "landfire": "LANDFIRE Product Service",
        },
    }
    manifest_path.write_text(json.dumps(result, indent=2))
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="LFPS request email")
    parser.add_argument("--landfire-release", required=True, help="Explicit release, for example LF2025")
    parser.add_argument("--bbox", type=parse_bbox, default=DEFAULT_BBOX, help="west,south,east,north in WGS84")
    parser.add_argument(
        "--output-resolution",
        type=int,
        help="Optional LFPS output resolution in metres; omit to retain provider-native resolution",
    )
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout in seconds")
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-wait-seconds", type=int, default=1800)
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent 3DEP tile downloads (default: 6; use 1 for serial downloads)",
    )
    parser.add_argument(
        "--landfire-job-id",
        help="Reuse an existing LFPS job ID instead of submitting a new request",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing source rasters")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    landfire_layers(args.landfire_release)
    output_dir = paths.STATIC_SOURCE_DIR
    expected = {product: output_dir / f"{product}.tif" for product in PRODUCTS}
    landfire_products = ("fbfm40", "canopy_cover", "canopy_height")
    if not args.force and all(expected[product].exists() for product in landfire_products):
        landfire_files = {product: expected[product] for product in landfire_products}
        print("Reusing existing LANDFIRE source rasters")
    else:
        landfire_files = None
    session = retrying_session()
    with TemporaryDirectory(prefix="showmefire-assets-") as temporary:
        temporary_path = Path(temporary)
        if landfire_files is None:
            if args.landfire_job_id:
                job_id = args.landfire_job_id
                print(f"Reusing LANDFIRE job: {job_id}")
            else:
                job_id, _ = submit_landfire(
                    session,
                    args.email,
                    args.landfire_release,
                    args.bbox,
                    args.timeout,
                    args.output_resolution,
                )
                print(f"LANDFIRE job submitted: {job_id}")
            landfire_url = wait_for_landfire(session, job_id, args.timeout, args.poll_seconds, args.max_wait_seconds)
            landfire_archive = temporary_path / "landfire.zip"
            download_file(session, landfire_url, landfire_archive, args.timeout)
            landfire_files = extract_landfire(landfire_archive, output_dir, args.landfire_release)
        else:
            landfire_url = None
        dem_is_complete = (
            expected["dem"].exists()
            and raster_valid_fraction(expected["dem"]) >= 0.80
        )
        if not args.force and dem_is_complete:
            existing_manifest = paths.STATIC_SOURCE_DIR / "source_manifest.json"
            try:
                dem_urls = json.loads(existing_manifest.read_text())["products"]["dem"]["url"]
            except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
                dem_urls = []
            dem_file = expected["dem"]
            print("Reusing existing DEM source raster")
        else:
            if expected["dem"].exists() and not args.force:
                print(
                    f"Existing DEM covers only {raster_valid_fraction(expected['dem']):.1%} "
                    "of its extent; regenerating it"
                )
            dem_urls = discover_dem_urls(session, args.bbox, args.timeout)
            dem_file = output_dir / "dem.tif"
            mosaic_dem(session, dem_urls, dem_file, temporary_path, args.timeout, args.workers)
    files = {"dem": dem_file, **landfire_files}
    urls = {
        "dem": dem_urls,
        "fbfm40": landfire_url,
        "canopy_cover": landfire_url,
        "canopy_height": landfire_url,
    }
    print(register(files, urls, args.landfire_release, args.bbox))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
