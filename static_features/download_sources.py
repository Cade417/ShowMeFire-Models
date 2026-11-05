from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

BBOX = (-96.8, 34.8, -88.1, 41.8)
PRODUCTS = ("dem", "nlcd_class", "nlcd_confidence", "fbfm40", "fvt", "canopy_cover", "canopy_height")
DEFAULT_UNITS = {
    "dem": "m",
    "nlcd_class": "code",
    "nlcd_confidence": "percent",
    "fbfm40": "code",
    "fvt": "code",
    "canopy_cover": "percent",
    "canopy_height": "m",
}


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def fetch(url: str, target: Path):
    """Resume a direct official GeoTIFF/zip download using HTTP Range."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=180) as response:
        response.raise_for_status()
        mode = "ab" if existing and response.status_code == 206 else "wb"
        with open(partial, mode) as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk: output.write(chunk)
    partial.replace(target)


def discover_3dep_urls():
    response = requests.get("https://tnmaccess.nationalmap.gov/api/v1/products", params={
        "datasets": "National Elevation Dataset (NED) 1/3 arc-second",
        "bbox": ",".join(map(str, BBOX)), "outputFormat": "JSON", "max": 1000,
    }, timeout=120)
    response.raise_for_status()
    return [item.get("downloadURL") for item in response.json().get("items", []) if item.get("downloadURL")]


def _materialize_raster(downloaded: Path, product: str) -> Path:
    if downloaded.suffix.lower() != ".zip": return downloaded
    with zipfile.ZipFile(downloaded) as archive:
        candidates = [name for name in archive.namelist() if Path(name).suffix.lower() in (".tif", ".tiff")]
        if len(candidates) != 1: raise ValueError(f"{product} archive must contain exactly one GeoTIFF; found {len(candidates)}")
        target = paths.STATIC_SOURCE_DIR / f"{product}.tif"
        with archive.open(candidates[0]) as source, open(target, "wb") as output: shutil.copyfileobj(source, output)
    return target


def _raster_metadata(path: Path):
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path} has no CRS")
        if src.count != 1:
            raise ValueError(f"{path} must contain exactly one raster band")
        if src.nodata is None:
            raise ValueError(f"{path} must declare an explicit nodata value")
        return {
            "crs": src.crs.to_string(),
            "bounds": list(src.bounds),
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }


def main():
    parser = argparse.ArgumentParser(description="Acquire official static rasters or register local overrides")
    for product in PRODUCTS:
        parser.add_argument(f"--{product.replace('_', '-')}-url")
        parser.add_argument(f"--{product.replace('_', '-')}-file")
        parser.add_argument(
            f"--{product.replace('_', '-')}-units",
            default=DEFAULT_UNITS[product],
            help=f"Confirmed native units (default: {DEFAULT_UNITS[product]})",
        )
    parser.add_argument("--discover-3dep", action="store_true")
    parser.add_argument("--release", default="explicit-v1", help="Human-readable source release label")
    args = parser.parse_args()
    output = paths.STATIC_SOURCE_DIR / "source_manifest.json"
    try:
        existing_manifest = json.loads(output.read_text())
        records = dict(existing_manifest.get("products") or {})
    except (FileNotFoundError, json.JSONDecodeError):
        records = {}
    updated_products = []
    for product in PRODUCTS:
        local = getattr(args, f"{product}_file")
        url = getattr(args, f"{product}_url")
        if local:
            source = Path(local).resolve(); target = paths.STATIC_SOURCE_DIR / f"{product}{source.suffix}"
            shutil.copy2(source, target)
        elif url:
            suffix = Path(url.split("?")[0]).suffix or ".tif"
            target = paths.STATIC_SOURCE_DIR / f"{product}{suffix}"
            fetch(url, target)
        else:
            continue
        target = _materialize_raster(target, product)
        records[product] = {
            "path": str(target),
            "url": url,
            "sha256": _sha(target),
            "size": target.stat().st_size,
            "units": getattr(args, f"{product}_units"),
            "raster": _raster_metadata(target),
            "source_release": args.release,
        }
        updated_products.append(product)
    if args.discover_3dep and "dem" not in records:
        records["dem_tiles"] = {"urls": discover_3dep_urls(), "note": "Download/mosaic these tiles or pass --dem-file"}
    manifest = {"release": args.release, "bbox": BBOX, "acquired_at": datetime.now(timezone.utc).isoformat(),
                "updated_products": updated_products, "products": records,
                "official_sources": {"dem": "USGS 3DEP/TNM", "nlcd": "USGS Annual NLCD/MRLC", "landfire": "LANDFIRE Product Service"}}
    output.write_text(json.dumps(manifest, indent=2)); print(output)


if __name__ == "__main__": main()
