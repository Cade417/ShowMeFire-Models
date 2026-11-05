"""Download a cropped HRRR NetCDF suitable as a static-bundle grid reference.

This downloads only one HRRR surface field.  The fire-behavior bundle needs
the HRRR latitude/longitude grid and CF projection metadata, not a forecast
sequence, so downloading all weather variables would waste storage.

Example (PowerShell):

    python static_features/download_reference_hrrr.py `
      --date 2026-08-31 `
      --cycle 12
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from herbie import Herbie

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths


BBOX = (-96.8, 34.8, -88.1, 41.8)  # west, south, east, north


def _as_dataset(value: xr.Dataset | list[xr.Dataset]) -> xr.Dataset:
    if isinstance(value, list):
        if not value:
            raise RuntimeError("Herbie returned no HRRR datasets")
        value = xr.merge(value, compat="override")
    return value


def crop_to_domain(ds: xr.Dataset, bbox: tuple[float, float, float, float] = BBOX) -> xr.Dataset:
    if "latitude" not in ds or "longitude" not in ds:
        raise ValueError("HRRR dataset has no latitude/longitude coordinates")
    west, south, east, north = bbox
    longitude = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
    mask = (longitude >= west) & (longitude <= east) & (ds.latitude >= south) & (ds.latitude <= north)
    if mask.ndim != 2:
        raise ValueError("HRRR latitude/longitude coordinates must be two-dimensional")
    rows, cols = np.where(mask.values)
    if not len(rows):
        raise ValueError("HRRR grid does not intersect the configured domain")
    ydim, xdim = mask.dims
    return ds.isel({ydim: slice(rows.min(), rows.max() + 1), xdim: slice(cols.min(), cols.max() + 1)})


def sanitize(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy(deep=False)
    for name in ds.variables:
        ds[name].attrs.pop("dtype", None)
        ds[name].attrs.pop("source", None)
        ds[name].encoding.pop("dtype", None)
    return ds


def download(
    date: str,
    cycle: int,
    fxx: int,
    output: Path | None = None,
) -> Path:
    # Herbie currently compares this value with a timezone-naive pandas UTC
    # timestamp. Keep the input naive while treating it as UTC by convention.
    run = datetime.strptime(f"{date} {cycle:02d}", "%Y-%m-%d %H")
    output = output or paths.CACHE_HRRR_DIR / f"hrrr_{run:%Y%m%d_%H}z_reference.nc"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"Reusing existing HRRR reference: {output}")
        return output

    print(f"Downloading HRRR {run:%Y-%m-%d %H:00 UTC}, forecast hour {fxx}")
    hrrr = Herbie(run, fxx=fxx, model="hrrr", product="sfc")
    dataset = _as_dataset(hrrr.xarray(":TMP:2 m"))
    dataset = sanitize(crop_to_domain(dataset))
    dataset.load()
    temporary = output.with_suffix(output.suffix + ".partial")
    dataset.to_netcdf(temporary, engine="netcdf4")
    temporary.replace(output)
    print(output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="HRRR initialization date, YYYY-MM-DD")
    parser.add_argument("--cycle", type=int, default=12, choices=range(0, 24), help="UTC cycle hour")
    parser.add_argument("--fxx", type=int, default=0, help="Forecast hour to use for the grid reference")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.fxx < 0:
        parser.error("--fxx must be non-negative")
    download(args.date, args.cycle, args.fxx, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
