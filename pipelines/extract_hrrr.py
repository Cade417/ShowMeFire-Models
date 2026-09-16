import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_unprocessed_snapshots, save_hrrr_features, get_all_stations, mark_snapshot_processed, set_hrrr_filename
import paths
from spatial.precipitation import decode_forecast_precipitation

logger = logging.getLogger(__name__)

HRRR_DIR = paths.CACHE_HRRR_DIR


def open_hrrr_dataset(nc_path):
    """Open one cached HRRR NetCDF file, tolerating a decode fragility seen
    in a handful of cached files (e.g. hrrr_20260809_12z_f04-15.nc): their
    'step' variable's attrs already contain a stray 'dtype' key (apparently
    left over from whatever wrote that particular file), which xarray
    refuses to silently overwrite when decode_timedelta=True tries to
    CF-decode it - see the ValueError's own message ("remove this key from
    the variable's attributes manually").

    Fixed by stripping that one stray attr and re-running normal CF
    decoding, rather than falling back to decode_timedelta=False for these
    files - the raw (undecoded) 'step' values here are in nanoseconds per
    their own 'units' attr, not hours, so a naive "not timedelta64 => must
    already be hours" fallback (the assumption extract_at_indices's dtype
    branch and spatial/precipitation.py's _lead_hours both make) would
    silently produce lead hours wrong by a factor of ~3.6e12 for exactly
    these files. Stripping the bad attr and decoding properly avoids ever
    needing that fallback to be unit-aware.
    """
    try:
        return xr.open_dataset(nc_path, engine='netcdf4', decode_timedelta=True)
    except ValueError as exc:
        if "'step'" not in str(exc) or "dtype" not in str(exc):
            raise
        logger.warning(f"{nc_path.name}: stray 'dtype' attr on 'step' blocked normal decoding; "
                        "stripping it and retrying")
        raw = xr.open_dataset(nc_path, engine='netcdf4', decode_cf=False, mask_and_scale=False)
        raw['step'].attrs.pop('dtype', None)
        decoded = xr.decode_cf(raw, decode_timedelta=True)
        decoded.set_close(raw.close)  # keep the underlying file handle alive as long as `decoded` is used
        return decoded


def get_nearest_indices(ds, lat, lon):
    """Finds the x, y indices for a given lat/lon with 0-360 normalization."""
    # HRRR longitude is usually 0-360. 
    # If the file shows values > 180, convert our negative lon to positive.
    ds_lon_max = float(ds.longitude.max())
    search_lon = lon + 360 if (ds_lon_max > 180 and lon < 0) else lon
    
    # Standard distance math
    dist = (ds.latitude - lat)**2 + (ds.longitude - search_lon)**2
    obj = dist.argmin(dim=['y', 'x'])
    return int(obj['x']), int(obj['y'])

def extract_at_indices(ds, x, y):
    leads = np.asarray(ds.step.values)
    lead_hours = leads / np.timedelta64(1, 'h') if np.issubdtype(leads.dtype, np.timedelta64) else leads.astype(float)
    eligible = np.flatnonzero(lead_hours >= 4)
    lead_index = int(eligible[0]) if len(eligible) else 0
    point = ds.isel(x=x, y=y, step=lead_index)
    
    u = float(point['u10'].values)
    v = float(point['v10'].values)
    
    precipitation = decode_forecast_precipitation(ds)
    selector = dict(x=x, y=y, step=lead_index)
    precip_mm = float(precipitation.cumulative_mm.isel(**selector).values)
    precip_interval_mm = float(precipitation.interval_mm.isel(**selector).values)
    precip_interval_hours = float(precipitation.interval_hours.isel(**selector).values)
    
    return {
        "temp_c": float(point['t2m'].values) - 273.15,
        "rel_humidity": float(point['r2'].values),
        "wind_speed_ms": (u**2 + v**2)**0.5,
        "precip_mm": precip_mm,
        "precip_interval_mm": precip_interval_mm,
        "precip_interval_hours": precip_interval_hours,
    }

def run_miner(since_date=None):
    """since_date ('YYYY-MM-DD', optional): only mine snapshots on/after this
    date. Without it, mines EVERY unprocessed snapshot regardless of age -
    see get_unprocessed_snapshots's docstring for why that can mean chewing
    through years of historical backlog on what was meant to be a routine
    incremental run."""
    stations = get_all_stations()
    to_process = get_unprocessed_snapshots(since_date=since_date)

    logger.info(f"Found {len(to_process)} unprocessed snapshots.")
    if not to_process:
        logger.info("No new snapshots to process.")
        return

    station_indices = None

    for row in to_process:
        logger.info(f"ROW: {dict(row)}")
        clean_date = row['snapshot_date'].replace('-', '')
        logger.info(f"Looking for files with: {clean_date}")
        matching_files = list(HRRR_DIR.glob(f"*{clean_date}*.nc"))
        logger.info(f"Found files: {[str(f) for f in matching_files]}")
        if not matching_files:
            logger.warning(f"No HRRR file found for {row['snapshot_date']} (id={row['id']})")
            continue

        nc_path = matching_files[0]
        logger.info(f"Using HRRR file: {nc_path}")

        if not nc_path.exists():
            logger.warning(f"HRRR file does not exist on disk: {nc_path}")
            continue

        try:
            with open_hrrr_dataset(nc_path) as ds:
                if station_indices is None:
                    logger.info("Calibrating station grid indices...")
                    station_indices = {
                        s['id']: get_nearest_indices(ds, s['lat'], s['lon']) 
                        for s in stations
                    }
                    logger.info(f"Station indices: {station_indices}")

                for station in stations:
                    x, y = station_indices[station['id']]
                    logger.info(f"Extracting for station {station['id']} at indices x={x}, y={y}")

                    extracted_dict = extract_at_indices(ds, x, y)
                    logger.info(f"Extracted data for station {station['id']}: {extracted_dict}")

                    if extracted_dict:
                        save_hrrr_features(row['id'], extracted_dict, station['id'])
                        logger.info(f"Saved features for snapshot {row['id']} and station {station['id']}")

            set_hrrr_filename(row['id'], nc_path.name)  # <-- Add this line
            mark_snapshot_processed(row['id'])
            logger.info(f"✅ Successfully finished all stations for: {row['snapshot_date']} (snapshot id: {row['id']})")

        except Exception as e:
            logger.error(f"Error processing {row['hrrr_filename']}: {e}", exc_info=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract HRRR weather features for unprocessed snapshots")
    parser.add_argument("--since-date", help="Only mine snapshots on/after this date (YYYY-MM-DD). "
                         "Default: no filter, mines every unprocessed snapshot regardless of age.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_miner(since_date=args.since_date)
