from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from spatial.model import SpatialQuantileModel
from spatial.static_inputs import load_static_inputs


def predict(checkpoint_path: Path, tensor_path: Path, static_bundle: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False); contract = checkpoint["static_contract"]
    scont, scat, loaded_contract = load_static_inputs(static_bundle, contract["feature_set"])
    if loaded_contract["bundle_sha256"] != contract["bundle_sha256"]: raise ValueError("static bundle mismatch")
    with np.load(tensor_path) as item:
        dynamic = item["dynamic_sequence"].astype("float32"); physics = item["physics"].astype("float32"); metadata = json.loads(str(item["metadata"]))
    model = SpatialQuantileModel(checkpoint["dynamic_channels"], checkpoint["static_continuous_channels"], checkpoint["category_sizes"], embedding_dims=checkpoint["embedding_dims"])
    model.load_state_dict(checkpoint["state_dict"]); model.eval(); normalized = (dynamic - checkpoint["dynamic_mean"]) / checkpoint["dynamic_std"]
    with torch.no_grad(): output = model(torch.from_numpy(normalized[None]), torch.from_numpy(scont[None]), torch.from_numpy(scat[None]), torch.from_numpy(physics[None])).numpy()[0]
    with xr.open_dataset(static_bundle) as static_ds: latitude, longitude = static_ds.latitude.values, static_ds.longitude.values
    channels = metadata["dynamic_channels"]; distance = dynamic[0, channels.index("nearest_station_distance_deg")]; effective = dynamic[0, channels.index("effective_station_count")]
    width = np.maximum(0, output[:, 2] - output[:, 0]); confidence = np.exp(-distance[None] / 2) / (1 + width / 10)
    ds = xr.Dataset({"fm_p10": (("lead", "y", "x"), output[:, 0]), "fm_p50": (("lead", "y", "x"), output[:, 1]), "fm_p90": (("lead", "y", "x"), output[:, 2]),
                     "confidence": (("lead", "y", "x"), confidence), "nearest_station_distance_deg": (("y", "x"), distance), "effective_station_count": (("y", "x"), effective)},
                    coords={"lead": np.arange(len(output)), "latitude": (("y", "x"), latitude), "longitude": (("y", "x"), longitude)},
                    attrs={"forecast_init_time": metadata["forecast_init_time"], "static_bundle_sha256": contract["bundle_sha256"], "warning": "Unobserved cells are model estimates."})
    output_path = paths.ALIGNED_DIR / f"fuel_moisture_grid_{metadata['run_id']}.nc"; ds.to_netcdf(output_path, engine="netcdf4"); print(output_path); return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--tensor", required=True, type=Path); parser.add_argument("--static-bundle", required=True, type=Path)
    args = parser.parse_args(); predict(args.checkpoint, args.tensor, args.static_bundle)
