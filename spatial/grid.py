from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def nearest_mapping(source_lat, source_lon, target_lat, target_lon):
    """Return reusable flat indices mapping a fixed curvilinear grid."""
    source = np.column_stack((np.asarray(source_lat).ravel(), _lon180(source_lon).ravel()))
    target = np.column_stack((np.asarray(target_lat).ravel(), _lon180(target_lon).ravel()))
    distance, index = cKDTree(source).query(target)
    return index.reshape(np.asarray(target_lat).shape), distance.reshape(np.asarray(target_lat).shape)


def apply_mapping(source_values, mapping):
    return np.asarray(source_values).ravel()[mapping]


def _lon180(values):
    values = np.asarray(values)
    return np.where(values > 180, values - 360, values)


def idw_initial_analysis(station_lat, station_lon, station_fm, grid_lat, grid_lon, power=2.0):
    """Transparent FM analysis plus distance and effective-station coverage."""
    stations = np.column_stack((np.asarray(station_lat), _lon180(station_lon)))
    targets = np.column_stack((np.asarray(grid_lat).ravel(), _lon180(grid_lon).ravel()))
    tree = cKDTree(stations)
    k = min(8, len(stations))
    distance, index = tree.query(targets, k=k)
    if k == 1:
        distance, index = distance[:, None], index[:, None]
    weights = 1.0 / np.maximum(distance, 1e-4) ** power
    weights /= weights.sum(axis=1, keepdims=True)
    analysis = (weights * np.asarray(station_fm)[index]).sum(axis=1)
    effective = 1.0 / np.square(weights).sum(axis=1)
    shape = np.asarray(grid_lat).shape
    return analysis.reshape(shape), distance[:, 0].reshape(shape), effective.reshape(shape)
