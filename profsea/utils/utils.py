from __future__ import annotations

import dask.array as da
import numpy as np
from scipy.spatial.distance import cdist
import xarray as xr


def sample_members_2D(array: np.ndarray, percentiles: list | np.ndarray) -> np.ndarray:
    """Sample real ensemble members from a 2D numpy array."""
    # Caculate statistical timeseries, then match with closest real timeseries
    array_percentiles = np.percentile(array, percentiles, axis=0)
    distances = cdist(array_percentiles, array)
    mem_indices = np.argmin(distances, axis=1)
    return array[mem_indices]


def interpolate(data: da.array, lats: int, lons: int) -> np.ndarray:
    """ """
    original_da = xr.DataArray(
        data.data,
        coords=[("lat", data[data.dims[0]].values), ("lon", data[data.dims[1]].values)],
        name="v",
    )

    target_lat = np.linspace(-90, 90, lats, endpoint=False) + 0.5
    target_lon = np.linspace(-180, 180, lons, endpoint=False) + 0.5
    data_interp = original_da.interp(
        lat=target_lat, lon=target_lon, method="linear"
    ).data
    return data_interp


def interpolate_to_grid(
    data: xr.DataArray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    grid_interpolation: str = "linear",
) -> xr.DataArray:
    """
    Interpolate an xarray DataArray to a target grid defined by target_lats and target_lons.
    Safely handles longitude wrapping mismatches (e.g., [0, 360) vs [-180, 180)).
    """
    # Normalize source longitudes to [-180, 180) and sort monotonically
    data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
    data = data.sortby("lon")

    # Normalize target longitudes to [-180, 180) and sort
    target_lons_norm = np.sort(((target_lons + 180) % 360) - 180)

    # Interpolate!
    data_interp = data.interp(
        lat=target_lats, lon=target_lons_norm, method=grid_interpolation
    )
    return data_interp


def check_shapes(array: np.ndarray, n_time: int) -> None:
    """Check that the input arrays have the correct shape.

    Parameters
    ----------
    array: np.ndarray
        Input array of some kind.
    n_time: int
        Expected number of time steps.

    Returns
    -------
    None
    """
    if array.ndim == 1:
        array = array[np.newaxis, :]

    if array.shape[1] != n_time:
        raise ValueError(
            f"Array should have shape (realisation, time) with time "
            f"dimension of length {n_time}. Got {array.shape}."
        )
