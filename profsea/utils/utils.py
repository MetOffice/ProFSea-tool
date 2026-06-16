from __future__ import annotations

import dask.array as da
import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist


def sample_members_2D(array: np.ndarray, percentiles: list | np.ndarray) -> np.ndarray:
    """
    Sample real ensemble members from a 2D numpy array.

    Parameters
    ----------
    array: np.ndarray
        Input 2D array of shape (realisation, time).
    percentiles: list | np.ndarray
        List of percentiles to sample from the input array.

    Returns
    -------
    np.ndarray
        Sampled array of shape (len(percentiles), time) corresponding to the closest real ensemble members to the specified percentiles.
    """
    # Caculate statistical timeseries, then match with closest real timeseries
    array_percentiles = np.nanpercentile(array, percentiles, axis=0)
    distances = cdist(array_percentiles, array)
    mem_indices = np.argmin(distances, axis=1)
    return array[mem_indices]


def interpolate(data: da.array, lats: int, lons: int) -> da.array:
    """
    Interpolate a 2D dask array to a target grid defined by lats and lons.

    Parameters
    ----------
    data: da.array
        Input 2D dask array to be interpolated.
    lats: int
        Number of latitude points in the target grid.
    lons: int
        Number of longitude points in the target grid.

    Returns
    -------
    da.array
        Interpolated 2D dask array on the target grid.
    """
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

    Parameters
    ----------
    data: xr.DataArray
        Input xarray DataArray to be interpolated. Must have 'lat' and 'lon' dimensions.
    target_lats: np.ndarray
        1D array of target latitude values.
    target_lons: np.ndarray
        1D array of target longitude values.
    grid_interpolation: str, optional
        Interpolation method to use. Default is 'linear'. Other options include 'nearest', 'cubic', etc.

    Returns
    -------
    xr.DataArray
        Interpolated xarray DataArray on the target grid.
    """
    import regionmask

    # Normalize source longitudes to [-180, 180) and sort monotonically
    data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
    data = data.sortby(["lat", "lon"])

    # Normalize target longitudes to [-180, 180) and sort
    target_lons_norm = np.sort(((target_lons + 180) % 360) - 180)

    # Pad longitude with one points from each end to handle periodicity in zonal direction
    data_padded = data.pad(
        lon=1, mode="wrap"
    )  # need more padding for higher-order interpolation
    lon = data.lon.values
    lon_padded = np.concatenate([[lon[-1] - 360], lon, [lon[0] + 360]])
    data_padded["lon"] = lon_padded
    data_padded = data_padded.sortby(["lat", "lon"])

    # Now interpolate
    data_padded = data_padded.chunk({"lat": -1, "lon": -1})
    for dim in ["lat", "lon"]:
        data_padded = data_padded.interpolate_na(
            dim=dim,
            method="nearest",
        )  # this to handle nan values or land mask

    data_interp = data_padded.interp(
        lat=target_lats, lon=target_lons_norm, method=grid_interpolation
    )

    # Account for land mask (using regionmask library)
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask = land.mask_3D(data_interp)
    is_land = land_mask.squeeze("region", drop=True)
    data_interp = data_interp.where(~is_land)

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
