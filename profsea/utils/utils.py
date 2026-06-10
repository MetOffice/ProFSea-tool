import dask.array as da
import numpy as np
from scipy.spatial.distance import cdist
import xarray as xr


def sample_members_2D(array: np.ndarray, percentiles: list | np.ndarray) -> np.ndarray:
    """Sample real ensemble members from a 2D numpy array."""
    # Caculate statistical timeseries, then match with closest real timeseries
    array_percentiles = np.nanpercentile(array, percentiles, axis=0)
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

    # Pad longitude with one points from each end to handle periodicity 
    data_padded = data.pad(lon=1, mode='wrap') # need more padding for higher-order interpolation

    # Fix the longitude coordinate after padding
    lon = data.lon.values
    lon_padded = np.concatenate([[lon[-1] - 360], lon, [lon[0] + 360]])
    data_padded['lon'] = lon_padded

    # Now interpolate (maybe add land_mask if condition here)
    for dim in ["lat", "lon"]:
        data_padded = data_padded.interpolate_na(
            dim=dim, method=grid_interpolation,
            fill_value="extrapolate"
        ) # this to handle nan values or land mask 
        
    data_interp = data_padded.interp(
        lat=target_lats, lon=target_lons_norm, method=grid_interpolation
    )

    data_interp = data_interp.where

    

    # Acount for land mask (1 where NaN, 0 elsewhere)
    #if mask_present:
    #    nan_mask = data.isnull().astype(float).interp(
    #        lat=target_lats, lon=target_lons_norm, method=grid_interpolation
    #    )

    #   # Mask out any grid point that had NaN influence
    #   data_interp = data_interp.where(nan_mask == 0)
    
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
        # Split over lines for readability
        raise ValueError(
            f"Array should have shape (realisation, time) with time \
                dimension of length {n_time}. Got {array.shape}."
        )
