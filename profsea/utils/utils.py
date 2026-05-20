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
    data: da.array, target_lats: np.ndarray, target_lons: np.ndarray
) -> da.array:
    """
    Interpolate a dask array to a target grid defined by target_lats and target_lons.
    Assumes the input data has dimensions (lat, lon) and coordinates named 'lat' and 'lon'.

    Parameters
    ----------
    data: da.array
        Input dask array with dimensions (lat, lon) and coordinates 'lat' and 'lon'.
    target_lats: np.ndarray
        1D array of target latitudes to interpolate to.
    target_lons: np.ndarray
        1D array of target longitudes to interpolate to.
    Returns
    -------
    da.array        Interpolated dask array on the target grid.
    """
    original_da = xr.DataArray(
        data.data,
        coords=[("lat", data[data.dims[0]].values), ("lon", data[data.dims[1]].values)],
        name="v",
    )

    data_interp = original_da.interp(
        lat=target_lats, lon=target_lons, method="linear"
    ).data

    return da.from_array(data_interp, chunks=data.chunks)


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
