from unittest.mock import patch

import dask.array as da
import numpy as np
import xarray as xr

from profsea.components.core.spatial_model import Spatial


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_spatial_init_blocks_download(mock_fetch):
    spatial = Spatial(components={}, end_year=2050)
    mock_fetch.assert_called_once()
    assert spatial.end_year == 2050


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_arr_to_xr_metadata(mock_fetch):
    # Initializes with default output_percentiles=[5, 17, 50, 83, 95] (Length 5)
    spatial = Spatial(components={}, end_year=2010)

    arr = da.zeros((5, 4, 180, 360))
    arr_dict = {"mock_spatial": arr}

    xr_dict = spatial._arr_to_xr(arr_dict)

    assert "mock_spatial" in xr_dict
    da_out = xr_dict["mock_spatial"]

    assert isinstance(da_out, xr.DataArray)
    assert list(da_out.dims) == ["percentile", "time", "lat", "lon"]
    np.testing.assert_array_equal(da_out.time.values, [2006, 2007, 2008, 2009])


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_sum_spatial_components(mock_fetch):
    spatial = Spatial(components={}, end_year=2010)

    # Two identical arrays of 1s
    da1 = xr.DataArray(np.ones((2, 2)), dims=["lat", "lon"])
    da2 = xr.DataArray(np.ones((2, 2)), dims=["lat", "lon"])

    components = {"comp1": da1, "comp2": da2}
    total = spatial.sum_components(components)

    # 1 + 1 = 2 everywhere
    np.testing.assert_array_equal(total.values, np.ones((2, 2)) * 2)
    assert "total_rsl" in components
