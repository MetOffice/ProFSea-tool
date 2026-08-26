from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import xarray as xr

from profsea.components.core.spatial_model import Spatial


def get_dummy_components(nt=2, num_members=3, n_years=4):
    """Helper to create a dummy component dictionary with the expected shape."""
    mock_comp = MagicMock()
    # 3D Shape: (climate_member, process_member, time)
    mock_comp.global_projection.shape = (nt, num_members, n_years)
    return {"mock_comp": mock_comp}


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_spatial_init_blocks_download_and_extracts_shapes(mock_fetch):
    components = get_dummy_components(nt=5, num_members=10)
    spatial = Spatial(components=components, end_year=2050)

    mock_fetch.assert_called_once()
    assert spatial.end_year == 2050
    assert spatial.nt == 5
    assert spatial.num_members == 10


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_arr_to_xr_metadata_with_percentiles(mock_fetch):
    # Initializes with default output_percentiles=[5, 17, 50, 83, 95] (Length 5)
    components = get_dummy_components(n_years=4)
    spatial = Spatial(components=components, end_year=2010)

    # 4D Array: (percentiles, time, lat, lon)
    arr = da.zeros((5, 4, 180, 360))
    arr_dict = {"mock_spatial": arr}

    xr_dict = spatial._arr_to_xr(arr_dict)

    assert "mock_spatial" in xr_dict
    da_out = xr_dict["mock_spatial"]

    assert isinstance(da_out, xr.DataArray)
    assert list(da_out.dims) == ["percentile", "time", "lat", "lon"]
    np.testing.assert_array_equal(da_out.time.values, [2006, 2007, 2008, 2009])


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_arr_to_xr_metadata_no_percentiles(mock_fetch):
    # Initializes without percentiles, triggering the 5D ensemble output
    components = get_dummy_components(nt=2, num_members=3, n_years=4)
    spatial = Spatial(components=components, end_year=2010, output_percentiles=None)

    # 5D Array: (climate_member, process_member, time, lat, lon)
    arr = da.zeros((2, 3, 4, 180, 360))
    arr_dict = {"mock_spatial": arr}

    xr_dict = spatial._arr_to_xr(arr_dict)
    da_out = xr_dict["mock_spatial"]

    assert isinstance(da_out, xr.DataArray)
    assert list(da_out.dims) == [
        "climate_member",
        "process_member",
        "time",
        "lat",
        "lon",
    ]
    np.testing.assert_array_equal(da_out.climate_member.values, [0, 1])
    np.testing.assert_array_equal(da_out.process_member.values, [0, 1, 2])
    np.testing.assert_array_equal(da_out.time.values, [2006, 2007, 2008, 2009])


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_sum_spatial_components(mock_fetch):
    components = get_dummy_components()
    spatial = Spatial(components=components, end_year=2010)

    # Two identical arrays of 1s
    da1 = xr.DataArray(np.ones((2, 2)), dims=["lat", "lon"])
    da2 = xr.DataArray(np.ones((2, 2)), dims=["lat", "lon"])

    components_dict = {"comp1": da1, "comp2": da2}
    total = spatial.sum_components(components_dict)

    # 1 + 1 = 2 everywhere
    np.testing.assert_array_equal(total.values, np.ones((2, 2)) * 2)
    assert "total_rsl" in components_dict
