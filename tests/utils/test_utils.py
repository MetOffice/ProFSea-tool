import numpy as np
import pytest
import xarray as xr

from profsea.utils import check_shapes, interpolate_to_grid, sample_members_2D


def test_check_shapes_valid_2d():
    # 5 realisations, 10 time steps
    arr = np.zeros((5, 10))
    # Should not raise an exception
    check_shapes(arr, n_time=10)


def test_check_shapes_promotes_1d():
    # 1 realisation, 10 time steps
    arr = np.zeros(10)
    # The function should silently promote this to (1, 10)
    check_shapes(arr, n_time=10)


def test_check_shapes_raises_value_error():
    arr = np.zeros((5, 10))
    # Passing the wrong time dimension length
    with pytest.raises(
        ValueError, match="Array should have shape .* time dimension of length 12"
    ):
        check_shapes(arr, n_time=12)


def test_interpolate_to_grid_longitude_wrapping():
    # Create mock data with 0-360 longitude format
    lats = np.array([-45, 0, 45])
    lons_360 = np.array([0, 180, 350])

    data = xr.DataArray(np.random.rand(3, 3), coords=[("lat", lats), ("lon", lons_360)])

    # Target grid using -180 to 180 format (includes 180, which should wrap)
    target_lats = np.array([-45, 0, 45])
    target_lons = np.array([-10, 0, 180])

    result = interpolate_to_grid(data, target_lats, target_lons)

    # Check that output dimensions match target dimensions
    assert result.shape == (3, 3)

    # Check that coordinates were successfully wrapped to [-180, 180) and sorted
    expected_lons = np.array([-180, -10, 0])
    np.testing.assert_array_almost_equal(result.lon.values, expected_lons)


def test_sample_members_2D_extracts_real_member():
    # 5 ensemble members over 3 time steps
    arr = np.array(
        [
            [1.0, 1.0, 1.0],  # Min
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],  # Median
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],  # Max
        ]
    )

    # Request the 50th percentile (median)
    result = sample_members_2D(arr, percentiles=[50])

    # The closest real member to the median is [3., 3., 3.]
    assert result.shape == (1, 3)
    np.testing.assert_array_equal(result[0], [3.0, 3.0, 3.0])
