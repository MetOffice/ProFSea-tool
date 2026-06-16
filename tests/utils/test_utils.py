from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from profsea.utils.utils import check_shapes, interpolate_to_grid, sample_members_2D


def test_check_shapes_valid_2d():
    arr = np.zeros((5, 10))
    check_shapes(arr, n_time=10)


def test_check_shapes_promotes_1d():
    arr = np.zeros(10)
    check_shapes(arr, n_time=10)


def test_check_shapes_raises_value_error():
    arr = np.zeros((5, 10))
    with pytest.raises(
        ValueError, match="Array should have shape .* time dimension of length 12"
    ):
        check_shapes(arr, n_time=12)


def test_interpolate_to_grid_longitude_wrapping():
    # Target grid using -180 to 180 format (includes 180, which should wrap)
    target_lats = np.array([-45, 0, 45])
    target_lons = np.array([-10, 0, 180])

    # 1. Setup mock regionmask to act as if there is no land
    mock_regionmask = MagicMock()
    mock_land = MagicMock()
    mock_mask_da = xr.DataArray(
        np.zeros((1, 3, 3), dtype=bool),
        dims=["region", "lat", "lon"],
        coords={"region": [0], "lat": target_lats, "lon": [-180, -10, 0]},
    )
    mock_land.mask_3D.return_value = mock_mask_da
    mock_regionmask.defined_regions.natural_earth_v5_0_0.land_110 = mock_land

    # 2. Setup mock input data
    lats = np.array([-45, 0, 45])
    lons_360 = np.array([0, 180, 350])
    data = xr.DataArray(np.random.rand(3, 3), coords=[("lat", lats), ("lon", lons_360)])

    # 3. Intercept the local import using sys.modules
    with patch.dict("sys.modules", {"regionmask": mock_regionmask}):
        result = interpolate_to_grid(data, target_lats, target_lons)

    assert result.shape == (3, 3)
    expected_lons = np.array([-180, -10, 0])
    np.testing.assert_array_almost_equal(result.lon.values, expected_lons)


def test_sample_members_2D_extracts_real_member():
    arr = np.array(
        [
            [1.0, 1.0, 1.0],  # Min
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],  # Median
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],  # Max
        ]
    )

    result = sample_members_2D(arr, percentiles=[50])
    assert result.shape == (1, 3)
    np.testing.assert_array_equal(result[0], [3.0, 3.0, 3.0])
