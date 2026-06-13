import numpy as np
from unittest.mock import patch
import xarray as xr

from profsea.components.core.state import SpatialState
from profsea.components.spatial.sterodynamic import SterodynamicCMIP6


@patch("profsea.components.spatial.sterodynamic.xr.open_dataset")
@patch("profsea.components.spatial.sterodynamic.Path.glob")
def test_load_cmip6_slopes(mock_glob, mock_open_dataset):
    # 1. Setup mock file paths
    mock_glob.return_value = ["dummy_path_1.nc", "dummy_path_2.nc"]

    # 2. Setup a mock xarray dataset that has the required variable
    mock_data = xr.DataArray(
        np.random.rand(5, 5),  # 5 lats, 5 lons
        dims=["lat", "lon"],
    )
    mock_dataset = {"zos_zostoga_regression_slope": mock_data}
    mock_open_dataset.return_value = mock_dataset

    # 3. Initialize component and call the method
    # global_projection shape: (members, years)
    dummy_global = np.zeros((10, 100))
    stero = SterodynamicCMIP6(global_projection=dummy_global)

    slopes = stero._load_CMIP6_slopes()

    # 4. Assertions
    # We expect 2 models (because we mocked 2 paths), and 5x5 spatial dimensions
    assert slopes.shape == (2, 5, 5)
    assert "model" in slopes.dims


@patch.object(SterodynamicCMIP6, "_load_CMIP6_slopes")
def test_expansion_contribution_storyline_mode(mock_load):
    # 1. Mock the slope data (3 models, 2 lats, 2 lons)
    # Model 1 is all 1s, Model 2 is all 2s, Model 3 is all 3s
    mock_coeffs = xr.DataArray(
        np.array([np.ones((2, 2)), np.ones((2, 2)) * 2, np.ones((2, 2)) * 3]),
        coords=[("model", [0, 1, 2]), ("lat", [0, 1]), ("lon", [0, 1])],
    )
    mock_load.return_value = mock_coeffs

    # 2. Create the state object
    state = SpatialState(
        grid_lats=np.array([0, 1]),
        grid_lons=np.array([0, 1]),
        n_years=100,
        n_members=5,
        grid_interpolation="nearest",
        output_percentiles=None,
        baseline_yrs=(1995, 2014),
    )

    # 3. Test Storyline Mode (sample_spatial = False)
    stero_storyline = SterodynamicCMIP6(
        global_projection=np.zeros((5, 100)), sample_spatial=False
    )

    rng = np.random.default_rng(42)
    result_storyline = stero_storyline._calc_expansion_contribution(rng, state)

    # In storyline mode, it calculates the ensemble mean of the models: (1+2+3)/3 = 2
    # Therefore, every member should receive a pattern of 2s.
    assert result_storyline.shape == (5, 2, 2)

    # Convert Dask array to NumPy for assertion
    computed_result = result_storyline.compute()
    np.testing.assert_array_equal(computed_result, np.ones((5, 2, 2)) * 2)


@patch.object(SterodynamicCMIP6, "_load_CMIP6_slopes")
def test_expansion_contribution_sampled_mode(mock_load):
    # Use the same setup as above
    mock_coeffs = xr.DataArray(
        np.array([np.ones((2, 2)), np.ones((2, 2)) * 2, np.ones((2, 2)) * 3]),
        coords=[("model", [0, 1, 2]), ("lat", [0, 1]), ("lon", [0, 1])],
    )
    mock_load.return_value = mock_coeffs

    state = SpatialState(
        grid_lats=np.array([0, 1]),
        grid_lons=np.array([0, 1]),
        n_years=100,
        n_members=5,
        grid_interpolation="nearest",
        output_percentiles=None,
        baseline_yrs=(1995, 2014),
    )

    # Test Sampled Mode (sample_spatial = True)
    stero_sampled = SterodynamicCMIP6(
        global_projection=np.zeros((5, 100)), sample_spatial=True
    )

    rng = np.random.default_rng(42)  # Seeded for determinism

    result_sampled = np.asarray(stero_sampled._calc_expansion_contribution(rng, state))

    # In sampled mode, the array should NOT be uniform across the 5 members.
    assert result_sampled.shape == (5, 2, 2)

    # Check that member 0 and member 1 are not guaranteed to be identical
    # (Though with a seed of 42 and 3 choices, they might be, so we check the unique values)
    unique_values = np.unique(result_sampled)
    assert len(unique_values) > 1  # Proves it didn't just take the mean
