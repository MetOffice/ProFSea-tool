from pathlib import Path
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from profsea.components.core.base import SpatialComponent
from profsea.components.core.state import LocalState, SpatialState
from profsea.components.spatial.sterodynamic import (
    SterodynamicCMIP5,
    SterodynamicCMIP6,
)


@pytest.fixture(autouse=True)
def block_assets():
    with patch.object(SpatialComponent, "fetch_assets"):
        yield


class TestSterodynamicInit:
    @patch.object(SpatialComponent, "fetch_assets")
    def test_fetches_assets_on_initialisation(
        self,
        mock_fetch_assets,
    ):
        """Should fetch assets during Sterodynamic initialisation."""
        dummy_global = xr.DataArray(np.zeros((3, 4, 100)))

        SterodynamicCMIP6(global_projection=dummy_global)

        mock_fetch_assets.assert_called_once()


@patch("profsea.components.spatial.sterodynamic.xr.open_dataset")
@patch("profsea.components.spatial.sterodynamic.Path.glob")
def test_load_cmip6_slopes(mock_glob, mock_open_dataset):
    # Mock file paths (glob is called twice: once for slopes, once for masks)
    mock_glob.side_effect = [
        [
            Path("dummy/zos_regression_ssp585_1.nc"),
            Path("dummy/zos_regression_ssp585_2.nc"),
        ],
        [Path("dummy/zos_mask_ssp585_1.nc"), Path("dummy/zos_mask_ssp585_2.nc")],
    ]

    # Mock datasets
    mock_data = xr.DataArray(np.random.rand(5, 5), dims=["lat", "lon"])
    mock_mask_data = xr.DataArray(np.zeros((5, 5)), dims=["lat", "lon"])

    def mock_open(f, **kwargs):
        if "regression" in str(f):
            return {"zos_zostoga_regression_slope": mock_data}
        elif "mask" in str(f):
            return {"zos_mask": mock_mask_data}

    mock_open_dataset.side_effect = mock_open

    # Updated shape to (climate_members, process_members, time)
    dummy_global = xr.DataArray(np.zeros((3, 4, 100)))
    stero = SterodynamicCMIP6(global_projection=dummy_global)

    slopes, masks = stero._load_CMIP6_slopes()

    assert slopes.shape == (2, 5, 5)
    assert "model" in slopes.dims
    assert masks is not None
    assert masks.shape == (2, 5, 5)
    assert stero.land_mask_present is True


@patch.object(SpatialComponent, "extract_spatial")
@patch.object(SterodynamicCMIP6, "_load_CMIP6_slopes")
def test_expansion_contribution_storyline_mode(mock_load, mock_interp):
    # Mock the loaded data and masks
    mock_coeffs = xr.DataArray(
        np.array([np.ones((2, 2)), np.ones((2, 2)) * 2, np.ones((2, 2)) * 3]),
        coords=[("model", [0, 1, 2]), ("lat", [0, 1]), ("lon", [0, 1])],
    )
    mock_masks = xr.DataArray(
        np.zeros((3, 2, 2)),
        coords=[("model", [0, 1, 2]), ("lat", [0, 1]), ("lon", [0, 1])],
    )
    mock_load.return_value = (mock_coeffs, mock_masks)

    # Let the mocked interpolator just return the array passed into it
    mock_interp.return_value = mock_coeffs

    # Test Storyline Mode with 3D global array: (5 climate, 2 process, 100 time)
    dummy_global = xr.DataArray(np.zeros((5, 2, 100)))
    stero_storyline = SterodynamicCMIP6(
        global_projection=dummy_global, sample_spatial=False
    )

    state = SpatialState(
        grid_lats=np.array([0, 1]),
        grid_lons=np.array([0, 1]),
        n_years=100,
        num_localMC_members=10,
        grid_interpolation="nearest",
        output_percentiles=None,
        baseline_yrs=(1995, 2014),
        endofhistory=2006,
        num_output_members=10,  # 5 climate_members * 2 process_members
    )

    stero_storyline.land_mask_present = (
        True  # Simulate the flag being set during the load phase
    )

    rng = np.random.default_rng(42)
    result_storyline = stero_storyline._calc_expansion_contribution(rng, state)

    # In storyline mode, it takes the ensemble mean: (1+2+3)/3 = 2
    # Expect flattened output members on the spatial dimension
    assert result_storyline.shape == (10, 2, 2)
    computed_result = result_storyline.compute()
    np.testing.assert_array_equal(computed_result, np.ones((10, 2, 2)) * 2)


@patch.object(SpatialComponent, "extract_spatial")
@patch.object(SterodynamicCMIP6, "_load_CMIP6_slopes")
def test_expansion_contribution_sampled_mode(mock_load, mock_interp):
    mock_coeffs = xr.DataArray(
        np.array([np.ones((2, 2)), np.ones((2, 2)) * 2, np.ones((2, 2)) * 3]),
        coords=[("model", [0, 1, 2]), ("lat", [0, 1]), ("lon", [0, 1])],
    )
    mock_masks = xr.DataArray(
        np.zeros((3, 2, 2)),
        coords=[("model", [0, 1, 2]), ("lat", [0, 1]), ("lon", [0, 1])],
    )
    mock_load.return_value = (mock_coeffs, mock_masks)
    mock_interp.return_value = mock_coeffs

    dummy_global = xr.DataArray(np.zeros((5, 2, 100)))
    stero_sampled = SterodynamicCMIP6(
        global_projection=dummy_global, sample_spatial=True
    )

    state = SpatialState(
        grid_lats=np.array([0, 1]),
        grid_lons=np.array([0, 1]),
        n_years=100,
        num_localMC_members=10,
        grid_interpolation="nearest",
        output_percentiles=None,
        baseline_yrs=(1995, 2014),
        endofhistory=2006,
        num_output_members=10,
    )

    stero_sampled.land_mask_present = True

    rng = np.random.default_rng(42)
    result_sampled = np.asarray(stero_sampled._calc_expansion_contribution(rng, state))

    assert result_sampled.shape == (10, 2, 2)
    unique_values = np.unique(result_sampled)
    assert len(unique_values) > 1


@pytest.mark.parametrize("at_sites", [False, True])
@pytest.mark.parametrize("sample_spatial", [False, True])
@patch.object(SterodynamicCMIP6, "_load_CMIP6_slopes")
@patch.object(SpatialComponent, "extract_spatial")
def test_project_resamples_global_members(
    mock_extract, mock_load, at_sites, sample_spatial
):
    kwargs = dict(
        n_years=3,
        num_localMC_members=4,
        num_output_members=1,
        output_percentiles=[50],
        baseline_yrs=(1995, 2014),
        endofhistory=2006,
        random_fraction=np.random.default_rng(42).uniform(size=4),
    )
    if at_sites:
        state = LocalState(
            target_lats=[0, 1],
            target_lons=[0, 1],
            interpolation_method="bilinear",
            **kwargs,
        )
        spatial_shape = (2,)
    else:
        state = SpatialState(
            grid_lats=np.array([0, 1]),
            grid_lons=np.array([0, 1]),
            grid_interpolation="linear",
            **kwargs,
        )
        spatial_shape = (2, 2)
    patterns = xr.DataArray(da.ones((3, *spatial_shape)) * 2)
    mock_load.return_value = (patterns, None)
    mock_extract.return_value = patterns
    raw = np.arange(12, dtype=float).reshape(2, 2, 3)
    comp = SterodynamicCMIP6(xr.DataArray(raw), sample_spatial=sample_spatial)
    comp.land_mask_present = False

    result = comp.project(state, np.random.default_rng(42))

    indices = (state.random_fraction * 4).astype(int)
    expected = raw.reshape(4, 3)[indices] * 2
    expected = expected.reshape((4, 3) + (1,) * len(spatial_shape))
    assert isinstance(result, da.Array)
    assert result.shape == (4, 3, *spatial_shape)
    np.testing.assert_allclose(
        result.compute(), np.broadcast_to(expected, result.shape)
    )


def test_cmip5_placeholder_raises_clear_error():
    """The CMIP5 placeholder should fail loudly rather than return zeros."""
    with pytest.raises(NotImplementedError, match="SterodynamicCMIP6"):
        SterodynamicCMIP5()
