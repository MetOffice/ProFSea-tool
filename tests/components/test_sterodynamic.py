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
    _SterodynamicCMIP,
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


@pytest.mark.parametrize("component_cls", [SterodynamicCMIP5, SterodynamicCMIP6])
@patch("profsea.components.spatial.sterodynamic.xr.open_dataset")
@patch("profsea.components.spatial.sterodynamic.Path.glob")
def test_load_cmip6_slopes(mock_glob, mock_open_dataset, component_cls):
    # Mock file paths (glob is called twice: once for slopes, once for masks)
    """Slope and land-mask files are stacked along a new model dimension."""
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
    stero = component_cls(global_projection=dummy_global)

    slopes, masks = stero._load_slopes()

    assert slopes.shape == (2, 5, 5)
    assert "model" in slopes.dims
    assert masks is not None
    assert masks.shape == (2, 5, 5)


def make_state(at_sites: bool, n_years: int = 3) -> SpatialState | LocalState:
    """A four-member state on a 2x2 grid or at two sites."""
    kwargs = dict(
        n_years=n_years,
        num_spatial_members=4,
        output_percentiles=None,
        baseline_yrs=(1995, 2014),
        endofhistory=2006,
        member_indices=np.array([3, 0, 2, 1]),
        fingerprint_fraction=np.array([0.1, 0.6, 0.9, 0.4]),
    )
    if at_sites:
        return LocalState(
            target_lats=[0, 1],
            target_lons=[0, 1],
            interpolation_method="bilinear",
            **kwargs,
        )
    return SpatialState(
        grid_lats=np.array([0, 1]),
        grid_lons=np.array([0, 1]),
        grid_interpolation="linear",
        **kwargs,
    )


def mock_patterns(mock_load, mock_extract, spatial_shape):
    """Three CMIP model patterns with values 1, 2 and 3."""
    values = np.stack([np.full(spatial_shape, v, dtype=float) for v in (1, 2, 3)])
    patterns = xr.DataArray(da.from_array(values))
    mock_load.return_value = (patterns, None)
    mock_extract.return_value = patterns
    return values


@pytest.mark.parametrize("component_cls", [SterodynamicCMIP5, SterodynamicCMIP6])
@pytest.mark.parametrize("at_sites", [False, True])
@patch.object(_SterodynamicCMIP, "_load_slopes")
@patch.object(SpatialComponent, "extract_spatial")
def test_storyline_mode_uses_pattern_mean(
    mock_extract, mock_load, at_sites, component_cls
):
    """With sample_spatial=False every member uses the mean CMIP pattern."""
    spatial_shape = (2,) if at_sites else (2, 2)
    mock_patterns(mock_load, mock_extract, spatial_shape)
    raw = np.arange(12, dtype=float).reshape(2, 2, 3)
    comp = component_cls(xr.DataArray(raw), sample_spatial=False)
    state = make_state(at_sites)

    result = comp.project(state, np.random.default_rng(42)).compute()

    # Pattern mean is (1 + 2 + 3) / 3 = 2
    expected = raw.reshape(4, 3)[state.member_indices] * 2
    np.testing.assert_allclose(
        result,
        np.broadcast_to(expected[(...,) + (None,) * len(spatial_shape)], result.shape),
    )


@pytest.mark.parametrize("component_cls", [SterodynamicCMIP5, SterodynamicCMIP6])
@pytest.mark.parametrize("at_sites", [False, True])
@patch.object(_SterodynamicCMIP, "_load_slopes")
@patch.object(SpatialComponent, "extract_spatial")
def test_sampled_mode_draws_a_model_per_member(
    mock_extract, mock_load, at_sites, component_cls
):
    """With sample_spatial=True each member uses the slope of the model it drew."""
    spatial_shape = (2,) if at_sites else (2, 2)
    values = mock_patterns(mock_load, mock_extract, spatial_shape)
    raw = np.arange(12, dtype=float).reshape(2, 2, 3)
    comp = component_cls(xr.DataArray(raw), sample_spatial=True)
    state = make_state(at_sites)

    separable = comp.project_separable(state, np.random.default_rng(7))
    result = comp.project(state, np.random.default_rng(7)).compute()

    assert separable.pattern_index.shape == (4,)
    expected = (
        raw.reshape(4, 3)[state.member_indices][(...,) + (None,) * len(spatial_shape)]
        * values[separable.pattern_index][:, None]
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("component_cls", "expected_dir"),
    [
        (SterodynamicCMIP5, "cmip5-patterns"),
        (SterodynamicCMIP6, "cmip6-patterns"),
    ],
)
def test_selects_correct_pattern_directory(component_cls, expected_dir):
    dummy_global = xr.DataArray(np.zeros((3, 4, 100)))

    stero = component_cls(global_projection=dummy_global)

    assert stero.patterns_dir.name == expected_dir
