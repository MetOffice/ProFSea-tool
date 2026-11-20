from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from profsea.components.core.base import SpatialComponent
from profsea.components.core.spatial_model import Spatial


def get_dummy_components(nt=2, num_members=3, n_years=4):
    """Helper to create a dummy component dictionary with the expected shape."""
    mock_comp = MagicMock(spec=SpatialComponent)
    # 3D Shape: (climate_member, process_member, time)
    mock_comp.global_projection.shape = (nt, num_members, n_years)
    return {"mock_comp": mock_comp}


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_spatial_init_blocks_download_and_configures_local_mc(mock_fetch):
    components = get_dummy_components(nt=5, num_members=10)
    spatial = Spatial(components=components, end_year=2050)

    mock_fetch.assert_called_once()
    assert spatial.end_year == 2050
    assert spatial.num_localMC_members == 1000
    assert spatial.num_members == 5


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
    # Full output uses a single local Monte Carlo axis.
    components = get_dummy_components(nt=2, num_members=3, n_years=4)
    spatial = Spatial(
        components=components,
        end_year=2010,
        output_percentiles=None,
        num_localMC_members=6,
    )

    arr = da.zeros((6, 4, 180, 360))
    arr_dict = {"mock_spatial": arr}

    xr_dict = spatial._arr_to_xr(arr_dict)
    da_out = xr_dict["mock_spatial"]

    assert isinstance(da_out, xr.DataArray)
    assert list(da_out.dims) == [
        "process_member",
        "time",
        "lat",
        "lon",
    ]
    assert da_out.shape == (6, 4, 180, 360)
    np.testing.assert_array_equal(da_out.process_member.values, np.arange(6))
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


@pytest.mark.parametrize("percentiles", [None, []])
@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_run_full_local_mc_ensemble(mock_fetch, percentiles, capsys):
    components = get_dummy_components()
    comp = components["mock_comp"]

    def project(state, rng):
        assert state.num_output_members == state.num_localMC_members == 7
        assert state.random_fraction.shape == (7,)
        return da.broadcast_to(
            state.random_fraction[:, None, None, None],
            (7, state.n_years, len(state.grid_lats), len(state.grid_lons)),
        )

    comp.project.side_effect = project
    model = Spatial(
        components=components,
        end_year=2009,
        output_percentiles=percentiles,
        num_localMC_members=7,
        grid_config=dict(
            start_lat=0,
            end_lat=1,
            step_lat=1,
            start_lon=0,
            end_lon=1,
            step_lon=1,
        ),
    )
    results = model.run(member_seed=42)
    assert "Full Distribution (7 members)" in capsys.readouterr().out
    assert set(results) == {"mock_comp", "total_rsl"}
    assert results["mock_comp"].shape == (7, 3, 2, 2)
    assert results["mock_comp"].dims == ("process_member", "time", "lat", "lon")
    assert isinstance(results["mock_comp"].data, da.Array)
    np.testing.assert_array_equal(
        results["total_rsl"].compute(), results["mock_comp"].compute()
    )
    repeated = model.run(member_seed=42)
    np.testing.assert_array_equal(
        results["mock_comp"].compute(), repeated["mock_comp"].compute()
    )


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_run_preserves_time_chunks_and_memberwise_total(mock_fetch):
    rng = np.random.default_rng(42)
    temporal = rng.random((7, 4))
    patterns = rng.random((7, 2, 2))
    expected = temporal[:, :, None, None] * patterns[:, None, :, :]
    components = get_dummy_components()
    components["other"] = MagicMock(spec=SpatialComponent)

    def project(state, rng):
        return SpatialComponent.broadcast_spatiotemporal(
            components["mock_comp"],
            da.from_array(temporal, chunks=(7, 1)),
            da.from_array(patterns, chunks=(7, 2, 2)),
        )

    for component in components.values():
        component.project.side_effect = project

    model = Spatial(
        components=components,
        end_year=2010,
        num_localMC_members=7,
        output_percentiles=[17, 50, 83],
        grid_config=dict(
            start_lat=0,
            end_lat=1,
            step_lat=1,
            start_lon=0,
            end_lon=1,
            step_lon=1,
        ),
    )
    results = model.run()

    assert results["total_rsl"].shape == (3, 4, 2, 2)
    assert results["total_rsl"].data.chunks[1] == (1, 1, 1, 1)
    np.testing.assert_allclose(
        results["mock_comp"].compute(), np.nanpercentile(expected, [17, 50, 83], axis=0)
    )
    np.testing.assert_allclose(
        results["total_rsl"].compute(),
        np.nanpercentile(2 * expected, [17, 50, 83], axis=0),
    )
