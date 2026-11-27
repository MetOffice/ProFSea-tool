from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from profsea.components.core.base import SeparableProjection, SpatialComponent
from profsea.components.core.spatial_model import Spatial


def get_dummy_components(nt=2, num_members=3, n_years=4):
    """Helper to create a dummy component dictionary with the expected shape."""
    mock_comp = MagicMock(spec=SpatialComponent)
    # 3D Shape: (climate_member, process_member, time)
    mock_comp.global_projection.shape = (nt, num_members, n_years)
    # Not separable: the model calls project() and uses the member array
    mock_comp.project_separable.return_value = None
    return {"mock_comp": mock_comp}


GRID_2X2 = dict(start_lat=0, end_lat=1, step_lat=1, start_lon=0, end_lon=1, step_lon=1)


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_spatial_init_blocks_download_and_configures_spatial_members(mock_fetch):
    """Construction checks the assets and records the global ensemble shape."""
    components = get_dummy_components(nt=5, num_members=10, n_years=44)
    spatial = Spatial(components=components, start_yr=2006, end_yr=2050)

    mock_fetch.assert_called_once()
    assert spatial.start_yr == 2006
    assert spatial.end_yr == 2050
    assert spatial.num_spatial_members == 1000
    assert spatial.num_members == 5
    assert (spatial.n_climate, spatial.n_process) == (5, 10)


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_spatial_rejects_mismatched_global_ensembles(mock_fetch):
    """Members are shared across components, so their ensembles must match."""
    components = get_dummy_components(nt=2, num_members=3, n_years=4)
    components.update(
        {"other": get_dummy_components(nt=2, num_members=5, n_years=4)["mock_comp"]}
    )
    with pytest.raises(ValueError, match="same global ensemble"):
        Spatial(components=components, end_yr=2010)


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_spatial_rejects_wrong_time_axis(mock_fetch):
    """Global projections must cover the model's years."""
    components = get_dummy_components(n_years=4)
    with pytest.raises(ValueError, match="4 years but the model expects 44"):
        Spatial(components=components, end_yr=2050)


@pytest.mark.parametrize(
    "start_lon, end_lon, step_lon, expected",
    [
        (-179.5, 179.5, 1, np.arange(-179.5, 180)),  # default grid kept exactly
        (0, 270, 90, [-180, -90, 0, 90]),
        (-180, 180, 90, [-180, -90, 0, 90]),  # closed interval: 180 is -180
        (170, 190, 10, [-180, -170, 170]),  # crosses the dateline
    ],
)
@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_spatial_grid_longitudes_are_normalised_and_sorted(
    mock_fetch, start_lon, end_lon, step_lon, expected
):
    """The grid uses the regridded patterns' longitudes: [-180, 180), ascending."""
    grid = dict(
        start_lat=0,
        end_lat=1,
        step_lat=1,
        start_lon=start_lon,
        end_lon=end_lon,
        step_lon=step_lon,
    )
    spatial = Spatial(
        components=get_dummy_components(n_years=4), grid_config=grid, end_yr=2010
    )
    np.testing.assert_array_equal(spatial.grid_lons, expected)


class FixedPattern(SpatialComponent):
    """Every member is one pattern, regridded as the real components do."""

    def __init__(self, pattern: xr.DataArray):
        self.pattern = pattern

    def project_separable(self, state, rng):
        regridded = self.extract_spatial(self.pattern, state).data
        return SeparableProjection(np.ones(state.n_years), regridded[None])


def _no_land_regionmask() -> MagicMock:
    """Stand-in for regionmask (its shapes are downloaded on first use) for a
    grid with no land. Like the real ``mask_3D``, it drops the land region
    unless ``drop=False``."""

    def mask_3D(obj, drop=True):
        mask = xr.DataArray(
            np.zeros((1, obj.lat.size, obj.lon.size), dtype=bool),
            dims=["region", "lat", "lon"],
            coords={"region": [0], "lat": obj.lat.values, "lon": obj.lon.values},
        )
        return mask.isel(region=slice(0, 0)) if drop else mask

    fake = MagicMock()
    fake.defined_regions.natural_earth_v5_0_0.land_110.mask_3D.side_effect = mask_3D
    return fake


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_spatial_output_longitudes_label_their_values(mock_fetch):
    """A 0-360 grid comes back on -180-180 with each value at its own longitude
    (previously the labels kept the 0-360 order and were shifted by 180)."""
    lat = np.arange(-89.5, 90)
    lon = np.arange(-179.5, 180)
    pattern = xr.DataArray(
        np.tile(np.sin(np.radians(lon)), (lat.size, 1)),
        coords=[("lat", lat), ("lon", lon)],
    )
    grid = dict(
        start_lat=-0.5,
        end_lat=0.5,
        step_lat=1,
        start_lon=0.5,
        end_lon=359.5,
        step_lon=1,
    )
    spatial = Spatial(
        components={"pattern": FixedPattern(pattern)},
        grid_config=grid,
        start_yr=2006,
        end_yr=2007,
        output_percentiles=None,
        num_spatial_members=1,
        compute_total=False,
    )

    with patch.dict("sys.modules", {"regionmask": _no_land_regionmask()}):
        out = spatial.run()["pattern"].isel(member=0, time=0).compute()

    assert out.shape == (2, 360)
    np.testing.assert_array_equal(out.lon.values, lon)
    expected = np.tile(np.sin(np.radians(out.lon.values)), (2, 1))
    np.testing.assert_allclose(out.values, expected, atol=1e-6)


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_arr_to_xr_metadata_with_percentiles(mock_fetch):
    # Initializes with default output_percentiles=[5, 17, 50, 83, 95] (Length 5)
    """Percentile outputs get a percentile axis and calendar years."""
    components = get_dummy_components(n_years=4)
    spatial = Spatial(components=components, end_yr=2010)

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
def test_full_ensemble_output_labels_global_members(mock_fetch):
    """Full output has one member axis labelled with each member's position in
    the global ensemble."""
    components = get_dummy_components(nt=2, num_members=3, n_years=4)
    components["mock_comp"].project.side_effect = lambda state, rng: da.zeros(
        (6, 4, 2, 2)
    )
    spatial = Spatial(
        components=components,
        end_yr=2010,
        output_percentiles=None,
        num_spatial_members=6,
        grid_config=GRID_2X2,
    )
    da_out = spatial.run(member_seed=1)["mock_comp"]

    assert list(da_out.dims) == ["member", "time", "lat", "lon"]
    np.testing.assert_array_equal(da_out.member.values, np.arange(6))
    # 6 members from a 2 x 3 ensemble: every member used once
    np.testing.assert_array_equal(da_out.climate_member.values, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(da_out.process_member.values, [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(da_out.time.values, [2006, 2007, 2008, 2009])


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_sum_spatial_components(mock_fetch):
    """sum_components adds full-member components when no total exists yet."""
    components = get_dummy_components()
    spatial = Spatial(components=components, end_yr=2010)

    # Two identical arrays of 1s
    da1 = xr.DataArray(np.ones((2, 2)), dims=["lat", "lon"])
    da2 = xr.DataArray(np.ones((2, 2)), dims=["lat", "lon"])

    components_dict = {"comp1": da1, "comp2": da2}
    total = spatial.sum_components(components_dict)

    # 1 + 1 = 2 everywhere
    np.testing.assert_array_equal(total.values, np.ones((2, 2)) * 2)
    assert "total_rsl" in components_dict


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_sum_components_does_not_double_count_or_sum_percentiles(mock_fetch):
    """sum_components returns an existing total and refuses to add percentiles."""
    spatial = Spatial(components=get_dummy_components(), end_yr=2010)
    ones = xr.DataArray(np.ones((3, 2)), dims=["percentile", "lat"])

    existing = {"comp1": ones, "total_rsl": ones * 5}
    assert spatial.sum_components(existing) is existing["total_rsl"]

    with pytest.raises(ValueError, match="Cannot sum percentiles"):
        spatial.sum_components({"comp1": ones, "comp2": ones})


@pytest.mark.parametrize("percentiles", [None, []])
@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_run_full_spatial_member_ensemble(mock_fetch, percentiles, capsys):
    """Members are shared by all components, stratified by trajectory and reproducible from the seed."""
    components = get_dummy_components(nt=2, num_members=3, n_years=3)
    comp = components["mock_comp"]
    seen = {}

    def project(state, rng):
        seen["indices"] = state.member_indices
        assert state.num_spatial_members == 7
        assert state.member_indices.shape == state.fingerprint_fraction.shape == (7,)
        return da.broadcast_to(
            state.member_indices[:, None, None, None].astype(float),
            (7, state.n_years, len(state.grid_lats), len(state.grid_lons)),
        )

    comp.project.side_effect = project
    model = Spatial(
        components=components,
        end_yr=2009,
        output_percentiles=percentiles,
        num_spatial_members=7,
        grid_config=GRID_2X2,
    )
    results = model.run(member_seed=42)
    assert "Full Distribution (7 members)" in capsys.readouterr().out
    assert set(results) == {"mock_comp", "total_rsl"}
    assert results["mock_comp"].shape == (7, 3, 2, 2)
    assert results["mock_comp"].dims == ("member", "time", "lat", "lon")
    assert isinstance(results["mock_comp"].data, da.Array)
    np.testing.assert_array_equal(
        results["total_rsl"].compute(), results["mock_comp"].compute()
    )
    # Stratified: 7 members over 2 trajectories gives 4 and 3 (or 3 and 4)
    assert sorted(np.bincount(seen["indices"] // 3, minlength=2)) == [3, 4]
    repeated = model.run(member_seed=42)
    np.testing.assert_array_equal(
        results["mock_comp"].compute(), repeated["mock_comp"].compute()
    )


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_run_memberwise_total_and_percentiles(mock_fetch):
    """The total is summed member by member before taking percentiles."""
    rng = np.random.default_rng(42)
    temporal = rng.random((7, 4))
    patterns = rng.random((7, 2, 2))
    expected = temporal[:, :, None, None] * patterns[:, None, :, :]
    components = get_dummy_components()
    components["other"] = get_dummy_components()["mock_comp"]

    for component in components.values():
        component.project.side_effect = lambda state, rng: da.from_array(
            expected, chunks=(7, 1, 1, 1)
        )

    model = Spatial(
        components=components,
        end_yr=2010,
        num_spatial_members=7,
        output_percentiles=[17, 50, 83],
        grid_config=GRID_2X2,
    )
    results = model.run()

    assert results["total_rsl"].shape == (3, 4, 2, 2)
    # Components are rechunked to the model's single chunk plan
    assert results["total_rsl"].data.chunks == results["mock_comp"].data.chunks
    np.testing.assert_allclose(
        results["mock_comp"].compute(),
        np.nanpercentile(expected, [17, 50, 83], axis=0),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        results["total_rsl"].compute(),
        np.nanpercentile(2 * expected, [17, 50, 83], axis=0),
        rtol=1e-6,
    )


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_component_percentiles_without_total_avoid_member_expansion(mock_fetch):
    """With compute_total=False, separable components skip building member arrays."""
    from profsea.components.core.base import SeparableProjection

    temporal = np.arange(24, dtype=np.float32).reshape(6, 4)
    projection = SeparableProjection(temporal, da.ones((1, 2, 2)))
    components = get_dummy_components()
    components["mock_comp"].project_separable.return_value = projection
    model = Spatial(
        components=components,
        end_yr=2010,
        output_percentiles=[25, 75],
        num_spatial_members=6,
        compute_total=False,
        grid_config=GRID_2X2,
    )
    with patch.object(
        projection, "to_array", side_effect=AssertionError("Expanded members")
    ):
        results = model.run()
        actual = results["mock_comp"].compute()
    assert set(results) == {"mock_comp"}
    assert actual.shape == (2, 4, 2, 2)
    expected = np.nanpercentile(temporal, [25, 75], axis=0)[:, :, None, None]
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape))


@patch("profsea.components.core.spatial_model.fetch_zenodo_fingerprints")
def test_output_size_estimate_counts_every_output(mock_fetch, capsys):
    """The estimate covers one array per component plus the total, not just one."""
    components = get_dummy_components(n_years=4)
    components["other"] = get_dummy_components(n_years=4)["mock_comp"]
    spatial = Spatial(
        components=components, end_yr=2010, grid_config=GRID_2X2
    )  # 5 percentiles, float32, 2 x 2 grid

    per_array_gb = 5 * 4 * 2 * 2 * 4 / 1e9
    assert spatial.n_output_arrays == 3  # two components + total_rsl
    assert spatial.output_gb_per_array == pytest.approx(per_array_gb)
    assert spatial.output_gb == pytest.approx(3 * per_array_gb)

    without_total = Spatial(
        components=components, end_yr=2010, grid_config=GRID_2X2, compute_total=False
    )
    assert without_total.n_output_arrays == 2
