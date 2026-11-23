from types import SimpleNamespace
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

# Adjust import path based on your package structure
from profsea.components.core.base import SpatialComponent


class DummySpatial(SpatialComponent):
    """Concrete implementation for testing abstract SpatialComponent."""

    @property
    def global_projection(self):
        return np.array([1, 2, 3])

    def project(self, state, rng):
        pass


@pytest.fixture
def dummy_component():
    return DummySpatial()


@pytest.fixture
def sample_da():
    """Returns a simple 3x4 grid from lat [0, 10, 20], lon [-180, -90, 0, 90]."""
    lats = np.array([0, 10, 20])
    lons = np.array([-180, -90, 0, 90])
    # Values represent their linear index for easy tracking
    data = np.arange(12).reshape(3, 4).astype(float)
    return xr.DataArray(data, coords=[lats, lons], dims=["lat", "lon"])


def test_fetch_assets_calls_zenodo_download(dummy_component):
    """Test that fetch_zenodo_fingerprints() is called from SpatialComponent"""

    with patch("profsea.components.core.base.fetch_zenodo_fingerprints") as mock_fetch:
        dummy_component.fetch_assets()

        mock_fetch.assert_called_once()


class TestExtractSpatialRouting:
    def test_routes_to_extract_points(self, dummy_component, sample_da):
        """Test that extract_spatial routes to point extraction when target_lats exist."""
        state = SimpleNamespace(target_lats=np.array([5]), target_lons=np.array([45]))

        with patch.object(
            dummy_component, "_extract_points", return_value="points"
        ) as mock_ext:
            result = dummy_component.extract_spatial(sample_da, state)
            mock_ext.assert_called_once()
            assert result == "points"

    def test_routes_to_interpolate_grid(self, dummy_component, sample_da):
        """Test that extract_spatial routes to grid interpolation when grid_lats exist."""
        state = SimpleNamespace(grid_lats=np.array([5]), grid_lons=np.array([45]))

        with patch(
            "profsea.utils.utils.interpolate_to_grid", return_value="grid"
        ) as mock_interp:
            result = dummy_component.extract_spatial(sample_da, state)
            mock_interp.assert_called_once()
            assert result == "grid"

    def test_missing_attributes_raises_error(self, dummy_component, sample_da):
        """Test that extract_spatial raises ValueError if state lacks spatial attrs."""
        state = SimpleNamespace()
        with pytest.raises(ValueError, match="missing required spatial attributes"):
            dummy_component.extract_spatial(sample_da, state)


class TestExtractPoints:
    def test_standard_interpolation(self, dummy_component, sample_da):
        """Test basic bilinear interpolation inside the grid bounds."""
        # Midpoint between lat (0, 10) and lon (0, 90)
        # Coordinates: (lat 5, lon 45)
        # Indices: lat(0,1), lon(2,3) -> Values: [2, 3] and [6, 7]
        # Average of 2, 3, 6, 7 is 4.5
        state = SimpleNamespace(target_lats=np.array([5]), target_lons=np.array([45]))

        result = dummy_component._extract_points(
            sample_da, state.target_lats, state.target_lons
        )

        assert result.dims == ("site",)
        assert len(result) == 1
        np.testing.assert_allclose(result.values, [4.5])

    def test_descending_latitudes(self, dummy_component, sample_da):
        """Test that descending latitude coordinates are correctly flipped and handled."""
        # Create a DataArray with flipped latitudes
        flipped_da = sample_da.isel(lat=slice(None, None, -1))
        # Ensure coordinates are actually descending
        assert flipped_da.lat.values[0] > flipped_da.lat.values[-1]

        state = SimpleNamespace(target_lats=np.array([5]), target_lons=np.array([45]))
        result = dummy_component._extract_points(
            flipped_da, state.target_lats, state.target_lons
        )

        # Should yield the exact same interpolation result as ascending lats
        np.testing.assert_allclose(result.values, [4.5])

    def test_longitude_wrapping(self, dummy_component, sample_da):
        """Test that targets across the -180/180 boundary interpolate correctly."""
        # Sample DA lons: [-180, -90, 0, 90]
        # To test wrapping, target lon = 135 (midway between 90 and -180)
        # Lat = 0 -> values at lon 90 (idx 3) = 3; lon -180 (idx 0) = 0
        # Expected midpoint value: 1.5
        state = SimpleNamespace(target_lats=np.array([0]), target_lons=np.array([135]))

        result = dummy_component._extract_points(
            sample_da, state.target_lats, state.target_lons
        )
        np.testing.assert_allclose(result.values, [1.5])

    def test_nan_handling_and_weight_renormalization(self, dummy_component, sample_da):
        """Test that NaNs in corner points correctly re-distribute weights."""
        # Introduce a NaN at (lat=10, lon=90)
        da_nan = sample_da.copy()
        da_nan.loc[dict(lat=10, lon=90)] = np.nan

        # Target lat 5, lon 45. Corners: (0,0)=2, (0,90)=3, (10,0)=6, (10,90)=NaN
        # Standard weights are 0.25 each.
        # With NaN, remaining 3 corners should get weights 1/3.
        # Mean of (2, 3, 6) is 3.666...
        state = SimpleNamespace(target_lats=np.array([5]), target_lons=np.array([45]))

        result = dummy_component._extract_points(
            da_nan, state.target_lats, state.target_lons
        )
        np.testing.assert_allclose(result.values, [11.0 / 3.0])

    def test_all_nans_returns_nan(self, dummy_component, sample_da):
        """Test that if all 4 bounding points are NaN, the result is NaN."""
        da_all_nan = sample_da.copy()
        da_all_nan.loc[dict(lat=slice(0, 10), lon=slice(0, 90))] = np.nan

        state = SimpleNamespace(target_lats=np.array([5]), target_lons=np.array([45]))
        result = dummy_component._extract_points(
            da_all_nan, state.target_lats, state.target_lons
        )

        assert np.isnan(result.values[0])


@pytest.mark.parametrize("spatial_shape", [(4,), (2, 2)])
@pytest.mark.parametrize("sample_patterns", [False, True])
@pytest.mark.parametrize("shared_time", [False, True])
def test_separable_percentiles_match_expanded_members(
    spatial_shape, sample_patterns, shared_time
):
    """Every percentile path (single pattern, shared time series, or full
    ranking) must equal nanpercentile over the expanded member array."""
    from dask.callbacks import Callback

    from profsea.components.core.base import SeparableProjection

    rng = np.random.default_rng(42)
    if shared_time:
        # One time series for all members, crossing zero (like GIA before and
        # after its baseline midpoint)
        temporal = np.array([-1.5, 0.0, 2.0], dtype=np.float32)
    else:
        temporal = rng.normal(size=(7, 3)).astype(np.float32)
        temporal[0, 1] = np.nan
    patterns = np.array([2.0, -3.0, 0.0, np.nan], dtype=np.float32).reshape(
        (1, *spatial_shape)
    )
    index = None
    if sample_patterns:
        patterns = np.concatenate((patterns, -patterns))
        index = rng.integers(2, size=7)
    projection = SeparableProjection(
        da.from_array(temporal, chunks=2),
        da.from_array(patterns, chunks=1),
        index,
    )
    state = SimpleNamespace(
        dtype=np.float32,
        time_chunk=2,
        spatial_chunks=(1,) * len(spatial_shape),
        num_spatial_members=7,
    )
    q = [0, 17, 50, 83, 100]
    with Callback(start=lambda graph: pytest.fail("Projection computed eagerly")):
        result = projection.percentiles(q, state)
        members = projection.to_array(state)
    selected = patterns[0] if index is None else patterns[index][:, None]
    temporal_members = np.broadcast_to(temporal, (7, 3))
    expanded = temporal_members[(...,) + (None,) * len(spatial_shape)] * selected
    expanded = np.broadcast_to(expanded, (7, 3, *spatial_shape))
    with pytest.warns(RuntimeWarning, match="All-NaN slice"):
        expected = np.nanpercentile(expanded, q, axis=0)
    assert members.shape == (7, 3, *spatial_shape)
    assert members.chunks[0] == (7,)
    np.testing.assert_allclose(members.compute(), expanded, rtol=1e-6)
    assert result.shape == (len(q), 3, *spatial_shape)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result.compute(), expected, rtol=1e-6, atol=1e-7)


def test_separable_rejects_mismatched_pattern_index():
    """A pattern index that does not match the members, or is missing for several patterns, is an error."""
    from profsea.components.core.base import SeparableProjection

    with pytest.raises(ValueError, match="pattern_index has 3 entries"):
        SeparableProjection(np.zeros((4, 2)), da.zeros((2, 5)), np.zeros(3, int))
    with pytest.raises(ValueError, match="pattern_index is required"):
        SeparableProjection(np.zeros((4, 2)), da.zeros((2, 5)))


def test_independent_member_array_has_own_graph_and_same_values():
    """The total's member arrays must not share Dask keys with a component's own
    output, or Dask holds every shared block until both are computed (72 GB at
    1 degree when saving all outputs together)."""
    from profsea.components.core.base import SeparableProjection

    rng = np.random.default_rng(1)
    projection = SeparableProjection(
        rng.normal(size=(5, 3)),
        da.from_array(rng.normal(size=(2, 4)), chunks=(1, 2)),
        np.array([0, 1, 1, 0, 1]),
    )
    state = SimpleNamespace(
        dtype=np.float32, time_chunk=2, spatial_chunks=(2,), num_spatial_members=5
    )

    shared = projection.to_array(state)
    independent = projection.to_array(state, independent=True)

    assert independent.name != shared.name
    assert independent.chunks == shared.chunks
    np.testing.assert_array_equal(independent.compute(), shared.compute())
