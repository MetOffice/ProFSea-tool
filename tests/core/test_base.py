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


class TestBroadcastSpatiotemporal:
    def test_broadcast_2d_temporal_1d_spatial(self, dummy_component):
        """Test broadcasting 2D temporal (members, years) with 1D spatial (members, sites)."""
        temporal = da.ones((2, 3))
        spatial = da.ones((2, 4)) * 2

        result = dummy_component.broadcast_spatiotemporal(temporal, spatial)

        assert result.ndim == 3
        assert result.shape == (2, 3, 4)  # (members, years, site)
        assert result.compute()[0, 0, 0] == 2.0

    def test_broadcast_2d_temporal_2d_spatial(self, dummy_component):
        """Test broadcasting 2D temporal (members, years) with 2D spatial (members, lat, lon)."""
        temporal = da.ones((2, 3))
        spatial = da.ones((2, 4, 5)) * 3

        result = dummy_component.broadcast_spatiotemporal(temporal, spatial)

        assert result.ndim == 4
        assert result.shape == (2, 3, 4, 5)  # (members, years, lat, lon)
        assert result.compute()[0, 0, 0, 0] == 3.0

    def test_broadcast_3d_temporal_1d_spatial(self, dummy_component):
        """Test broadcasting 3D temporal (climate, process, years) with 1D spatial (total_members, site)."""
        # Shape: (2 climate_members, 3 process_members, 4 years)
        temporal = da.ones((2, 3, 4))

        # Spatial comes in flattened: 2*3 = 6 output members. Shape: (6 members, 5 sites)
        spatial = da.ones((6, 5)) * 4

        result = dummy_component.broadcast_spatiotemporal(temporal, spatial)

        # Output should be un-flattened: (climate, process, years, site)
        assert result.ndim == 4
        assert result.shape == (2, 3, 4, 5)
        assert result.compute()[0, 0, 0, 0] == 4.0

    def test_broadcast_3d_temporal_2d_spatial(self, dummy_component):
        """Test broadcasting 3D temporal (climate, process, years) with 2D spatial (total_members, lat, lon)."""
        # Shape: (2 climate_members, 3 process_members, 4 years)
        temporal = da.ones((2, 3, 4))

        # Spatial comes in flattened: (6 members, 5 lats, 6 lons)
        spatial = da.ones((6, 5, 6)) * 5

        result = dummy_component.broadcast_spatiotemporal(temporal, spatial)

        # Output should be un-flattened: (climate, process, years, lat, lon)
        assert result.ndim == 5
        assert result.shape == (2, 3, 4, 5, 6)
        assert result.compute()[0, 0, 0, 0, 0] == 5.0

    def test_invalid_spatial_dimensions(self, dummy_component):
        """Test ValueError when spatial dimensions are invalid."""
        temporal = da.ones((2, 3))
        spatial = da.ones((2,))  # Missing spatial dimension entirely

        with pytest.raises(ValueError, match="Unexpected spatial dimensions"):
            dummy_component.broadcast_spatiotemporal(temporal, spatial)

    def test_invalid_temporal_dimensions(self, dummy_component):
        """Test ValueError when temporal dimensions are invalid."""
        temporal = da.ones((2,))  # 1D temporal array
        spatial = da.ones((2, 4))

        with pytest.raises(ValueError, match="Unexpected temporal dimensions"):
            dummy_component.broadcast_spatiotemporal(temporal, spatial)
