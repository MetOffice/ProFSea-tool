import pytest
import numpy as np
import dask.array as da
from pathlib import Path
from unittest.mock import patch

from profsea.components.spatial.gia import GIA
from profsea.components.spatial.fingerprint import Fingerprint
from profsea.components.core.state import SpatialState, LocalState


@pytest.fixture
def real_spatial_state():
    """Returns a fully instantiated SpatialState using the real constructor."""
    return SpatialState(
        grid_lats=np.array([0, 10, 20]),
        grid_lons=np.array([-180, 0, 180]),
        n_years=3,
        nt=2,
        num_members=2,
        num_output_members=4,  # nt * num_members
        grid_interpolation="bilinear",
        output_percentiles=None,
        baseline_yrs=(1995, 2014),
        endofhistory=2006,
    )


@pytest.fixture
def real_local_state():
    """Returns a fully instantiated LocalState using the real constructor."""
    return LocalState(
        target_lats=[5.0, 15.0],
        target_lons=[45.0, 55.0],
        n_years=3,
        nt=2,
        num_members=2,
        num_output_members=4,  # nt * num_members
        interpolation_method="bilinear",
        output_percentiles=None,
        baseline_yrs=(1995, 2014),
        endofhistory=2006,
    )


class TestComponentIntegration:
    @patch("profsea.components.spatial.gia.GIA._load_and_interpolate_rates")
    def test_gia_spatial_integration(self, mock_load, real_spatial_state):
        """Test that GIA can process a real SpatialState (Grid)."""
        # Mocking a 2D spatial grid (lat, lon) -> shape (1 pattern, 3 lats, 3 lons)
        mock_load.return_value = da.ones((1, 3, 3))

        # Mocking filesystem checks so we don't need real NetCDF files
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.glob", return_value=[Path("dummy.nc")]),
        ):
            gia = GIA(sample_spatial=False)

        # Should execute successfully without throwing attribute errors
        result = gia.project(real_spatial_state, np.random.default_rng(42))

        # Expected shape when output_percentiles is None: (nt, num_members, years, lats, lons)
        assert isinstance(result, da.Array)
        assert result.ndim == 5
        assert result.shape == (
            real_spatial_state.nt,
            real_spatial_state.num_members,
            real_spatial_state.n_years,
            3,
            3,
        )

    @patch("profsea.components.spatial.gia.GIA._load_and_interpolate_rates")
    def test_gia_local_integration(self, mock_load, real_local_state):
        """Test that GIA can process a real LocalState (Points)."""
        # Mocking a 1D spatial grid (sites) -> shape (1 pattern, 2 sites)
        mock_load.return_value = da.array([[10.0, 20.0]])

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.glob", return_value=[Path("dummy.nc")]),
        ):
            gia = GIA(sample_spatial=False)

        result = gia.project(real_local_state, np.random.default_rng(42))

        # Expected shape when output_percentiles is None: (nt, num_members, years, sites)
        assert isinstance(result, da.Array)
        assert result.ndim == 4
        assert result.shape == (
            real_local_state.nt,
            real_local_state.num_members,
            real_local_state.n_years,
            2,
        )

    @patch("profsea.components.spatial.fingerprint.Fingerprint._load_and_interpolate")
    def test_fingerprint_spatial_integration(self, mock_load, real_spatial_state):
        """Test that Fingerprint can process a real SpatialState with 3D global projections."""
        import xarray as xr

        # 3D global projection: (climate_members, process_members, time)
        global_proj = xr.DataArray(
            np.ones(
                (
                    real_spatial_state.nt,
                    real_spatial_state.num_members,
                    real_spatial_state.n_years,
                )
            ),
            dims=["climate_member", "process_member", "year"],
        )

        # Mock spatial fingerprint: 1 pattern, 3 lats, 3 lons
        mock_load.return_value = da.ones((1, 3, 3))

        with patch("pathlib.Path.exists", return_value=True):
            fp = Fingerprint(
                global_projection=global_proj,
                fingerprint_component="custom",
                fingerprint_paths=["dummy.nc"],
                sample_spatial=False,
            )

        result = fp.project(real_spatial_state, np.random.default_rng(42))

        # The temporal_array (3D) and spatial_array (2D) should combine into a 5D array
        # Shape: (climate_members, process_members, years, lats, lons)
        assert isinstance(result, da.Array)
        assert result.ndim == 5
        assert result.shape == (
            real_spatial_state.nt,
            real_spatial_state.num_members,
            real_spatial_state.n_years,
            3,
            3,
        )
