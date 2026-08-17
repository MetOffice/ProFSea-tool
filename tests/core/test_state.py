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
        num_members=2,
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
        num_members=2,
        interpolation_method="bilinear",
        output_percentiles=None,
        baseline_yrs=(1995, 2014),
        endofhistory=2006,
    )


class TestComponentIntegration:
    @patch("profsea.components.spatial.gia.GIA._load_and_interpolate_rates")
    def test_gia_spatial_integration(self, mock_load, real_spatial_state):
        """Test that GIA can process a real SpatialState (Grid)."""
        # Note: This test will fail with an AttributeError until `endofhistory`
        # is added to the SpatialState dataclass.

        mock_load.return_value = da.array([[10.0, 20.0, 30.0]])  # Dummy spatial grid

        # Mocking filesystem checks so we don't need real NetCDF files
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.glob", return_value=[Path("dummy.nc")]),
        ):
            gia = GIA(sample_spatial=False)

        # Should execute successfully without throwing attribute errors
        result = gia.project(real_spatial_state, np.random.default_rng(42))

        # Expected shape: (members, years, lats, lons)
        # Because mock_load returned 1D array here to simplify, broadcast handles it
        assert isinstance(result, da.Array)
        assert result.shape[0] == real_spatial_state.num_members
        assert result.shape[1] == real_spatial_state.n_years

    @patch("profsea.components.spatial.gia.GIA._load_and_interpolate_rates")
    def test_gia_local_integration(self, mock_load, real_local_state):
        """Test that GIA can process a real LocalState (Points)."""
        # Mocking a 2-site extraction
        mock_load.return_value = da.array([[10.0, 20.0]])

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.glob", return_value=[Path("dummy.nc")]),
        ):
            gia = GIA(sample_spatial=False)

        result = gia.project(real_local_state, np.random.default_rng(42))

        # Expected shape: (members, years, sites)
        assert isinstance(result, da.Array)
        assert result.shape == (2, 3, 2)

    @patch("profsea.components.spatial.fingerprint.Fingerprint._load_and_interpolate")
    def test_fingerprint_spatial_integration(self, mock_load, real_spatial_state):
        """Test that Fingerprint can process a real SpatialState."""
        # Dummy global projection: shape (members=2, years=3)
        import xarray as xr

        global_proj = xr.DataArray(np.ones((2, 3)), dims=["member", "year"])

        mock_load.return_value = da.array([[1.0, 1.0, 1.0]])

        with patch("pathlib.Path.exists", return_value=True):
            fp = Fingerprint(
                global_projection=global_proj,
                fingerprint_component="custom",
                fingerprint_paths=["dummy.nc"],
                sample_spatial=False,
            )

        result = fp.project(real_spatial_state, np.random.default_rng(42))

        assert isinstance(result, da.Array)
        assert result.shape[0] == real_spatial_state.num_members
