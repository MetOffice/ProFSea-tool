import pytest
import numpy as np
import xarray as xr
import dask.array as da
from pathlib import Path
from unittest.mock import patch

from profsea.components.spatial.fingerprint import Fingerprint, FP_PATH_MAP
from profsea.components.core.state import SpatialState


def get_dummy_spatial_state(
    num_output_members: int | None = None,
    num_members: int = 2,
    output_percentiles: list | None = None,
    use_target_points: bool = True,
) -> SpatialState:
    """
    Helper to generate a SpatialState for fingerprint testing.
    Uses __new__ to bypass potential validation/I_O in the real __init__.
    """
    state = SpatialState.__new__(SpatialState)
    state.num_output_members = num_output_members
    state.num_members = num_members
    state.output_percentiles = output_percentiles

    if use_target_points:
        state.target_lats = np.array([5])
        state.target_lons = np.array([45])
    else:
        # If testing interpolation to grid rather than specific points
        state.grid_lats = np.array([0, 10, 20])
        state.grid_lons = np.array([-180, 0, 180])

    return state


@pytest.fixture
def sample_global_proj():
    """Returns a dummy global projection of shape (climate_members=2, process_members=2, years=3)."""
    data = np.arange(12).reshape(2, 2, 3).astype(float)
    return xr.DataArray(data, dims=["climate_member", "process_member", "year"])


class TestFingerprintInit:
    def test_default_paths(self, sample_global_proj):
        """Default paths should load from FP_PATH_MAP accurately."""
        fp = Fingerprint(
            global_projection=sample_global_proj,
            fingerprint_component="greenland",
        )
        assert len(fp.fp_paths) == len(FP_PATH_MAP["greenland"])
        assert isinstance(fp.fp_paths[0], Path)
        assert fp.fp_paths[0].name == "greenland_ar6.nc"

    def test_invalid_default_component(self, sample_global_proj):
        """Requesting an unmapped component should raise a ValueError."""
        with pytest.raises(ValueError, match="No default fingerprint paths found"):
            Fingerprint(
                global_projection=sample_global_proj,
                fingerprint_component="invalid_ice_sheet",
            )

    @patch("pathlib.Path.exists", return_value=True)
    def test_custom_paths_string(self, mock_exists, sample_global_proj):
        """Custom string paths should be normalized into Path objects."""
        fp = Fingerprint(
            global_projection=sample_global_proj,
            fingerprint_component="custom",
            fingerprint_paths="/fake/path/fp.nc",
        )
        assert len(fp.fp_paths) == 1
        assert fp.fp_paths[0] == Path("/fake/path/fp.nc")
        mock_exists.assert_called_once()

    @patch("pathlib.Path.exists", return_value=False)
    def test_custom_paths_missing(self, mock_exists, sample_global_proj):
        """Missing custom files should raise a FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Missing fingerprint file"):
            Fingerprint(
                global_projection=sample_global_proj,
                fingerprint_component="custom",
                fingerprint_paths=["/fake/path/fp.nc"],
            )


class TestFingerprintLoadAndInterpolate:
    @patch("xarray.open_dataarray")
    @patch("profsea.components.spatial.fingerprint.Fingerprint.extract_spatial")
    def test_load_and_interpolate_scaling(
        self, mock_extract, mock_open, sample_global_proj
    ):
        """Fingerprints should be loaded, spatially extracted, and scaled correctly."""
        state = get_dummy_spatial_state()

        mock_da = xr.DataArray(np.array([[1, 2], [3, 4]]))
        mock_open.return_value = mock_da

        # Emulate extract_spatial returning a 1D site array
        mock_extract.return_value = xr.DataArray(np.array([5.0]))

        with patch("pathlib.Path.exists", return_value=True):
            fp = Fingerprint(
                global_projection=sample_global_proj,
                fingerprint_component="custom",
                fingerprint_paths=["fake1.nc", "fake2.nc"],
                scaling_factor=2.0,
            )

        result = fp._load_and_interpolate(state)

        # 2 files stacked -> shape (2, 1); scaled by 2.0 -> 5.0 * 2.0 = 10.0
        assert isinstance(result, da.Array)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result.compute(), [[10.0], [10.0]])
        assert mock_extract.call_count == 2


class TestFingerprintProject:
    @patch("profsea.components.spatial.fingerprint.Fingerprint._load_and_interpolate")
    def test_project_single_fingerprint(self, mock_load, sample_global_proj):
        """Projection should broadcast properly when only 1 fingerprint is mapped."""
        # For a 2x2 global projection array, total output members = 4
        state = get_dummy_spatial_state(num_output_members=4)
        mock_load.return_value = da.array([[1.0, 2.0, 3.0]])

        fp = Fingerprint(
            global_projection=sample_global_proj, fingerprint_component="greenland"
        )

        rng = np.random.default_rng(42)

        with patch.object(
            fp, "broadcast_spatiotemporal", return_value="done"
        ) as mock_bcast:
            result = fp.project(state, rng)

            assert result == "done"
            mock_bcast.assert_called_once()

            _, selected_fps = mock_bcast.call_args[0]

            # The single FP should be broadcast across all 4 flattened members
            assert selected_fps.shape == (4, 3)
            np.testing.assert_allclose(
                selected_fps.compute(),
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            )

    @patch("profsea.components.spatial.fingerprint.Fingerprint._load_and_interpolate")
    def test_project_storyline_mode(self, mock_load, sample_global_proj):
        """Storyline mode should collapse fingerprints into an unweighted mean."""
        state = get_dummy_spatial_state(num_output_members=4)
        mock_load.return_value = da.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        with patch("pathlib.Path.exists", return_value=True):
            fp = Fingerprint(
                global_projection=sample_global_proj,
                fingerprint_component="custom",
                fingerprint_paths=["1.nc", "2.nc", "3.nc"],
                sample_spatial=False,
            )

        rng = np.random.default_rng(42)

        with patch.object(fp, "broadcast_spatiotemporal") as mock_bcast:
            fp.project(state, rng)

            _, selected_fps = mock_bcast.call_args[0]

            # Mean of [1,3,5] is 3; Mean of [2,4,6] is 4.
            assert selected_fps.shape == (4, 2)
            np.testing.assert_allclose(
                selected_fps.compute(), [[3.0, 4.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]
            )

    @patch("profsea.components.spatial.fingerprint.Fingerprint._load_and_interpolate")
    def test_project_probabilistic_mode(self, mock_load, sample_global_proj):
        """Probabilistic mode should assign distinct random fingerprints to members."""
        state = get_dummy_spatial_state(num_output_members=4)

        # 3D array representing (n_fps, lat, lon)
        fps = np.array(
            [
                [[1.0, 1.0]],  # FP 0 (shape 1x2)
                [[2.0, 2.0]],  # FP 1
                [[3.0, 3.0]],  # FP 2
            ]
        )
        mock_load.return_value = da.array(fps)

        with patch("pathlib.Path.exists", return_value=True):
            fp = Fingerprint(
                global_projection=sample_global_proj,
                fingerprint_component="custom",
                fingerprint_paths=["1.nc", "2.nc", "3.nc"],
                sample_spatial=True,
            )

        rng = np.random.default_rng(42)

        with patch.object(fp, "broadcast_spatiotemporal") as mock_bcast:
            fp.project(state, rng)

            _, selected_fps = mock_bcast.call_args[0]

            assert selected_fps.shape == (4, 1, 2)
            # Make sure we got a mix of indices
            unique_values = np.unique(selected_fps.compute())
            assert len(unique_values) > 1

    @patch("profsea.components.spatial.fingerprint.reformat_global_projection")
    @patch("profsea.components.spatial.fingerprint.Fingerprint._load_and_interpolate")
    def test_project_with_percentiles(
        self, mock_load, mock_reformat, sample_global_proj
    ):
        """State requesting percentiles should trigger the global array reformatting."""
        state = get_dummy_spatial_state(
            output_percentiles=[5, 50, 95], num_output_members=3
        )
        mock_load.return_value = da.array([[1.0]])
        mock_reformat.return_value = da.array([[99.0]])

        fp = Fingerprint(
            global_projection=sample_global_proj, fingerprint_component="greenland"
        )

        with patch.object(fp, "broadcast_spatiotemporal") as mock_bcast:
            fp.project(state, np.random.default_rng(42))

            mock_reformat.assert_called_once_with(fp.global_projection, state)

            global_proj, _ = mock_bcast.call_args[0]
            assert global_proj.compute() == 99.0
