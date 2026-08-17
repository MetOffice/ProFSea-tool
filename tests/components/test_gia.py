from pathlib import Path
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from profsea.components.core.state import SpatialState
from profsea.components.spatial.gia import GIA


def get_dummy_spatial_state(
    num_output_members: int = 4,
    nt: int = 2,
    num_members: int = 2,
    n_years: int = 3,
    use_target_points: bool = True,
    output_percentiles: list | None = None,
) -> SpatialState:
    """Helper to generate a SpatialState for GIA testing."""
    state = SpatialState.__new__(SpatialState)
    state.num_output_members = num_output_members
    state.nt = nt
    state.num_members = num_members
    state.n_years = n_years
    state.baseline_yrs = [1995, 2014]  # 20-year period, midpoint 2005
    state.endofhistory = 2006
    state.output_percentiles = output_percentiles

    if use_target_points:
        state.target_lats = np.array([5])
        state.target_lons = np.array([45])
    else:
        state.grid_lats = np.array([0, 10, 20])
        state.grid_lons = np.array([-180, 0, 180])

    return state


class TestGIAInit:
    @patch("pathlib.Path.glob", return_value=[Path("dummy.nc")])
    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_default_gia_paths(self, mock_exists, mock_isdir, mock_glob):
        """Should fall back to default GIA_DIR and resolve files."""
        gia = GIA()
        assert len(gia.gia_files) == 1
        assert gia.gia_files[0] == Path("dummy.nc")
        assert isinstance(gia.global_projection, da.Array)
        assert gia.sample_spatial is False

    @patch("pathlib.Path.suffix", new_callable=lambda: ".nc")
    @patch("pathlib.Path.is_file", return_value=True)
    @patch("pathlib.Path.is_dir", return_value=False)
    @patch("pathlib.Path.exists", return_value=True)
    def test_custom_gia_paths_direct_file(
        self, mock_exists, mock_isdir, mock_isfile, mock_suffix
    ):
        """Should correctly resolve a direct file path instead of a directory."""
        custom_path = "/fake/gia_model.nc"
        with patch.object(Path, "suffix", ".nc"):
            gia = GIA(gia_paths=custom_path, sample_spatial=True)

        assert len(gia.gia_files) == 1
        assert gia.gia_files[0] == Path(custom_path)
        assert gia.sample_spatial is True

    @patch("pathlib.Path.glob", return_value=[])
    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_no_files_raises_error(self, mock_exists, mock_isdir, mock_glob):
        """Should raise FileNotFoundError during init if the directory has no .nc files."""
        with pytest.raises(FileNotFoundError, match="No GIA NetCDF files found"):
            GIA(gia_paths="/empty/dir")


class TestGIALoadAndInterpolateRates:
    @patch("xarray.open_dataarray")
    @patch("profsea.components.spatial.gia.GIA.extract_spatial")
    @patch("pathlib.Path.glob", return_value=[Path("file1.nc"), Path("file2.nc")])
    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_and_prepend_dimensions(
        self, mock_exists, mock_isdir, mock_glob, mock_extract, mock_open
    ):
        """Should correctly append 'model' dimension if missing and stack files."""
        state = get_dummy_spatial_state(use_target_points=True)
        mock_extract.return_value = xr.DataArray(np.array([5.0]))

        gia = GIA(gia_paths="/fake/dir")
        result = gia._load_and_interpolate_rates(state)

        # 2 files, 1 site per file -> shape (2, 1)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result.compute(), [[5.0], [5.0]])


class TestGIAProject:
    @pytest.fixture
    def mocked_gia(self):
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.glob", return_value=[Path("dummy.nc")]),
        ):
            return GIA()

    @pytest.fixture
    def mocked_gia_storyline(self):
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.glob", return_value=[Path("dummy.nc")]),
        ):
            return GIA(sample_spatial=False)

    @pytest.fixture
    def mocked_gia_probabilistic(self):
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.glob", return_value=[Path("dummy.nc")]),
        ):
            return GIA(sample_spatial=True)

    @patch("profsea.components.spatial.gia.GIA._load_and_interpolate_rates")
    def test_project_reshapes_to_5d(self, mock_load, mocked_gia):
        """Should reshape the flattened 4D broadcast array into a 5D array."""
        state = get_dummy_spatial_state(
            num_output_members=4, nt=2, num_members=2, n_years=3
        )
        mock_load.return_value = da.array([[10.0]])  # 1 pattern, 1 site
        rng = np.random.default_rng(42)

        # Mock the broadcaster to return a flattened 3D array (members, years, sites)
        dummy_bcast_out = da.ones((4, 3, 1))

        with patch.object(
            mocked_gia, "broadcast_spatiotemporal", return_value=dummy_bcast_out
        ):
            result = mocked_gia.project(state, rng)

            # The final result should be reshaped to 4D (climate, process, years, sites)
            assert result.shape == (2, 2, 3, 1)

    @patch("profsea.components.spatial.gia.GIA._load_and_interpolate_rates")
    def test_project_time_vector_logic(self, mock_load, mocked_gia):
        """Should accurately calculate the accumulation time vector."""
        state = get_dummy_spatial_state(num_output_members=4, n_years=3)
        mock_load.return_value = da.array([[10.0]])  # single pattern, single site
        rng = np.random.default_rng(42)

        dummy_bcast_out = da.ones((4, 3, 1))

        with patch.object(
            mocked_gia, "broadcast_spatiotemporal", return_value=dummy_bcast_out
        ) as mock_bcast:
            mocked_gia.project(state, rng)

            temporal_array, _ = mock_bcast.call_args[0]

            assert temporal_array.shape == (4, 3)  # (num_output_members, years)
            expected_time_series = [0.001, 0.002, 0.003]
            np.testing.assert_allclose(
                temporal_array.compute(), [expected_time_series] * 4
            )

    @patch("profsea.components.spatial.gia.GIA._load_and_interpolate_rates")
    def test_project_single_pattern(self, mock_load, mocked_gia):
        """Should broadcast automatically if only 1 GIA pattern exists."""
        state = get_dummy_spatial_state(num_output_members=4, n_years=1)
        mock_load.return_value = da.array([[10.0, 20.0]])  # 1 pattern, 2 sites

        dummy_bcast_out = da.ones((4, 1, 2))

        with patch.object(
            mocked_gia, "broadcast_spatiotemporal", return_value=dummy_bcast_out
        ) as mock_bcast:
            mocked_gia.project(state, np.random.default_rng(42))

            _, spatial_array = mock_bcast.call_args[0]
            assert spatial_array.shape == (4, 2)  # (num_output_members, sites)
            np.testing.assert_allclose(spatial_array.compute(), [[10.0, 20.0]] * 4)

    @patch("profsea.components.spatial.gia.GIA._load_and_interpolate_rates")
    def test_project_storyline_mode_with_nans(self, mock_load, mocked_gia_storyline):
        """Should take the nanmean of spatial patterns in storyline mode."""
        state = get_dummy_spatial_state(num_output_members=4, n_years=1)
        mock_load.return_value = da.array([[10.0, 20.0], [np.nan, 40.0]])

        dummy_bcast_out = da.ones((4, 1, 2))

        with patch.object(
            mocked_gia_storyline,
            "broadcast_spatiotemporal",
            return_value=dummy_bcast_out,
        ) as mock_bcast:
            mocked_gia_storyline.project(state, np.random.default_rng(42))

            _, spatial_array = mock_bcast.call_args[0]

            np.testing.assert_allclose(spatial_array.compute(), [[10.0, 30.0]] * 4)

    @patch("profsea.components.spatial.gia.GIA._load_and_interpolate_rates")
    def test_project_probabilistic_mode(self, mock_load, mocked_gia_probabilistic):
        """Should randomly sample patterns in probabilistic mode."""
        state = get_dummy_spatial_state(num_output_members=4, n_years=1)
        fps = np.array(
            [
                [[10.0, 10.0]],
                [[20.0, 20.0]],
                [[30.0, 30.0]],
            ]
        )
        mock_load.return_value = da.array(fps)
        rng = np.random.default_rng(42)

        dummy_bcast_out = da.ones((4, 1, 2))

        with patch.object(
            mocked_gia_probabilistic,
            "broadcast_spatiotemporal",
            return_value=dummy_bcast_out,
        ) as mock_bcast:
            mocked_gia_probabilistic.project(state, rng)

            _, spatial_array = mock_bcast.call_args[0]

            assert spatial_array.shape == (4, 1, 2)
            unique_values = np.unique(spatial_array.compute())
            assert len(unique_values) > 1
