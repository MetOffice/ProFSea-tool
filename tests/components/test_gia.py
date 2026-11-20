from pathlib import Path
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from profsea.components.core.base import SpatialComponent
from profsea.components.core.state import LocalState, SpatialState
from profsea.components.spatial.gia import GIA


def get_dummy_spatial_state(
    num_output_members: int = 4,
    n_years: int = 3,
    use_target_points: bool = True,
    output_percentiles: list | None = None,
) -> LocalState | SpatialState:
    """Build a complete spatial or local Monte Carlo state."""
    kwargs = dict(
        n_years=n_years,
        num_output_members=num_output_members,
        num_localMC_members=4,
        output_percentiles=output_percentiles,
        baseline_yrs=(1995, 2014),
        endofhistory=2006,
        random_fraction=np.random.default_rng(42).uniform(size=4),
    )
    if use_target_points:
        return LocalState(
            target_lats=[5],
            target_lons=[45],
            interpolation_method="bilinear",
            **kwargs,
        )
    return SpatialState(
        grid_lats=np.array([0, 10, 20]),
        grid_lons=np.array([-180, 0, 180]),
        grid_interpolation="linear",
        **kwargs,
    )


@pytest.fixture(autouse=True)
def block_assets():
    with patch.object(SpatialComponent, "fetch_assets"):
        yield


class TestGIAInit:
    @patch.object(SpatialComponent, "fetch_assets")
    @patch("pathlib.Path.glob", return_value=[Path("dummy.nc")])
    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_fetches_assets_on_initialisation(
        self,
        mock_exists,
        mock_isdir,
        mock_glob,
        mock_fetch_assets,
    ):
        """Should fetch profsea assets before resolving GIA files."""
        GIA()

        mock_fetch_assets.assert_called_once()

    @patch("pathlib.Path.glob", return_value=[Path("dummy.nc")])
    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_default_gia_paths(self, mock_exists, mock_isdir, mock_glob):
        """Should fall back to default GIA_DIR and resolve files."""
        gia = GIA()
        assert len(gia.gia_files) == 1
        assert gia.gia_files[0] == Path("dummy.nc")
        assert isinstance(gia.global_projection, da.Array)
        assert gia.sample_spatial is True

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
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("sample_spatial", [False, True])
    @pytest.mark.parametrize("at_sites", [False, True])
    @patch.object(GIA, "_load_and_interpolate_rates")
    def test_project_aligns_chunks_before_broadcast(
        self, mock_load, mocked_gia, sample_spatial, at_sites, dtype
    ):
        state = get_dummy_spatial_state(use_target_points=at_sites)
        state.dtype = dtype
        spatial_shape = (1,) if at_sites else (3, 3)
        patterns = np.arange(2 * np.prod(spatial_shape), dtype=float).reshape(
            2, *spatial_shape
        )
        mock_load.return_value = da.from_array(patterns, chunks=1)
        mocked_gia.sample_spatial = sample_spatial

        with patch.object(
            mocked_gia,
            "broadcast_spatiotemporal",
            wraps=mocked_gia.broadcast_spatiotemporal,
        ) as mock_broadcast:
            result = mocked_gia.project(state, np.random.default_rng(42))

        temporal, spatial = mock_broadcast.call_args.args
        assert temporal.dtype == spatial.dtype == np.dtype(dtype)

        selected = (
            patterns[(state.random_fraction * 2).astype(int)]
            if sample_spatial
            else np.broadcast_to(patterns.mean(axis=0), (4, *spatial_shape))
        )
        expected = selected[:, None] * np.array([0.001, 0.002, 0.003]).reshape(
            (1, 3) + (1,) * len(spatial_shape)
        )
        assert result.shape == (4, 3, *spatial_shape)
        assert result.dtype == np.dtype(dtype)
        assert result.chunks == (
            (4,),
            (1, 1, 1),
            *((1,) * size for size in spatial_shape),
        )
        np.testing.assert_allclose(result.compute(), expected)

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
    def test_project_preserves_local_mc_axis(self, mock_load, mocked_gia):
        """Return the local Monte Carlo axis without restoring global axes."""
        state = get_dummy_spatial_state(num_output_members=4, n_years=3)
        mock_load.return_value = da.array([[10.0]])  # 1 pattern, 1 site
        rng = np.random.default_rng(42)

        # Mock the broadcaster to return a flattened 3D array (members, years, sites)
        dummy_bcast_out = da.ones((4, 3, 1))

        with patch.object(
            mocked_gia, "broadcast_spatiotemporal", return_value=dummy_bcast_out
        ):
            result = mocked_gia.project(state, rng)

            assert result.shape == (4, 3, 1)

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
