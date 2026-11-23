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
    n_years: int = 3,
    use_target_points: bool = True,
    output_percentiles: list | None = None,
) -> LocalState | SpatialState:
    """Build a complete spatial or local Monte Carlo state with four members."""
    kwargs = dict(
        n_years=n_years,
        num_spatial_members=4,
        output_percentiles=output_percentiles,
        baseline_yrs=(1995, 2014),
        endofhistory=2006,
        member_indices=np.arange(4),
        fingerprint_fraction=np.array([0.1, 0.6, 0.9, 0.4]),
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
        assert gia.global_projection is None  # GIA has no global driver
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
    @pytest.fixture
    def mocked_gia(self):
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.glob", return_value=[Path("dummy.nc")]),
        ):
            return GIA()

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("sample_spatial", [False, True])
    @pytest.mark.parametrize("at_sites", [False, True])
    @patch.object(GIA, "_load_and_interpolate_rates")
    def test_project_values_dtype_and_chunks(
        self, mock_load, mocked_gia, sample_spatial, at_sites, dtype
    ):
        """Accumulated GIA = years since the baseline midpoint (mm to m) x rate."""
        state = get_dummy_spatial_state(use_target_points=at_sites)
        state.dtype = dtype
        state.time_chunk = 2
        spatial_shape = (1,) if at_sites else (3, 3)
        patterns = np.arange(2 * np.prod(spatial_shape), dtype=float).reshape(
            2, *spatial_shape
        )
        mock_load.return_value = da.from_array(patterns, chunks=1)
        mocked_gia.sample_spatial = sample_spatial

        rng = np.random.default_rng(42)
        separable = mocked_gia.project_separable(state, np.random.default_rng(42))
        result = mocked_gia.project(state, rng)

        if sample_spatial:
            selected = patterns[separable.pattern_index]
        else:
            assert separable.pattern_index is None
            selected = np.broadcast_to(patterns.mean(axis=0), (4, *spatial_shape))

        # Baseline 1995-2014 has midpoint 2005, so 2006 is one year after it
        time_series = np.arange(1, 4) * 0.001
        expected = (
            selected[:, None] * time_series.reshape((1, 3) + (1,) * len(spatial_shape))
        ).astype(dtype)

        assert result.shape == (4, 3, *spatial_shape)
        assert result.dtype == np.dtype(dtype)
        assert result.chunks[:2] == ((4,), (2, 1))
        np.testing.assert_allclose(result.compute(), expected, rtol=1e-6)

    @patch.object(GIA, "_load_and_interpolate_rates")
    def test_project_storyline_mode_with_nans(self, mock_load):
        """Storyline mode takes the nanmean of the GIA patterns."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.glob", return_value=[Path("dummy.nc")]),
        ):
            gia = GIA(sample_spatial=False)
        state = get_dummy_spatial_state(n_years=1)
        mock_load.return_value = da.array([[10.0, 20.0], [np.nan, 40.0]])

        separable = gia.project_separable(state, np.random.default_rng(42))

        np.testing.assert_allclose(separable.patterns.compute(), [[10.0, 30.0]])

    @patch.object(GIA, "_load_and_interpolate_rates")
    def test_project_probabilistic_draws_from_component_rng(
        self, mock_load, mocked_gia
    ):
        """GIA model draws come from the component's own random stream."""
        state = get_dummy_spatial_state(n_years=1)
        mock_load.return_value = da.array([[[10.0]], [[20.0]], [[30.0]]])

        first = mocked_gia.project_separable(state, np.random.default_rng(1))
        second = mocked_gia.project_separable(state, np.random.default_rng(1))

        np.testing.assert_array_equal(first.pattern_index, second.pattern_index)
        assert first.pattern_index.shape == (4,)
        assert set(first.pattern_index) <= {0, 1, 2}
