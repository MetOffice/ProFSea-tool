from pathlib import Path
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from profsea.components.core.base import SpatialComponent
from profsea.components.core.state import LocalState, SpatialState
from profsea.components.spatial.fingerprint import FP_PATH_MAP, Fingerprint


def get_dummy_spatial_state(
    output_percentiles: list | None = None,
    use_target_points: bool = True,
    member_indices: np.ndarray | None = None,
    fingerprint_fraction: np.ndarray | None = None,
) -> LocalState | SpatialState:
    """Build a complete state with four Monte Carlo members."""
    kwargs = dict(
        n_years=3,
        num_spatial_members=4,
        output_percentiles=output_percentiles,
        baseline_yrs=(1995, 2014),
        endofhistory=2006,
        member_indices=(
            np.array([3, 0, 2, 1]) if member_indices is None else member_indices
        ),
        fingerprint_fraction=(
            np.array([0.1, 0.6, 0.9, 0.4])
            if fingerprint_fraction is None
            else fingerprint_fraction
        ),
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


@pytest.fixture
def sample_global_proj():
    """Returns a dummy global projection of shape (climate_members=2, process_members=2, years=3)."""
    data = np.arange(12).reshape(2, 2, 3).astype(float)
    return xr.DataArray(data, dims=["climate_member", "process_member", "year"])


class TestFingerprintInit:
    @pytest.mark.parametrize(
        ("shape", "stored"), [((4, 3), (1, 4, 3)), ((2, 2, 3), (2, 2, 3))]
    )
    def test_global_projection_stored_as_climate_process_time(self, shape, stored):
        """2D (members, time) inputs are treated as a single climate trajectory."""
        data = np.arange(12, dtype=np.float32).reshape(shape)
        fp = Fingerprint(
            global_projection=xr.DataArray(data),
            fingerprint_component="greenland",
        )

        assert fp.global_projection.shape == stored
        np.testing.assert_array_equal(fp.global_projection.reshape(shape), data)

    def test_rejects_percentile_global_projection(self):
        """Percentile curves are not an ensemble that can be sampled."""
        percentiles = xr.DataArray(np.zeros((3, 5)), dims=["percentile", "time"])
        with pytest.raises(ValueError, match="full global ensemble"):
            Fingerprint(percentiles, fingerprint_component="greenland")

    @patch.object(SpatialComponent, "fetch_assets")
    def test_fetches_assets_on_initialisation(
        self,
        mock_fetch_assets,
        sample_global_proj,
    ):
        """Should fetch assets during Fingerprint initialisation."""

        Fingerprint(
            global_projection=sample_global_proj,
            fingerprint_component="greenland",
        )

        mock_fetch_assets.assert_called_once()

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
        assert mock_exists

    @patch.object(SpatialComponent, "fetch_assets")
    @patch("pathlib.Path.exists", return_value=False)
    def test_custom_paths_missing(
        self, mock_exists, mock_fetch_assets, sample_global_proj
    ):
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
    @patch.object(Fingerprint, "_load_and_interpolate")
    def test_project_samples_members_with_single_fingerprint(
        self, mock_load, sample_global_proj
    ):
        """Each member is its sampled global series times the single fingerprint."""
        state = get_dummy_spatial_state()
        mock_load.return_value = da.array([[1.0, 2.0]])

        fp = Fingerprint(sample_global_proj, fingerprint_component="greenland")
        separable = fp.project_separable(state, np.random.default_rng(42))
        result = fp.project(state, np.random.default_rng(42))

        assert separable.pattern_index is None
        flat = sample_global_proj.values.reshape(4, 3)
        expected = flat[state.member_indices][:, :, None] * np.array([1.0, 2.0])
        assert result.shape == (4, 3, 2)
        np.testing.assert_allclose(result.compute(), expected)

    @patch.object(Fingerprint, "_load_and_interpolate")
    def test_project_storyline_mode(self, mock_load, sample_global_proj):
        """Storyline mode should collapse fingerprints into an unweighted mean."""
        state = get_dummy_spatial_state()
        mock_load.return_value = da.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        with patch("pathlib.Path.exists", return_value=True):
            fp = Fingerprint(
                global_projection=sample_global_proj,
                fingerprint_component="custom",
                fingerprint_paths=["1.nc", "2.nc", "3.nc"],
                sample_spatial=False,
            )

        separable = fp.project_separable(state, np.random.default_rng(42))

        assert separable.pattern_index is None
        np.testing.assert_allclose(separable.patterns.compute(), [[3.0, 4.0]])

    @patch.object(Fingerprint, "_load_and_interpolate")
    def test_project_draws_fingerprints_from_shared_fraction(
        self, mock_load, sample_global_proj
    ):
        """Fingerprint draws come from the state's fingerprint fraction, not from
        the member indices, so they are independent of the global member."""
        state = get_dummy_spatial_state()
        patterns = np.array([[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]])
        mock_load.return_value = da.array(patterns)

        with patch("pathlib.Path.exists", return_value=True):
            fp = Fingerprint(
                global_projection=sample_global_proj,
                fingerprint_component="custom",
                fingerprint_paths=["1.nc", "2.nc", "3.nc"],
                sample_spatial=True,
            )

        separable = fp.project_separable(state, np.random.default_rng(42))
        result = fp.project(state, np.random.default_rng(42)).compute()

        expected_index = (state.fingerprint_fraction * 3).astype(int)
        np.testing.assert_array_equal(separable.pattern_index, expected_index)
        flat = sample_global_proj.values.reshape(4, 3)
        expected = (
            flat[state.member_indices][:, :, None, None]
            * patterns[expected_index][:, None]
        )
        np.testing.assert_allclose(result, expected)

    @patch.object(Fingerprint, "_load_and_interpolate")
    def test_project_uses_state_chunk_plan(self, mock_load, sample_global_proj):
        """The member axis stays in one chunk; time and space follow the state."""
        state = get_dummy_spatial_state(use_target_points=False)
        state.time_chunk = 2
        state.spatial_chunks = (2, 2)
        mock_load.return_value = da.ones((2, 3, 3), chunks=(1, 1, 1))

        with patch("pathlib.Path.exists", return_value=True):
            fp = Fingerprint(
                sample_global_proj, "custom", ["a.nc", "b.nc"], sample_spatial=True
            )

        result = fp.project(state, np.random.default_rng(42))
        assert result.chunks == ((4,), (2, 1), (2, 1), (2, 1))
