from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from profsea.utils import (
    check_shapes,
    fetch_zenodo_fingerprints,
    interpolate,
    interpolate_to_grid,
    reformat_global_projection,
    sample_members_2D,
    save_components,
)


class DummyClimateState:
    """Mock state object for testing projection reformatting."""

    def __init__(self, percentiles=None, nt=2, num_members=3, n_years=4):
        self.output_percentiles = percentiles
        self.nt = nt
        self.num_members = num_members
        self.n_years = n_years


# --- check_shapes ---


def test_check_shapes_valid_2d():
    arr = np.zeros((5, 10))
    check_shapes(arr, n_time=10)


def test_check_shapes_promotes_1d():
    arr = np.zeros(10)
    check_shapes(arr, n_time=10)


def test_check_shapes_raises_value_error():
    arr = np.zeros((5, 10))
    with pytest.raises(
        ValueError, match="Array should have shape .* time dimension of length 12"
    ):
        check_shapes(arr, n_time=12)


# --- interpolate & interpolate_to_grid ---


def test_interpolate():
    # interpolate expects an xarray DataArray (despite the da.array type hint in the signature)
    # because it accesses data.dims and data.data
    data = xr.DataArray(
        da.ones((10, 20)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-90, 90, 10), "lon": np.linspace(-180, 180, 20)},
    )
    result = interpolate(data, lats=180, lons=360)

    assert isinstance(result, np.ndarray) or isinstance(result, da.Array)
    assert result.shape == (180, 360)


def test_interpolate_to_grid_longitude_wrapping():
    # Target grid using -180 to 180 format (includes 180, which should wrap)
    target_lats = np.array([-45, 0, 45])
    target_lons = np.array([-10, 0, 180])

    # Setup mock regionmask to act as if there is no land
    mock_regionmask = MagicMock()
    mock_land = MagicMock()
    mock_mask_da = xr.DataArray(
        np.zeros((1, 3, 3), dtype=bool),
        dims=["region", "lat", "lon"],
        coords={"region": [0], "lat": target_lats, "lon": [-180, -10, 0]},
    )
    mock_land.mask_3D.return_value = mock_mask_da
    mock_regionmask.defined_regions.natural_earth_v5_0_0.land_110 = mock_land

    # Setup mock input data
    lats = np.array([-45, 0, 45])
    lons_360 = np.array([0, 180, 350])
    data = xr.DataArray(np.random.rand(3, 3), coords=[("lat", lats), ("lon", lons_360)])

    # Intercept the local import using sys.modules
    with patch.dict("sys.modules", {"regionmask": mock_regionmask}):
        result = interpolate_to_grid(data, target_lats, target_lons)

    assert result.shape == (3, 3)
    expected_lons = np.array([-180, -10, 0])
    np.testing.assert_array_almost_equal(result.lon.values, expected_lons)


# --- sample_members_2D ---


def test_sample_members_2D_extracts_real_member():
    arr = np.array(
        [
            [1.0, 1.0, 1.0],  # Min
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],  # Median
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],  # Max
        ]
    )

    result = sample_members_2D(arr, percentiles=[50])
    assert result.shape == (1, 3)
    np.testing.assert_array_equal(result[0], [3.0, 3.0, 3.0])


def test_sample_members_2D_preserves_lazy_evaluation():
    arr = da.ones((10, 5))
    result = sample_members_2D(arr, percentiles=[5, 50, 95])

    assert isinstance(result, da.Array)
    assert result.shape == (3, 5)


# --- reformat_global_projection ---


def test_reformat_global_projection_none_percentiles():
    """Should return raw array unmodified if percentiles are None or empty."""
    state = DummyClimateState(percentiles=None)
    raw_arr = np.ones((2, 3, 4))

    result = reformat_global_projection(raw_arr, state)
    assert result.shape == (2, 3, 4)
    assert result is raw_arr

    state.output_percentiles = []
    result_empty = reformat_global_projection(raw_arr, state)
    assert result_empty is raw_arr


@patch("profsea.utils.utils.sample_members_2D")
def test_reformat_global_projection_3d_flattens(mock_sample):
    """Should dynamically flatten a 3D array to 2D before sampling."""
    state = DummyClimateState(percentiles=[50])
    raw_arr = np.ones((2, 3, 4))  # (climate_members, process_members, years)

    mock_sample.return_value = "sampled_array"

    result = reformat_global_projection(raw_arr, state)

    # Check that sample_members_2D was called with a flattened (6, 4) array
    flat_args, _ = mock_sample.call_args
    passed_array = flat_args[0]

    assert passed_array.shape == (6, 4)
    assert result == "sampled_array"


@patch("profsea.utils.utils.sample_members_2D")
def test_reformat_global_projection_2d_direct_sample(mock_sample):
    """Should sample directly if array is already 2D."""
    state = DummyClimateState(percentiles=[5, 95])
    raw_arr = np.ones((6, 4))

    mock_sample.return_value = "sampled_array"
    result = reformat_global_projection(raw_arr, state)

    flat_args, _ = mock_sample.call_args
    passed_array = flat_args[0]

    assert passed_array.shape == (6, 4)
    assert result == "sampled_array"


# --- fetch_zenodo_fingerprints ---


def test_fetch_zenodo_early_return(tmp_path, capsys):
    """Should return immediately if the target directory exists and is populated."""
    target_dir = tmp_path / "profsea-assets"
    target_dir.mkdir()
    (target_dir / "dummy_file.nc").touch()

    fetch_zenodo_fingerprints("http://fake.url", tmp_path, "profsea-assets")

    # Rich logger writes to stdout, so we use capsys instead of caplog
    captured = capsys.readouterr()
    assert "ProFSea assets found locally!" in captured.out


@patch("requests.get")
@patch("zipfile.ZipFile")
def test_fetch_zenodo_downloads_and_extracts(mock_zip, mock_get, tmp_path):
    """Should download and extract if the directory is missing."""
    mock_response = MagicMock()
    mock_response.headers = {"content-length": "100"}
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_get.return_value = mock_response

    mock_zip_instance = MagicMock()
    mock_zip_instance.namelist.return_value = ["file1.nc", "__MACOSX/hidden", "._file2"]
    mock_zip.return_value.__enter__.return_value = mock_zip_instance

    fetch_zenodo_fingerprints("http://fake.url", tmp_path, "profsea-assets")

    mock_get.assert_called_once_with("http://fake.url", stream=True)
    # Ensure hidden macOS files are filtered out during extraction
    mock_zip_instance.extractall.assert_called_once_with(tmp_path, members=["file1.nc"])
    # Zip file should be cleaned up
    assert not (tmp_path / "temp_fingerprints.zip").exists()


# --- save_components ---


class DummyInstance:
    """Mock instance to pass as 'self' to save_components."""

    pass


def test_save_components_empty_dict(capsys):
    """Should return early and warn if no components are provided."""
    save_components(DummyInstance(), {}, "ssp119")

    captured = capsys.readouterr()
    # Check both stdout and stderr since rich handles warnings differently
    assert "No components provided to save" in captured.out + captured.err


def test_save_components_invalid_format():
    """Should reject unknown output formats."""
    components = {"comp1": xr.DataArray([1, 2, 3])}
    with pytest.raises(
        ValueError, match="output_format must be either 'netcdf' or 'zarr'"
    ):
        save_components(DummyInstance(), components, "ssp119", output_format="csv")


def test_save_components_netcdf(tmp_path):
    """Should successfully stream and save a NetCDF file."""
    components = {
        "comp1": xr.DataArray(np.ones((2, 3)), dims=["x", "y"]),
    }

    save_components(
        DummyInstance(),
        components,
        scenario_name="ssp245",
        output_prefix="test",
        output_dir=str(tmp_path),
        output_format="netcdf",
    )

    expected_file = tmp_path / "ssp245_test.nc"
    assert expected_file.exists()

    ds = xr.open_dataset(expected_file)
    assert "comp1" in ds
    assert ds.attrs["scenario"] == "ssp245"
    assert ds.attrs["source"] == "ProFSea v3.0"


def test_save_components_zarr(tmp_path):
    """Should successfully stream and save a Zarr store."""
    components = {
        "comp1": xr.DataArray(np.ones((2, 3)), dims=["x", "y"]),
    }

    save_components(
        DummyInstance(),
        components,
        scenario_name="ssp585",
        output_prefix="global",
        output_dir=str(tmp_path),
        output_format="zarr",
    )

    expected_store = tmp_path / "ssp585_global.zarr"
    assert expected_store.exists()
    assert expected_store.is_dir()

    ds = xr.open_zarr(expected_store)
    assert "comp1" in ds
