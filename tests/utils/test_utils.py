from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from profsea.utils import (
    check_shapes,
    fetch_zenodo_fingerprints,
    interpolate,
    interpolate_to_grid,
    read_antarctic_obs,
    read_thermal_expansion_obs,
    rebaseline_projection,
    reformat_global_projection,
    sample_members_2D,
    save_components,
    validate_projection_years,
)
from profsea.utils.utils import _load_zheng_component, _read_imbie3_obs


class DummyClimateState:
    """Mock state object for testing projection reformatting."""

    def __init__(
        self,
        percentiles=None,
        start_yr=2006,
        end_yr=2101,
        baseline_start=1996,
        baseline_end=2014,
        nt=2,
        num_members=3,
        n_years=4,
        dtype=np.float64,
    ):
        self.output_percentiles = percentiles
        self.start_yr = start_yr
        self.end_yr = end_yr
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end
        self.nt = nt
        self.num_members = num_members
        self.n_years = n_years
        self.dtype = dtype


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
    assert ds.attrs["source"] == "ProFSea v3.0.0"


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


def test_validate_projection_years():
    """Valid years should pass with and without observational initialisation."""
    validate_projection_years(
        start_yr=2006,
        end_yr=2101,
        baseline_start=1986,
        baseline_end=2005,
        initialise_from_observations=True,
    )

    validate_projection_years(
        start_yr=1850,
        end_yr=2101,
        baseline_start=1850,
        baseline_end=1900,
        initialise_from_observations=False,
    )


@pytest.mark.parametrize(
    "start_yr,end_yr",
    [
        (2100, 2000),
        (2100, 2100),
    ],
)
def test_validate_projection_years_invalid_projection_period(start_yr, end_yr):
    with pytest.raises(ValueError, match="must be earlier than end_yr"):
        validate_projection_years(
            start_yr=start_yr,
            end_yr=end_yr,
            baseline_start=1986,
            baseline_end=2005,
        )


@pytest.mark.parametrize(
    "baseline_start,baseline_end",
    [
        (2005, 1986),
        (2005, 2005),
    ],
)
def test_validate_projection_years_invalid_baseline_period(
    baseline_start,
    baseline_end,
):
    with pytest.raises(ValueError, match="must be earlier than baseline_end"):
        validate_projection_years(
            start_yr=2001,
            end_yr=2100,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
        )


@pytest.mark.parametrize(
    "start_yr,baseline_start,baseline_end",
    [
        (1979, 1986, 2005),
        (2001, 1979, 2005),
        (2024, 1986, 2005),
        (2001, 1986, 2024),
    ],
)
def test_validate_projection_years_observational_limits(
    start_yr,
    baseline_start,
    baseline_end,
):
    with pytest.raises(ValueError):
        validate_projection_years(
            start_yr=start_yr,
            end_yr=2100,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            initialise_from_observations=True,
        )


@pytest.mark.parametrize(
    "baseline_start,baseline_end",
    [
        (1849, 1900),
        (1850, 2101),
    ],
)
def test_validate_projection_years_baseline_outside_projection(
    baseline_start,
    baseline_end,
):
    with pytest.raises(
        ValueError,
        match="baseline period must fall within the projection period",
    ):
        validate_projection_years(
            start_yr=1850,
            end_yr=2100,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            initialise_from_observations=False,
        )


# --- observational data readers ---


def _write_mock_imbie3_csv(tmp_path):
    """Write a small synthetic IMBIE3-style dataset."""

    years = np.arange(1990, 2011)

    df = pd.DataFrame(
        {
            "Date": [f"{year}-01-01" for year in years],
            "Cumulative mass balance anomaly (mm)": years - 1990,
            "Cumulative mass balance anomaly uncertainty (mm)": np.full(
                years.size, 0.2
            ),
            "Mass balance (mm/yr)": np.full(years.size, 2.0),
            "Mass balance uncertainty (mm/yr)": np.full(years.size, 0.2),
            "Cumulative surface mass balance anomaly (mm)": (2 * (years - 1990)),
            "Cumulative surface mass balance anomaly uncertainty (mm)": (
                np.full(years.size, 0.3)
            ),
            "Surface mass balance anomaly (mm/yr)": np.full(years.size, 3.0),
            "Surface mass balance anomaly uncertainty (mm/yr)": (
                np.full(years.size, 0.3)
            ),
            "Cumulative dynamics mass balance anomaly (mm)": (3 * (years - 1990)),
            "Cumulative dynamics mass balance anomaly uncertainty (mm)": (
                np.full(years.size, 0.4)
            ),
            "Dynamics mass balance anomaly (mm/yr)": np.full(years.size, 4.0),
            "Dynamics mass balance anomaly uncertainty (mm/yr)": (
                np.full(years.size, 0.4)
            ),
        }
    )

    fpath = tmp_path / "imbie3.csv"
    df.to_csv(fpath, index=False)

    return fpath


@pytest.mark.parametrize(
    "component, expected_sle, expected_sle_unc, expected_rate, expected_rate_unc",
    [
        ("total", -0.003, 0.0002, -2.0, 0.2),
        ("smb", -0.006, 0.0003, -3.0, 0.3),
        ("dyn", -0.009, 0.0004, -4.0, 0.4),
    ],
)
def test_read_imbie3_obs_uses_baseline_mean(
    tmp_path,
    component,
    expected_sle,
    expected_sle_unc,
    expected_rate,
    expected_rate_unc,
):
    state = DummyClimateState(
        start_yr=2000,
        baseline_start=1995,
        baseline_end=1997,
    )
    fpath = _write_mock_imbie3_csv(tmp_path)

    ds = _read_imbie3_obs(
        fpath,
        component=component,
        baseline_start=1995,
        baseline_end=1997,
    )
    initial = ds.sel(time=state.start_yr - 1)

    assert initial["sle"].item() == pytest.approx(expected_sle)
    assert initial["sle_unc"].item() == pytest.approx(expected_sle_unc)
    assert initial["rate"].item() == pytest.approx(expected_rate)
    assert initial["rate_unc"].item() == pytest.approx(expected_rate_unc)


def test_read_imbie3_obs_explicit_baseline_overrides_state(
    tmp_path,
):
    state = DummyClimateState(
        start_yr=2000,
        baseline_start=1990,
        baseline_end=1991,
    )
    fpath = _write_mock_imbie3_csv(tmp_path)

    ds = _read_imbie3_obs(
        fpath,
        component="total",
        baseline_start=1995,
        baseline_end=1997,
    )

    sle = ds["sle"].sel(time=state.start_yr - 1).item()

    assert sle == pytest.approx(-0.003)


def _write_mock_zheng_dataset(tmp_path):
    """Write a small mock NetCDF dataset."""

    component_dir = tmp_path / "aux_data" / "component-obs"
    component_dir.mkdir(parents=True)

    ds = xr.Dataset(
        {
            "tws": (
                "time",
                np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            ),
            "tws_u": (
                "time",
                np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            ),
            "glacier": (
                "time",
                np.array([5.0, 10.0, 15.0, 20.0, 25.0]),
            ),
        },
        coords={
            "time": np.arange(1990, 1995),
        },
    )

    fpath = component_dir / "Zheng2026.nc"
    ds.to_netcdf(fpath)

    return fpath


def test_load_zheng_component_uses_state_baseline(
    tmp_path,
    monkeypatch,
):
    _write_mock_zheng_dataset(tmp_path)

    monkeypatch.setattr(
        "profsea.utils.utils.PROFSEA_DIR",
        tmp_path,
    )

    state = DummyClimateState(
        baseline_start=1991,
        baseline_end=1993,
    )

    ds = _load_zheng_component(
        variable="tws",
        uncertainty="tws_u",
        baseline_start=state.baseline_start,
        baseline_end=state.baseline_end,
    )

    np.testing.assert_allclose(
        ds["sle"].values,
        [-0.020, -0.010, 0.0, 0.010, 0.020],
    )
    np.testing.assert_allclose(
        ds["sle_unc"].values,
        [0.001, 0.002, 0.003, 0.004, 0.005],
    )


@patch("profsea.utils.utils._load_zheng_component")
def test_read_thermal_expansion_obs(
    mock_load,
):
    state = DummyClimateState()

    expected = xr.Dataset(
        {
            "sle": (
                "time",
                [0.010, 0.020],
            ),
            "sle_unc": (
                "time",
                [0.001, 0.002],
            ),
        },
        coords={
            "time": [1998, 1999],
        },
    )

    mock_load.return_value = expected
    result = read_thermal_expansion_obs(
        baseline_start=state.baseline_start, baseline_end=state.baseline_end
    )

    xr.testing.assert_identical(result, expected)

    mock_load.assert_called_once_with(
        variable="iapv4",
        uncertainty="iapv4_u",
        baseline_start=state.baseline_start,
        baseline_end=state.baseline_end,
    )


def test_read_imbie3_AIS_obs_invalid_region():

    with pytest.raises(
        ValueError,
        match="Unknown Antarctic region",
    ):
        read_antarctic_obs(region="invalid", baseline_start=1995, baseline_end=2005)


@pytest.mark.parametrize(
    ("region", "expected_filename"),
    [
        (None, "imbie3_antarctica_mm_partitioned.csv"),
        ("eais", "imbie3_east_antarctica_mm_partitioned.csv"),
        ("wais", "imbie3_west_antarctica_mm_partitioned.csv"),
        (
            "peninsula",
            "imbie3_antarctic_peninsula_mm_partitioned.csv",
        ),
    ],
)
@patch("profsea.utils.utils._read_imbie3_obs")
def test_read_imbie3_AIS_obs_selects_region_file(
    mock_read,
    region,
    expected_filename,
):
    state = DummyClimateState(
        baseline_start=1991,
        baseline_end=1993,
    )

    expected = xr.Dataset(
        {
            "sle": ("time", [0.001]),
        },
        coords={"time": [2000]},
    )

    mock_read.return_value = expected

    result = read_antarctic_obs(
        region=region,
        component="total",
        baseline_start=state.baseline_start,
        baseline_end=state.baseline_end,
    )

    xr.testing.assert_identical(result, expected)

    args = mock_read.call_args.args

    assert args[0].name == expected_filename
    assert args[1] == "total"
    assert args[2] == state.baseline_start
    assert args[3] == state.baseline_end


def test_rebaseline_projection():
    projection = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0]],
            [[2.0, 4.0, 6.0, 8.0, 10.0]],
        ]
    )

    result = rebaseline_projection(
        projection,
        start_yr=2000,
        baseline_start=2001,
        baseline_end=2003,
    )

    expected = np.array(
        [
            [[-2.0, -1.0, 0.0, 1.0, 2.0]],
            [[-4.0, -2.0, 0.0, 2.0, 4.0]],
        ]
    )

    np.testing.assert_allclose(result, expected)
