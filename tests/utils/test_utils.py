import subprocess
import sys
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from profsea.utils import (
    check_shapes,
    fetch_zenodo_fingerprints,
    interpolate,
    interpolate_to_grid,
    members_percentiles,
    read_antarctic_obs,
    read_thermal_expansion_obs,
    rebaseline_projection,
    sample_members_2D,
    save_components,
    validate_projection_years,
)
from profsea.utils.utils import (
    _load_zheng_component,
    _read_imbie3_obs,
    _rechunk_for_storage,
)


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


def _no_land_regionmask() -> MagicMock:
    """Stand-in for regionmask (its shapes are downloaded on first use) for a
    grid with no land. Like the real ``mask_3D``, it drops the land region
    unless ``drop=False``."""

    def mask_3D(obj, drop=True):
        mask = xr.DataArray(
            np.zeros((1, obj.lat.size, obj.lon.size), dtype=bool),
            dims=["region", "lat", "lon"],
            coords={"region": [0], "lat": obj.lat.values, "lon": obj.lon.values},
        )
        return mask.isel(region=slice(0, 0)) if drop else mask

    fake = MagicMock()
    fake.defined_regions.natural_earth_v5_0_0.land_110.mask_3D.side_effect = mask_3D
    return fake


def test_interpolate_to_grid_open_ocean_grid():
    """A target grid without land cells is regridded rather than rejected."""
    data = xr.DataArray(
        np.random.default_rng(42).random((3, 4)),
        coords=[("lat", [-60, -55, -50]), ("lon", [0, 90, 180, 270])],
    )

    with patch.dict("sys.modules", {"regionmask": _no_land_regionmask()}):
        result = interpolate_to_grid(
            data, np.array([-58.0, -52.0]), np.array([10.0, 100.0, -100.0])
        )

    assert result.shape == (2, 3)
    assert not result.isnull().any()


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


# --- members_percentiles ---


@pytest.mark.parametrize("shape", [(4,), (4, 3, 2), (4, 2, 2, 3)])
@pytest.mark.parametrize("lazy", [False, True])
def test_members_percentiles_first_axis(shape, lazy):
    array = np.arange(np.prod(shape), dtype=float).reshape(shape)
    array[0] = np.nan
    percentiles = [0, 25, 50, 100]
    expected = np.nanpercentile(array, percentiles, axis=0)
    if lazy:
        array = da.from_array(array, chunks=(2,) + (1,) * (len(shape) - 1))

    with Callback(start=lambda graph: pytest.fail("Percentile computation was eager")):
        result = members_percentiles(array, percentiles)

    assert isinstance(result, da.Array if lazy else np.ndarray)
    assert result.shape == (len(percentiles), *shape[1:])
    np.testing.assert_allclose(result.compute() if lazy else result, expected)


@pytest.mark.parametrize("lazy", [False, True])
def test_members_percentiles_all_nan_location(lazy):
    """Columns with no valid members stay NaN."""
    array = np.array([[0.0, np.nan], [10.0, np.nan], [20.0, np.nan]])
    if lazy:
        array = da.from_array(array, chunks=(1, 1))
    with (
        nullcontext()
        if lazy
        else pytest.warns(RuntimeWarning, match="All-NaN slice encountered")
    ):
        result = members_percentiles(array, [25, 75])
        values = result.compute() if lazy else result
    assert result.shape == (2, 2)
    np.testing.assert_allclose(values, [[5.0, np.nan], [15.0, np.nan]], equal_nan=True)


# --- fetch_zenodo_fingerprints ---


@patch("requests.get")
def test_fetch_zenodo_early_return(mock_get, tmp_path, capsys):
    """Should return quietly, without downloading, if the assets are present."""
    target_dir = tmp_path / "profsea-assets"
    target_dir.mkdir()
    (target_dir / "dummy_file.nc").touch()

    fetch_zenodo_fingerprints("http://fake.url", tmp_path, "profsea-assets")

    mock_get.assert_not_called()
    # Rich logger writes to stdout, so we use capsys instead of caplog
    assert capsys.readouterr().out == ""


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


def _mock_zenodo_download(mock_get, mock_zip, members):
    mock_response = MagicMock()
    mock_response.headers = {"content-length": "100"}
    mock_response.iter_content.return_value = [b"chunk"]
    mock_get.return_value = mock_response

    mock_zip_instance = MagicMock()
    mock_zip_instance.namelist.return_value = members
    mock_zip.return_value.__enter__.return_value = mock_zip_instance
    return mock_zip_instance


@patch("requests.get")
def test_fetch_zenodo_matching_version_early_return(mock_get, tmp_path, capsys):
    """A local copy with a matching version marker should not be downloaded again."""
    target_dir = tmp_path / "profsea-assets"
    target_dir.mkdir()
    (target_dir / "dummy_file.nc").touch()
    (target_dir / ".assets_version").write_text("1.1\n")

    fetch_zenodo_fingerprints(
        "http://fake.url", tmp_path, "profsea-assets", version="1.1"
    )

    mock_get.assert_not_called()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("old_marker", [None, "1.0"])
@patch("requests.get")
@patch("zipfile.ZipFile")
def test_fetch_zenodo_outdated_version_redownloads(
    mock_zip, mock_get, tmp_path, old_marker
):
    """A local copy without a marker, or with an old one, should be updated."""
    target_dir = tmp_path / "profsea-assets"
    target_dir.mkdir()
    (target_dir / "dummy_file.nc").touch()
    if old_marker is not None:
        (target_dir / ".assets_version").write_text(old_marker)
    mock_zip_instance = _mock_zenodo_download(mock_get, mock_zip, ["file1.nc"])

    fetch_zenodo_fingerprints(
        "http://fake.url", tmp_path, "profsea-assets", version="1.1"
    )

    mock_get.assert_called_once_with("http://fake.url", stream=True)
    mock_zip_instance.extractall.assert_called_once_with(tmp_path, members=["file1.nc"])
    assert (target_dir / ".assets_version").read_text().strip() == "1.1"


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


def test_save_components_zarr_merges_small_blocks(tmp_path):
    """Small compute blocks are written as larger Zarr chunks (fewer files),
    keeping the leading axis whole and the values unchanged."""
    values = np.arange(5 * 4 * 6 * 8, dtype=np.float32).reshape(5, 4, 6, 8)
    lazy = da.from_array(values, chunks=(5, 4, 1, 1))  # 48 tiny blocks
    components = {
        "comp1": xr.DataArray(lazy, dims=["percentile", "time", "lat", "lon"])
    }

    save_components(
        DummyInstance(),
        components,
        scenario_name="test",
        output_dir=str(tmp_path),
        output_format="zarr",
        zarr_chunk_mb=0.002,  # about 2 kB: several blocks per chunk, not all
    )

    stored = xr.open_zarr(tmp_path / "test_projection.zarr")["comp1"]
    chunks = stored.encoding["chunks"]
    assert chunks[0] == 5  # leading axis kept whole
    assert 1 < np.prod(chunks) / (5 * 4) < 48  # merged, but within the budget
    np.testing.assert_array_equal(stored.values, values)


def _is_zarr_compatible(chunks: tuple[tuple[int, ...], ...]) -> bool:
    """Zarr needs equal chunks along each axis, except a smaller final one."""
    return all(
        len(set(axis[:-1])) <= 1 and axis[-1] <= axis[0] for axis in chunks if axis
    )


@pytest.mark.parametrize(
    "leading, compute_block",
    [
        (3, 44),  # reported failure: Dask "auto" gave lat (88, 92)
        (3, 67),  # Dask "auto" gave lon (67, 67, 67, 67, 92)
        (5, 11),
        (1000, 3),
    ],
)
def test_rechunk_for_storage_gives_regular_chunks(leading, compute_block):
    """Storage chunks are regular whole multiples of the compute blocks.

    Lazy only: nothing is computed, so a full 1-degree shape stays cheap.
    """
    shape = (leading, 295, 180, 360)
    lazy = da.zeros(
        shape, dtype=np.float32, chunks=(leading, 295, compute_block, compute_block)
    )
    ds = xr.Dataset({"x": (("percentile", "time", "lat", "lon"), lazy)})

    stored = _rechunk_for_storage(ds, target_mb=32)["x"].data

    assert stored.shape == shape
    assert _is_zarr_compatible(stored.chunks)
    assert stored.chunks[0] == (leading,)  # leading axis kept whole
    for axis in (2, 3):  # merges compute blocks without splitting them
        size = stored.chunks[axis][0]
        assert size == shape[axis] or size % compute_block == 0
    chunk_bytes = np.prod([axis[0] for axis in stored.chunks]) * 4
    single_block = leading * 295 * compute_block**2 * 4
    assert chunk_bytes <= max(32e6, single_block)


def test_save_components_zarr_regularises_irregular_blocks(tmp_path):
    """Components computed in irregular blocks are still written to Zarr."""
    values = np.arange(3 * 4 * 7 * 5, dtype=np.float32).reshape(3, 4, 7, 5)
    lazy = da.from_array(values, chunks=((3,), (4,), (2, 3, 2), (1, 4)))
    components = {
        "comp1": xr.DataArray(lazy, dims=["percentile", "time", "lat", "lon"])
    }

    save_components(
        DummyInstance(),
        components,
        scenario_name="test",
        output_dir=str(tmp_path),
        output_format="zarr",
        zarr_chunk_mb=0.0002,  # 200 bytes: several chunks along lat
    )

    stored = xr.open_zarr(tmp_path / "test_projection.zarr")["comp1"]
    assert stored.shape == values.shape
    assert stored.encoding["chunks"][2] < values.shape[2]
    np.testing.assert_array_equal(stored.values, values)


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


def test_read_imbie3_obs_weights_months_by_days(tmp_path):
    """Annual means weight each month by its length (leap years included) and
    leave out missing months; a year with no data stays NaN."""
    dates = pd.date_range("2003-01-01", "2005-12-01", freq="MS")
    rate = np.where(dates.month == 2, 1.0, 0.0)  # mm/yr, February only
    rate[(dates.year == 2004) & (dates.month == 12)] = np.nan
    rate[dates.year == 2005] = np.nan
    columns = [
        "Cumulative mass balance anomaly (mm)",
        "Cumulative mass balance anomaly uncertainty (mm)",
        "Mass balance (mm/yr)",
        "Mass balance uncertainty (mm/yr)",
    ]
    fpath = tmp_path / "imbie3_monthly.csv"
    pd.DataFrame(
        {"Date": dates.strftime("%Y-%m-%d"), **{name: rate for name in columns}}
    ).to_csv(fpath, index=False)

    ds = _read_imbie3_obs(fpath, "total", baseline_start=2003, baseline_end=2004)

    assert list(ds["time"].values) == [2003, 2004, 2005]
    # Sign flips: IMBIE3 mass balance is negative for sea-level rise
    assert ds["rate"].sel(time=2003).item() == pytest.approx(-28 / 365)
    assert ds["rate"].sel(time=2004).item() == pytest.approx(-29 / (366 - 31))
    assert np.isnan(ds["rate"].sel(time=2005).item())
    assert ds["rate"].attrs["units"] == "mm yr-1"
    assert ds.attrs["component"] == "total"


def _write_mock_zheng_dataset(tmp_path):
    """Write a small mock NetCDF dataset."""

    component_dir = tmp_path / "auxiliary_data" / "component-obs"
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


def test_utils_importable_first():
    """profsea.utils must import in a fresh interpreter (no circular import)."""
    result = subprocess.run(
        [sys.executable, "-c", "import profsea.utils"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("lazy", [False, True])
def test_members_percentiles_integer_interpolation(lazy):
    """Integer inputs are interpolated in floating point, as NumPy does."""
    array = np.array([[0, 10], [3, 15]], dtype=np.int32)
    source = da.from_array(array, chunks=1) if lazy else array
    result = members_percentiles(source, [25, 50, 75])
    assert result.shape == (3, 2)
    np.testing.assert_allclose(
        result.compute() if lazy else result,
        np.nanpercentile(array, [25, 50, 75], axis=0),
    )


@pytest.mark.parametrize("q", [[-1], [101], [np.nan], [[50]]])
@pytest.mark.parametrize("lazy", [False, True])
def test_members_percentiles_rejects_invalid_requests(q, lazy):
    """Percentiles outside [0, 100], NaN or nested lists are rejected."""
    array = np.arange(6, dtype=float).reshape(3, 2)
    if lazy:
        array = da.from_array(array, chunks=1)
    with pytest.raises(ValueError, match="Percentiles"):
        members_percentiles(array, q)


@pytest.mark.parametrize("n_members", [2, 6, 17])
def test_stratified_members_balance_and_cover_ensemble(n_members):
    """Each trajectory gets an equal share, and process members cycle before repeating."""
    from profsea.utils import stratified_member_indices

    indices = stratified_member_indices(2, 3, n_members, np.random.default_rng(42))
    assert indices.shape == (n_members,)
    counts = np.bincount(indices // 3, minlength=2)
    assert counts.max() - counts.min() <= 1
    for climate, count in enumerate(counts):
        process = indices[indices // 3 == climate] % 3
        frequencies = np.bincount(process, minlength=3)
        assert frequencies.min() == count // 3
        assert frequencies.max() <= count // 3 + 1
    np.testing.assert_array_equal(
        indices,
        stratified_member_indices(2, 3, n_members, np.random.default_rng(42)),
    )


@pytest.mark.parametrize("spatial_shape", [(15,), (18, 36)])
@pytest.mark.parametrize("target_bytes", [32, 4096, 1000000])
def test_plan_chunks_respects_memory_budget(spatial_shape, target_bytes):
    """Chunks stay within the grid and, where possible, within the byte budget."""
    from profsea.utils import plan_chunks

    time_chunk, spatial_chunks = plan_chunks(7, 20, spatial_shape, 4, target_bytes)
    assert 1 <= time_chunk <= 20
    assert len(spatial_chunks) == len(spatial_shape)
    assert all(1 <= chunk <= size for chunk, size in zip(spatial_chunks, spatial_shape))
    assert 7 * time_chunk * np.prod(spatial_chunks) * 4 <= target_bytes
