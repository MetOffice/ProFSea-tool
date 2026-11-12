from __future__ import annotations

import logging
import os
import zipfile
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
import pandas as pd
import requests
import xarray as xr
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from profsea.components.core.state import ClimateState

console = Console()
logger = logging.getLogger(__name__)

try:
    PROFSEA_VERSION = package_version("profsea")
except PackageNotFoundError:
    PROFSEA_VERSION = "unknown"

PROFSEA_DIR = Path(__file__).resolve().parents[1]


def validate_component_map(
    components: dict,
    expected_type: type,
    model_name: str = "model",
) -> None:
    """Validate a component dictionary before model construction.

    Parameters
    ----------
    components: dict
        Dictionary keyed by component name and valued by component instances.
    expected_type: type
        Base component type expected in the mapping.
    model_name: str, optional
        Human-readable label used in error messages.
    """
    if not isinstance(components, dict):
        msg = (
            "'components' must be a dictionary of the form {name: component_instance}."
        )
        logger.error(msg)
        console.print(f"[red]{msg}[/red]")
        raise ValueError(msg)

    invalid_entries = []
    for name, comp in components.items():
        if isinstance(comp, expected_type):
            continue

        if isinstance(comp, type) and issubclass(comp, expected_type):
            reason = "component class provided; instantiate it with ()"
        else:
            reason = (
                f"expected {expected_type.__name__} instance, got {type(comp).__name__}"
            )

        invalid_entries.append(f"{name}: {reason}")

    if invalid_entries:
        msg = (
            f"Invalid entries in 'components' for {model_name}: "
            f"{'; '.join(invalid_entries)}. "
            "If you see NameError before this step, a component class was not "
            "imported in your script."
        )
        logger.error(msg)
        console.print(f"[red]{msg}[/red]")
        raise ValueError(msg)


def validate_projection_years(
    start_yr: int, end_yr: int, baseline_start: int, baseline_end: int
) -> None:
    """Validate the baseline and projections years before model construction.

    Parameters
    ----------
    start_yr : int
        First year of the projection.
    end_yr : int
        End year of the projection.
    baseline_start : int
        First year of the baseline period.
    baseline_end : int
        Last year of the baseline period.

    Raises
    ------
    ValueError
        If the projection or baseline periods are invalid, or if the
        requested years fall outside the period for which observational
        constraints are available for all components.
    """
    if start_yr >= end_yr:
        raise ValueError(
            f"start_yr ({start_yr}) must be earlier than end_yr ({end_yr})."
        )
    if baseline_start >= baseline_end:
        raise ValueError(
            f"baseline_start ({baseline_start}) must be earlier than "
            f"baseline_end ({baseline_end})."
        )
    if start_yr < 1980 or baseline_start < 1980:
        raise ValueError(
            f"start_yr ({start_yr}) and baseline_start ({baseline_start}) must both "
            f"be later than 1980 (so that observational constraints are available "
            f"for all components)."
        )
    if baseline_end > 2023:
        raise ValueError(
            f"baseline_end ({baseline_end}) cannot be later than 2023 (so that "
            f"observational constraints are available for all components)."
        )


def sample_members_2D(
    array: np.ndarray | da.Array,
    percentiles: list | np.ndarray,
    dtype: np.dtype = np.float32,
) -> np.ndarray | da.Array:
    """
    Sample real ensemble members from a 2D numpy or dask array lazily.

    Parameters
    ----------
    array: np.ndarray | da.Array
        Input 2D array of shape (realisation, time).
    percentiles: list | np.ndarray
        List of percentiles to sample from the input array.

    Returns
    -------
    np.ndarray | da.Array
        Sampled array of shape (len(percentiles), time). Maintains lazy
        evaluation if a dask array is provided.
    """

    def _eager_sample(arr, percs):
        # Calculate statistical timeseries, then match with closest real timeseries
        array_percentiles = np.nanpercentile(arr, percs, axis=0)
        distances = cdist(array_percentiles, arr)
        mem_indices = np.argmin(distances, axis=1)
        return arr[mem_indices].astype(dtype)

    if isinstance(array, da.Array):
        # Tell Dask to delay this operation until the graph is computed
        lazy_result = dask.delayed(_eager_sample)(array, percentiles)

        # Reconstruct into a Dask array so downstream Xarray operations continue to work lazily
        shape = (len(percentiles), array.shape[1])
        return da.from_delayed(lazy_result, shape=shape, dtype=array.dtype)

    else:
        # Fallback for standard numpy arrays
        return _eager_sample(array, percentiles)


def interpolate(data: da.array, lats: int, lons: int) -> da.array:
    """
    Interpolate a 2D dask array to a target grid defined by lats and lons.

    Parameters
    ----------
    data: da.array
        Input 2D dask array to be interpolated.
    lats: int
        Number of latitude points in the target grid.
    lons: int
        Number of longitude points in the target grid.

    Returns
    -------
    da.array
        Interpolated 2D dask array on the target grid.
    """
    original_da = xr.DataArray(
        data.data,
        coords=[("lat", data[data.dims[0]].values), ("lon", data[data.dims[1]].values)],
        name="v",
    )

    target_lat = np.linspace(-90, 90, lats, endpoint=False) + 0.5
    target_lon = np.linspace(-180, 180, lons, endpoint=False) + 0.5
    data_interp = original_da.interp(
        lat=target_lat, lon=target_lon, method="linear"
    ).data
    return data_interp


def interpolate_to_grid(
    data: xr.DataArray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    grid_interpolation: str = "linear",
) -> xr.DataArray:
    """
    Interpolate an xarray DataArray to a target grid defined by target_lats and target_lons.
    Safely handles longitude wrapping mismatches (e.g., [0, 360) vs [-180, 180)).

    Parameters
    ----------
    data: xr.DataArray
        Input xarray DataArray to be interpolated. Must have 'lat' and 'lon' dimensions.
    target_lats: np.ndarray
        1D array of target latitude values.
    target_lons: np.ndarray
        1D array of target longitude values.
    grid_interpolation: str, optional
        Interpolation method to use. Default is 'linear'. Other options include 'nearest', 'cubic', etc.

    Returns
    -------
    xr.DataArray
        Interpolated xarray DataArray on the target grid.
    """
    import regionmask

    # Normalize source longitudes to [-180, 180) and sort monotonically
    data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
    data = data.sortby(["lat", "lon"])

    # Normalize target longitudes to [-180, 180) and sort
    target_lons_norm = np.sort(((target_lons + 180) % 360) - 180)

    # Pad longitude with one points from each end to handle periodicity in zonal direction
    data_padded = data.pad(
        lon=1, mode="wrap"
    )  # need more padding for higher-order interpolation
    lon = data.lon.values
    lon_padded = np.concatenate([[lon[-1] - 360], lon, [lon[0] + 360]])
    data_padded["lon"] = lon_padded
    data_padded = data_padded.sortby(["lat", "lon"])

    # Now interpolate
    data_padded = data_padded.chunk({"lat": -1, "lon": -1})
    for dim in ["lat", "lon"]:
        data_padded = data_padded.interpolate_na(
            dim=dim,
            method="nearest",
        )  # this to handle nan values or land mask

    data_interp = data_padded.interp(
        lat=target_lats, lon=target_lons_norm, method=grid_interpolation
    )

    # Account for land mask (using regionmask library)
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    land_mask = land.mask_3D(data_interp)
    is_land = land_mask.squeeze("region", drop=True)
    data_interp = data_interp.where(~is_land)

    return data_interp


def check_shapes(array: np.ndarray, n_time: int) -> None:
    """Check that the input arrays have the correct shape.

    Parameters
    ----------
    array: np.ndarray
        Input array of some kind.
    n_time: int
        Expected number of time steps.

    Returns
    -------
    None
    """
    if array.ndim == 1:
        array = array[np.newaxis, :]

    if array.shape[1] != n_time:
        raise ValueError(
            f"Array should have shape (realisation, time) with time "
            f"dimension of length {n_time}. Got {array.shape}."
        )


def fetch_zenodo_fingerprints(
    zenodo_url: str, data_dir: Path, expected_folder_name: str
) -> None:
    """
    Downloads and extracts the ProFSea fingerprint dataset from Zenodo if it doesn't already exist locally.

    Parameters
    ----------
    zenodo_url: str
        The direct download URL for the fingerprint dataset on Zenodo.
    data_dir: Path
        The base directory where the dataset should be stored.
    expected_folder_name: str
        The name of the folder that should be created when the dataset is extracted. Used to check if the data already exists.
    """
    target_dir = data_dir / expected_folder_name

    # 1. Check if data already exists
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info("[bold green]✓ ProFSea assets found locally![/bold green]")
        return

    # Create the base directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "temp_fingerprints.zip"

    logger.info(f"Initiating download from {zenodo_url}...")

    # 2. Stream the download with a rich progress bar
    try:
        response = requests.get(zenodo_url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        total_size = int(response.headers.get("content-length", 0))

        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            download_task = progress.add_task(
                "Downloading dataset...", total=total_size
            )

            with open(zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        progress.update(download_task, advance=len(chunk))

    except requests.exceptions.RequestException as e:
        logger.error(f"[bold red]Failed to download data: {e}[/bold red]")
        if zip_path.exists():
            zip_path.unlink()  # Clean up partial downloads
        raise

    logger.info("Extracting data...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Filter out the __MACOSX directory and its contents
            valid_members = [
                member
                for member in zip_ref.namelist()
                if not member.startswith("__MACOSX/") and not member.startswith("._")
            ]
            zip_ref.extractall(data_dir, members=valid_members)

        logger.info(
            f"[bold green]✓ Successfully extracted data to {data_dir}[/bold green]"
        )
    except zipfile.BadZipFile:
        logger.error(
            "[bold red]Error: Downloaded file is not a valid zip archive.[/bold red]"
        )
        raise
    finally:
        # 4. Clean up the zip file
        if zip_path.exists():
            zip_path.unlink()


def save_components(
    self,
    components: dict[str, xr.DataArray],
    scenario_name: str,
    output_prefix: str = "projection",
    output_dir: str = ".",
    output_format: str = "zarr",
    output_dtype: str | None = None,
    description: str = "Spatial sea level rise projections",
) -> None:
    """
    Stream sea level projections to disk in a single file/store.

    Parameters
    ----------
    components: dict[str, xr.DataArray]
        Dictionary of component names and their corresponding Xarray DataArrays.
    scenario_name: str
        Name of the scenario you've run the emulator for.
    output_prefix: str
        Prefix for the output file name (e.g., 'projection' or 'global').
    output_dir: str
        Directory to save components to.
    output_format: str
        Format to save the output in. Must be either 'netcdf' or 'zarr'.
    output_dtype: str | None
        Data type to force the output to (e.g., 'float32'). If None, the dtype
        is dynamically inferred from each individual component.
    description: str
        Metadata description attached to the dataset attributes.

    Returns
    -------
    None
    """
    if not components:
        logger.warning("No components provided to save.")
        return

    ds = xr.Dataset(components)

    # Add ProFSea version and scenario metadata
    ds.attrs["source"] = f"ProFSea v{PROFSEA_VERSION}"
    ds.attrs["scenario"] = scenario_name
    ds.attrs["description"] = description

    output_format = output_format.lower()
    if output_format not in ["netcdf", "zarr"]:
        raise ValueError("output_format must be either 'netcdf' or 'zarr'.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    encoding = {}

    # Sort out Zarr encoding
    if output_format == "zarr":
        import numcodecs
        from zarr.codecs.numcodecs import Blosc

        # We use the Zarr 3 wrapper (Blosc), but feed it the integer constant from numcodecs
        compressor = Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    # Set the encoding dynamically, overriding with output_dtype if provided
    for name, component in components.items():
        comp_dtype = output_dtype if output_dtype else component.dtype.name

        if output_format == "netcdf":
            encoding[name] = {"zlib": True, "complevel": 1, "dtype": comp_dtype}
        elif output_format == "zarr":
            encoding[name] = {"compressors": [compressor], "dtype": comp_dtype}

    file_name = f"{scenario_name}_{output_prefix}"

    # Dynamically adjust console message string based on the prefix
    log_prefix = "Global " if "global" in output_prefix.lower() else ""

    # Stream the computation and write to disk
    if output_format == "netcdf":
        out_path = os.path.join(output_dir, f"{file_name}.nc")
        with console.status(
            f"[bold cyan]Computing and saving {log_prefix}NetCDF...[/bold cyan]",
            spinner="dots",
        ):
            # Let xarray handle streaming directly to avoid RAM overload
            ds.compute().to_netcdf(out_path, encoding=encoding)

        logger.info(f"[bold green]✓ Successfully saved NetCDF:[/bold green] {out_path}")

    elif output_format == "zarr":
        out_path = os.path.join(output_dir, f"{file_name}.zarr")
        with console.status(
            f"[bold cyan]Streaming computation and saving {log_prefix}Zarr...[/bold cyan]",
            spinner="dots",
        ):
            ds.to_zarr(
                out_path,
                encoding=encoding,
                mode="w",
                compute=True,
                zarr_format=3,
                consolidated=True,
            )

        logger.info(f"[bold green]✓ Successfully saved Zarr:[/bold green] {out_path}")

    # Safely get a sample name to log the shape (fixes the scope-leak bug)
    sample_name = "total_gmslr" if "total_gmslr" in ds else list(ds.data_vars.keys())[0]
    dims_str = ", ".join(ds[sample_name].dims)
    logger.info(
        f"{log_prefix}Output shape for '{sample_name}' was {ds[sample_name].shape} ({dims_str})"
    )


def reformat_global_projection(
    raw_projection: np.ndarray, state: ClimateState
) -> np.ndarray:

    if state.output_percentiles is None or len(state.output_percentiles) == 0:
        return raw_projection

    if raw_projection.ndim == 3:
        flat_projection = raw_projection.reshape(-1, raw_projection.shape[-1])
    else:
        flat_projection = raw_projection

    return sample_members_2D(flat_projection, state.output_percentiles)


def _read_imbie3_obs(
    fpath: Path,
    component: str,
    baseline_start: int,
    baseline_end: int,
):
    """Read IMBIE3 observations for ice-sheet components.

    Reads total, surface mass balance (SMB), and dynamic mass-balance
    contributions from the Antarctic/Greenland IMBIE3 datasets (Otosaka
    et al., 2026).

    Cumulative sea-level contributions are expressed relative to the mean
    contribution over the requested baseline period. The complete annual
    observational time series is returned so that it can be used both for
    projection initialisation and comparison with projections.

    Parameters
    ----------
    fpath : pathlib.Path
        Path to IMBIE3 observational dataset (.csv).
    baseline_start : int
        Start year of the baseline period.
    baseline_end : int
        End year of the baseline period.

    Returns
    -------
    xr.Dataset
        Dataset containing annual sea-level contribution, rate, and rate
        uncertainty for the total, SMB, and dynamic components.

        ``sle`` is expressed in metres relative to the requested baseline
        period. ``rate`` and ``rate_unc`` are expressed in mm/yr.

    References
    ----------
    Otosaka et al. (2026)
        Mass balance of the Greenland and Antarctic Ice Sheets from the 1970s to 2023.
        https://doi.org/10.5285/128c5e33-5224-4197-82f0-19dcc95b80a0
    """

    df = pd.read_csv(fpath)

    dates = pd.to_datetime(df["Date"], errors="coerce")
    # retain year as the common temporal coordinate
    time = dates.to_numpy()

    columns = {
        "total": {
            "sle": "Cumulative mass balance anomaly (mm)",
            "sle_unc": "Cumulative mass balance anomaly uncertainty (mm)",
            "rate": "Mass balance (mm/yr)",
            "rate_unc": "Mass balance uncertainty (mm/yr)",
        },
        "smb": {
            "sle": "Cumulative surface mass balance anomaly (mm)",
            "sle_unc": "Cumulative surface mass balance anomaly uncertainty (mm)",
            "rate": "Surface mass balance anomaly (mm/yr)",
            "rate_unc": "Surface mass balance anomaly uncertainty (mm/yr)",
        },
        "dyn": {
            "sle": "Cumulative dynamics mass balance anomaly (mm)",
            "sle_unc": "Cumulative dynamics mass balance anomaly uncertainty (mm)",
            "rate": "Dynamics mass balance anomaly (mm/yr)",
            "rate_unc": "Dynamics mass balance anomaly uncertainty (mm/yr)",
        },
    }

    try:
        component_columns = columns[component]
    except KeyError:
        raise ValueError(
            f"Unknown IMBIE3 component: {component!r}. Expected one of {list(columns)}."
        )

    sle = -pd.to_numeric(
        df[component_columns["sle"]],
        errors="coerce",
    ).to_numpy()

    sle_unc = pd.to_numeric(
        df[component_columns["sle_unc"]],
        errors="coerce",
    ).to_numpy()

    rate = -pd.to_numeric(
        df[component_columns["rate"]],
        errors="coerce",
    ).to_numpy()

    rate_unc = pd.to_numeric(
        df[component_columns["rate_unc"]],
        errors="coerce",
    ).to_numpy()

    ds = xr.Dataset(
        data_vars={
            "sle": (
                "time",
                sle / 1e3,
                {
                    "long_name": "Sea-level contribution",
                    "units": "m",
                },
            ),
            "sle_unc": (
                "time",
                sle_unc / 1e3,
                {
                    "long_name": "Sea-level contribution uncertainty",
                    "units": "m",
                },
            ),
            "rate": (
                "time",
                rate,
                {
                    "long_name": "Sea-level contribution rate",
                    "units": "mm yr-1",
                },
            ),
            "rate_unc": (
                "time",
                rate_unc,
                {
                    "long_name": "Sea-level contribution rate uncertainty",
                    "units": "mm yr-1",
                },
            ),
        },
        coords={"time": time},
        attrs={
            "baseline_start": baseline_start,
            "baseline_end": baseline_end,
            "component": component,
            "reference": "IMBIE3 (Otosaka et al., 2026)",
        },
    )

    # Convert the monthly observational record to annual means and use
    # integer years as the time coordinate.
    ds = ds.groupby("time.year").mean("time")
    ds = ds.rename({"year": "time"})

    # Reference cumulative SLE to its mean over the requested baseline
    # period.
    baseline_mean = ds["sle"].sel(time=slice(baseline_start, baseline_end)).mean("time")

    ds["sle"] = ds["sle"] - baseline_mean

    return ds
    # # AR5 used the average ice-sheet loss rate over 2005--2010
    # # (0.63 mm/yr and 0.41mm/yr SLE; AR5 Chapter 4, Table 4.6). Here, we apply
    # # the equivalent six-year window to support flexible start years.
    # rate_mask = (
    #     (time >= state.start_yr - 1)
    #     & (time <= state.start_yr + 4)
    # )


def read_greenland_obs(
    baseline_start: int, baseline_end: int, component: str = "total"
) -> xr.DataSet:
    """Read IMBIE3 Greenland Ice Sheet observations."""

    fpath = (
        PROFSEA_DIR
        / "aux_data"
        / "component-obs"
        / "imbie3_greenland_mm_partitioned.csv"
    )

    return _read_imbie3_obs(fpath, component, baseline_start, baseline_end)


def read_antarctic_obs(
    baseline_start: int,
    baseline_end: int,
    component: str = "total",
    region: str | None = None,
) -> xr.DataSet:
    """Read IMBIE3 Antarctic Ice Sheet observations.

    baseline_start : int
        Start year of the baseline period.
    baseline_end : int
        End year of the baseline period.
    region : str, optional
        Antarctic region to load. One of ``"eais"``, ``"wais"``,
        ``"peninsula"``, or ``None`` for the total Antarctic Ice Sheet.
    """

    root = PROFSEA_DIR / "aux_data" / "component-obs"

    files = {
        None: "imbie3_antarctica_mm_partitioned.csv",
        "eais": "imbie3_east_antarctica_mm_partitioned.csv",
        "wais": "imbie3_west_antarctica_mm_partitioned.csv",
        "peninsula": "imbie3_antarctic_peninsula_mm_partitioned.csv",
    }

    try:
        fpath = root / files[region]
    except KeyError:
        raise ValueError(f"Unknown Antarctic region: {region!r}")

    return _read_imbie3_obs(fpath, component, baseline_start, baseline_end)


def _load_zheng_component(
    variable: str,
    uncertainty: str,
    baseline_start: int,
    baseline_end: int,
) -> xr.Dataset:
    """Load and rebaseline an observational sea-level component.

    Loads land-water storage, thermal expansion, or glacier observational
    data from Zheng et al. (2026 Sci. Adv.), and expresses the sea-level
    contribution relative to the requested baseline period.

    The complete annual observational time series is returned so that it
    can be used both for projection initialisation and comparison with
    projections.

    Parameters
    ----------
    state : ClimateState
        Climate state containing the baseline period and numerical dtype.
    variable : str
        Name of the sea-level variable to load.
    uncertainty : str
        Name of the corresponding uncertainty variable.
    baseline_start : int
        Start year of the baseline period.
    baseline_end : int
        End year of the baseline period.

    Returns
    -------
    xr.Dataset
        Dataset containing the annual sea-level contribution and uncertainty
        in metres, relative to the requested baseline period.
    """

    filepath = PROFSEA_DIR / "aux_data" / "component-obs" / "Zheng2026.nc"

    with xr.open_dataset(filepath) as ds:
        sle = ds[variable].load()
        sle_unc = ds[uncertainty].load()

    # Standardise the observational coordinate to integer years.
    years = sle["time"].values.astype(int)

    sle = sle.assign_coords(time=years)
    sle_unc = sle_unc.assign_coords(time=years)

    # Reference cumulative SLE to its mean over the requested
    # baseline period.
    baseline_mean = sle.sel(time=slice(baseline_start, baseline_end)).mean("time")

    sle = (sle - baseline_mean) / 1e3
    sle_unc = sle_unc / 1e3

    ds = xr.Dataset(
        data_vars={
            "sle": sle,
            "sle_unc": sle_unc,
        },
        attrs={
            "baseline_start": baseline_start,
            "baseline_end": baseline_end,
            "reference": "Zheng et al. (2026)",
        },
    )

    ds["sle"].attrs = {
        "long_name": "Sea-level contribution",
        "units": "m",
    }

    ds["sle_unc"].attrs = {
        "long_name": "Sea-level contribution uncertainty",
        "units": "m",
    }

    return ds


def read_thermal_expansion_obs(
    baseline_start: int,
    baseline_end: int,
) -> xr.Dataset:
    """Read annual observed thermal-expansion sea-level contribution."""

    return _load_zheng_component(
        variable="iapv4",
        uncertainty="iapv4_u",
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )


def read_glacier_obs(
    baseline_start: int,
    baseline_end: int,
) -> xr.Dataset:
    """Read annual observed glacier sea-level contribution."""

    return _load_zheng_component(
        variable="glacier",
        uncertainty="glacier_u",
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )


def read_landwater_obs(
    baseline_start: int,
    baseline_end: int,
) -> xr.Dataset:
    """Read annual observed land-water-storage sea-level contribution."""

    return _load_zheng_component(
        variable="tws",
        uncertainty="tws_u",
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )


def read_GMSL_obs(
    baseline_start: int,
    baseline_end: int,
) -> xr.Dataset:
    """Read annual observed GMSL timeseries."""

    return _load_zheng_component(
        variable="gmsl_m",
        uncertainty="gmsl_m_u",
        baseline_start=baseline_start,
        baseline_end=baseline_end,
    )
