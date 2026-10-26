from __future__ import annotations

import logging
import os
import zipfile
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

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
    start_yr: int,
    end_yr: int,
    baseline_start: int,
    baseline_end: int
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
            f"start_yr ({start_yr}) must be earlier than "
            f"end_yr ({end_yr})."
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
    state: ClimateState,
    fpath: Path,
    ar5_baseline: bool = False,
    baseline_start: int | None = None,
    baseline_end: int | None = None,
):
    """Read IMBIE3 observations for ice-sheet component initialisation.

    Reads total, surface mass balance (SMB), and dynamic mass-balance
    contributions from the Antarctic/Greenland IMBIE3 datasets (from Otosaka+
    2026).

    Sea-level contributions are expressed relative to the requested baseline
    period. The initial contribution is evaluated in the year immediately
    preceding `state.start_yr`.

    If `ar5_baseline` is True, the initial contribution is calculated
    following the AR5 methodology, using the change in cumulative SLE relative
    to the specified baseline. If False, contributions are expressed relative
    to the mean SLE over the baseline period (following AR6 methodology).

    Parameters
    ----------
    state : ClimateState
        Climate state containing the projection start year and baseline period.
    fpath : pathlib.Path
        Path to IMBIE3 observational dataset (.csv).
    ar5_baseline : bool, optional
        If True, SLE contributions are given relative to `state.baseline_start` (
        following the AR5-style baseline methodology). If False, SLE contributions are
        given relative to its mean in the range `[baseline_start, baseline_end]`.
    baseline_start : int, optional
        Start year of the baseline period. Defaults to `state.baseline_start`.
    baseline_end : int, optional
        End year of the baseline period. Defaults to `state.baseline_end`.

    Returns
    -------
    offsets : tuple of float
        Initial total, SMB, and dynamic sea-level contributions, in metres.
    rates : tuple of float
        Mean total, SMB, and dynamic sea-level contribution rates, in mm/yr.
    rate_unc : tuple of float
        Uncertainty in the total, SMB, and dynamic rates, respectively, in
        mm/yr.

    References
    ----------
    Otosaka et al. (2026)
        Mass balance of the Greenland and Antarctic Ice Sheets from the 1970s to 2023.
        https://doi.org/10.5285/128c5e33-5224-4197-82f0-19dcc95b80a0
    """

    baseline_start = (
        state.baseline_start if baseline_start is None else baseline_start
    )
    baseline_end = (
        state.baseline_end if baseline_end is None else baseline_end
    )

    df = pd.read_csv(fpath)

    dates = pd.to_datetime(df["Date"], errors="coerce")

    time = (
        dates.dt.year
        + (dates.dt.dayofyear - 1) / 365.25
    ).to_numpy()

    columns = {
        "total": {
            "sle": "Cumulative mass balance anomaly (mm)",
            "rate": "Mass balance (mm/yr)",
            "rate_unc": "Mass balance uncertainty (mm/yr)",
        },
        "smb": {
            "sle": "Cumulative surface mass balance anomaly (mm)",
            "rate": "Surface mass balance anomaly (mm/yr)",
            "rate_unc": "Surface mass balance anomaly uncertainty (mm/yr)",
        },
        "dyn": {
            "sle": "Cumulative dynamics mass balance anomaly (mm)",
            "rate": "Dynamics mass balance anomaly (mm/yr)",
            "rate_unc": "Dynamics mass balance anomaly uncertainty (mm/yr)",
        },
    }

    data = {}

    for component, cols in columns.items():
        data[component] = {
            name: -pd.to_numeric(
                df[column],
                errors="coerce",
            ).to_numpy()
            for name, column in cols.items()
        }

    # Initialise from the observed contribution in the year immediately
    # preceding the projection start year.
    idx_start = np.where(
        time == state.start_yr - 1
    )[0][0]

    if ar5_baseline:

        # AR5-style baseline: calculate the cumulative SLE change relative to
        # the baseline reference year
        idx_ref = np.where(
            time == baseline_start
        )[0][0]

        offsets = {
            component:
                values["sle"][idx_start]
                - values["sle"][idx_ref]
            for component, values in data.items()
        }

    else:

        # Reference cumulative SLE to its mean over the requested baseline period
        baseline_mask = (
            (time >= baseline_start)
            & (time <= baseline_end)
        )

        offsets = {}

        for component, values in data.items():

            sle = values["sle"]

            sle = (
                sle
                - np.nanmean(sle[baseline_mask])
            )

            offsets[component] = sle[idx_start]

    # For the original AR5 setup, these represent the observed contribution
    # between 1996 and 2005 (AR5 Chapter 4).
    offsets = {
        component: value / 1e3
        for component, value in offsets.items()
    }

    # AR5 used the average ice-sheet loss rate over 2005--2010
    # (0.63 mm/yr and 0.41mm/yr SLE; AR5 Chapter 4, Table 4.6). Here, we apply
    # the equivalent six-year window to support flexible start years.
    rate_mask = (
        (time >= state.start_yr - 1)
        & (time <= state.start_yr + 4)
    )

    rates = {
        component:
            np.nanmean(values["rate"][rate_mask])
        for component, values in data.items()
    }

    rate_unc = {
        component:
            np.nanmean(values["rate_unc"][rate_mask])
        for component, values in data.items()
    }

    return (
        (
            offsets["total"],
            offsets["smb"],
            offsets["dyn"],
        ),
        (
            rates["total"],
            rates["smb"],
            rates["dyn"],
        ),
        (
            rate_unc["total"],
            rate_unc["smb"],
            rate_unc["dyn"],
        ),
    )


def read_imbie3_GIS_obs(
    state: ClimateState,
    ar5_baseline: bool = False,
    baseline_start: int | None = None,
    baseline_end: int | None = None,
):
    """Read IMBIE3 Greenland Ice Sheet observations.

    Parameters
    ----------
    state : ClimateState
        Climate state containing the projection start year and baseline
        period.
    ar5_baseline : bool, optional
        If True, use the AR5-style baseline methodology for calculating
        the initial sea-level contribution. Default is False.
    baseline_start : int, optional
        Start year of the baseline period. Defaults to
        `state.baseline_start`.
    baseline_end : int, optional
        End year of the baseline period. Defaults to `state.baseline_end`.

    Returns
    -------
    offsets : tuple of float
        Initial total, SMB, and dynamic SLE contributions, respectively,
        in metres.
    rates : tuple of float
        Mean total, SMB, and dynamic SLE contribution rates, respectively,
        in mm/yr.
    rate_unc : tuple of float
        Uncertainty in the total, SMB, and dynamic rates, respectively, in
        mm/yr.
    """

    fpath = (
        PROFSEA_DIR
        / "profsea-assets"
        / "component-obs"
        / "imbie3_greenland_mm_partitioned.csv"
    )

    return _read_imbie3_obs(
        state,
        fpath,
        ar5_baseline,
        baseline_start,
        baseline_end
    )


def read_imbie3_AIS_obs(
    state: ClimateState,
    ar5_baseline: bool = False,
    region: str | None = None,
    baseline_start: int | None = None,
    baseline_end: int | None = None,
):
    """Read IMBIE3 Antarctic Ice Sheet observations.

    Parameters
    ----------
    state : ClimateState
        Climate state containing the projection start year and baseline
        period.
    ar5_baseline : bool, optional
        If True, use the AR5-style baseline methodology for calculating
        the initial sea-level contribution. Default is False.
    region : {"eais", "wais", "peninsula", None}, optional
        Antarctic region to read. If None, use the complete Antarctic
        Ice Sheet dataset. Default is None.
    baseline_start : int, optional
        Start year of the baseline period. Defaults to
        `state.baseline_start`.
    baseline_end : int, optional
        End year of the baseline period. Defaults to `state.baseline_end`.

    Returns
    -------
    offsets : tuple of float
        Initial total, SMB, and dynamic SLE contributions, respectively,
        in metres.
    rates : tuple of float
        Mean total, SMB, and dynamic SLE contribution rates, respectively,
        in mm/yr.
    rate_unc : tuple of float
        Uncertainty in the total, SMB, and dynamic rates, respectively, in mm/yr.

    Raises
    ------
    ValueError
        If `region` is not one of `"eais"`, `"wais"`,`"peninsula"`, or None.
    """

    root = (
        PROFSEA_DIR
        / "profsea-assets"
        / "component-obs"
    )

    files = {
        None: "imbie3_antarctica_mm_partitioned.csv",
        "eais": "imbie3_east_antarctica_mm_partitioned.csv",
        "wais": "imbie3_west_antarctica_mm_partitioned.csv",
        "peninsula": "imbie3_antarctic_peninsula_mm_partitioned.csv",
    }

    try:
        fpath = root / files[region]
    except KeyError:
        raise ValueError(
            f"Unknown Antarctic region: {region!r}"
        )

    return _read_imbie3_obs(
        state,
        fpath,
        ar5_baseline,
        baseline_start,
        baseline_end
    )


def _load_zheng_component(
    state: ClimateState,
    variable: str,
    uncertainty: str | None = None,
    baseline_start: int | None = None,
    baseline_end: int | None = None
):
    """Load and rebaseline an observational sea-level component.

    Loads landwater storage, thermal expansion, or glacier observational
    data from Zheng et al. (2026 Sci. Adv.), and expresses the sea-level
    contribution relative to the requested baseline period.

    Parameters
    ----------
    state : ClimateState
        Climate state containing the baseline period.
    variable : str
        Name of the sea-level variable to load.
    uncertainty : str, optional
        Name of the corresponding uncertainty variable. Default is None.
    baseline_start : int, optional
        Start year of the baseline period. Defaults to
        `state.baseline_start`.
    baseline_end : int, optional
        End year of the baseline period. Defaults to `state.baseline_end`.

    Returns
    -------
    sle : xarray.DataArray
        Sea-level contribution relative to the requested baseline period.
    sle_err : xarray.DataArray or None
        Uncertainty in the sea-level contribution, or None if not required
        for component.
    """
    filepath = (
        PROFSEA_DIR
        / "profsea-assets"
        / "component-obs"
        / "Zheng2026.nc"
    )

    with xr.open_dataset(filepath) as ds:
        sle = ds[variable].load()
        sle_err = ds[uncertainty].load() if uncertainty else None

    baseline_start = (
        state.baseline_start if baseline_start is None else baseline_start
    )
    baseline_end = (
        state.baseline_end if baseline_end is None else baseline_end
    )

    baseline_mean = sle.sel(
        time=slice(baseline_start, baseline_end)
    ).mean("time")

    sle = sle - baseline_mean

    return sle, sle_err


def read_expansion_obs(state: ClimateState):
    """Read observed thermal expansion for component initialisation.

    Parameters
    ----------
    state : ClimateState
        Climate state containing the projection start year and baseline period.

    Returns
    -------
    sle : float
        Thermal expansion contribution in the year preceding the projection
        start year, relative to the requested baseline period, in metres.
    sle_err : float
        Uncertainty in the thermal expansion contribution, in metres.
    """
    sle, sle_err = _load_zheng_component(
        state,
        variable="iapv4",
        uncertainty="iapv4_u",
    )

    return (
        sle.sel(time=state.start_yr - 1).item() / 1e3,
        sle_err.sel(state.start_yr - 1).item() / 1e3,
    )


def read_glacier_obs(state: ClimateState):
    """Read observed glacier sea-level contribution for initialisation.

    Parameters
    ----------
    state : ClimateState
        Climate state containing the projection start year and baseline period.

    Returns
    -------
    float
        Glacier SLE contribution in the year relative to the requested baseline
        period, in metres.
    """

    sle, _ = _load_zheng_component(
        state,
        variable="glacier",
    )

    return sle.sel(time=state.start_yr - 1).item() / 1e3


def read_LWS_obs(
    state: ClimateState,
    baseline_start: int | None = None,
    baseline_end: int | None = None,
):
    """Read observed land-water-storage contribution for initialisation.

    The AR5 land-water-storage component requires both the initial SLE
    contribution and the observed contribution rate. The rate is estimated
    using a weighted linear regression over the observational period from
    `state.start_yr - 13` to `state.start_yr + 4`. This follows the AR5
    methodology, which adopts an effective contribution rate of 0.38 [0.26
    to 0.49] mm/yr over the baseline 1993-2010 (Ar5 Ch13 Table 13.1).

    Parameters
    ----------
    state : ClimateState
        Climate state containing the projection start year and baseline period.
    baseline_start : int, optional
        Start year of the baseline period. Defaults to
        `state.baseline_start`.
    baseline_end : int, optional
        End year of the baseline period. Defaults to
        `state.baseline_end`.

    Returns
    -------
    d_sle : float
        Land-water-storage SLE contribution relative to the requested baseline period,
        in metres.
    sle_rate : float
        Estimated land-water-storage SLE contribution rate.
    sle_rate_err : float
        Standard error of the estimated SLE contribution rate.
    """

    sle, sle_err = _load_zheng_component(
        state,
        variable="tws",
        uncertainty="tws_u",
        baseline_start=baseline_start,
        baseline_end=baseline_end
    )

    rate_start = state.start_yr - 13
    rate_end = state.start_yr + 4

    rate_mask = (
        (sle.time >= rate_start)
        & (sle.time <= rate_end)
    )

    x = sle.time.values[rate_mask]
    y = sle.values[rate_mask]
    sigma = sle_err.values[rate_mask]

    coeffs, cov = np.polyfit(
        x,
        y,
        deg=1,
        w=1 / sigma,
        cov=True,
    )

    sle_rate = coeffs[0]
    sle_rate_err = np.sqrt(cov[0, 0])

    d_sle = (
        sle.sel(time=state.start_yr - 1).item()
        / 1e3
    )

    return d_sle, sle_rate, sle_rate_err
