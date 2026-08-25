from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import dask
import dask.array as da
import numpy as np
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
        msg = "'components' must be a dictionary of the form {name: component_instance}."
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
                f"expected {expected_type.__name__} instance, got "
                f"{type(comp).__name__}"
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
    ds.attrs["source"] = "ProFSea v3.0"
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
