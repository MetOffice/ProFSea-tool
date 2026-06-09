import os
from pathlib import Path
from typing import Dict
import zipfile

import dask.array as da
import numpy as np
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)
import requests
from scipy.spatial.distance import cdist
import xarray as xr

console = Console()


def sample_members_2D(array: np.ndarray, percentiles: list | np.ndarray) -> np.ndarray:
    """Sample real ensemble members from a 2D numpy array."""
    # Caculate statistical timeseries, then match with closest real timeseries
    array_percentiles = np.percentile(array, percentiles, axis=0)
    distances = cdist(array_percentiles, array)
    mem_indices = np.argmin(distances, axis=1)
    return array[mem_indices]


def interpolate(data: da.array, lats: int, lons: int) -> np.ndarray:
    """ """
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
    """
    # Normalize source longitudes to [-180, 180) and sort monotonically
    data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
    data = data.sortby("lon")

    # Normalize target longitudes to [-180, 180) and sort
    target_lons_norm = np.sort(((target_lons + 180) % 360) - 180)

    # Interpolate!
    data_interp = data.interp(
        lat=target_lats, lon=target_lons_norm, method=grid_interpolation
    )
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
        # Split over lines for readability
        raise ValueError(
            f"Array should have shape (realisation, time) with time \
                dimension of length {n_time}. Got {array.shape}."
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
        console.log("[bold green]✓ ProFSea assets found locally![/bold green]")
        return

    # Create the base directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "temp_fingerprints.zip"

    console.log(f"Initiating download from {zenodo_url}...")

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
        console.log(f"[bold red]Failed to download data: {e}[/bold red]")
        if zip_path.exists():
            zip_path.unlink()  # Clean up partial downloads
        raise

    console.log("Extracting data...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Filter out the __MACOSX directory and its contents
            valid_members = [
                member
                for member in zip_ref.namelist()
                if not member.startswith("__MACOSX/") and not member.startswith("._")
            ]
            zip_ref.extractall(data_dir, members=valid_members)

        console.log(
            f"[bold green]✓ Successfully extracted data to {data_dir}[/bold green]"
        )
    except zipfile.BadZipFile:
        console.log(
            "[bold red]Error: Downloaded file is not a valid zip archive.[/bold red]"
        )
        raise
    finally:
        # 4. Clean up the zip file
        if zip_path.exists():
            zip_path.unlink()


def save_components(
    self,
    components: Dict[str, xr.DataArray],
    scenario_name: str,
    output_prefix: str = "projection",
    output_dir: str = ".",
    output_format: str = "zarr",
) -> None:
    """
    Stream all regional sea level projections to disk in a single file/store.

    Parameters
    ----------
    components: Dict[str, xr.DataArray]
        Dictionary of component names and their corresponding Xarray DataArrays.
    output_format: str
        Format to save the output in. Must be either 'netcdf' or 'zarr'.
    output_dir: str
        Directory to save components to.
    scenario_name: str
        Name of the scenario you've run the emulator for.
    output_prefix: str
        Prefix for the output file name (e.g., 'projection' will result in 'ssp

    Returns
    -------
    None
    """
    ds = xr.Dataset(components)

    output_format = output_format.lower()
    if output_format not in ["netcdf", "zarr"]:
        raise ValueError("output_format must be either 'netcdf' or 'zarr'.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    encoding = {}

    # Sort out Zarr encoding
    if output_format == "zarr":
        import numcodecs
        from numcodecs.zarr3 import Blosc

        compressor = Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    # Set the encoding/compression for each variable based on the output format
    for name, component in components.items():
        if output_format == "netcdf":
            encoding[name] = {"zlib": True, "complevel": 1, "dtype": "float32"}
        elif output_format == "zarr":
            encoding[name] = {"compressor": compressor, "dtype": "float32"}

    file_name = f"{scenario_name}_{output_prefix}"

    # Stream the computation and write to disk
    if output_format == "netcdf":
        out_path = os.path.join(output_dir, f"{file_name}.nc")
        with console.status(
            "[bold cyan]Computing and saving NetCDF...[/bold cyan]", spinner="dots"
        ):
            ds.compute().to_netcdf(out_path, encoding=encoding)
        console.log(f"[bold green]✓ Successfully saved NetCDF:[/bold green] {out_path}")

    elif output_format == "zarr":
        out_path = os.path.join(output_dir, f"{file_name}.zarr")
        with console.status(
            "[bold cyan]Streaming computation and saving Zarr...[/bold cyan]",
            spinner="dots",
        ):
            ds.to_zarr(out_path, encoding=encoding, mode="w", compute=True)
        console.log(f"[bold green]✓ Successfully saved Zarr:[/bold green] {out_path}")

    dims_str = ", ".join(ds[name].dims)
    console.log(f"Output shape was {ds[name].shape} ({dims_str})")
