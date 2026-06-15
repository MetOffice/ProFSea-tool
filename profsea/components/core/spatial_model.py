from __future__ import annotations

import logging
import os
import warnings
import zipfile
from pathlib import Path

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
    track,
)

from .base import Component
from .state import SpatialState

logger = logging.getLogger(__name__)
console = Console()
warnings.filterwarnings("ignore")

PROFSEA_DIR = Path(__file__).resolve().parent.parent.parent
ZENODO_DOWNLOAD_LINK = (
    "https://zenodo.org/records/20427061/files/profsea-assets.zip?download=1"
)


class Spatial:
    """Spatial sea level rise component emulator."""

    def __init__(
        self,
        components: dict[str, Component],
        grid_config: dict = None,
        grid_interpolation: str = "linear",
        end_year: int = 2301,
        baseline_yrs: tuple = (1995, 2014),
        output_percentiles: list | np.ndarray = [5, 17, 50, 83, 95],
    ) -> None:
        """
        Parameters
        ----------
        components: dict
            Dictionary of spatial components to include in the model. Keys should be the component names and values should be instances of Component subclasses.
        grid_config: dict, optional
            Dictionary defining the grid configuration with keys 'start_lon', 'end_lon', 'step_lon
            'start_lat', 'end_lat', 'step_lat'. If None, defaults to a 1-degree global grid.
        grid_interpolation: str, optional
            Interpolation method to use when interpolating patterns to the target grid. Default is 'linear'.
        end_year: int, optional
            The final year of the projections. Default is 2301.
        baseline_yrs: tuple, optional
            Tuple defining the start and end years of the baseline period for calculating anomalies. Default is (1995, 2014).
        output_percentiles: list or np.ndarray, optional
            List or array of percentiles to sample from the ensemble for output. If None, outputs all members. Default is [5, 17, 50, 83, 95].
        """
        # Define the path where the data should live

        fetch_zenodo_fingerprints(
            zenodo_url=ZENODO_DOWNLOAD_LINK,
            data_dir=PROFSEA_DIR,
            expected_folder_name="profsea-assets",
        )

        self.components = components
        self.end_year = end_year
        self.baseline_yrs = baseline_yrs
        self.output_percentiles = output_percentiles
        self.start_year = 2006
        self.n_years = self.end_year - self.start_year

        if self.output_percentiles is not None and len(self.output_percentiles) > 0:
            self.num_members = len(self.output_percentiles)
        else:
            self.num_members = next(
                iter(self.components.values())
            ).global_projection.shape[0]

        if grid_config is None:
            grid_config: dict = {
                "start_lon": -179.5,
                "end_lon": 179.5,
                "step_lon": 1.0,
                "start_lat": -89.5,
                "end_lat": 89.5,
                "step_lat": 1.0,
            }

        # Define the grid coordinates
        self.grid_lons = np.arange(
            grid_config["start_lon"],
            grid_config["end_lon"] + grid_config["step_lon"],
            grid_config["step_lon"],
        )
        self.grid_lats = np.arange(
            grid_config["start_lat"],
            grid_config["end_lat"] + grid_config["step_lat"],
            grid_config["step_lat"],
        )

        logger.info(
            f"Baseline period = {self.baseline_yrs[0]} to {self.baseline_yrs[1]}"
        )

        # Log the size of each component and provide an estimate of their memory usage
        # Output shape will be (num_members, n_years, n_lats, n_lons)
        bytes_per_element = 8  # Assuming float64. Use 4 if strictly float32.

        future_size = (
            self.num_members
            * self.n_years
            * len(self.grid_lats)
            * len(self.grid_lons)
            * bytes_per_element
        ) / 1e9

        for name, comp in self.components.items():
            # Warn if memory usage is going to be large
            if future_size > 20:
                logger.warning(
                    f"[bold red]The output array for component "
                    f"[bold blue]'{name}'[/bold blue] requires a large amount "
                    f"of memory [bold blue]({future_size:.2f} GB)[/bold blue]. "
                    f"Consider reducing the number of members, the grid "
                    f"resolution or take percentiles.[/bold red]"
                )

    def _arr_to_xr(self, arr_dict: dict[str, da.Array]) -> dict[str, xr.DataArray]:
        """
        Convert a dictionary of Dask arrays to a dictionary of xarray DataArrays with appropriate coordinates and metadata.

        Parameters
        ----------
        arr_dict: Dict[str, da.Array]
            Dictionary where keys are component names and values are Dask arrays of shape (n_members, n_years, n_lats, n_lons).

        Returns
        -------
        Dict[str, xr.DataArray]
            Dictionary where keys are component names and values are xarray DataArrays with dimensions (member, time, lat, lon) and appropriate coordinates.
        """
        xr_dict = {}
        member_dim = "percentile" if self.output_percentiles is not None else "member"

        for name, arr in arr_dict.items():
            xr_dict[name] = xr.DataArray(
                arr,
                dims=[member_dim, "time", "lat", "lon"],
                coords={
                    member_dim: self.output_percentiles
                    if self.output_percentiles is not None
                    else np.arange(arr.shape[0]),
                    "time": np.arange(self.start_year, self.start_year + arr.shape[1]),
                    "lat": self.grid_lats,
                    "lon": self.grid_lons,
                },
                attrs={
                    "units": "m",
                    "long_name": f"Regional {name} sea-level projections",
                    "source": "ProFSea-Climate v0.1",
                },
            )

        return xr_dict

    def run(self, member_seed: int = 42) -> None:
        """
        Run the spatial model to generate regional sea level projections for each component.

        Parameters
        ----------
        member_seed: int, optional
            Seed for random number generation to ensure reproducibility of member sampling. Default is 42.

        Returns
        -------
        Dict[str, da.Array]
            Dictionary of spatial projections for each component, where keys are component names and values are Dask arrays of shape (n_members, n_years, n_lats, n_lons).
        """
        seed_seq = np.random.SeedSequence(member_seed)

        logger.info(
            f"Simulating {len(self.components)} sea-level components: {', '.join(self.components.keys())}"
        )

        state = SpatialState(
            n_years=self.n_years,
            n_members=self.num_members,
            grid_lats=self.grid_lats,
            grid_lons=self.grid_lons,
            grid_interpolation="linear",
            output_percentiles=self.output_percentiles,
            baseline_yrs=self.baseline_yrs,
        )

        child_seeds = seed_seq.spawn(len(self.components))
        comp_rngs = {
            name: np.random.default_rng(s)
            for name, s in zip(self.components.keys(), child_seeds)
        }

        spatial_projections = {}
        console.print()  # Add a blank line for better readability in the console output
        for name, comp in track(
            self.components.items(), description="Spatialising components..."
        ):
            lazy_projection = comp.project(state, comp_rngs[name])

            # Rechunk before saving to optimize memory during writing
            lazy_projection = lazy_projection.rechunk({0: -1, 1: -1, 2: 10, 3: 10})
            spatial_projections[name] = lazy_projection
        console.print()

        # Put into xarray datasets for easier saving and metadata handling
        spatial_projections_xr = self._arr_to_xr(spatial_projections)
        self.results = spatial_projections_xr
        return self.results

    def sum_components(self, components: dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Sum the spatial components to get total sea-level change.

        Parameters
        ----------
        components: Dict[str, xr.DataArray]
            Dictionary of spatial component DataArrays.

        Returns
        -------
        xr.DataArray
            DataArray of the summed spatial projections.
        """
        # Using xr.concat preserves all dimensions and coordinates, and summing
        # along the new dimension handles the underlying dask arrays cleanly.
        total_rsl = xr.concat(components.values(), dim="component").sum(dim="component")

        # Optionally, apply attributes so it matches the other DataArrays
        total_rsl.attrs = {
            "units": "m",
            "long_name": "Regional total sea-level projections",
            "source": "ProFSea-Climate v0.1",
        }

        components["total_rsl"] = total_rsl
        return total_rsl

    def save_components(
        self,
        components: dict[str, xr.DataArray],
        scenario_name: str,
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

        Returns
        -------
        None
        """
        # Wrap dataarrays in a single Dataset for saving
        ds = xr.Dataset(components)

        output_format = output_format.lower()
        if output_format not in ["netcdf", "zarr"]:
            raise ValueError("output_format must be either 'netcdf' or 'zarr'.")

        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        encoding = {}
        # Sort out Zarr encoding
        if output_format == "zarr":
            import numcodecs
            from numcodecs.zarr3 import Blosc

            compressor = Blosc(
                cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
            )

        # Set the encoding/compression for each variable based on the output format
        for name, component in components.items():
            if output_format == "netcdf":
                encoding[name] = {"zlib": True, "complevel": 1, "dtype": "float32"}
            elif output_format == "zarr":
                encoding[name] = {"compressor": compressor, "dtype": "float32"}

        file_header = f"{scenario_name}_spatial_projection"

        # Stream the computation and write to disk
        if output_format == "netcdf":
            out_path = os.path.join(output_dir, f"{file_header}.nc")

            with console.status(
                "[bold cyan]Computing and saving NetCDF...[/bold cyan]",
                spinner="dots",
            ):
                ds.compute().to_netcdf(out_path, encoding=encoding)

            logger.info(
                f"[bold green]✓ Successfully saved NetCDF:[/bold green] {out_path}"
            )

        elif output_format == "zarr":
            out_path = os.path.join(output_dir, f"{file_header}.zarr")

            with console.status(
                "[bold cyan]Streaming computation and saving Zarr...[/bold cyan]",
                spinner="dots",
            ):
                ds.to_zarr(out_path, encoding=encoding, mode="w", compute=True)

            logger.info(
                f"[bold green]✓ Successfully saved Zarr:[/bold green] {out_path}"
            )

        logger.info(
            "Output shape was " + str(ds[name].shape) + " (members, time, lat, lon)"
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
