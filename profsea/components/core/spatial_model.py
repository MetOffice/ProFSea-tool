from __future__ import annotations

import logging
import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from rich.console import Console
from rich.progress import track

from profsea.utils import fetch_zenodo_fingerprints, save_components
from profsea.utils.ui import print_spatial_preflight

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

    # Instance method!
    save_components = save_components

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
        print_spatial_preflight(self)
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
