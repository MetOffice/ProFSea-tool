import os
from pathlib import Path
from typing import Dict
import warnings

import dask.array as da
import numpy as np
from rich.console import Console
from rich.progress import track
import xarray as xr

from .state import SpatialState
from .base import Component

console = Console()
warnings.filterwarnings("ignore")


class Spatial:
    """Spatial sea level rise component emulator."""

    def __init__(
        self,
        components: Dict[str, Component],
        grid_config: dict = None,
        grid_interpolation: str = "linear",
        end_year: int = 2301,
        baseline_yrs: tuple = (1986, 2005),
        output_percentiles: list | np.ndarray = [5, 17, 50, 83, 95],
    ):
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
            Tuple defining the start and end years of the baseline period for calculating anomalies. Default is (1986, 2005).
        output_percentiles: list or np.ndarray, optional
            List or array of percentiles to sample from the ensemble for output. If None, outputs all members. Default is [5, 17, 50, 83, 95].
        """

        self.components = components
        self.end_year = end_year
        self.baseline_yrs = baseline_yrs
        self.output_percentiles = output_percentiles
        self.start_year = 2006
        self.n_years = self.end_year - self.start_year

        if output_percentiles:
            self.num_members = len(output_percentiles)
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

        console.log(
            f"Baseline period = {self.baseline_yrs[0]} to {self.baseline_yrs[1]}"
        )

        # Log the size of each component and provide an estimate of their memory usage
        for name, comp in self.components.items():
            if not output_percentiles:
                comp_size = comp.global_projection.nbytes / 1e9
                future_size = (
                    comp_size
                    * self.num_members
                    * len(self.grid_lats)
                    * len(self.grid_lons)
                )
            else:
                comp_size = (
                    comp.global_projection[: len(output_percentiles)].nbytes / 1e9
                )
                future_size = comp_size * len(self.grid_lats) * len(self.grid_lons)

            # Warn if memory usage is going to be large
            if future_size > 20:
                console.log(
                    f"[bold red]Warning: the output array for component "
                    f"[bold blue]'{name}'[/bold blue] requires a large amount "
                    f"of memory [bold blue]({future_size:.2f} GB)[/bold blue]. "
                    f"Consider reducing the number of members, the grid "
                    f"resolution or take percentiles.[/bold red]"
                )

    def _calc_baseline_period(
        self,
    ) -> float:  # TODO: move this to the GIA componenent once made.
        """
        Baseline years used for IPCC AR5 and Palmer et al 2020 -- 1986-2005
        :param yrs: years of the projections
        :return: baseline years
        """
        midyr = (
            self.baseline_yrs[1] - self.baseline_yrs[0] + 1
        ) * 0.5 + self.baseline_yrs[0]
        return self.start_year - midyr

    def run(self, scenario: str, member_seed: int = 42) -> None:
        """
        Calculates global and regional component part contributions to sea level
        change.
        :param mcdir: location of Monte Carlo time series for new projections
        :param components: sea level components
        :param scenario: emission scenario
        :param yrs: years of the projections
        :param array_dims: Array of nesm, nsmps and nyrs
            nesm --> Number of ensemble members in time series
            nsmps --> Determine the number of samples you wish to make
            nyrs --> Number of years in each projection time series
        :return: montecarlo_G (global contribution to sea level rise) and
            montecarlo_R (regional contribution to sea level change)
        """
        seed_seq = np.random.SeedSequence(member_seed)

        console.log(
            f"Simulating {len(self.components)} sea-level components...: {', '.join(self.components.keys())}"
        )

        state = SpatialState(
            scenario=scenario,
            n_years=self.n_years,
            n_members=self.num_members,
            grid_lats=self.grid_lats,
            grid_lons=self.grid_lons,
            grid_interpolation="linear",
            output_percentiles=self.output_percentiles,
        )

        child_seeds = seed_seq.spawn(len(self.components))
        comp_rngs = {
            name: np.random.default_rng(s)
            for name, s in zip(self.components.keys(), child_seeds)
        }

        spatial_projections = {}
        for name, comp in track(
            self.components.items(), description="Spatialising components..."
        ):
            lazy_projection = comp.project(state, comp_rngs[name])

            # Rechunk before saving to optimize memory during writing
            lazy_projection = lazy_projection.rechunk({0: -1, 1: -1, 2: 10, 3: 10})
            spatial_projections[name] = lazy_projection

        self.results = spatial_projections
        return spatial_projections

    def save_components(
        self,
        components: Dict[str, da.Array],
        scenario_name: str,
        output_dir: str = ".",
        output_format: str = "netcdf",
    ) -> None:
        """
        Stream all regional sea level projections to disk in a single file/store.

        Parameters
        ----------
        components: Dict[str, da.Array]
            Dictionary of component names and their corresponding Dask arrays.
        output_format: str
            Format to save the output in. Must be either 'netcdf' or 'zarr'.
        output_dir: str
            Directory to save components to.
        scenario_name: str
            Name of the scenario you've run the emulator for.
        """
        output_format = output_format.lower()
        if output_format not in ["netcdf", "zarr"]:
            raise ValueError("output_format must be either 'netcdf' or 'zarr'.")

        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        ds = xr.Dataset()
        member_dim = "percentile" if self.output_percentiles is not None else "member"

        # Build the shared coordinates once to ensure alignment
        # Extracting time dynamically based on the shape of the first component
        sample_shape = next(iter(components.values())).shape

        coords = {
            member_dim: self.output_percentiles
            if self.output_percentiles is not None
            else np.arange(sample_shape[0]),
            "time": np.arange(2006, sample_shape[1] + 2006),
            "lat": self.grid_lats,
            "lon": self.grid_lons,
        }

        encoding = {}
        if output_format == "zarr":
            import numcodecs

            compressor = numcodecs.Blosc(
                cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
            )

        # Loop through the isDask arrays and add them to the single Dataset
        for name, component in components.items():
            xr_dataArray = xr.DataArray(
                component,
                dims=[member_dim, "time", "lat", "lon"],
                coords=coords,
            )
            xr_dataArray.attrs["units"] = "m"
            xr_dataArray.attrs["long_name"] = f"Regional {name} sea-level projections"
            xr_dataArray.attrs["source"] = "ProFSea-Climate v0.1"

            ds[name] = xr_dataArray

            # Populate the encoding dictionary variable-by-variable
            if output_format == "netcdf":
                encoding[name] = {"zlib": True, "complevel": 5, "dtype": "float32"}
            elif output_format == "zarr":
                encoding[name] = {"compressor": compressor, "dtype": "float32"}

        # Define output paths
        file_header = f"{scenario_name}_spatial_projection"

        # Stream the computation and write to disk
        if output_format == "netcdf":
            out_path = os.path.join(output_dir, f"{file_header}.nc")

            # The spinner will animate while to_netcdf is blocking
            with console.status(
                "[bold cyan]Streaming computation and saving NetCDF...[/bold cyan]",
                spinner="dots",
            ):
                ds.to_netcdf(out_path, encoding=encoding, compute=True)

            console.log(
                f"[bold green]✓ Successfully saved NetCDF:[/bold green] {out_path}"
            )

        elif output_format == "zarr":
            out_path = os.path.join(output_dir, f"{file_header}.zarr")

            with console.status(
                "[bold cyan]Streaming computation and saving Zarr...[/bold cyan]",
                spinner="dots",
            ):
                ds.to_zarr(out_path, encoding=encoding, mode="w", compute=True)

            console.log(
                f"[bold green]✓ Successfully saved Zarr:[/bold green] {out_path}"
            )

        console.log(
            "Output shape was " + str(ds[name].shape) + " (members, time, lat, lon)"
        )
