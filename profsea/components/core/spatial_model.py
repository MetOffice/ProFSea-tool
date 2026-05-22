import os
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
            spatial_projections[name] = comp.project(state, comp_rngs[name])

        self.results = spatial_projections
        return spatial_projections

    def _save_projections(self, montecarlo_R: da.array, component: str) -> None:
        """
        Save the regional sea level projections to a file.
        :param montecarlo_R: regional sea level projections
        :param component: sea level component
        :param scenario: emission scenario
        :param percentile: percentiles used for spatial projections
        """
        # Save data in netcdf format (Assuming first dimension is percentile, but can be more general percentile/ensemble)
        xr_dataArray = xr.DataArray(
            montecarlo_R,
            dims=["percentile", "time", "lat", "lon"],
            coords={
                "percentile": self.output_percentiles,
                "time": np.arange(2006, montecarlo_R.shape[1] + 2006),
                "lat": self.grid_lats,
                "lon": self.grid_lons,
            },
        )
        xr_dataArray.attrs["units"] = "m"
        xr_dataArray.attrs["long_name"] = f"Regional {component} sea-level projections"
        xr_dataArray.attrs["source"] = "ProFSea-Climate v0.1"
        ds = xr_dataArray.to_dataset(name=component)

        file_header = f"{component}_{self.scenario}_projection_{self.end_year}"
        R_file = "_".join([file_header, "regional"]) + ".nc"
        encoding = {component: {"zlib": True, "complevel": 5, "dtype": "float32"}}
        ds.to_netcdf(
            os.path.join(self.output_dir, R_file), encoding=encoding, compute=True
        )
