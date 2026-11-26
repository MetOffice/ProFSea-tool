from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from profsea.utils import (
    PROFSEA_ASSETS_URL,
    PROFSEA_ASSETS_VERSION,
    fetch_zenodo_fingerprints,
    save_components,
)
from profsea.utils.ui import print_spatial_preflight

from .base import SpatialComponent
from .regional import RegionalEnsemble
from .state import SpatialState

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PROFSEA_DIR = Path(__file__).resolve().parent.parent.parent


class Spatial(RegionalEnsemble):
    """Regional sea-level projections on a regular latitude-longitude grid.

    Takes spatial components (fingerprints, sterodynamics, GIA) built from a
    full ``Global`` ensemble and runs the regional Monte Carlo described in
    :class:`~profsea.components.core.regional.RegionalEnsemble`. Outputs have
    dimensions (percentile, time, lat, lon), or (member, time, lat, lon) when
    every member is kept.
    """

    def __init__(
        self,
        components: dict[str, SpatialComponent],
        grid_config: dict = None,
        grid_interpolation: str = "linear",
        start_yr: int = 2006,
        end_yr: int = 2301,
        baseline_yrs: tuple = (1995, 2014),
        output_percentiles: list | np.ndarray = [5, 17, 50, 83, 95],
        num_spatial_members: int = 1000,
        dtype: np.dtype = np.float32,
        compute_total: bool = True,
        chunk_size_mb: float = 64,
    ) -> None:
        """
        Parameters
        ----------
        components: dict
            Spatial components keyed by name, e.g.
            ``{"glacier": Fingerprint(...), "gia": GIA()}``. All components
            driven by a global projection must come from the same ``Global``
            run.
        grid_config: dict, optional
            Grid definition with keys 'start_lon', 'end_lon', 'step_lon',
            'start_lat', 'end_lat' and 'step_lat' (cell centres, in degrees).
            Longitudes may use either convention; outputs are on ascending
            longitudes in [-180, 180), so a 0-360 grid comes back as -180-180.
            Default is a 1-degree global grid.
        grid_interpolation: str, optional
            Interpolation method for regridding patterns. Currently not applied:
            patterns are always interpolated linearly. Default is 'linear'.
        start_yr: int, optional
            The projections run from start_yr to end_yr - 1. Default is 2006.
        end_yr: int, optional
            The projections run from start_yr to end_yr - 1. Default is 2301.
        baseline_yrs: tuple, optional
            First and last year of the baseline period. Default is (1995, 2014).
        output_percentiles: list or np.ndarray, optional
            Percentiles to output, taken over the Monte Carlo members at every
            grid point and year. If None or empty, every member is output.
            Default is [5, 17, 50, 83, 95].
        num_spatial_members: int, optional
            Number of Monte Carlo members, drawn from the global ensemble with an
            equal share per climate trajectory. If equal to the global ensemble
            size, every global member is used exactly once. Default is 1000.
        dtype: np.dtype, optional
            Data type of the outputs. Default is np.float32.
        compute_total: bool, optional
            If True, add "total_rsl": the components summed member by member,
            before any percentiles are taken. Default is True.
        chunk_size_mb: float, optional
            Approximate memory, in MB, for one block of the total together with
            the blocks of every component summed into it. Each Dask worker
            thread needs a few times this. Default is 64.
        """
        fetch_zenodo_fingerprints(
            zenodo_url=PROFSEA_ASSETS_URL,
            data_dir=PROFSEA_DIR,
            expected_folder_name="profsea-assets",
            version=PROFSEA_ASSETS_VERSION,
        )

        self.grid_interpolation = grid_interpolation
        self.start_yr = start_yr
        self.end_yr = end_yr
        self.baseline_yrs = baseline_yrs
        self.output_percentiles = output_percentiles
        self.n_years = self.end_yr - self.start_yr
        self.num_spatial_members = num_spatial_members
        self.dtype = dtype
        self.compute_total = compute_total
        self.chunk_size_mb = chunk_size_mb

        self._set_components(components, "Spatial")

        # Length of the outputs' leading axis: percentiles or members
        if self._has_percentiles():
            self.num_members = len(self.output_percentiles)
        else:
            self.num_members = self.num_spatial_members

        if grid_config is None:  # 1-degree global grid of cell centres
            grid_config: dict = {
                "start_lon": -179.5,
                "end_lon": 179.5,
                "step_lon": 1.0,
                "start_lat": -89.5,
                "end_lat": 89.5,
                "step_lat": 1.0,
            }

        # Define the grid coordinates. Patterns are regridded onto longitudes
        # in [-180, 180) in ascending order (interpolate_to_grid), so the grid
        # takes that form too and the output labels match the data: wrap
        # values outside the range (values inside are kept exactly), sort, and
        # drop the duplicate a closed interval gives (e.g. -180 and 180).
        lons = np.arange(
            grid_config["start_lon"],
            grid_config["end_lon"] + grid_config["step_lon"],
            grid_config["step_lon"],
        )
        in_range = (lons >= -180) & (lons < 180)
        self.grid_lons = np.unique(np.where(in_range, lons, (lons + 180) % 360 - 180))
        self.grid_lats = np.arange(
            grid_config["start_lat"],
            grid_config["end_lat"] + grid_config["step_lat"],
            grid_config["step_lat"],
        )

        logger.info(
            f"Baseline period = {self.baseline_yrs[0]} to {self.baseline_yrs[1]}"
        )

        # Size of all outputs together: one (percentiles or members, years, lat,
        # lon) array per component, plus the total
        self.n_output_arrays = len(self.components) + int(self.compute_total)
        self.output_gb_per_array = (
            self.num_members
            * self.n_years
            * len(self.grid_lats)
            * len(self.grid_lons)
            * np.dtype(self.dtype).itemsize
        ) / 1e9
        self.output_gb = self.n_output_arrays * self.output_gb_per_array
        if self.output_gb > 20:
            logger.warning(
                f"[bold red]The {self.n_output_arrays} outputs total about "
                f"[bold blue]{self.output_gb:.1f} GB[/bold blue] uncompressed, all of "
                f"which is held in memory if they are computed at once or saved to "
                f"NetCDF. Consider fewer members, a coarser grid, output percentiles "
                f"or saving to Zarr.[/bold red]"
            )

    # Shared output writer (profsea.utils.save_components), used as a method
    save_components = save_components

    def _spatial_coords(self) -> tuple[list[str], dict, str]:
        """Grid dimensions and coordinates (see RegionalEnsemble._spatial_coords)."""
        return (
            ["lat", "lon"],
            {"lat": self.grid_lats, "lon": self.grid_lons},
            "Regional",
        )

    def run(self, member_seed: int = 42) -> dict[str, xr.DataArray]:
        """
        Run the spatial model to generate regional sea level projections for each component.

        Parameters
        ----------
        member_seed: int, optional
            Seed for the Monte Carlo member and pattern draws. Default is 42.

        Returns
        -------
        Dict[str, xr.DataArray]
            Lazy projections for each component, plus "total_rsl" if
            compute_total, of shape (percentiles or members, n_years, n_lats,
            n_lons). Nothing is computed until the results are used or saved.
        """
        print_spatial_preflight(self)
        logger.info(
            f"Simulating {len(self.components)} sea-level components: {', '.join(self.components.keys())}"
        )
        self.results = self._run_ensemble(
            member_seed,
            SpatialState,
            (len(self.grid_lats), len(self.grid_lons)),
            "Spatialising components...",
            grid_lats=self.grid_lats,
            grid_lons=self.grid_lons,
            grid_interpolation=self.grid_interpolation,
        )
        return self.results
