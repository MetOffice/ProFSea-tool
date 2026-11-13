from __future__ import annotations
 
import logging
import warnings
from pathlib import Path
import pdb

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
#        end_year: int = 2301,
        end_year: int = 2100,
        baseline_yrs: tuple = (1995, 2014),
        output_percentiles: list | np.ndarray = [5, 17, 50, 83, 95],
        num_local_members: int = 100000,    # LA added for local MC
        dtype: np.dtype = np.float32,
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
        num_local_members: int, optional   # (LA added for local MC)
            Number of members to use for the local Monte Carlo ensemble .
            Default is 100000.
        dtype: np.dtype, optional
            Data type for the output arrays. Default is np.float32.
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
        self.num_local_members = num_local_members    # LA added for local MC
        self.start_year = 2006
        self.n_years = self.end_year - self.start_year
        self.dtype = dtype
        
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
        Run the spatial model to generate regional sea level projections for each component.  [LA: might want to change the "-> None" to "-> xr.DataArray" or similar, as it does return something? But _arr_to_xr() may return an xr.Dataset rather than a DataArray, especially since it is receiving a dictionary containing multiple components, so think about the most appropriate type later.]

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



        #-------------------------------------------------------------------------
        # SET UP THE RANDOM NUMBER GENERATORS & SEEDS FOR THE LOCAL MONTE CARLO:
        #-------------------------------------------------------------------------
        # 1) First (shared) RNG: which member to select from the global Monte Carlo (available to all spatial component functions).
        # 2) Second (shared) RNG: which GRD method to use for that member for consistency across barystatic components (available to all spatial component functions).
        # 3) Remaining RNGs:  For each of the spatial component functions (component-specific).
        # Each RNG gets its own statistically independent stream of random numbers while still being reproducible from the original seed (default 42).
        
        child_seeds = seed_seq.spawn(len(self.components) + 2)    # LA added the +2
        
        global_member_rng = np.random.default_rng(child_seeds[0])    # LA added this line for the local MC
        grd_method_rng = np.random.default_rng(child_seeds[1])       # LA added this line for the local MC
        
        comp_rngs = {
            name: np.random.default_rng(s)
            for name, s in zip(self.components.keys(), child_seeds[2:])  # LA added the [2:]
        }

        #-------------------------------------------------------------------------
        # MAKE THE RANDOM DRAW OF GLOBAL MEMBER INDICES (LA added for local MC)
        #-------------------------------------------------------------------------
        # Establish how many global members are available and verify that the components with global_projection agree on that number:
        global_member_counts = {
            name: comp.global_projection.shape[0]
            for name, comp in self.components.items()
            if name !="gia"     # exclude GIA beacuse it does not have a global projection to select from
        }
        if not global_member_counts:
            raise ValueError(
                "No components with a global_projection were found."
            )

        if len(set(global_member_counts.values())) != 1:
            raise ValueError(
                "Global component ensembles do not have a common member count: "
                f"{global_member_counts}"
            )
        n_global_members = next(iter(global_member_counts.values()))

        # RANDOMLY SELECT n_global_members INDICES FOR THE LOCAL MONTE CARLO
        # Doing this once, to be used by all spatial components.
        global_member_indices = global_member_rng.choice(
            n_global_members,
            size=self.num_local_members,
            replace=True,
        )
        #-------------------------------------------------------------------------
        # MAKE THE RANDOM DRAW OF GRD METHODS (LA added for local MC)
        #-------------------------------------------------------------------------

        # Components that share GRD methods
        shared_grd_components = [
            "greenland_dyn",
            "greenland_smb",
            "antarctica_dyn",
            "antarctica_smb",
            "glacier",
        ]


        shared_grd_methods = sorted(
            set.intersection(
                *[
                    set(self.components[name].available_grd_methods)
                    for name in shared_grd_components
                ]
            )
        )

        if not shared_grd_methods:
            raise ValueError(
                "No common GRD methods are available across the "
                "components that require shared GRD sampling."
            )


        grd_method_choices = grd_method_rng.choice(
            shared_grd_methods,
            size=self.num_local_members,
            replace=True,
        )
    
        #--------------------------------------------------------------------------



        state = SpatialState(
            n_years=self.n_years,
            n_members=self.num_members,
            n_local_members=self.num_local_members,    # LA added for local MC
            global_member_indices=global_member_indices,    # LA added for local MC
            grd_method_choices=grd_method_choices,     # LA added for local MC
            grid_lats=self.grid_lats,
            grid_lons=self.grid_lons,
            grid_interpolation="linear",
            output_percentiles=self.output_percentiles,
            baseline_yrs=self.baseline_yrs,
            dtype=self.dtype,
            endofhistory=self.start_year,
        )

        
        # LA added for local MC (indices to be created once, before spatial tiles are processed)
        if "sterodynamic" in self.components:
            state.sterodynamic_pattern_indices = (
                self.components["sterodynamic"].sample_pattern_indices(
                    comp_rngs["sterodynamic"],
                    state.n_local_members,
                )
            )

        # LA: sample one GIA spatial pattern for each local MC member.
        # Use the reproducible GIA RNG spawned from member_seed.
        if "gia" in self.components:
            gia_component = self.components["gia"]

            if gia_component.sample_spatial:
                state.gia_pattern_indices = (
                    gia_component.sample_pattern_indices(
                        comp_rngs["gia"],
                        state.n_local_members,
                    )
                )

        
#        tile_components = {
#            "sterodynamic",
#            "greenland_dyn",
#            "greenland_smb",
#            "landwater",
#            "antarctica_dyn",
#            "antarctica_smb",
#            "glacier",
#            "gia",
#        }

        lat_slice=slice(50, 55)
        lon_slice=slice(20, 25)

        n_block_lats = len(self.grid_lats[lat_slice])
        n_block_lons = len(self.grid_lons[lon_slice])
        
        spatial_projections = {}
        console.print()  # Add a blank line for better readability in the console output
        #for name, comp in track(
        #    self.components.items(), description="Spatialising components..."
        #):

        for name, comp in self.components.items():   # LA removed progress tracker for easier use of pdb
            # lazy_projection does not calculate the projection yet - it returns an object describing the calculation which will be executed later
            #lazy_projection = comp.project(state, comp_rngs[name])  
            # Rechunk before saving to optimize memory during writing
            #lazy_projection = lazy_projection.rechunk({0: -1, 1: -1, 2: 10, 3: 10})   # keep dims 0 and 1 together, but split the spatial dims into 10x10 tiles


            lazy_projection = comp.project(
                state,
                comp_rngs[name],
                lat_slice=lat_slice,
                lon_slice=lon_slice,
            )

            
            expected_shape = (
                state.n_local_members,
                state.n_years,
                n_block_lats,
                n_block_lons,
            )

            if lazy_projection.shape != expected_shape:
                raise ValueError(
                    f"{name} returned shape {lazy_projection.shape}; "
                    f"expected {expected_shape}"
                )
            
            lazy_projection = lazy_projection.rechunk(
                {
                    0: 2000,  # local MC members per chunk
                    1: 10,     # years per chunk
                    2: 5,      # latitudes per chunk
                    3: 5,      # longitudes per chunk
                }
            )
            spatial_projections[name] = lazy_projection
            
        console.print()

        # Sum all components member by member before percentile reduction.
        total_members = self._sum_component_members(spatial_projections)

        # Calculate percentiles from each component's local Monte Carlo
        component_percentiles = {
            name: self._calculate_member_percentiles(member_array)
            for name, member_array in spatial_projections.items()
        }

        # Calculate percentiles from the component-total local Monte Carlo
        # (independent from the member-wise total)
        total_percentiles = self._calculate_member_percentiles(
            total_members
        )

        # Collect the percentiles together
        percentile_projections = {**component_percentiles, "total_rsl": total_percentiles,}


        pdb.set_trace()
        
        logger.info(
            "Local MC member-wise total shape: %s",
            total_members.shape,
        )

        pdb.set_trace()
        
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
        total_rsl = xr.concat(components.values(), dim="component").sum(
            dim="component", skipna=False
        )
        total_rsl.attrs["units"] = "m"
        components["total_rsl"] = total_rsl
        return total_rsl

    
    def _sum_component_members(
        self,
        component_members: dict[str, da.Array],
    ) -> da.Array:
        """
        Sum local Monte Carlo spatial components member by member.

        Parameters
        ----------
        component_members : dict[str, da.Array]
            Component arrays with common shape
            (n_local_members, n_years, n_lats, n_lons).

        Returns
        -------
        da.Array
            Member-wise total with the same shape as each component.
        """
        if not component_members:
            raise ValueError(
                "No local Monte Carlo component arrays were supplied."
            )

        shapes = {
            name: arr.shape
            for name, arr in component_members.items()
        }

        expected_shape = next(iter(shapes.values()))

        if any(shape != expected_shape for shape in shapes.values()):
            raise ValueError(
                "All local Monte Carlo component arrays must have the "
                f"same shape. Received: {shapes}"
            )

        if expected_shape[0] != self.num_local_members:
            raise ValueError(
                "Local Monte Carlo member dimension does not match "
                f"num_local_members={self.num_local_members}. "
                f"Received shape {expected_shape}."
            )

        total_members = da.stack(
            list(component_members.values()),
            axis=0,
        ).sum(axis=0)

        return total_members


    def _calculate_member_percentiles(
        self,
        member_array: da.Array,
    ) -> da.Array:
        """
        Calculate requested percentiles across the local Monte Carlo
        member dimension.

        Parameters
        ----------
        member_array : da.Array
            Local Monte Carlo array with shape
            (n_local_members, n_years, n_lats, n_lons).

        Returns
        -------
        da.Array
            Percentile-reduced array with shape
            (n_percentiles, n_years, n_lats, n_lons).
        """
        if self.output_percentiles is None:
            raise ValueError(
                "output_percentiles must be set for percentile reduction."
            )

        if member_array.ndim != 4:
            raise ValueError(
                "Expected a 4-D local Monte Carlo array "
                "(member, time, lat, lon), "
                f"but received shape {member_array.shape}."
            )

        if member_array.shape[0] != self.num_local_members:
            raise ValueError(
                "Local member dimension does not match "
                f"num_local_members={self.num_local_members}. "
                f"Received shape {member_array.shape}."
            )

        # Exact percentile reduction requires all members for each
        # time/lat/lon cell to be available in one member chunk.
        member_array = member_array.rechunk({0: -1})

        return da.nanpercentile(
            member_array,
            q=self.output_percentiles,
            axis=0,
        )
