from __future__ import annotations
 
import logging
import warnings
from pathlib import Path
import pdb
import shutil

import dask
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

    # Now pass lat_slice and lon_slice so _arr_to_xr knows which spatial block the reduced arrays belong to
    def _arr_to_xr(self, arr_dict: dict[str, da.Array], lat_slice: slice, lon_slice: slice,) -> dict[str, xr.DataArray]:
        """
        Convert percentile-reduced Dask arrays to xarray DataArrays with
        appropriate coordinates and metadata.

        Parameters
        ----------
        arr_dict : dict[str, da.Array]
            Dictionary where keys are component names and values are
            percentile-reduced Dask arrays with shape
            (n_percentiles, n_years, n_lats, n_lons).

        lat_slice : slice
            Slice defining the latitude indices represented by the arrays.

        lon_slice : slice
            Slice defining the longitude indices represented by the arrays.

        Returns
        -------
        dict[str, xr.DataArray]
            Dictionary where keys are component names and values are xarray
            DataArrays with dimensions (percentile, time, lat, lon) and
            coordinates corresponding to the requested percentiles, projection
            years, and spatial block.

        """
        
        output_lats = self.grid_lats[lat_slice]
        output_lons = self.grid_lons[lon_slice]

        expected_shape = (len(self.output_percentiles), self.n_years, len(output_lats), len(output_lons),)

        xr_dict = {}

        for name, arr in arr_dict.items():

            if arr.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {arr.shape}; "
                    f"expected percentile-reduced shape {expected_shape}"
                )
            
            xr_dict[name] = xr.DataArray(
                arr,
                dims=["percentile", "time", "lat", "lon"],
                coords={
                    "percentile": self.output_percentiles,
                    "time": np.arange(self.start_year, self.start_year + arr.shape[1]),
                    "lat": output_lats,
                    "lon": output_lons,
                },
                attrs={
                    "units": "m",
                    "long_name": f"Regional {name} sea-level projections",
                    "source": "ProFSea-Climate v0.1",
                },
            )

        return xr_dict



    def _initialise_local_mc_zarr(
        self,
        output_path: str | Path,
        output_lats: np.ndarray,
        output_lons: np.ndarray,
        output_names: list[str],
    ) -> None:
        """
        Initialise an empty Zarr store for percentile-reduced local Monte
        Carlo spatial projections.
        """

        output_path = Path(output_path)
          
        time_coord = np.arange(
            self.start_year,
            self.start_year + self.n_years,
        )

        shape = (
            len(self.output_percentiles),
            self.n_years,
            len(output_lats),
            len(output_lons),
        )

        # Create empty Dask-backed arrays. These define the complete output
        # structure without materialising the complete result in memory.
        empty_arrays = {
            name: xr.DataArray(
                da.full(
                    shape,
                    np.nan,
                    dtype=np.float64,
                    chunks=(
                        len(self.output_percentiles),
                        10,
                        5,
                        5,
                    ),
                ),
                name=name,
                dims=[
                    "percentile",
                    "time",
                    "lat",
                    "lon",
                ],
                coords={
                    "percentile": self.output_percentiles,
                    "time": time_coord,
                    "lat": output_lats,
                    "lon": output_lons,
                },
                attrs={
                    "units": "m",
                    "long_name": (
                        f"Regional {name} sea-level projections"
                    ),
                    "source": "ProFSea-Climate v0.1",
                },
            )
            for name in output_names
        }

        ds = xr.Dataset(empty_arrays)

        # Compute=False because we are creating the structure of the Zarr store:
        # percentile x time x lat x lon
        # but we do not want Dask to calculate and write all those np.nan values.
        # The actual data will be supplied strip by strip afterwards.
        ds.to_zarr(
            output_path,
            mode="w",
            compute=False,
            consolidated=False,
        )
       

    def _write_local_mc_strip(
        self,
        output_path: str | Path,
        computed_strips: dict[str, xr.DataArray],
        output_lat_start: int,
        output_lat_stop: int,
    ) -> None:
        """
        Write one completed latitude strip into an existing local-MC
        Zarr output store.

        Parameters
        ----------
        output_path : str | Path
            Path to the existing Zarr store.

        computed_strips : dict[str, xr.DataArray]
            Computed percentile-reduced arrays for the current latitude
            strip.

        output_lat_start : int
            Starting latitude index in the output store.

        output_lat_stop : int
            Stopping latitude index in the output store.
        """

        strip_ds = xr.Dataset(computed_strips)

        strip_ds.to_zarr(
            output_path,
            mode="r+",
            region={
                "percentile": slice(
                    0,
                    len(self.output_percentiles),
                ),
                "time": slice(
                    0,
                    self.n_years,
                ),
                "lat": slice(
                    output_lat_start,
                    output_lat_stop,
                ),
                "lon": slice(
                    0,
                    strip_ds.sizes["lon"],
                ),
            },
            consolidated=False,
        )

        
    
    def run(
        self,
        member_seed: int = 42,
        output_path: str | Path | None = None,
        lat_slice: slice | None = None,
        lon_slice: slice | None = None,
    ):
        """
        Run the spatial model to generate regional sea level projections for each component.  [LA: might want to change the "-> None" to "-> xr.DataArray" or similar, as it does return something? But _arr_to_xr() may return an xr.Dataset rather than a DataArray, especially since it is receiving a dictionary containing multiple components, so think about the most appropriate type later.]

        Parameters
        ----------
        member_seed: int, optional
            Seed for random number generation to ensure reproducibility of member sampling. Default is 42.

        output_path : str | Path | None, optional
            Path for the local Monte Carlo Zarr output store.
            If None, defaults to "local_mc_development.zarr".

        lat_slice : slice | None, optional
            Latitude-index range to process. Currently defaults to the
            development range slice(50, 60).

        lon_slice : slice | None, optional
            Longitude-index range to process. Currently defaults to the
            development range slice(20, 35).

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

        


        console.print()  # Add a blank line for better readability in the console output
        #for name, comp in track(
        #    self.components.items(), description="Spatialising components..."
        #):


        #-----------------------------------------------
        # Loop over spatial tiles
        #-----------------------------------------------

        #block_results_list = []

        #lat_output_start = 50
        #lat_output_stop = 60

        #lon_output_start = 20
        #lon_output_stop = 35

        # Default to the complete spatial grid.
        #if lat_slice is None:
        #    lat_slice = slice(0, len(self.grid_lats))
        #if lon_slice is None:
        #    lon_slice = slice(0, len(self.grid_lons))

        # Temporary development defaults.
        if lat_slice is None:
            lat_slice = slice(50, 60)
        if lon_slice is None:
            lon_slice = slice(20, 35)

        # Derive the controller bounds
        lat_output_start = lat_slice.start
        lat_output_stop = lat_slice.stop
        lon_output_start = lon_slice.start
        lon_output_stop = lon_slice.stop

            
        lat_block_size = 5
        lon_block_size = 5

        output_lats = self.grid_lats[
            lat_output_start:lat_output_stop
        ]

        output_lons = self.grid_lons[
            lon_output_start:lon_output_stop
        ]

        final_shape = (
            len(self.output_percentiles),
            self.n_years,
            len(output_lats),
            len(output_lons),
        )

        output_names = [
            *self.components.keys(),
            "total_rsl",
        ]


        if output_path is None:
            output_path = Path(
                "local_mc_development.zarr"
            )
        else:
            output_path = Path(output_path)

            
        #zarr_output_path = Path(
        #    "local_mc_development.zarr"
        #)

        # Remove any Zarr store left over from a previous development run.
        if output_path.exists():
            shutil.rmtree(output_path)

        # Initialise a fresh empty Zarr store once, before processing strips
        self._initialise_local_mc_zarr(
            output_path=output_path,
            output_lats=output_lats,
            output_lons=output_lons,
            output_names=output_names,
        )


            
        #final_arrays = {
        #    name: np.full(
        #        final_shape,
        #        np.nan,
        #        dtype=np.float64,
        #    )
        #    for name in output_names
        #}

        
        #for lat_start in range(0, len(self.grid_lats), lat_block_size,):
        #for lat_start in range(50, 60, lat_block_size,):
        for lat_start in range(lat_output_start, lat_output_stop, lat_block_size,):
            
            #lat_stop = min(
            #    lat_start + lat_block_size,
            #    len(self.grid_lats),
            #)

            lat_stop = min(
                lat_start + lat_block_size,
                lat_output_stop,
            )

            lat_slice = slice(lat_start, lat_stop)

            # Retain only the percentile-reduced longitude tiles for this latitude strip.
            strip_tile_results = []
    
            #for lon_start in range(0, len(self.grid_lons), lon_block_size,):
            #for lon_start in range(20, 35, lon_block_size,):
            #    lon_stop = min(
            #        lon_start + lon_block_size,
            #        len(self.grid_lons),
            #    )
            
            for lon_start in range(lon_output_start, lon_output_stop, lon_block_size,):
                
                lon_stop = min(lon_start + lon_block_size, lon_output_stop,)
                
                lon_slice = slice(lon_start, lon_stop)



                logger.info(
                    "Processing spatial block: "
                    "lat indices %d:%d, lon indices %d:%d",
                    lat_start,
                    lat_stop,
                    lon_start,
                    lon_stop,
                )
                
                print(
                    f"Processing block: "
                    f"lat {lat_start}:{lat_stop}, "
                    f"lon {lon_start}:{lon_stop}"
                )

                n_block_lats = len(self.grid_lats[lat_slice])
                n_block_lons = len(self.grid_lons[lon_slice])


                # Create new dictionary for each tile (member-level arrays from previous tile can be released from memory)
                spatial_projections = {}
        

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

                logger.info("Local MC member-wise total shape: %s", total_members.shape,)

                # Calculate percentiles from each component's local Monte Carlo
                component_percentiles = {
                    name: self._calculate_member_percentiles(member_array)
                    for name, member_array in spatial_projections.items()
                }

                # Calculate total percentiles from the member-wise component total local Monte Carlo
                total_percentiles = self._calculate_member_percentiles(
                    total_members
                )

                # Collect the percentiles together
                percentile_projections = {**component_percentiles, "total_rsl": total_percentiles,}


                # Convert percentile-reduced component and total projections for this tile to xarray.
                block_results = self._arr_to_xr(
                    percentile_projections,
                    lat_slice=lat_slice,
                    lon_slice=lon_slice,
                )

                # Append this tile's results to the block list
                #block_results_list.append(block_results)
                strip_tile_results.append(block_results)

                
            # Inside lat loop but outside lon loop, join the current strip along longitude:
            strip_results = {
                name: xr.concat(
                    [
                        tile_results[name]
                        for tile_results in strip_tile_results
                    ],
                    dim="lon",
                )
                
                for name in output_names
            }

            expected_strip_shape = (
                len(self.output_percentiles),
                self.n_years,
                lat_stop - lat_start,
                len(output_lons),
            )

            for name, strip_array in strip_results.items():
                if strip_array.shape != expected_strip_shape:
                    raise ValueError(
                        f"{name} strip has shape {strip_array.shape}; "
                        f"expected {expected_strip_shape}"
                    )

            output_lat_start = (lat_start - lat_output_start)

            output_lat_stop = (lat_stop - lat_output_start)
                

            
            # (Still inside lat loop)
            strip_names = list(strip_results)

            computed_strips = dask.compute(
                *[
                    strip_results[name]
                    for name in strip_names
                ]
            )

            # computed_strips is a tuple, whereas the latitude strip Zarr writer method wants dict[str, xr.DataArray]
            # so prepare computed_strip_dict in the required format
            # computed_strip_dict can go to either final_arrays or Zarr writer
            computed_strip_dict = dict(
                zip(
                    strip_names,
                    computed_strips,
                )
            )

            # Existing in-memory reference path - not needed now that the Zarr writer is ready
            #for name, computed_strip in zip(strip_names, computed_strips,):
            #    final_arrays[name][
            #        :,
            #        :,
            #        output_lat_start:output_lat_stop,
            #        :,
            #    ] = computed_strip.values

                
            # Write the completed latitude strip to Zarr
            self._write_local_mc_strip(
                output_path=output_path,
                computed_strips=computed_strip_dict,
                output_lat_start=output_lat_start,
                output_lat_stop=output_lat_stop,
            )



            logger.info(
                "Placed reduced latitude strip %d:%d "
                "into output positions %d:%d",
                lat_start,
                lat_stop,
                output_lat_start,
                output_lat_stop,
            )


            # Release references to this strip before processing the next one
            del strip_tile_results
            del strip_results
            del computed_strips
            del computed_strip_dict

            
        # -------------------------------------------------
        # OUTSIDE THE LATITUDE LOOP
        # All strips have now been written to Zarr.
        # Reopen the completed store lazily.
        # -------------------------------------------------        
        #time_coord = np.arange(
        #    self.start_year,
        #    self.start_year + self.n_years,
        #)

        # This dictionary comprehension (constructing self.results from final_arrays) is not needed
        # Instead use zarr_results after all latitude strips have been written
        #self.results = {
        #    name: xr.DataArray(
        #        array,
        #        name=name,
        #        dims=[
        #            "percentile",
        #            "time",
        #            "lat",
        #            "lon",
        #        ],
        #        coords={
        #            "percentile": self.output_percentiles,
        #            "time": time_coord,
        #            "lat": output_lats,
        #            "lon": output_lons,
        #        },
        #        attrs={
        #            "units": "m",
        #            "long_name": (
        #                f"Regional {name} sea-level projections"
        #            ),
        #            "source": "ProFSea-Climate v0.1",
        #        },
        #    )
        #    for name, array in final_arrays.items()
        #}

    
        zarr_results = xr.open_zarr(
            output_path,
            consolidated=False,
        )

        
        self.results = {
            name: zarr_results[name]
            for name in output_names
        }

        
        pdb.set_trace()
        



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
