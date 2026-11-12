from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
import pdb

from profsea.components.core.base import SpatialComponent
from profsea.components.core.state import SpatialState
from profsea.utils import sample_members_2D

PROFSEA_DIR = Path(__file__).resolve().parents[2]
FP_DIR = PROFSEA_DIR / "profsea-assets" / "grd-fingerprints"

#FP_PATH_MAP = {
#    "greendyn": [
#        FP_DIR / "greendyn_klemann.nc",
#        FP_DIR / "greendyn_slangen.nc",
#        FP_DIR / "greendyn_spada.nc",
#    ],
#    "greensmb": [
#        FP_DIR / "greensmb_klemann.nc",
#        FP_DIR / "greensmb_slangen.nc",
#        FP_DIR / "greensmb_spada.nc",
#    ],
#    "greenland": [FP_DIR / "greenland_ar6.nc"],   # LA: consider renaming the fingerprint option as "greenland_AR6"?
#    "landwater": [FP_DIR / "landwater_slangen.nc"],
#    "wais": [FP_DIR / "wais.nc"],
#    "eais": [FP_DIR / "eais.nc"],
#    "antdyn": [
#        FP_DIR / "antdyn_klemann.nc",
#        FP_DIR / "antdyn_slangen.nc",
#        FP_DIR / "antdyn_spada.nc",
#    ],
#    "antsmb": [
#        FP_DIR / "antsmb_klemann.nc",
#        FP_DIR / "antsmb_slangen.nc",
#        FP_DIR / "antsmb_spada.nc",
#    ],
#    "glacier": [
#        FP_DIR / "glacier_klemann.nc",
#        FP_DIR / "glacier_slangen.nc",
#        FP_DIR / "glacier_spada.nc",
#    ],
#}

# LA updated FP_PATH_MAP to map GRD method names to filepaths, rather than an index-based mapping method.
# This removes reliance on the ordering of the GRD methods and allows
# the local Monte Carlo to select methods by name rather than by array index.
# The available pool of GRD methods can therefore be changed through the fingerprint
# configuration without changing the Monte Carlo sampling logic.

FP_PATH_MAP = {
    "greendyn": {
        "klemann": FP_DIR / "greendyn_klemann.nc",
        "slangen": FP_DIR / "greendyn_slangen.nc",
        "spada": FP_DIR / "greendyn_spada.nc",
    },

    "greensmb": {
        "klemann": FP_DIR / "greensmb_klemann.nc",
        "slangen": FP_DIR / "greensmb_slangen.nc",
        "spada": FP_DIR / "greensmb_spada.nc",
    },

    "greenland": {
        "greenland_ar6": FP_DIR / "greenland_ar6.nc",
    },

    "landwater": {
        "slangen": FP_DIR / "landwater_slangen.nc",
    },

    "wais": {
        "wais": FP_DIR / "wais.nc",
    },

    "eais": {
        "eais": FP_DIR / "eais.nc",
    },

    "antdyn": {
        "klemann": FP_DIR / "antdyn_klemann.nc",
        "slangen": FP_DIR / "antdyn_slangen.nc",
        "spada": FP_DIR / "antdyn_spada.nc",
    },

    "antsmb": {
        "klemann": FP_DIR / "antsmb_klemann.nc",
        "slangen": FP_DIR / "antsmb_slangen.nc",
        "spada": FP_DIR / "antsmb_spada.nc",
    },

    "glacier": {
        "klemann": FP_DIR / "glacier_klemann.nc",
        "slangen": FP_DIR / "glacier_slangen.nc",
        "spada": FP_DIR / "glacier_spada.nc",
    },
}



class Fingerprint(SpatialComponent):
    """
    Spatial fingerprint component that applies spatial patterns to global projections.
    """

    def __init__(
        self,
        global_projection: xr.DataArray,
        fingerprint_component: str,
        #fingerprint_paths: str | Path | list[str | Path] = None,
        fingerprint_paths: (str | Path | list[str | Path] | dict[str, str | Path] | None) = None,   # LA extended to allow named mapping
        scaling_factor: float = 1.0,
        sample_spatial: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        global_projection: xr.DataArray
             A 2D array (members x years) of global projections to apply the fingerprints to.
        fingerprint_paths: str, Path, or list of str/Path
             Path(s) to NetCDF files containing the spatial fingerprint patterns. Each file should contain a DataArray with dimensions (lat, lon).
        scaling_factor: float, optional
             Optional multiplier to apply to the fingerprint patterns (e.g., to convert from m to mm). Default is 1.0 (no scaling).
        sample_spatial: bool, optional
             If True, randomly sample a different fingerprint pattern for each member. If False, use the mean of all provided fingerprints for all members (storyline mode). Default is False.
        """
        self._global_projection = da.from_array(global_projection.data, chunks="auto")
        self.scaling_factor = scaling_factor
        self.sample_spatial = sample_spatial
        self.fingerprint_component = fingerprint_component

        
#        # Default paths if not provided (can be overridden by user input)
#        if fingerprint_paths is None:
#            # Determine default paths based on the component type
#            try:
#                self.fp_paths = [Path(p) for p in FP_PATH_MAP[fingerprint_component]]
#            except KeyError:
#                raise ValueError(
#                    f"No default fingerprint paths found for fingerprint component '{fingerprint_component}'. Please provide explicit paths."
#                )
#        # Normalize the input to always be a list of Path objects
#        elif isinstance(fingerprint_paths, (str, Path)):
#            self.fp_paths = [Path(fingerprint_paths)]
#        else:
#            self.fp_paths = [Path(p) for p in fingerprint_paths]
#
#        if fingerprint_paths is not None:
#            for p in self.fp_paths:
#                if not p.exists():
#                    raise FileNotFoundError(f"Missing fingerprint file: {p}")


        # LA Updated fingerprint path-handling to use a dictionary so the GRD method name stays explicity assoicated with its file.
        # It still supports a user supplying a single custom file, a named dictionary, or an old-style list,
        # and it retains self.fp_paths temporarily so existing loading code continues to work. 

        # Default fingerprint configuration
        if fingerprint_paths is None:
            try:
                self.fp_method_paths = {
                    method: Path(path)
                    for method, path in FP_PATH_MAP[fingerprint_component].items()
                }
            except KeyError:
                raise ValueError(
                    f"No default fingerprint paths found for fingerprint component "
                    f"'{fingerprint_component}'. Please provide explicit paths."
                )

        # User supplied one unnamed fingerprint file
        elif isinstance(fingerprint_paths, (str, Path)):
            self.fp_method_paths = {
                "custom": Path(fingerprint_paths)
            }

        # User supplied a named mapping: method -> filepath
        elif isinstance(fingerprint_paths, dict):
            self.fp_method_paths = {
                method: Path(path)
                for method, path in fingerprint_paths.items()
            }

        # Backwards compatibility with an existing list of paths
        else:
            self.fp_method_paths = {
                f"custom_{i}": Path(path)
                for i, path in enumerate(fingerprint_paths)
            }

        # Retain a list of paths for existing code that still uses self.fp_paths
        self.fp_paths = list(self.fp_method_paths.values())

        # Record the available named methods
        self.available_grd_methods = list(
            self.fp_method_paths.keys()
        )

        # Check that all configured files exist
        for path in self.fp_method_paths.values():
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing fingerprint file: {path}"
                )


    
                
    @property
    def global_projection(self):
        return self._global_projection

#    def _load_and_interpolate(self, state: SpatialState) -> da.Array:
#    def _load_and_interpolate(self, state: SpatialState,) -> tuple[da.Array, list[str]]:

    def _load_and_interpolate(
        self,
        state: SpatialState,
        lat_slice: slice | None = None,   # add optional lat/lon slices
        lon_slice: slice | None = None,
    ) -> tuple[da.Array, list[str]]:
        
        """
        Lazily load and regrid all provided fingerprints.

        Parameters
        ----------
        state: SpatialState
            The state object containing the target grid information.

        Returns
        -------
        fps: da.Array
            Fingerprint fields with shape
            (n_fingerprints, lat, lon).

        methods: list[str]
        GRD method name corresponding to each field in fps.

        """
        grids = []
        methods = []  # LA added
        
#        for path in self.fp_paths:
#            fp_da = xr.open_dataarray(path, chunks={"lat": 45, "lon": 45})
#            fp_interp = self.extract_spatial(fp_da, state)
#            grids.append(fp_interp.data * self.scaling_factor)

#        # Stack them into a 3D array: (n_fingerprints, lat, lon)
#        return da.stack(grids, axis=0)


        for method, path in self.fp_method_paths.items():
            
            fp_da = xr.open_dataarray(path, chunks={"lat": 45, "lon": 45},)

            #fp_interp = self.extract_spatial(fp_da, state)
            fp_interp = self.extract_spatial(
                fp_da,
                state,
                lat_slice=lat_slice,
                lon_slice=lon_slice,
            )

            grids.append(fp_interp.data * self.scaling_factor)

            methods.append(method)

        fps = da.stack(grids, axis=0)

        return fps, methods


    
    #def project(self, state: SpatialState, rng: np.random.Generator) -> da.Array:
    def project(
        self,
        state: SpatialState,
        rng: np.random.Generator,
        lat_slice: slice | None = None,    # Make Fingerprint.project() accept spatial slices
        lon_slice: slice | None = None,
    ) -> da.Array:
        
        """
        Calculate the spatial projection by applying the fingerprints to the global projection.

        Parameters
        ----------
        state: SpatialState
            The state object containing the target grid information and number of members.
        rng: np.random.Generator
            Random number generator for sampling fingerprints if sample_spatial is True.

        Returns
        -------
        da.Array
            A 4D array of shape (members, years, lat, lon) containing the spatial projections for each member and year.
        """
        # Load call:  LA added fp_methods (also returned), and added lat/lon slices
        fps, fp_methods = self._load_and_interpolate(
            state,
            lat_slice=lat_slice,
            lon_slice=lon_slice,
        )  
        
        spatial_shape = fps.shape[1:]

#        # Handle the global projection
#        if state.output_percentiles is not None:
#            global_proj = sample_members_2D(
#                self.global_projection, state.output_percentiles
#            )
#        else:
#            global_proj = self.global_projection

        #-------------------------------------------------------------------
        # Select the shared global projection rows (LA added for local MC)
        if state.global_member_indices is not None:
            # Local-Monte-Carlo route: use the same global rows as all
            # other aligned global components.
            global_proj = self.global_projection[
                state.global_member_indices,
                :
            ]

        elif state.output_percentiles is not None:
            # Existing percentile-representative pathway.
            global_proj = sample_members_2D(
                self.global_projection,
                state.output_percentiles,
            )

        else:
            global_proj = self.global_projection
        #-------------------------------------------------------------------


         
#        # Determine the spatial fingerprint for each member
#        n_fps = fps.shape[0]
#
#        if n_fps == 1:
#            # Only one fingerprint available
#            selected_fps = da.broadcast_to(
#                fps[0],
#                (state.n_members, *spatial_shape),
#            )
#        elif self.sample_spatial:
#            # Probabilistic mode: pick a random fingerprint per member
#            fp_indices = rng.integers(0, n_fps, size=state.n_members)
#            selected_fps = fps[fp_indices, ...]
#        else:
#            # Storyline mode: take the mean of the available fingerprints
#            mean_fp = da.nanmean(fps, axis=0)
#            selected_fps = da.broadcast_to(
#                mean_fp,
#                (state.n_members, *spatial_shape),
#            )

        #-----------------------------------------------------
        # New code for local MC:
        
        n_fps = fps.shape[0]
        using_local_mc = state.global_member_indices is not None

        if using_local_mc:
            if n_fps == 1:
                # Components such as landwater have only one available
                # fingerprint, so use that field for every local member.
                selected_fps = da.broadcast_to(
                    fps[0],
                    (state.n_local_members, *spatial_shape),
                )

            else:
                if state.grd_method_choices is None:
                    raise RuntimeError(
                        "Shared GRD method choices have not been created."
                    )

                method_to_index = {
                    method: index
                    for index, method in enumerate(fp_methods)
                }

                unavailable_methods = sorted(
                    set(state.grd_method_choices)
                    - set(method_to_index)
                )

                if unavailable_methods:
                    raise ValueError(
                        f"Fingerprint component "
                        f"'{self.fingerprint_component}' does not provide "
                        f"the following selected GRD methods: "
                        f"{unavailable_methods}"
                    )

                fp_indices = np.array(
                    [
                        method_to_index[method]
                        for method in state.grd_method_choices
                    ],
                    dtype=int,
                )

                selected_fps = da.take(
                    fps,
                    fp_indices,
                    axis=0,
                )

        else:
            # Existing non-local-MC behaviour.
            if n_fps == 1:
                selected_fps = da.broadcast_to(
                    fps[0],
                    (state.n_members, *spatial_shape),
                )

            elif self.sample_spatial:
                fp_indices = rng.integers(
                    0,
                    n_fps,
                    size=state.n_members,
                )

                selected_fps = da.take(
                    fps,
                    fp_indices,
                    axis=0,
                )

            else:
                mean_fp = da.nanmean(fps, axis=0)

                selected_fps = da.broadcast_to(
                    mean_fp,
                    (state.n_members, *spatial_shape),
                )
        #-----------------------------------------------------

        # Broadcast and multiply: (members, years) * (members, lat, lon)
        return self.broadcast_spatiotemporal(global_proj, selected_fps)
