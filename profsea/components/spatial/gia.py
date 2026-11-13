from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
import pdb

from profsea.components.core.base import SpatialComponent
from profsea.components.core.state import SpatialState

PROFSEA_DIR = Path(__file__).resolve().parents[2]
GIA_DIR = PROFSEA_DIR / "profsea-assets" / "gia"


class GIA(SpatialComponent):
    """
    Handles Glacial Isostatic Adjustment.
    GIA accumulates linearly over time and has no global warming driver input.
    Supports loading single files, lists of files, or directories of files.
    """

    def __init__(
        self,
        gia_paths: str | Path = None,
        sample_spatial: bool = False,
    ) -> None:
        """
        gia_paths: str, Path, list of str/Path, or None
            Path(s) to a directory containing GIA files or direct paths to specific GIA NetCDF files.
            Defaults to the main GIA_DIR if None.
        sample_spatial: bool, optional
            Whether to sample spatial patterns probabilistically. Default is False.
        """
        self.sample_spatial = sample_spatial

        if gia_paths is None:
            raw_paths = [GIA_DIR]
        elif isinstance(gia_paths, (str, Path)):
            raw_paths = [Path(gia_paths)]
        else:
            raw_paths = [Path(p) for p in gia_paths]

        # Resolve directories into files, and keep direct file paths
        self.gia_files = []
        for p in raw_paths:
            if not p.exists():
                raise FileNotFoundError(f"Provided GIA path does not exist: {p}")

            if p.is_dir():
                # Extract all .nc files if a directory is passed
                self.gia_files.extend(list(p.glob("*.nc")))
            elif p.is_file() and p.suffix == ".nc":
                # Keep direct NetCDF files
                self.gia_files.append(p)

        if not self.gia_files:
            raise FileNotFoundError(
                f"No GIA NetCDF files found in the provided paths: {raw_paths}"
            )

        self._global_projection = da.zeros((1, 1))

    @property
    def global_projection(self):
        return self._global_projection

    # LA added:
    def get_n_patterns(self) -> int:
        """
        Return the total number of available GIA spatial patterns.

        A GIA NetCDF file may contain either one spatial pattern or
        multiple patterns along a 'model' dimension.
        """
        n_patterns = 0

        for path in self.gia_files:
            with xr.open_dataarray(path) as gia_da:
                if "model" in gia_da.dims:
                    n_patterns += gia_da.sizes["model"]
                else:
                    n_patterns += 1

        return n_patterns

    
    # LA added:
    def sample_pattern_indices(
        self,
        rng: np.random.Generator,
        n_local_members: int,
    ) -> np.ndarray:
        """
        Select one GIA spatial pattern for each local Monte Carlo member.

        The indices are generated once per Spatial.run() and then reused
        across the whole spatial domain.
        """
        n_patterns = self.get_n_patterns()

        if n_patterns < 1:
            raise ValueError(
                "No GIA spatial patterns are available for sampling."
            )

        return rng.integers(
            low=0,
            high=n_patterns,
            size=n_local_members,
        )

    
    
    #def _load_and_interpolate_rates(self, state: SpatialState) -> da.Array:
    def _load_and_interpolate_rates(
            self,
            state: SpatialState,
            lat_slice: slice | None = None,
            lon_slice: slice | None = None,
    ) -> da.Array:
        """
        Lazily loads all GIA files, regrids them, and stacks them into a
        single 3D array of shape (total_models, lat, lon).

        Parameters
        ----------
        state: SpatialState
            The spatial state containing the target grid information.

        Returns
        -------
        da.Array
            A Dask array of shape (total_models, lat, lon) containing the regridded GIA rates.
        """
        grids = []
        for path in self.gia_files:
            gia_da = xr.open_dataarray(path, chunks={"lat": 45, "lon": 45})
            
            #interp_da = self.extract_spatial(gia_da, state)
            interp_da = self.extract_spatial(
                gia_da,
                state,
                lat_slice=lat_slice,
                lon_slice=lon_slice,
            )
            
            data = interp_da.data

            # Determine expected spatial dims based on state
            spatial_dims = 2 if hasattr(state, "grid_lats") else 1

            # If the raw file lacks a 'model' dimension, prepend it
            if data.ndim == spatial_dims:
                data = data[None, ...]

            grids.append(data)

        # Concatenate along the model axis (axis 0)
        return da.concatenate(grids, axis=0)

    #def project(self, state: SpatialState, rng: np.random.Generator) -> da.Array:
    def project(
            self,
            state: SpatialState,
            rng: np.random.Generator,
            lat_slice: slice | None = None,
            lon_slice: slice | None = None,
    ) -> da.Array:
            
        """
        Project the GIA component by multiplying the accumulation time vector with the spatial rates.

        Parameters
        ----------
        state: SpatialState
            The spatial state containing the target grid information.
        rng: np.random.Generator
            Random number generator for sampling GIA models if sample_spatial is True.

        Returns
        -------
        da.Array
            A 4D array of shape (members, years, lat, lon) containing the spatial projections for each member and year.
        """
        #gia_rates = self._load_and_interpolate_rates(state)
        gia_rates = self._load_and_interpolate_rates(
            state,
            lat_slice=lat_slice,
            lon_slice=lon_slice,
        )
        
        n_patterns = gia_rates.shape[0]

        using_local_mc = state.global_member_indices is not None

        n_output_members = (
            state.n_local_members
            if using_local_mc
            else state.n_members
        )
        
        # Dynamically capture spatial shape (site,) or (lat, lon)
        spatial_shape = gia_rates.shape[1:]

        # Calculate the accumulation time vector (mm/yr to m/yr)
        midyr = (
            state.baseline_yrs[1] - state.baseline_yrs[0] + 1
        ) * 0.5 + state.baseline_yrs[0]
        Tdelta = state.endofhistory - midyr
        unit_series = (np.arange(state.n_years) + Tdelta) * 0.001

        # Broadcast 1D time series to match expected (members, years) signature
        #temporal_array = da.broadcast_to(unit_series, (state.n_members, state.n_years))
        temporal_array = da.broadcast_to(
            unit_series,
            (n_output_members, state.n_years),
        )
        
#        # Handle sampling if required
#        if n_patterns == 1:
#            selected_gia = da.broadcast_to(
#                gia_rates[0],
#                (state.n_members, *spatial_shape),
#            )
#        else:
#            if self.sample_spatial:
#                rgiai = rng.integers(n_patterns, size=state.n_members)
#                selected_gia = gia_rates[rgiai, ...]
#            else:
#                mean_gia = da.nanmean(gia_rates, axis=0)
#                selected_gia = da.broadcast_to(
#                    mean_gia,
#                    (state.n_members, *spatial_shape),
#                )


        # LA: local Monte Carlo uses the GIA model indices that were sampled
        # once in Spatial.run(), so each member keeps the same GIA model
        # across the whole spatial domain.
        if using_local_mc and self.sample_spatial:

            if state.gia_pattern_indices is None:
                raise RuntimeError(
                    "GIA pattern indices have not been sampled."
                )

            if len(state.gia_pattern_indices) != state.n_local_members:
                raise ValueError(
                    "Number of GIA pattern indices does not match "
                    "state.n_local_members."
                )

            selected_gia = da.take(
                gia_rates,
                state.gia_pattern_indices,
                axis=0,
            )

        # Existing behaviour for calculations that are not using the new local MC
        elif n_patterns == 1:
            selected_gia = da.broadcast_to(
                gia_rates[0],
                (n_output_members, *spatial_shape),
            )

        elif self.sample_spatial:
            rgiai = rng.integers(
                low=0,
                high=n_patterns,
                size=n_output_members,
            )

            selected_gia = da.take(
                gia_rates,
                rgiai,
                axis=0,
            )

        else:
            mean_gia = da.nanmean(
                gia_rates,
                axis=0,
            )

            selected_gia = da.broadcast_to(
                mean_gia,
                (n_output_members, *spatial_shape),
            )


        # Delegate dimensional multiplication to the base class
        #return self.broadcast_spatiotemporal(temporal_array, selected_gia)
        result = self.broadcast_spatiotemporal(
            temporal_array,
            selected_gia,
        )


        return result
