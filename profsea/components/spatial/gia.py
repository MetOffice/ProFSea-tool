from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

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

    def _load_and_interpolate_rates(self, state: SpatialState) -> da.Array:
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
            interp_da = self.extract_spatial(gia_da, state)
            data = interp_da.data

            # Determine expected spatial dims based on state
            spatial_dims = 2 if hasattr(state, "grid_lats") else 1

            # If the raw file lacks a 'model' dimension, prepend it
            if data.ndim == spatial_dims:
                data = data[None, ...]

            grids.append(data)

        # Concatenate along the model axis (axis 0)
        return da.concatenate(grids, axis=0)

    def project(self, state: SpatialState, rng: np.random.Generator) -> da.Array:
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
        gia_rates = self._load_and_interpolate_rates(state)
        n_patterns = gia_rates.shape[0]

        # Dynamically capture spatial shape (site,) or (lat, lon)
        spatial_shape = gia_rates.shape[1:]

        # Calculate the accumulation time vector (mm/yr to m/yr)
        midyr = (
            state.baseline_yrs[1] - state.baseline_yrs[0] + 1
        ) * 0.5 + state.baseline_yrs[0]
        Tdelta = state.endofhistory - midyr
        unit_series = (np.arange(state.n_years) + Tdelta) * 0.001

        # Broadcast 1D time series to match expected (members, years) signature
        temporal_array = da.broadcast_to(
            unit_series, (state.num_output_members, state.n_years)
        )

        # Handle sampling if required
        if n_patterns == 1:
            selected_gia = da.broadcast_to(
                gia_rates[0],
                (state.num_output_members, *spatial_shape),
            )
        else:
            if self.sample_spatial:
                rgiai = rng.integers(n_patterns, size=state.num_output_members)
                selected_gia = gia_rates[rgiai, ...]
            else:
                mean_gia = da.nanmean(gia_rates, axis=0)
                selected_gia = da.broadcast_to(
                    mean_gia,
                    (state.num_output_members, *spatial_shape),
                )

        # Delegate dimensional multiplication to the base class
        return self.broadcast_spatiotemporal(temporal_array, selected_gia)
