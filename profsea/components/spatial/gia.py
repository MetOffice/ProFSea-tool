from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from profsea.components.core.base import SpatialComponent
from profsea.components.core.state import SpatialState
from profsea.utils import interpolate_to_grid

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
        gia_dir: str | Path = None,
        sample_spatial: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        gia_dir: str, Path, or list of str/Path
            Path to a directory containing GIA files.
        sample_spatial: bool, optional
            Whether to sample spatial patterns probabilistically. Default is False.
        """
        self.sample_spatial = sample_spatial

        if gia_dir is None:
            self.gia_dir = GIA_DIR
        else:
            self.gia_dir = Path(gia_dir)

        # Dummy property required by the base Spatial architecture
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
        gia_paths = list(self.gia_dir.glob("*.nc"))

        if not gia_paths:
            raise FileNotFoundError(f"No GIA NetCDF files found in {self.gia_dir}")

        grids = []
        for path in gia_paths:
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
        Tdelta = 2006 - midyr
        unit_series = (np.arange(state.n_years) + Tdelta) * 0.001

        # Broadcast 1D time series to match expected (members, years) signature
        temporal_array = da.broadcast_to(unit_series, (state.n_members, state.n_years))

        # Handle sampling if required
        if n_patterns == 1:
            selected_gia = da.broadcast_to(
                gia_rates[0],
                (state.n_members, *spatial_shape),
            )
        else:
            if self.sample_spatial:
                rgiai = rng.integers(n_patterns, size=state.n_members)
                selected_gia = gia_rates[rgiai, ...]
            else:
                mean_gia = da.mean(gia_rates, axis=0)
                selected_gia = da.broadcast_to(
                    mean_gia,
                    (state.n_members, *spatial_shape),
                )

        # Delegate dimensional multiplication to the base class
        return self.broadcast_spatiotemporal(temporal_array, selected_gia)
