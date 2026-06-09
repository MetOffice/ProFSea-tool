from abc import ABC, abstractmethod
import dask.array as da
import numpy as np
import xarray as xr

from .state import ClimateState


class Component(ABC):
    @abstractmethod
    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """Calculate and return the SLR projection for this component"""
        pass


class SpatialComponent(Component):
    def extract_spatial(self, da_input: xr.DataArray, state) -> xr.DataArray:
        """
        Spatial extractor for both Grid (2D) and Site (1D) states.

        Parameters
        ----------
        da_input: xarray.DataArray
            The input data array containing spatial coordinates (lat/lon).
        state: SpatialState
            The state object which may contain either target_lats/lons for point extraction or grid_lats/lons for grid interpolation.
        """
        if "lon" in da_input.coords and da_input.lon.max() > 180:
            da_input = da_input.assign_coords(
                lon=(((da_input.lon + 180) % 360) - 180)
            ).sortby("lon")

        if hasattr(state, "target_lats"):
            # Point extraction for LocalState
            lats = xr.DataArray(state.target_lats, dims="site")
            lons = xr.DataArray(state.target_lons, dims="site")

            return da_input.interp(
                lat=lats,
                lon=lons,
                method=getattr(state, "interpolation_method", "linear"),
            )
        elif hasattr(state, "grid_lats"):
            from profsea.utils import interpolate_to_grid

            return interpolate_to_grid(da_input, state.grid_lats, state.grid_lons)
        else:
            raise ValueError(
                "State object is missing required spatial attributes "
                "(needs either target_lats/lons or grid_lats/lons)."
            )

    def broadcast_spatiotemporal(
        self, temporal_array: da.Array, spatial_array: da.Array
    ) -> da.Array:
        """
        Dynamically multiplies a temporal series by a spatial pattern.

        Parameters
        ----------
        temporal_array: dask.array.Array
            2D array of shape (members, years)
        spatial_array: dask.array.Array
            Either 2D (members, site) or 3D (members, lat,
        """
        # Determine how many spatial dimensions we are dealing with
        spatial_dims = spatial_array.ndim - 1

        if spatial_dims == 1:
            # 3D Output: (members, years, site)
            return temporal_array[:, :, None] * spatial_array[:, None, :]
        elif spatial_dims == 2:
            # 4D Output: (members, years, lat, lon)
            return temporal_array[:, :, None, None] * spatial_array[:, None, :, :]
        else:
            raise ValueError(
                f"Unexpected spatial dimensions. Expected 1 or 2, got {spatial_dims}."
            )

    @property
    @abstractmethod
    def global_projection(self):
        """Spatial component must define a global_projection attribute or property"""
        pass
