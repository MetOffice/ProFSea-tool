from __future__ import annotations

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
        if hasattr(state, "target_lats"):
            return self._extract_points(da_input, state.target_lats, state.target_lons)

        elif hasattr(state, "grid_lats"):
            from profsea.utils.utils import interpolate_to_grid

            return interpolate_to_grid(da_input, state.grid_lats, state.grid_lons)

        else:
            raise ValueError("State object is missing required spatial attributes.")

    def _extract_points(
        self, da_input: xr.DataArray, target_lats: np.ndarray, target_lons: np.ndarray
    ) -> xr.DataArray:
        target_lats = np.asarray(target_lats)
        target_lons = np.asarray(target_lons)

        # Normalize longitudes to [-180, 180) lazy metadata update
        if "lon" in da_input.coords:
            da_input = da_input.assign_coords(lon=(((da_input.lon + 180) % 360) - 180))
        target_lons = ((target_lons + 180) % 360) - 180

        # Extract 1D coordinates eagerly
        lats = da_input.lat.values
        lons = da_input.lon.values

        # Handle decreasing latitudes
        is_descending_lat = False
        if lats[0] > lats[-1]:
            is_descending_lat = True
            lats = lats[::-1]

        # Sort lons
        lon_sort_idx = np.argsort(lons)
        lons = lons[lon_sort_idx]

        # Sort the lazy array to match the coordinate math
        da_input = da_input.isel(lon=lon_sort_idx)
        if is_descending_lat:
            da_input = da_input.isel(lat=slice(None, None, -1))

        # Calculate bounded indices locally
        lat_idx_hi = np.clip(np.searchsorted(lats, target_lats), 1, len(lats) - 1)
        lat_idx_lo = lat_idx_hi - 1

        # Handle longitude wrapping for global datasets
        idx = np.searchsorted(lons, target_lons)
        wrap_lo = idx == 0
        wrap_hi = idx == len(lons)

        # Modulo arithmetic perfectly wraps indices over the array bounds
        lon_idx_hi = idx % len(lons)
        lon_idx_lo = (idx - 1) % len(lons)

        # Extract physical coordinates
        lat_lo, lat_hi = lats[lat_idx_lo], lats[lat_idx_hi]
        lon_lo, lon_hi = lons[lon_idx_lo], lons[lon_idx_hi]

        # Fix coordinates for wrapped bounds so distance calculations work
        lon_lo = np.where(wrap_lo, lons[-1] - 360, lon_lo)
        lon_hi = np.where(wrap_hi, lons[0] + 360, lon_hi)

        dlat = np.where(lat_hi == lat_lo, 1.0, lat_hi - lat_lo)
        dlon = np.where(lon_hi == lon_lo, 1.0, lon_hi - lon_lo)

        # Calculate weights locally
        w_lat_lo = (lat_hi - target_lats) / dlat
        w_lat_hi = (target_lats - lat_lo) / dlat
        w_lon_lo = (lon_hi - target_lons) / dlon
        w_lon_hi = (target_lons - lon_lo) / dlon

        # Build DataArrays for the 4 corners so they broadcast properly with the lazy data
        sites = xr.DataArray(np.arange(len(target_lats)), dims=["site"])

        w_lo_lo = xr.DataArray(
            w_lat_lo * w_lon_lo, dims=["site"], coords={"site": sites}
        )
        w_hi_lo = xr.DataArray(
            w_lat_hi * w_lon_lo, dims=["site"], coords={"site": sites}
        )
        w_lo_hi = xr.DataArray(
            w_lat_lo * w_lon_hi, dims=["site"], coords={"site": sites}
        )
        w_hi_hi = xr.DataArray(
            w_lat_hi * w_lon_hi, dims=["site"], coords={"site": sites}
        )

        # Pull the 4 corners
        da_lo_lo = da_input.isel(
            lat=xr.DataArray(lat_idx_lo, dims="site"),
            lon=xr.DataArray(lon_idx_lo, dims="site"),
        )
        da_hi_lo = da_input.isel(
            lat=xr.DataArray(lat_idx_hi, dims="site"),
            lon=xr.DataArray(lon_idx_lo, dims="site"),
        )
        da_lo_hi = da_input.isel(
            lat=xr.DataArray(lat_idx_lo, dims="site"),
            lon=xr.DataArray(lon_idx_hi, dims="site"),
        )
        da_hi_hi = da_input.isel(
            lat=xr.DataArray(lat_idx_hi, dims="site"),
            lon=xr.DataArray(lon_idx_hi, dims="site"),
        )

        # Handle NaNs and renormalization within the Xarray/Dask graph
        # Create a boolean mask of valid data
        valid_lo_lo = da_lo_lo.notnull()
        valid_hi_lo = da_hi_lo.notnull()
        valid_lo_hi = da_lo_hi.notnull()
        valid_hi_hi = da_hi_hi.notnull()

        # Zero out weights where data is NaN
        w_lo_lo_masked = w_lo_lo.where(valid_lo_lo, 0.0)
        w_hi_lo_masked = w_hi_lo.where(valid_hi_lo, 0.0)
        w_lo_hi_masked = w_lo_hi.where(valid_lo_hi, 0.0)
        w_hi_hi_masked = w_hi_hi.where(valid_hi_hi, 0.0)

        # Sum the active weights
        w_sum = w_lo_lo_masked + w_hi_lo_masked + w_lo_hi_masked + w_hi_hi_masked

        # Prevent division-by-zero warnings, but we will restore NaNs later where all 4 corners were NaN
        w_sum_safe = w_sum.where(w_sum != 0, 1.0)

        # Renormalize weights using the safe denominator
        w_lo_lo_norm = w_lo_lo_masked / w_sum_safe
        w_hi_lo_norm = w_hi_lo_masked / w_sum_safe
        w_lo_hi_norm = w_lo_hi_masked / w_sum_safe
        w_hi_hi_norm = w_hi_hi_masked / w_sum_safe

        # Calculate final weighted sum
        result = (
            (da_lo_lo.fillna(0) * w_lo_lo_norm)
            + (da_hi_lo.fillna(0) * w_hi_lo_norm)
            + (da_lo_hi.fillna(0) * w_lo_hi_norm)
            + (da_hi_hi.fillna(0) * w_hi_hi_norm)
        )

        # Restore pure NaNs where all 4 corners were originally NaN (where w_sum was 0)
        result = result.where(w_sum != 0, np.nan)

        # Re-attach target coordinate metadata
        result = result.assign_coords(
            lat=("site", target_lats), lon=("site", target_lons)
        )

        return result

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
