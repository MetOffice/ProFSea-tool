from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from dask.array.core import elemwise
from dask.base import tokenize

from profsea.utils import (
    PROFSEA_ASSETS_URL,
    PROFSEA_ASSETS_VERSION,
    fetch_zenodo_fingerprints,
    members_percentiles,
)

from .state import ClimateState

PROFSEA_DIR = Path(__file__).resolve().parent.parent.parent


class Component(ABC):
    @abstractmethod
    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """Calculate and return the SLR projection for this component"""
        pass


@dataclass
class SeparableProjection:
    """A spatial projection written as "time series x spatial pattern".

    Every built-in spatial component has this form. For Monte Carlo member
    ``m`` at time ``t`` and location ``x``::

        value[m, t, x] = temporal[m, t] * patterns[pattern_index[m], x]

    For example, a ``Fingerprint`` component multiplies each member's global
    ice-sheet contribution (``temporal``) by the GRD fingerprint that member
    drew (``patterns[pattern_index[m]]``). GIA is the other way round: every
    member shares one time series (years since the baseline) and members
    differ only in which GIA model they drew.

    Keeping the two factors separate, instead of handing the models a ready
    made (members, time, *spatial) array, has two benefits:

    - the spatial models choose the Dask chunking in one place (``to_array``);
    - percentiles can often be computed exactly without building and sorting
      the full member array (``percentiles``), which is much faster.

    Parameters
    ----------
    temporal: np.ndarray | da.Array
        Time series of each member, shape (members, time), in metres; or a
        single time series shared by every member, shape (time,).
    patterns: da.Array
        The available spatial patterns, shape (n_patterns, *spatial), where
        *spatial is (lat, lon) on a grid or (site,) at sites.
    pattern_index: np.ndarray | None
        Which pattern each member uses, shape (members,). Must be None when
        there is only one pattern, in which case every member uses it.
    """

    temporal: np.ndarray | da.Array
    patterns: da.Array
    pattern_index: np.ndarray | None = None

    def __post_init__(self):
        if self.pattern_index is None:
            if self.patterns.shape[0] != 1:
                raise ValueError(
                    "pattern_index is required when there is more than one pattern."
                )
        elif np.ndim(self.temporal) == 2 and len(self.pattern_index) != len(
            self.temporal
        ):
            raise ValueError(
                f"pattern_index has {len(self.pattern_index)} entries but temporal "
                f"has {len(self.temporal)} members."
            )

    def _chunked(self, state) -> tuple[da.Array, da.Array]:
        """Cast both factors to the output dtype and apply the model's chunk plan.

        The member axis is always one chunk (percentiles need every member at
        once); time and space are split as ``state.time_chunk`` and
        ``state.spatial_chunks`` say. If the plan does not match the pattern
        dimensions (e.g. a hand-built test state), space is left unsplit.
        """
        temporal = da.asarray(self.temporal).astype(state.dtype)
        member_axes = (-1,) * (temporal.ndim - 1)  # () when temporal is shared
        temporal = temporal.rechunk(member_axes + (state.time_chunk,))

        patterns = da.asarray(self.patterns).astype(state.dtype)
        spatial_chunks = tuple(state.spatial_chunks)
        if len(spatial_chunks) != patterns.ndim - 1:
            spatial_chunks = (-1,) * (patterns.ndim - 1)
        patterns = patterns.rechunk((-1, *spatial_chunks))
        return temporal, patterns

    def _member_patterns(self, patterns: da.Array) -> da.Array:
        """The pattern each member uses, shape (members, *spatial).

        With a single pattern this returns it once, shape (1, *spatial), and
        relies on broadcasting rather than copying it for every member.
        """
        if self.pattern_index is None:
            return patterns[:1]
        # Gather each member's pattern block by block. Dask's own fancy
        # indexing (patterns[pattern_index]) would split the member axis into
        # many small chunks, which then have to be merged again before the
        # components can be summed into the total.
        return patterns.map_blocks(
            np.take,
            self.pattern_index,
            axis=0,
            chunks=((len(self.pattern_index),), *patterns.chunks[1:]),
            dtype=patterns.dtype,
        )

    def to_array(self, state, independent: bool = False) -> da.Array:
        """Build the lazy member array, shape (members, time, *spatial).

        Parameters
        ----------
        state: SpatialState | LocalState
            Provides the dtype, chunk plan and number of members.
        independent: bool, optional
            If True, build the array under a Dask graph name of its own. Use
            this when the same members feed two outputs that are computed
            together (a component's percentiles and the total): with a shared
            name, Dask computes each member block once and keeps it in memory
            until both outputs have used it, which at 1 degree can mean holding
            most of a 76 GB array. Independent copies are recomputed, which is
            cheap (one multiplication), and freed as soon as they are used.

        Returns
        -------
        da.Array
            One value per member, time and location.
        """
        temporal, patterns = self._chunked(state)
        add_spatial_axes = (None,) * (patterns.ndim - 1)
        if temporal.ndim == 1:
            temporal = temporal[None]  # shared series: (1, time)

        # (members or 1, time, 1, ...) * (members or 1, 1, *spatial)
        temporal = temporal[(...,) + add_spatial_axes]
        member_patterns = self._member_patterns(patterns)[:, None]
        if independent:
            name = "independent-members-" + tokenize(temporal, member_patterns)
            members = elemwise(operator.mul, temporal, member_patterns, name=name)
        else:
            members = temporal * member_patterns

        # A shared time series with a single pattern makes every member
        # identical; broadcast to the member count without copying.
        n_members = state.num_spatial_members
        if members.shape[0] == 1 and n_members > 1:
            members = da.broadcast_to(
                members,
                (n_members, *members.shape[1:]),
                chunks=((n_members,), *members.chunks[1:]),
            )
        return members

    def percentiles(self, percentiles, state) -> da.Array:
        """Lazy percentiles over members at every time and location.

        If members differ in only one of the two factors, the other factor is
        common to all members at a given time and location, and multiplying
        by a common number does not change the members' order, except that a
        negative number reverses it. So for a common factor ``c`` and a
        varying factor ``v``:

        - where ``c >= 0``, the q-th percentile is ``c * (q-th percentile of v)``;
        - where ``c < 0``, it is ``c * ((100 - q)-th percentile of v)``.

        For example, near an ice sheet the fingerprint is negative, so the 5th
        percentile of local sea level comes from the 95th percentile of the
        global ice loss. This is exact for linear interpolation, and only
        ``v`` needs ranking:

        - one pattern for all members: rank the time series (members, time);
        - one time series for all members: rank the patterns (members, *spatial).

        If members differ in both factors, the full member array is ranked at
        every point instead.

        Parameters
        ----------
        percentiles: list | np.ndarray
            Percentiles to compute, between 0 and 100.
        state: SpatialState | LocalState
            Provides the dtype, chunk plan and number of members.

        Returns
        -------
        da.Array
            Shape (n_percentiles, time, *spatial).
        """
        temporal, patterns = self._chunked(state)
        shared_time = temporal.ndim == 1
        if self.pattern_index is not None and not shared_time:
            # Members differ in both factors: no shortcut
            return members_percentiles(self.to_array(state), percentiles)

        q = np.atleast_1d(np.asarray(percentiles, dtype=np.float64))
        add_spatial_axes = (None,) * (patterns.ndim - 1)
        if shared_time:
            varying_factor = self._member_patterns(patterns)  # (members, *spatial)
            common_factor = temporal[(None, slice(None)) + add_spatial_axes]
            # ranked percentiles (n_q, *spatial) -> (n_q, 1, *spatial)
            to_output_shape = (slice(None), None)
        else:
            varying_factor = temporal  # (members, time)
            common_factor = patterns[0][None, None]  # (1, 1, *spatial)
            # ranked percentiles (n_q, time) -> (n_q, time, 1, ...)
            to_output_shape = (...,) + add_spatial_axes

        # Rank once for q and 100 - q, covering both signs of the common factor
        ranked = members_percentiles(varying_factor, np.concatenate((q, 100.0 - q)))
        for_positive_factor = ranked[: len(q)][to_output_shape]
        for_negative_factor = ranked[len(q) :][to_output_shape]
        return (
            da.where(common_factor >= 0, for_positive_factor, for_negative_factor)
            * common_factor
        )


class SpatialComponent(Component):
    """Base class for components that map global projections onto space.

    Subclasses normally implement :meth:`project_separable`; :meth:`project`
    then builds the full member array from it. Components that do not
    factorise can override :meth:`project` instead and return an array of
    shape (members, time, *spatial).
    """

    def fetch_assets(self):
        fetch_zenodo_fingerprints(
            zenodo_url=PROFSEA_ASSETS_URL,
            data_dir=PROFSEA_DIR,
            expected_folder_name="profsea-assets",
            version=PROFSEA_ASSETS_VERSION,
        )

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

    @property
    def global_projection(self):
        """The global ensemble driving this component, shape (climate, process,
        time), or None for components without a global driver (e.g. GIA)."""
        return getattr(self, "_global_projection", None)

    def _set_global_projection(self, global_projection) -> None:
        """Check the global ensemble and store it as (climate, process, time).

        Parameters
        ----------
        global_projection: xr.DataArray | np.ndarray | da.Array
            Output of ``Global`` for this component, shape (climate_member,
            process_member, time). A 2D (members, time) array is treated as a
            single climate trajectory.

        Raises
        ------
        ValueError
            If it holds percentiles rather than ensemble members, or has the
            wrong number of dimensions.
        """
        data = global_projection
        if isinstance(global_projection, xr.DataArray):
            if "percentile" in global_projection.dims:
                raise ValueError(
                    "Spatial components need the full global ensemble, not "
                    "percentiles. Run Global with output_percentiles=None."
                )
            data = global_projection.data
        if not isinstance(data, da.Array):
            data = np.asarray(data)
        if data.ndim == 2:
            data = data[None]
        if data.ndim != 3:
            raise ValueError(
                "global_projection must have shape (climate, process, time) or "
                f"(members, time); got {data.shape}."
            )
        self._global_projection = data

    def sample_global_members(self, state) -> np.ndarray | da.Array:
        """Global time series of this run's Monte Carlo members.

        ``state.member_indices`` index the global ensemble flattened to
        (climate x process) members. Every component uses the same indices,
        so member ``m`` of each component comes from the same global member.

        Returns
        -------
        np.ndarray | da.Array
            Shape (members, time).
        """
        projection = self.global_projection
        flat = projection.reshape(-1, projection.shape[-1])
        return flat[state.member_indices]

    def choose_patterns(
        self,
        patterns: da.Array,
        state,
        rng: np.random.Generator,
        sample_spatial: bool,
        fraction: np.ndarray | None = None,
    ) -> tuple[da.Array, np.ndarray | None]:
        """Decide which spatial pattern each Monte Carlo member uses.

        Parameters
        ----------
        patterns: da.Array
            Available patterns (e.g. one per GRD or CMIP model), shape
            (n_patterns, *spatial).
        state: SpatialState | LocalState
            Provides the number of members.
        rng: np.random.Generator
            This component's random stream, used when ``fraction`` is None.
        sample_spatial: bool
            If True, each member draws one pattern with equal probability. If
            False, all members use the mean pattern ("storyline" mode).
        fraction: np.ndarray | None
            Optional uniform [0, 1) value per member. Components given the same
            fraction (and the same number of patterns, in the same order) draw
            the same pattern for each member; Fingerprint components use this
            so a member keeps one GRD model throughout.

        Returns
        -------
        tuple[da.Array, np.ndarray | None]
            The patterns to use and each member's pattern index, or None when
            every member uses the same single pattern.
        """
        n_patterns = patterns.shape[0]
        if n_patterns == 1:
            return patterns, None
        if not sample_spatial:
            return da.nanmean(patterns, axis=0, keepdims=True), None
        if fraction is not None:
            # floor(fraction * n) is uniform over 0..n-1; the minimum guards
            # against rounding up to n
            index = np.minimum((fraction * n_patterns).astype(np.intp), n_patterns - 1)
        else:
            index = rng.integers(n_patterns, size=state.num_spatial_members)
        return patterns, index

    def project_separable(self, state, rng) -> SeparableProjection | None:
        """Return this component's projection as time series x patterns.

        Override this in new spatial components. Return None only if the
        component cannot be written in that form, and override
        :meth:`project` instead.
        """
        return None

    def project(self, state, rng: np.random.Generator) -> da.Array:
        """Lazy member array of shape (members, time, *spatial).

        Built from :meth:`project_separable`. Components that cannot be
        written as time series x patterns override this method directly.
        """
        separable = self.project_separable(state, rng)
        if separable is None:
            raise NotImplementedError(
                f"{type(self).__name__} must implement project_separable or project."
            )
        return separable.to_array(state)
