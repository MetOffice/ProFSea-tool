"""Monte Carlo machinery shared by the Spatial (grid) and Local (site) models.

The regional stage follows the local-projection method of Palmer et al. (2020,
Earth's Future, doi:10.1029/2019EF001413, section 3.2). For each Monte Carlo
member:

1. take one member of the global ensemble, the same for every component, so
   the correlations between components are kept;
2. give it one GRD fingerprint model, the same for every Fingerprint
   component;
3. give it one CMIP sterodynamic pattern and one GIA estimate, drawn at random;
4. add the components to get the member's total;

then take percentiles over all members for each component and for the total.
"""

from __future__ import annotations

import logging
import operator
from functools import reduce

import dask.array as da
import numpy as np
import xarray as xr
from rich.console import Console
from rich.progress import track

from profsea.utils import (
    PROFSEA_VERSION,
    members_percentiles,
    plan_chunks,
    stratified_member_indices,
    validate_component_map,
)

from .base import SpatialComponent

logger = logging.getLogger(__name__)
console = Console()

TOTAL_NAME = "total_rsl"


class RegionalEnsemble:
    """Monte Carlo machinery shared by the Spatial and Local models.

    A run (:meth:`_run_ensemble`) does the following:

    1. ``_draw_ensemble``: draw the members, once for all components.
    2. ``_chunk_plan``: choose Dask chunk sizes from the memory budget.
    3. ``_project_components``: project each component, sum the total member
       by member, and take percentiles if requested.
    4. ``_arr_to_xr``: label the results as xarray DataArrays.

    The random draws reach the components through the state:

    - ``member_indices``: which global members are used, with an equal share
      per climate trajectory. Shared by all components, so the total is a sum
      over consistent members.
    - ``fingerprint_fraction``: one uniform number per member for choosing GRD
      fingerprints, shared by all Fingerprint components and independent of
      ``member_indices``.
    - one random stream per component, for any other draws (CMIP model, GIA
      model).

    The class using this mixin must set ``n_years``, ``start_year``,
    ``end_year``, ``baseline_yrs``, ``num_spatial_members``,
    ``output_percentiles``, ``compute_total``, ``chunk_size_mb`` and
    ``dtype``, call :meth:`_set_components`, and implement
    :meth:`_spatial_coords`.
    """

    def _set_components(
        self, components: dict[str, SpatialComponent], model_name: str
    ) -> None:
        """Check the components and record the shape of their global ensemble.

        Sets ``self.components``, ``self.n_climate`` and ``self.n_process``.
        """
        if not components:
            raise ValueError(
                "The components dictionary is empty. Please provide at least one component."
            )
        validate_component_map(components, SpatialComponent, model_name)
        self.components = components
        self.n_climate, self.n_process = self._global_ensemble_shape()

    def _has_percentiles(self) -> bool:
        """True if outputs are percentiles; False if they keep every member."""
        return self.output_percentiles is not None and len(self.output_percentiles) > 0

    def _global_ensemble_shape(self) -> tuple[int, int]:
        """Check the components' global ensembles and return their shape.

        Every component with a global driver must come from the same global
        ensemble, so that one member index refers to the same global member in
        every component, and its time axis must match this model's years.

        Returns
        -------
        tuple[int, int]
            Number of climate trajectories and process members. (1, 1) when no
            component has a global driver (e.g. GIA alone).
        """
        shapes = {
            name: tuple(comp.global_projection.shape)
            for name, comp in self.components.items()
            if comp.global_projection is not None
        }
        if not shapes:
            return 1, 1

        if len(set(shapes.values())) > 1:
            raise ValueError(
                "All global projections must come from the same global ensemble, "
                f"but their (climate, process, time) shapes differ: {shapes}."
            )

        n_climate, n_process, n_time = next(iter(shapes.values()))
        if n_time != self.n_years:
            raise ValueError(
                f"Global projections have {n_time} years but the model expects "
                f"{self.n_years} ({self.start_year} to {self.end_year - 1})."
            )
        return n_climate, n_process

    def _chunk_plan(self, spatial_shape: tuple[int, ...]) -> tuple[int, tuple]:
        """Choose time and spatial chunk sizes from the memory budget.

        Computing one block of the total needs the matching block of every
        component at the same time, so the ``chunk_size_mb`` budget is shared
        between all components plus the total.

        Returns
        -------
        tuple[int, tuple]
            Time chunk length and spatial chunk sizes (see ``plan_chunks``).
        """
        n_arrays = len(self.components) + int(self.compute_total)
        target_bytes = self.chunk_size_mb * 1e6 / max(1, n_arrays)
        return plan_chunks(
            self.num_spatial_members,
            self.n_years,
            spatial_shape,
            np.dtype(self.dtype).itemsize,
            target_bytes,
        )

    def _draw_ensemble(self, member_seed: int) -> dict[str, np.random.Generator]:
        """Draw the members and fingerprint choices shared by all components.

        Sets ``self.member_indices`` and ``self.fingerprint_fraction``. The
        seed is split into independent streams: one for the member indices,
        one for the fingerprint fraction and one per component. Each
        component's draws therefore do not depend on the others', and stay the
        same if components are added after it in the dictionary.

        Returns
        -------
        dict[str, np.random.Generator]
            One random number generator per component, keyed by name.
        """
        seed_seq = np.random.SeedSequence(member_seed)
        member_seed_seq, fingerprint_seed_seq, *component_seed_seqs = seed_seq.spawn(
            2 + len(self.components)
        )

        self.member_indices = stratified_member_indices(
            self.n_climate,
            self.n_process,
            self.num_spatial_members,
            np.random.default_rng(member_seed_seq),
        )
        self.fingerprint_fraction = np.random.default_rng(fingerprint_seed_seq).random(
            self.num_spatial_members
        )
        return {
            name: np.random.default_rng(child)
            for name, child in zip(self.components, component_seed_seqs)
        }

    def _project_components(
        self, state, component_rngs: dict[str, np.random.Generator], description: str
    ) -> dict[str, da.Array]:
        """Project every component, add the total, and take percentiles.

        Returns
        -------
        dict[str, da.Array]
            Lazy arrays of shape (members, time, *spatial), or
            (percentiles, time, *spatial) when output percentiles are
            requested, keyed by component name, plus "total_rsl" if
            ``compute_total``.
        """
        want_percentiles = self._has_percentiles()
        outputs = {}
        # Member arrays to sum into the total. For separable components these
        # are independent copies (see SeparableProjection.to_array), so no large
        # block is shared between the total and a component's own output.
        # Sharing would make Dask keep each block until both outputs have used
        # it, which can grow to a whole member array when everything is
        # computed together (e.g. by save_components).
        total_members = []

        console.print()
        for name, comp in track(self.components.items(), description=description):
            projection = comp.project_separable(state, component_rngs[name])

            if projection is None:
                # The component builds its own (members, time, *spatial) array.
                # It cannot be rebuilt independently, so it is shared with the
                # total and costs more memory than a separable component.
                array = da.asarray(comp.project(state, component_rngs[name]))
                array = array.astype(state.dtype).rechunk(
                    (-1, state.time_chunk, *state.spatial_chunks)
                )
                outputs[name] = (
                    members_percentiles(array, self.output_percentiles)
                    if want_percentiles
                    else array
                )
                total_members.append(array)
                continue

            # Percentiles come straight from the factors where possible, so the
            # member array is only built when it is needed
            outputs[name] = (
                projection.percentiles(self.output_percentiles, state)
                if want_percentiles
                else projection.to_array(state)
            )
            if self.compute_total:
                total_members.append(projection.to_array(state, independent=True))
        console.print()

        if self.compute_total:
            # Sum member by member, before any percentiles are taken
            total = reduce(operator.add, total_members)
            outputs[TOTAL_NAME] = (
                members_percentiles(total, self.output_percentiles)
                if want_percentiles
                else total
            )
        return outputs

    def _run_ensemble(
        self,
        member_seed: int,
        state_cls: type,
        spatial_shape: tuple[int, ...],
        description: str,
        **state_kwargs,
    ) -> dict[str, xr.DataArray]:
        """Run the regional Monte Carlo and return labelled outputs.

        Parameters
        ----------
        member_seed: int
            Seed for every random draw in the run.
        state_cls: type
            SpatialState or LocalState.
        spatial_shape: tuple[int, ...]
            (n_lat, n_lon) for a grid or (n_sites,) for sites.
        description: str
            Progress bar label.
        **state_kwargs
            Grid or site fields for the state (e.g. grid_lats, target_lats).

        Returns
        -------
        dict[str, xr.DataArray]
            Lazy projections keyed by component name, plus "total_rsl".
        """
        component_rngs = self._draw_ensemble(member_seed)
        time_chunk, spatial_chunks = self._chunk_plan(spatial_shape)
        state = state_cls(
            n_years=self.n_years,
            num_spatial_members=self.num_spatial_members,
            output_percentiles=self.output_percentiles,
            baseline_yrs=self.baseline_yrs,
            dtype=self.dtype,
            endofhistory=self.start_year,
            member_indices=self.member_indices,
            fingerprint_fraction=self.fingerprint_fraction,
            time_chunk=time_chunk,
            spatial_chunks=spatial_chunks,
            **state_kwargs,
        )
        return self._arr_to_xr(
            self._project_components(state, component_rngs, description)
        )

    def _spatial_coords(self) -> tuple[list[str], dict, str]:
        """Describe the spatial axes of the outputs; implemented by each model.

        Returns
        -------
        tuple[list[str], dict, str]
            Spatial dimension names (e.g. ["lat", "lon"]), their coordinates,
            and a word used in each output's long_name (e.g. "Regional").
        """
        raise NotImplementedError

    def _arr_to_xr(self, arr_dict: dict[str, da.Array]) -> dict[str, xr.DataArray]:
        """
        Wrap arrays of shape (percentiles or members, time, *spatial) as DataArrays.

        The leading dimension is "percentile", or "member" with each member's
        climate_member and process_member in the global ensemble.
        """
        spatial_dims, spatial_coords, label = self._spatial_coords()
        ensemble_dim = "percentile" if self._has_percentiles() else "member"
        xr_dict = {}
        for name, arr in arr_dict.items():
            xr_dict[name] = xr.DataArray(
                arr,
                dims=[ensemble_dim, "time", *spatial_dims],
                coords={
                    **self._ensemble_coords(arr.shape[0]),
                    "time": np.arange(self.start_year, self.start_year + arr.shape[1]),
                    **spatial_coords,
                },
                attrs={
                    "units": "m",
                    "long_name": f"{label} {name} sea-level projections",
                    "source": f"ProFSea v{PROFSEA_VERSION}",
                },
            )
        return xr_dict

    def _ensemble_coords(self, n_members: int) -> dict:
        """Coordinates for the leading axis of the outputs.

        For full-member output, each member also records where it came from in
        the global ensemble: member indices count through the flattened
        (climate, process) ensemble, so index // n_process is the climate
        trajectory and index % n_process the process member.
        """
        if self._has_percentiles():
            return {"percentile": self.output_percentiles}
        return {
            "member": np.arange(n_members),
            "climate_member": ("member", self.member_indices // self.n_process),
            "process_member": ("member", self.member_indices % self.n_process),
        }

    def sum_components(self, components: dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Total relative sea-level change, added to ``components`` as "total_rsl".

        ``run`` already computes the total by summing members when
        ``compute_total`` is True, so this returns that total if it is present.
        Percentiles cannot be summed (the percentile of a sum is not the sum of
        percentiles), so summing percentile outputs raises an error.

        Parameters
        ----------
        components: dict[str, xr.DataArray]
            Component projections, e.g. ``model.results``.

        Returns
        -------
        xr.DataArray
            Total relative sea-level change.
        """
        if TOTAL_NAME in components:
            logger.info(f"'{TOTAL_NAME}' was already computed by run().")
            return components[TOTAL_NAME]

        if any("percentile" in comp.dims for comp in components.values()):
            raise ValueError(
                "Cannot sum percentiles: the percentile of a sum is not the sum of "
                "percentiles. Use compute_total=True, or output_percentiles=None and "
                "then sum the members."
            )

        total = xr.concat(components.values(), dim="component").sum(
            dim="component", skipna=False
        )
        total.attrs["units"] = "m"
        components[TOTAL_NAME] = total
        return total
