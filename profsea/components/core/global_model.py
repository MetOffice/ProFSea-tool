from __future__ import annotations

import concurrent.futures
import logging

import numpy as np
import xarray as xr
from rich.console import Console
from rich.progress import track

from profsea.utils import (
    check_shapes,
    sample_members_2D,
    save_components,
    validate_component_map,
)
from profsea.utils.ui import print_global_preflight

from .base import Component
from .state import ClimateState

console = Console()
logger = logging.getLogger(__name__)


class Global:
    """Global sea level rise component emulator.

    Parameters
    ----------
    components: Dict
        List of SLR components for projections
    end_yr: int
        End year of the projections.
    nt: int
        Number of realisations of the input timeseries
    num_members: int
        Number of realisations of for each component.
        Must be a multiple of the number of glacier methods.
    tcv: float
        Multiplier for the standard deviation in the input fields.
    parallel: bool
        If True, project SLR components in parallel.
    output_percentiles: list|np.ndarray
        If not None, calculate percentiles from a 1D list/array for each
        component
    palmer_method: bool
        If True, allow integration to end in any year up to 2300,
        with the contributions to GMLSR from ice-sheet dynamics,
        Greenland SMB and land water storage held at the 2100 rate
        beyond 2100.
    random_sample: bool
        If True, randomly sample a single ensemble member across all
        components

    Attributes
    ----------
    endofhistory: int
        First year of AR5 projections.
    endofAR5: int
        Last year of AR5 projections.
    n_years: int
        Length of projections.
    """

    def __init__(
        self,
        components: dict[str, Component],
        end_yr: int,
        nt: int = 100,
        num_members: int = 1000,
        tcv: float = 1.0,
        parallel: bool = True,
        output_percentiles: list | np.ndarray = None,
        palmer_method: bool = True,
        random_sample: bool = False,
        dtype: np.dtype | str = np.float32,
    ):
        self.components = components
        self.end_yr = end_yr
        self.nt = nt
        self.num_members = num_members
        self.tcv = tcv
        self.parallel = parallel
        self.output_percentiles = output_percentiles
        self.palmer_method = palmer_method
        self.random_sample = random_sample
        self.dtype = np.dtype(dtype)

        self.endofhistory = 2006
        self.endofAR5 = 2100
        self.n_years = self.end_yr - self.endofhistory

        validate_component_map(self.components, Component, "Global")

    # Inject method!
    save_components = save_components

    def _arr_to_xr(self, arr_dict: dict[str, np.ndarray]) -> dict[str, xr.DataArray]:
        """Convert a dictionary of numpy/dask arrays to xarray DataArrays.

        Parameters
        ----------
        arr_dict: dict
            Dictionary of arrays, where keys are component names and values are arrays.

        Returns
        -------
        dict
            Dictionary of xarray DataArrays, where keys are component names and values are xarray DataArrays.
        """
        xr_dict = {}
        if self.output_percentiles is not None and len(self.output_percentiles) > 0:
            for name, arr in arr_dict.items():
                xr_dict[name] = xr.DataArray(
                    arr,
                    dims=["percentile", "time"],
                    coords={
                        "percentile": self.output_percentiles,
                        "time": np.arange(
                            self.endofhistory, self.endofhistory + arr.shape[1]
                        ),
                    },
                )
                xr_dict[name].attrs["units"] = "m"

        else:
            for name, arr in arr_dict.items():
                xr_dict[name] = xr.DataArray(
                    arr,
                    dims=["climate_member", "process_member", "time"],
                    coords={
                        "climate_member": np.arange(arr.shape[0]),
                        "process_member": np.arange(arr.shape[1]),
                        "time": np.arange(
                            self.endofhistory, self.endofhistory + arr.shape[2]
                        ),
                    },
                )
                xr_dict[name].attrs["units"] = "m"

        return xr_dict

    def run(
        self,
        scenario: str,
        T_change: np.ndarray | xr.DataArray,
        member_seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """Run the emulator to project GMSLR components for a specific state.
        Parameters
        ----------
        scenario: str
            Name of the scenario.
        T_change: np.ndarray | xr.DataArray
            Array/DataArray of temperature change values.
        member_seed: int
            Seed for numpy.random.
        """

        if isinstance(T_change, xr.DataArray):
            T_change = T_change.to_numpy()
        else:
            T_change = np.asarray(T_change)

        seed_seq = np.random.SeedSequence(member_seed)
        run_rng = np.random.default_rng(seed_seq)

        check_shapes(T_change, self.n_years)

        # Standardize T_change shape to (nt, n_years)
        if T_change.ndim > 2:
            T_change = np.squeeze(T_change)
        if T_change.ndim == 1:
            T_change = np.expand_dims(T_change, axis=0)

        self.nt = T_change.shape[0]

        print_global_preflight(self, scenario)

        T_change = T_change.astype(self.dtype)
        T_ens, T_int_ens, T_int_med = self._calculate_drivers(T_change)

        # Shared physical correlation state
        fraction = run_rng.random((self.nt, self.num_members)).astype(self.dtype)

        state = ClimateState(
            scenario=scenario,
            T_ens=T_ens,
            T_int_ens=T_int_ens,
            T_int_med=T_int_med,
            fraction=fraction,
            palmer_method=self.palmer_method,
            endofAR5=self.endofAR5,
            endofhistory=self.endofhistory,
            end_yr=self.end_yr,
            n_years=self.n_years,
            nt=self.nt,
            num_members=self.num_members,
            dtype=self.dtype,
        )

        # Child RNGs for each component
        child_seeds = seed_seq.spawn(len(self.components))
        comp_rngs = {
            name: np.random.default_rng(s)
            for name, s in zip(self.components.keys(), child_seeds)
        }

        results = {}
        if self.parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(comp.project, state, comp_rngs[name]): name
                    for name, comp in self.components.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    comp_name = futures[future]
                    try:
                        results[comp_name] = future.result()
                    except Exception as e:
                        raise RuntimeError(f"Component '{comp_name}' failed.") from e
        else:
            for name, comp in track(
                self.components.items(), description="Projecting components..."
            ):
                results[name] = comp.project(state, comp_rngs[name])

        # Random Sampling
        if self.random_sample:
            random_idx = run_rng.integers(low=0, high=self.nt * self.num_members)
            for comp_name, data in results.items():
                if data.ndim > 1:
                    results[comp_name] = data[random_idx][None, :]

        # Output percentiles
        if self.output_percentiles is not None:
            logger.info(
                f"Sampling {len(self.output_percentiles)} members per component..."
            )
            for comp_name, data in results.items():
                data = data.reshape(
                    self.nt * self.num_members, data.shape[-1]
                )  # reshape to 2D
                results[comp_name] = sample_members_2D(
                    data, self.output_percentiles, dtype=self.dtype
                )

        self.results = self._arr_to_xr(results)
        return self.results

    def sum_components(self, components: dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Sum the components in-place to get total GMSLR.

        Parameters
        ----------
        components: dict[str, xr.DataArray]
            Dictionary of component names and their corresponding Xarray DataArrays.

        Returns
        -------
        xr.DataArray
            DataArray of the summed global projections.
        """

        iterator = iter(components.values())
        gmslr = next(iterator).copy()

        for comp in iterator:
            gmslr += comp

        gmslr.attrs["units"] = "m"
        gmslr.attrs["description"] = "Total global mean sea level rise"
        components["total_gmslr"] = gmslr
        return gmslr

    def _calculate_drivers(self, T_change: np.ndarray) -> tuple:
        """Calculate the drivers of GMSLR: temperature change and
        thermosteric sea level rise.

        Returns
        -------
        T_ens: np.ndarray
            Ensemble of temperature changes.
        T_int_ens: np.ndarray
            Ensemble of time-integral temperature anomalies.
        T_int_med: np.ndarray
            Median of time-integral temperature anomalies.
        """
        T_ens = T_change.copy()

        # Time-integral of temperature anomaly
        T_int_ens = np.cumsum(T_ens, axis=1)
        T_int_med = np.cumsum(np.median(T_ens, axis=0))
        return (T_ens, T_int_ens, T_int_med)
