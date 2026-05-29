import concurrent.futures
from pathlib import Path
import os
from typing import Dict

import numpy as np
from rich.console import Console
import xarray as xr

from .state import ClimateState
from .base import Component
from profsea.utils import sample_members_2D, check_shapes

console = Console()


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
    nyr: int
        Length of projections.
    """

    def __init__(
        self,
        components: Dict[str, Component],
        end_yr: int,
        nt: int = 100,
        num_members: int = 1000,
        tcv: float = 1.0,
        parallel: bool = True,
        output_percentiles: list | np.ndarray = None,
        palmer_method: bool = True,
        random_sample: bool = False,
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

        self.endofhistory = 2006
        self.endofAR5 = 2100
        self.nyr = self.end_yr - self.endofhistory

    def run(
        self,
        scenario: str,
        T_change: np.ndarray,
        member_seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """Run the emulator to project GMSLR components for a specific state.
        Parameters
        ----------
        scenario: str
            Name of the scenario.
        T_change: np.ndarray
            Array of temperature change values.
        member_seed: int
            Seed for numpy.random.
        """

        seed_seq = np.random.SeedSequence(member_seed)
        run_rng = np.random.default_rng(seed_seq)

        check_shapes(T_change, self.nyr)

        # Standardize T_change shape to (nt, nyr)
        if T_change.ndim > 2:
            T_change = np.squeeze(T_change)
        if T_change.ndim == 1:
            T_change = np.expand_dims(T_change, axis=0)

        self.nt = T_change.shape[0]

        T_ens, T_int_ens, T_int_med = self._calculate_drivers(T_change)

        # Shared physical correlation state
        fraction = run_rng.random(self.num_members * self.nt)

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
            nyr=self.nyr,
            nt=self.nt,
            num_members=self.num_members,
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
            for name, comp in self.components.items():
                results[name] = comp.project(state, comp_rngs[name])

        # Random Sampling
        if self.random_sample:
            random_idx = run_rng.integers(low=0, high=self.nt * self.num_members)
            for comp_name, data in results.items():
                if data.ndim > 1:
                    results[comp_name] = data[random_idx][None, :]

        # Output percentiles
        if self.output_percentiles is not None:
            console.log(
                f"Sampling {len(self.output_percentiles)} members per component..."
            )
            for comp_name, data in results.items():
                results[comp_name] = sample_members_2D(data, self.output_percentiles)

        self.results = results

        return results

    def sum_components(self, components: Dict[str, np.ndarray]) -> np.ndarray:
        """Sum the components to get total GMSLR."""
        components["gmslr"] = np.sum(
            [np.atleast_2d(c) for c in components.values()], axis=0
        )
        return components["gmslr"]

    def save_components(
        self, components: Dict[str, np.ndarray], output_dir: str, scenario_name: str
    ) -> None:
        """Save SLR components as nc files to a directory.

        Parameters
        ----------
        components: Dict[str, np.ndarray]
            Dictionary of component names and their corresponding arrays.
        output_directory: str
            Directory to save components to.
        scenario_name: str
            Name of the scenario you've run the emulator for.

        Returns
        -------
        None
        """
        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save data in netcdf format
        ds = xr.Dataset()
        member_dim = "percentile" if self.output_percentiles is not None else "member"
        for name, component in components.items():
            xr_dataArray = xr.DataArray(
                component,
                dims=[member_dim, "time"],
                coords={
                    member_dim: self.output_percentiles
                    if self.output_percentiles is not None
                    else np.arange(
                        component.shape[0]
                    ),  # handle if no output percentiles
                    "time": np.arange(2006, component.shape[1] + 2006),
                },
            )
            xr_dataArray.attrs["units"] = "m"
            ds[name] = xr_dataArray
        ds.to_netcdf(os.path.join(output_dir, f"{scenario_name}_global.nc"))

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
        return T_ens, T_int_ens, T_int_med
