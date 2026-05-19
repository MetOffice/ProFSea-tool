"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

# Calculate Monte Carlo projections of GMSLR using methods
# from Jonathon Gregory and AR5. Staying close to JG's original
# code where possible.
import os
import concurrent
from collections.abc import Sequence
import functools
import atexit
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from scipy.stats import norm, truncnorm
import xarray as xr

from profsea.utils import sample_members_2D
from .antarctica import AntarcticaISMIP6

console = Console()


@functools.lru_cache(maxsize=1)
def load_greenland_calibration():
    """Loads the CSV once and keeps it in memory."""
    path = Path(__file__).parent / "aux_data" / "ISMIP_GIS_calibration.csv"
    return pd.read_csv(path)


@functools.lru_cache(maxsize=1)
def load_landwater_projection():
    """Loads the NetCDF once and keeps the VALUES in memory."""
    path = Path(__file__).parent / "aux_data" / "ssp_global_landwater_projections.nc"
    with xr.open_dataset(path) as ds:
        ds.load()
    return ds


_SHARED_EXECUTOR = None


def get_shared_executor():
    """Returns a singleton ThreadPoolExecutor."""
    global _SHARED_EXECUTOR
    if _SHARED_EXECUTOR is None:
        max_workers = min(8, os.cpu_count() or 1)
        _SHARED_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )
    return _SHARED_EXECUTOR


@atexit.register
def shutdown_executor():
    """Ensures the executor is cleaned up when the script exits."""
    global _SHARED_EXECUTOR
    if _SHARED_EXECUTOR:
        _SHARED_EXECUTOR.shutdown(wait=True)


class Global:
    """Global sea level rise component emulator.

    Parameters
    ----------
    T_change: np.ndarray
        Array of temperature change values.
    OHC_change: np.ndarray
        Array of ocean heat content change values.
    scenario: str
        Name of the scenario.
    end_yr: int
        End year of the projections.
    seed: int
        Seed for numpy.random.
    nt: int
        Number of realisations of the input timeseries
    nm: int
        Number of realisations of for each component.
        Must be a multiple of the number of glacier methods.
    tcv: float
        Multiplier for the standard deviation in the input fields.
    glaciermip: bool | int
        If False, use AR5 parameters. If 1, use GlacierMIP
        (Hock et al., 2019). If 2, use GlacierMIP2 (Marzeion et al., 2020).
    parallel: bool
        If True, project SLR components in parallel.
    input_ensemble: bool
        If True, use an input ensemble of temperature and
        ocean heat content change.
    output_percentiles: list|np.ndarray
        If not None, calculate percentiles from a 1D list/array for each
        component
    random_sample: bool
        If True, randomly sample a single ensemble member across all
        components
    T_percentile_95: np.ndarray
        95th percentile of temperature change timeseries.
    OHC_percentile_95: np.ndarray
        95th percentile of ocean heat content change timeseries.
    cum_emissions_total: float
        Total cumulative emissions from 2015 to 2100.
    palmer_method: bool
        If True, allow integration to end in any year up to 2300,
        with the contributions to GMLSR from ice-sheet dynamics,
        Greenland SMB and land water storage held at the 2100 rate
        beyond 2100.

    Attributes
    ----------
    endofhistory: int
        First year of AR5 projections.
    endofAR5: int
        Last year of AR5 projections.
    nyr: int
        Length of projections.
    fgreendyn: float
        Fraction of SLE from Greenland during 1996 to 2005 assumed
        to result from rapid dynamical change.
    dgreen: float
        m SLE from Greenland during 1996 to 2005 according to AR5 chapter 4.
    dant: float
        m SLE from Antarctica during 1996 to 2005 according to AR5 chapter 4.
    mSLEoGt: float
        Conversion factor for Gt to m SLE.
    exp_efficiency: float
        Sensitivity of thermosteric SLR to ocean heat content change.
    """

    def __init__(
        self,
        end_yr: int,
        seed: int = 1234,
        nt: int = 100,
        nm: int = 1000,
        tcv: float = 1.0,
        glaciermip: bool | int = 2,
        parallel: bool = True,
        input_ensemble: bool = True,
        output_percentiles: list | np.ndarray = None,
        random_sample: bool = False,
        T_percentile_95: np.ndarray = None,
        OHC_percentile_95: np.ndarray = None,
        cum_emissions_total: float = None,
        palmer_method: bool = True,
        active_components: list[str] = None,
    ) -> None:

        self.end_yr = end_yr
        self.nt = nt
        self.nm = nm
        self.tcv = tcv
        self.glaciermip = glaciermip
        self.parallel = parallel
        self.input_ensemble = input_ensemble
        self.output_percentiles = output_percentiles
        self.random_sample = random_sample
        self.T_percentile_95 = T_percentile_95
        self.OHC_percentile_95 = OHC_percentile_95
        self.cum_emissions_total = cum_emissions_total
        self.palmer_method = palmer_method

        # First year of AR5 projections
        self.endofhistory = 2006

        # Last year of AR5 projections
        self.endofAR5 = 2100

        # Length of projections
        self.nyr = self.end_yr - self.endofhistory

        # Fraction of SLE from Greenland during 1996 to 2005 assumed to result from
        # rapid dynamical change, with the remainder assumed to result from SMB change
        self.fgreendyn = 0.5

        # m SLE from Greenland during 1996 to 2005 according to AR5 chapter 4
        self.dgreen = (3.21 - 0.30) * 1e-3

        # m SLE from Antarctica during 1996 to 2005 according to AR5 chapter 4
        self.dant = (2.37 + 0.13) * 1e-3

        # Conversion factor for Gt to m SLE
        self.mSLEoGt = 1e12 / 3.61e14 * 1e-3

        self.seed_seq = np.random.SeedSequence(seed if seed is not None else 1234)
        self.rng = np.random.default_rng(self.seed_seq)
        self._stochastic_components = [
            "glacier",
            "antsmb",
            "greenland_ar6",
            "greendyn",
            "greensmb",
            "antdyn",
            "antarctica",
        ]
        child_seeds = self.seed_seq.spawn(len(self._stochastic_components))
        self.component_rngs = {
            comp: np.random.default_rng(s)
            for comp, s in zip(self._stochastic_components, child_seeds)
        }

        self.active_components = active_components or [
            "expansion",
            "glacier",
            "greenland_ar6",
            "antarctica",
            "landwater",
        ]

        # Setup AIS emulator, if needed
        if "antarctica" in self.active_components:
            wais_path = Path(__file__).parent / "aux_data" / "wais_params_APR_nodrift.nc"
            eais_path = Path(__file__).parent / "aux_data" / "eais_params_APR_nodrift.nc"
            aispen_path = Path(__file__).parent / "aux_data" / "pen_params_APR_nodrift.nc"
            self.wais_model = AntarcticaISMIP6(wais_path)
            self.eais_model = AntarcticaISMIP6(eais_path)
            self.aispen_model = AntarcticaISMIP6(aispen_path)

    def get_components(self) -> dict:
        """Get all GMSLR components as a dictionary."""
        components_dict = {}
        # Explicitly check for sub-components alongside active ones
        export_list = self.active_components + ["gmslr", "wais", "eais", "aispen"]

        for comp in export_list:
            if hasattr(self, comp):
                components_dict[comp] = getattr(self, comp)
        return components_dict

    def list_components(self) -> list:
        """List the available SLR components."""
        component_dict = self.get_components()
        print(list(component_dict.keys()))

    def save_components(self, output_dir: str, scenario_name: str) -> None:
        """Save all SLR components as .npy files to a directory.

        Parameters
        ----------
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

        # Save data in netcdf format (store all components in an xarray dataset)
        # Assumes first dimension is percentile, but can be more general (percentile/ensemble)
        ds = xr.Dataset()
        for name, component in self.get_components().items():
            xr_dataArray = xr.DataArray(
                component,
                dims=["percentile", "time"],
                coords={
                    "percentile": self.output_percentiles,
                    "time": np.arange(2006, component.shape[1] + 2006),
                },
            )
            xr_dataArray.attrs["units"] = "m"
            ds[name] = xr_dataArray
        ds.to_netcdf(os.path.join(output_dir, f"{scenario_name}_global.nc"))

    def check_shapes(
        self, T_change: np.ndarray, OHC_change: np.ndarray, n_time: int
    ) -> None:
        """Check that the input arrays have the correct shape.

        Parameters
        ----------
        T_change: np.ndarray
            Array of surface temperature changes.
        OHC_change: np.ndarray
            Array of ocean heat content changes.
        n_time: int
            Expected number of time steps.

        Returns
        -------
        None
        """
        if T_change.ndim == 1:
            T_change = T_change[np.newaxis, :]
        if OHC_change.ndim == 1:
            OHC_change = OHC_change[np.newaxis, :]

        if T_change.shape[1] != n_time:
            # Split over lines for readability
            raise ValueError(
                f"T_change should have shape (realisation, time) with time \
                dimension of length {n_time}. Got {T_change.shape}."
            )
        if OHC_change.shape[1] != n_time:
            raise ValueError(
                f"OHC_change should have shape (realisation, time) with time \
                dimension of length {n_time}. Got {OHC_change.shape}."
            )

    def project(
        self,
        scenario: str,
        T_change: np.ndarray,
        OHC_change: np.ndarray,
        cum_emissions_total: int = None,
    ) -> None:
        """Run the emulator to project GMSLR components."""
        self.scenario = scenario
        self.T_change = np.asarray(T_change)
        self.OHC_change = np.asarray(OHC_change)
        self.cum_emissions_total = cum_emissions_total

        # Check input shapes are correct
        self.check_shapes(self.T_change, self.OHC_change, self.nyr)

        if self.input_ensemble:
            self.nt = self.T_change.shape[0]

        T_ens, Exp_ens, T_int_ens, T_int_med = self.calculate_drivers()
        fraction = self.rng.random(
            self.nm * self.nt
        )  # correlation between antsmb and antdyn

        # Evaluate Inline Components
        if "expansion" in self.active_components:
            self.expansion = np.tile(Exp_ens, (self.nm, 1))

        # Evaluate Standard Components
        if self.parallel:
            self.run_parallel_projections(T_int_med, T_int_ens, T_ens, fraction)
        else:
            self.run_serial_projections(T_int_med, T_int_ens, T_ens, fraction)

        # Handle Random Sampling
        if self.random_sample:
            random_idx = self.rng.integers(low=0, high=self.nm)

            for comp in self.active_components + ["wais", "eais", "aispen"]:
                if hasattr(self, comp):
                    comp_data = getattr(self, comp)
                    if comp_data.ndim > 1:
                        setattr(self, comp, comp_data[random_idx][None, :])

        # Calculate GMSLR
        components_to_sum = [
            getattr(self, comp)
            for comp in self.active_components
            if hasattr(self, comp)
        ]
        if not components_to_sum:
            raise RuntimeError(
                "No active components were evaluated. Cannot compute GMSLR."
            )
        self.gmslr = np.sum(components_to_sum, axis=0)

        # Output percentiles
        if self.output_percentiles is not None:
            console.log(
                f"Sampling {len(self.output_percentiles)} members per component..."
            )
            self.gmslr = sample_members_2D(self.gmslr, self.output_percentiles)

            for comp in self.active_components + ["wais", "eais", "aispen"]:
                if hasattr(self, comp):
                    sampled_data = sample_members_2D(
                        getattr(self, comp), self.output_percentiles
                    )
                    setattr(self, comp, sampled_data)

    def _build_task_registry(
        self,
        T_int_med: np.ndarray,
        T_int_ens: np.ndarray,
        T_ens: np.ndarray,
        fraction: np.ndarray,
    ) -> dict:
        """Helper to map active component names to their functions and arguments."""
        registry = {}
        for comp in self.active_components:
            rng = self.component_rngs.get(comp)

            if comp == "glacier":
                registry[comp] = (self.project_glacier, (T_int_med, T_int_ens, rng))
            elif comp == "antsmb":
                registry[comp] = (self.project_antsmb, (T_int_ens, rng, fraction))
            elif comp == "greenland_ar6":
                registry[comp] = (self.project_greenland_AR6, (T_ens, rng))
            elif comp == "greendyn":
                registry[comp] = (self.project_greendyn_AR5, (rng,))
            elif comp == "greensmb":
                registry[comp] = (self.project_greensmb_AR5, (T_ens, rng))
            elif comp == "antdyn":
                registry[comp] = (self.project_antdyn, (rng, fraction))
            elif comp == "landwater":
                registry[comp] = (self.project_landwater_ar6, ())
            elif comp == "antarctica":
                registry[comp] = (self.project_antarctica_ismip6, (T_ens, rng))

        return registry

    def run_serial_projections(
        self,
        T_int_med: np.ndarray,
        T_int_ens: np.ndarray,
        T_ens: np.ndarray,
        fraction: np.ndarray,
    ) -> None:
        """Run components of the emulator sequentially."""
        registry = self._build_task_registry(T_int_med, T_int_ens, T_ens, fraction)

        for comp in self.active_components:
            if comp in registry:
                func, args = registry[comp]
                setattr(self, comp, func(*args))

        # Handle AR5 greenland combination if both are active
        if (
            "greendyn" in self.active_components
            and "greensmb" in self.active_components
        ):
            self.greenland_ar5 = self.greendyn + self.greensmb

    def run_parallel_projections(
        self,
        T_int_med: np.ndarray,
        T_int_ens: np.ndarray,
        T_ens: np.ndarray,
        fraction: np.ndarray,
    ) -> None:
        """Run components of the emulator in parallel."""
        registry = self._build_task_registry(T_int_med, T_int_ens, T_ens, fraction)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for comp in self.active_components:
                if comp in registry:
                    func, args = registry[comp]
                    futures[executor.submit(func, *args)] = comp

            for future in concurrent.futures.as_completed(futures):
                comp_name = futures[future]
                try:
                    setattr(self, comp_name, future.result())
                except Exception as e:
                    raise RuntimeError(
                        f"Component '{comp_name}' failed during projection."
                    ) from e

        # Handle AR5 greenland combination if both are active
        if (
            "greendyn" in self.active_components
            and "greensmb" in self.active_components
        ):
            self.greenland_ar5 = self.greendyn + self.greensmb

    def calculate_drivers(self) -> tuple:
        """Calculate the drivers of GMSLR: temperature change and
        thermosteric sea level rise.

        Returns
        -------
        T_ens: np.ndarray
            Ensemble of temperature changes.
        therm_ens: np.ndarray
            Ensemble of thermosteric sea level rise.
        T_int_ens: np.ndarray
            Ensemble of time-integral temperature anomalies.
        T_int_med: np.ndarray
            Median of time-integral temperature anomalies.
        """
        # Sensitivity of thermosteric SLR to ocean heat content change
        # From Turner et al. (2023)
        exp_efficiency = (
            self.rng.normal(loc=0.113, scale=0.013, size=self.nt)[:, None] * 1e-24
        )  # m/YJ

        if self.input_ensemble:
            # Check if dimensions are the right way around
            if self.T_change.shape[1] != self.nyr:
                self.T_change = self.T_change.T
            if self.OHC_change.shape[1] != self.nyr:
                self.OHC_change = self.OHC_change.T

            T_ens = self.T_change
            therm_ens = self.OHC_change * exp_efficiency
            T_int_ens = np.cumsum(T_ens, axis=1)
            T_int_med = np.percentile(T_int_ens, 50, axis=0)

        #     T_med = np.percentile(self.T_change, 50, axis=0)
        #     T_std = np.std(self.T_change, axis=0)

        #     therm_med = np.percentile(self.OHC_change, 50, axis=0) * exp_efficiency
        #     therm_std = np.std(self.OHC_change * exp_efficiency, axis=0)

        # else:
        #     if self.T_percentile_95 is not None:
        #         T_med = self.T_change
        #         therm_med = self.OHC_change * exp_efficiency

        #         T_std = (self.T_percentile_95 - self.T_change) / 1.645
        #         therm_std = (
        #             (self.OHC_percentile_95 - self.OHC_change) * exp_efficiency / 1.645
        #         )

        #     else:
        #         raise ValueError(
        #             "If input_ensemble is False, and T_change and OHC_change "
        #             "are not 2D arrays, you must provide a 95th percentile "
        #             "timeseries for T_change and OHC_change. Add this using "
        #             "T_percentile_95 and OHC_percentile_95 keyword arguments."
        #         )

        # Time-integral of temperature anomaly
        # T_int_med = np.cumsum(T_med)
        # T_int_std = np.cumsum(T_std)

        # # Generate a sample of perfectly correlated timeseries fields of temperature,
        # # time-integral temperature and expansion, each of them [realisation,time]
        # z = self.rng.standard_normal(self.nt) * self.tcv

        # # For each quantity, mean + standard deviation * normal random number
        # # reshape to [realisation,time]
        # T_ens = z[:, np.newaxis] * T_std + T_med
        # therm_ens = z[:, np.newaxis] * therm_std + therm_med
        # T_int_ens = z[:, np.newaxis] * T_int_std + T_int_med
        return T_ens, therm_ens, T_int_ens, T_int_med

    def project_antarctica_ismip6(self, T_ens: np.ndarray, rng) -> np.ndarray:
        wais_raw = self.wais_model.predict(T_ens.squeeze(), display_progress=False)
        eais_raw = self.eais_model.predict(T_ens.squeeze(), display_progress=False)
        aispen_raw = self.aispen_model.predict(T_ens.squeeze(), display_progress=False)

        random_ais_idx = rng.integers(low=0, high=17)

        # Match the correct output shape
        self.wais = np.repeat(wais_raw[random_ais_idx, :, :], self.nt, axis=0)
        self.eais = np.repeat(eais_raw[random_ais_idx, :, :], self.nt, axis=0)
        self.aispen = np.repeat(aispen_raw[random_ais_idx, :, :], self.nt, axis=0)
        return self.wais + self.eais + self.aispen

    def project_glacier(
        self, T_int_med: np.ndarray, T_int_ens: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Project glacier contribution to GMSLR.

        Parameters
        ----------
        T_int_med: np.ndarray
            Time-integral of median temperature anomaly timeseries.
        T_int_ens: np.ndarray
            Ensemble of time-integral temperature anomaly timeseries.

        Returns
        -------
        glacier: np.ndarray
            Glacier contribution to GMSLR.
        """
        # glaciermip -- False => AR5 parameters, 1 => fit to Hock et al. (2019),
        #   2 => fit to Marzeion et al. (2020)
        dmzdtref = 0.95  # mm yr-1 in Marzeion's CMIP5 ensemble mean for AR5 ref period
        dmz = (
            dmzdtref * (self.endofhistory - 1996) * 1e-3
        )  # m from glacier at start wrt AR5 ref period
        glmass = 412.0 - 96.3  # initial glacier mass, used to set a limit, from Tab 4.2
        glmass = 1e-3 * glmass  # m SLE

        glacier = np.full((self.nm, self.nt, self.nyr), np.nan)

        if self.glaciermip:
            if self.glaciermip == 1:
                glparm = [
                    dict(name="SLA2012", factor=3.39, exponent=0.722, cvgl=0.15),
                    dict(name="MAR2012", factor=4.35, exponent=0.658, cvgl=0.13),
                    dict(name="GIE2013", factor=3.57, exponent=0.665, cvgl=0.13),
                    dict(name="RAD2014", factor=6.21, exponent=0.648, cvgl=0.17),
                    dict(name="GloGEM", factor=2.88, exponent=0.753, cvgl=0.13),
                ]
                cvgl = 0.15  # unnecessary default
            elif self.glaciermip == 2:
                glparm = [
                    dict(name="GLIMB", factor=3.70, exponent=0.662, cvgl=0.206),
                    dict(name="GloGEM", factor=4.08, exponent=0.716, cvgl=0.161),
                    dict(name="JULES", factor=5.50, exponent=0.564, cvgl=0.188),
                    dict(name="Mar-12", factor=4.89, exponent=0.651, cvgl=0.141),
                    dict(name="OGGM", factor=4.26, exponent=0.715, cvgl=0.164),
                    dict(name="RAD2014", factor=5.18, exponent=0.709, cvgl=0.135),
                    dict(name="WAL2001", factor=2.66, exponent=0.730, cvgl=0.206),
                ]
                cvgl = 0.20  # unnecessary default
            else:
                raise KeyError("glaciermip must be 1 or 2")
        else:
            glparm = [
                dict(name="Marzeion", factor=4.96, exponent=0.685),
                dict(name="Radic", factor=5.45, exponent=0.676),
                dict(name="Slangen", factor=3.44, exponent=0.742),
                dict(name="Giesen", factor=3.02, exponent=0.733),
            ]
            cvgl = 0.20  # random methodological error

        ngl = len(glparm)  # number of glacier methods

        r_per_model = self.nm // ngl
        r_remainder = self.nm % ngl
        r = self.rng.standard_normal(self.nm)
        r = r[:, np.newaxis, np.newaxis]

        # Precompute mgl and zgl for all glacier methods
        mgl_all = np.array(
            [
                self._project_glacier1(
                    T_int_med, glparm[igl]["factor"], glparm[igl]["exponent"]
                )
                for igl in range(ngl)
            ]
        )
        zgl_all = np.array(
            [
                self._project_glacier1(
                    T_int_ens, glparm[igl]["factor"], glparm[igl]["exponent"]
                )
                for igl in range(ngl)
            ]
        )
        cvgl_all = np.array(
            [glparm[igl]["cvgl"] if self.glaciermip else cvgl for igl in range(ngl)]
        )

        # Make an ensemble of projections for each method
        current_ensemble_idx = 0
        for igl in range(ngl):
            mgl = mgl_all[igl]
            zgl = zgl_all[igl]
            cvgl = cvgl_all[igl]

            num_reals_for_model_i = (
                r_per_model + 1 if igl < r_remainder else r_per_model
            )
            ifirst = current_ensemble_idx
            ilast = current_ensemble_idx + num_reals_for_model_i

            glacier[ifirst:ilast, ...] = zgl + (mgl * r[ifirst:ilast] * cvgl)
            current_ensemble_idx = ilast

        glacier += dmz
        np.clip(glacier, None, glmass, out=glacier)

        glacier = glacier.reshape(glacier.shape[0] * glacier.shape[1], glacier.shape[2])
        return glacier

    def _project_glacier1(
        self, T_int: np.ndarray, factor: float, exponent: float
    ) -> np.ndarray:
        """Project glacier contribution by one glacier method.

        Parameters
        ----------
        T_int: np.ndarray
            Time-integral temperature anomaly timeseries.
        factor: float
            Factor for the glacier method.
        exponent: float
            Exponent for the glacier method.

        Returns
        -------
        np.ndarray
            Projection of glacier contribution.
        """
        scale = 1e-3  # mm to m
        return scale * factor * (np.where(T_int < 0, 0, T_int) ** exponent)

    def project_greensmb_AR5(
        self, T_ens: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Project Greenland SMB contribution to GMSLR.

        Parameters
        ----------
        T_ens: np.ndarray
            Ensemble of temperature anomaly timeseries.

        Returns
        -------
        greensmb: np.ndarray
            Greenland SMB contribution to GMSLR.

        """
        dtgreen = -0.146  # Delta_T of Greenland ref period wrt AR5 ref period
        fnlogsd = 0.4  # random methodological error of the log factor
        febound = [1, 1.15]  # bounds of uniform pdf of SMB elevation feedback factor

        # random log-normal factor
        fn = np.exp(self.rng.standard_normal(self.nm) * fnlogsd)
        # elevation feedback factor
        fe = self.rng.random(self.nm) * (febound[1] - febound[0]) + febound[0]
        ff = fn * fe

        ztgreen = T_ens - dtgreen

        greensmb = ff[:, np.newaxis, np.newaxis] * self._fettweis(ztgreen)

        if self.palmer_method and self.end_yr > self.endofAR5:
            greensmb[:, :, 95:] = greensmb[:, :, 94:95]

        greensmb = np.cumsum(greensmb, axis=-1)

        greensmb += (1 - self.fgreendyn) * self.dgreen

        greensmb = greensmb.reshape(
            greensmb.shape[0] * greensmb.shape[1], greensmb.shape[2]
        )
        return greensmb

    def _fettweis(self, ztgreen: np.ndarray) -> np.ndarray:
        """Calculate Greenland SMB in m yr-1 SLE from global mean temperature
        anomaly, using Eq 2 of Fettweis et al. (2013).

        Parameters
        ----------
        ztgreen: np.ndarray
            Global mean temperature anomaly.

        Returns
        -------
        np.ndarray
            Greenland SMB in m yr-1 SLE.
        """
        return (
            71.5 * ztgreen + 20.4 * (ztgreen**2) + 2.8 * (ztgreen**3)
        ) * self.mSLEoGt

    def project_antsmb(
        self,
        T_int_ens: np.ndarray,
        rng: np.random.Generator,
        fraction: np.ndarray = None,
    ) -> np.ndarray:
        """Project Antarctic SMB contribution to GMSLR.

        Parameters
        ----------
        T_int_ens: np.ndarray
            Ensemble of time-integral temperature anomaly timeseries.
        fraction: np.ndarray
            Random numbers for the SMB-dynamic feedback.

        Returns
        -------
        antsmb: np.ndarray
            Antarctic SMB contribution to GMSLR.
        """
        # The following are [mean,SD]
        pcoK = [5.1, 1.5]  # % change in Ant SMB per K of warming from G&H06
        KoKg = [1.1, 0.2]  # ratio of Antarctic warming to global warming from G&H06

        # Generate a distribution of products of the above two factors
        pcoKg = (pcoK[0] + self.rng.standard_normal([self.nm, self.nt]) * pcoK[1]) * (
            KoKg[0] + self.rng.standard_normal([self.nm, self.nt]) * KoKg[1]
        )
        meansmb = 1923  # model-mean time-mean 1979-2010 Gt yr-1 from 13.3.3.2
        moaoKg = (
            -pcoKg * 1e-2 * meansmb * self.mSLEoGt
        )  # m yr-1 of SLE per K of global warming

        if fraction is None:
            fraction = self.rng.random((self.nm, self.nt))
        elif fraction.size != self.nm * self.nt:
            raise ValueError("fraction is the wrong size")
        else:
            fraction = fraction.reshape((self.nm, self.nt))

        smax = 0.35  # max value of S in 13.SM.1.5
        ainterfactor = 1 - fraction * smax

        z = moaoKg * ainterfactor
        z = z[:, :, np.newaxis]
        antsmb = z * T_int_ens
        antsmb = antsmb.reshape(antsmb.shape[0] * antsmb.shape[1], antsmb.shape[2])
        return antsmb

    def project_greenland_AR6(
        self, T_ens: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Project Greenland ice-sheet contribution to GMSLR.
        This follows the IPCC AR6 methodology as closely as possible.
        Projections are relative to 1996-2014 baseline.

        Returns
        -------
        np.ndarray
            Total GIS contribution to GMSLR.
        """
        df = load_greenland_calibration()
        b0 = df["b0"].values[None, :, None]
        b1 = df["b1"].values[None, :, None]
        b2 = df["b2"].values[None, :, None]
        b3 = df["b3"].values[None, :, None]
        b4 = df["b4"].values[None, :, None]
        b5 = df["b5"].values[None, :, None]
        sigma = df["sigma"].values
        time_delta = np.arange(self.nyr)

        # GIS trend values taken from FACTS GitHub repo
        trend_mean = 0.19
        trend_std = 0.1

        # Calculate trend contribution distribution
        trend = truncnorm.ppf(
            self.rng.random(self.nm), a=0.0, b=99999.9, loc=trend_mean, scale=trend_std
        )
        trend = trend[:, None] * time_delta[None, :]
        trend = trend[:, None, :]
        trend /= 1e3  # convert mm to m SLE

        # Calculate GIS contribution rate
        dsle = (
            b0
            + (b1 * T_ens[:, None, :])
            + (b2 * T_ens[:, None, :] ** 2)
            + (b3 * T_ens[:, None, :] ** 3)
            + (b4 * time_delta[None, None, :])
            + (b5 * time_delta[None, None, :] ** 2)
        )

        # Now integrate
        sle = np.cumsum(dsle, axis=2)  # mm SLE per K of global warming
        sle = sle * 1e-3  # convert mm to m SLE

        # Make a Monte Carlo ensemble of projections for each model in the calibration
        sle_ens = np.zeros((self.nm, self.nt, self.nyr))
        r_per_model = self.nm // sle.shape[1]
        r_remainder = self.nm % sle.shape[1]

        # We want to distribute the remainder evenly across the models
        unc = self.rng.normal(scale=sigma)
        current_ensemble_idx = 0
        for i in range(sle.shape[1]):
            num_reals_for_model_i = r_per_model + 1 if i < r_remainder else r_per_model
            ifirst = current_ensemble_idx
            ilast = current_ensemble_idx + num_reals_for_model_i
            model_term = sle[:, i, :]  # Shape (nt, nyr)
            uncertainty_term = (
                model_term * unc[None, i, None]
            )  # Shape (num_reals_for_model_i, nt, nyr_param)

            sle_ens[ifirst:ilast, :, :] = model_term[None, :, :] + uncertainty_term
            current_ensemble_idx = ilast

        sle_ens += trend

        # Persist 2100 rate of change
        rate = np.diff(sle_ens, axis=2)[:, :, 94]
        sle_ens[:, :, 95:] = sle_ens[:, :, 94:95] + (
            rate[:, :, None] * time_delta[None, None, 1 : self.nyr - 94]
        )
        sle_ens = sle_ens.reshape((self.nm * self.nt, self.nyr))
        return sle_ens

    def project_greendyn_AR5(self, rng: np.random.Generator) -> np.ndarray:
        """Project Greenland rapid ice-sheet dynamics contribution to GMSLR.

        Returns
        -------
        np.ndarray
            Greenland rapid ice-sheet dynamics contribution to GMSLR.
        """
        # For SMB+dyn during 2005-2010 Table 4.6 gives 0.63+-0.17 mm yr-1 (5-95% range)
        # For dyn at 2100 Chapter 13 gives [20,85] mm for rcp85, [14,63] mm otherwise
        if self.scenario in ["rcp85", "ssp585"]:
            finalrange = [0.020, 0.085]
        else:
            finalrange = [0.014, 0.063]
        return (
            self.time_projection(
                0.63 * self.fgreendyn, 0.17 * self.fgreendyn, finalrange, rng
            )
            + self.fgreendyn * self.dgreen
        )

    def project_antdyn(
        self, rng: np.random.Generator, fraction: np.ndarray = None
    ) -> np.ndarray:
        """Project Antarctic rapid ice-sheet dynamics contribution to GMSLR.

        Parameters
        ----------
        fraction: np.ndarray
            Random numbers for the dynamic contribution.

        Returns
        -------
        np.ndarray
            Antarctic rapid ice-sheet dynamics contribution to GMSLR.
        """
        # This is a naive solution to calculating the AntDyn contribution
        # for any given scenario. Basically linear regressions through existing data
        # to find rough relationship between cumulative emissions and AntDyn contribution.
        if self.cum_emissions_total:
            upper = (0.000110 * self.cum_emissions_total) + 0.375  # in metres
            lower = (1.363e-05 * self.cum_emissions_total) + 0.0392  # in metres
            final = [lower, upper]
            # print(final)
            # final=[-0.020, 0.185]
        else:
            lcoeff = dict(
                rcp26=[-2.881, 0.923, 0.000],
                rcp45=[-2.676, 0.850, 0.000],
                rcp60=[-2.660, 0.870, 0.000],
                rcp85=[-2.399, 0.860, 0.000],
            )
            lcoeff = lcoeff[self.scenario]

            ascale = norm.ppf(fraction)
            final = np.exp(lcoeff[2] * ascale**2 + lcoeff[1] * ascale + lcoeff[0])
            final = final.reshape(self.nm, self.nt)

            # final=[-0.020, 0.185]
            # final = [0.06, 0.49] # AR6, SSP2-4.5

            # For SMB+dyn during 2005-2010 Table 4.6 gives 0.41+-0.24 mm yr-1 (5-95% range)
            # For dyn at 2100 Chapter 13 gives [-20,185] mm for all scenarios
        return (
            self.time_projection(0.41, 0.20, final, rng, fraction=fraction) + self.dant
        )

    def project_landwater_ar6(self) -> np.ndarray:
        """Project land water storage contribution to GMSLR.

        Returns
        -------
        np.ndarray
            Land water storage contribution to GMSLR.
        """
        # Read in AR6 landwater contributions
        lw_ds = load_landwater_projection()

        # Interpolate to annual projections
        interp_ds = lw_ds.interp(
            years=np.arange(2005, 2301, 1), method="linear"
        ).squeeze()
        lw = interp_ds["sea_level_change"].values * 1e-3  # mm to m

        del interp_ds

        # Go from shape (20000, 294) to (101000, 294)
        full_repeats = (self.nt * self.nm) // lw.shape[0]
        remainder = (self.nt * self.nm) % lw.shape[0]
        lw = np.vstack([np.tile(lw, (full_repeats, 1)), lw[:remainder]])
        lw = lw.reshape(self.nt * self.nm, lw.shape[1])
        lw = lw[:, 1:]  # Start at 2006, end at 2299
        return lw

    def project_landwater_ar5(self, rng: np.random.Generator) -> np.ndarray:
        """Old projection function. Project land water storage
        contribution to GMSLR.

        Returns
        -------
        np.ndarray
            Land water storage contribution to GMSLR.
        """
        # The rate at start is the one for 1993-2010 from the budget table.
        # The final amount is the mean for 2081-2100.
        nyr = 2100 - 2081 + 1  # number of years of the time-mean of the final amount
        final = [-0.01, 0.09]  # AR5
        # final = [0.01, 0.04] # AR6
        return self.time_projection(0.38, 0.49 - 0.38, final, rng, nfinal=nyr)

    def time_projection(
        self,
        startratemean: float,
        startratepm: float,
        final,
        rng: np.random.Generator,
        nfinal: int = 1,
        fraction: np.ndarray = None,
    ) -> np.ndarray:
        """Project a quantity which is a quadratic function of time.

        Parameters
        ----------
        startratemean: float
            Rate of GMSLR at the start in mm yr-1.
        startratepm: float
            Start rate error in mm yr-1.
        final: list | np.ndarray
            Likely range in m for GMSLR at the end of AR5.
        nfinal: int
            Number of years at the end over which final is a time-mean.
        fraction: np.ndarray
            Random numbers in the range 0 to 1.

        Returns
        -------
        np.ndarray
            Projection of the quantity.
        """
        # Create a field of elapsed time since start in years
        timeendofAR5 = self.endofAR5 - self.endofhistory + 1
        time = np.arange(self.end_yr - self.endofhistory) + 1

        if fraction is None:
            fraction = self.rng.random((self.nm, self.nt))
        elif fraction.size != self.nm * self.nt:
            raise ValueError("fraction is the wrong size")

        fraction = fraction.reshape(self.nm, self.nt)

        # Convert inputs to startrate (m yr-1) and afinal (m), where both are
        # arrays with the size of fraction
        startrate = (
            startratemean + startratepm * np.array([-1, 1], dtype=float)
        ) * 1e-3  # convert mm yr-1 to m yr-1
        finalisrange = isinstance(final, Sequence)

        if finalisrange:
            if len(final) != 2:
                raise ValueError("final range is the wrong size")
            afinal = (1 - fraction) * final[0] + fraction * final[1]
        else:
            if final.shape != fraction.shape:
                raise ValueError("final array is the wrong shape")
            afinal = final

        startrate = (1 - fraction) * startrate[0] + fraction * startrate[1]

        # For terms where the rate increases linearly in time t, we can write GMSLR as
        #   S(t) = a*t**2 + b*t
        # where a is 0.5*acceleration and b is start rate. Hence
        #   a = S/t**2-b/t = (S-b*t)/t**2
        # If nfinal=1, the following two lines are equivalent to
        # halfacc=(final-startyr*nyr)/nyr**2
        finalyr = np.arange(nfinal) - nfinal + 94 + 1  # last element ==nyr
        halfacc = (afinal - startrate * finalyr.mean()) / (finalyr**2).mean()
        quadratic = halfacc[:, :, np.newaxis] * (time**2)
        linear = startrate[:, :, np.newaxis] * time

        # If acceleration ceases for t>t0, the rate is 2*a*t0+b thereafter, so
        #   S(t) = a*t0**2 + b*t0 + (2*a*t0+b)*(t-t0)
        #        = a*t0*(2*t - t0) + b*t
        # i.e. the quadratic term is replaced, the linear term unaffected
        # The quadratic also = a*t**2-a*(t-t0)**2

        if self.palmer_method:
            y = halfacc[:, :, np.newaxis] * timeendofAR5 * ((2 * time) - timeendofAR5)
            quadratic[:, :, 95:] = y[:, :, 95:]

        quadratic += linear

        quadratic = quadratic.reshape(
            quadratic.shape[0] * quadratic.shape[1], quadratic.shape[2]
        )

        return quadratic
