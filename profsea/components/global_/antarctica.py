from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.signal import fftconvolve
from scipy.stats import norm

from profsea.components.core.base import Component
from profsea.components.core.global_model import ClimateState
from profsea.components.core.time_projection import time_projection

PROFSEA_DIR = Path(__file__).resolve().parents[2]
AUX_DATA_DIR = PROFSEA_DIR / "aux_data"

PARAMS_MAP = {
    "wais": AUX_DATA_DIR / "wais_params_expanded.nc",
    "eais": AUX_DATA_DIR / "eais_params_expanded.nc",
    "peninsula": AUX_DATA_DIR / "pen_params_expanded.nc",
}


class AntarcticaISMIP6(Component):
    """
    ISMIP6 2300 Antarctic ice-sheet emulator with two-timescale response.

    This emulation of the ISMIP6 ice-sheet model ensemble aims to capture
    slow, fast and drift responses of the Antarctic ice sheet to GMST.
    The slow response is modelled as the impulse response to temperature
    forcing, convolved with two exponential decay kernels representing
    different ice-sheet response timescales, depending on the region being modelled.
    The fast response is modelled as a direct proportionality to the integrated
    temperature anomaly, while the drift term captures any linear time-dependent
    trends not explained by the temperature forcing.

    Parameters provided are for the WAIS, EAIS and AIS Peninsula regions, since each has
    different response characteristics to warming.
    """

    def __init__(self, region: str):
        """
        Parameters
        ----------
        region: str
            The Antarctic region to model. Must be one of "wais", "eais", or "peninsula".
        """
        params_path = PARAMS_MAP.get(region.lower())
        if not params_path:
            raise ValueError(f"Invalid region calibration: {region}")

        self.param_ds = xr.load_dataset(params_path)
        self.n_models = self.param_ds.coords["model"].shape[0]

    def _impulse_response_term(
        self,
        tas_1d: np.ndarray,
        tau1: float,
        tau2: float,
        gamma: float,
        params_1d: np.ndarray,
        dt: float,
        state: ClimateState,
    ) -> np.ndarray:
        """Computes the slow response term for a single trajectory."""
        n_time = tas_1d.shape[0]
        alpha1, alpha2, _ = params_1d

        forcing_base = np.sign(tas_1d) * (np.abs(tas_1d) ** gamma)

        # decay_factors1 = np.exp(-np.arange(n_time) * dt / tau1) * (dt / tau1)
        t_arr = np.arange(n_time, dtype=state.dtype)
        decay_factors1 = (t_arr * dt / tau1**2) * np.exp(-t_arr * dt / tau1) * dt
        rate_delayed1 = fftconvolve(forcing_base, decay_factors1, mode="full")[:n_time]
        term_slow1 = alpha1 * (np.cumsum(rate_delayed1) * dt)

        # decay_factors2 = np.exp(-np.arange(n_time) * dt / tau2) * (dt / tau2)
        decay_factors2 = (t_arr * dt / tau2**2) * np.exp(-t_arr * dt / tau2) * dt
        rate_delayed2 = fftconvolve(forcing_base, decay_factors2, mode="full")[:n_time]
        term_slow2 = alpha2 * (np.cumsum(rate_delayed2) * dt)

        return term_slow1 + term_slow2

    def _precompute_delayed_rates(
        self, tas: np.ndarray, dt: float, state: ClimateState
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Precomputes the cumulative delayed rates for all models across all trajectories.
        Returns two arrays of shape (n_models, n_traj, n_time).
        """
        n_traj, n_time = tas.shape
        cum_rate1 = np.zeros((self.n_models, n_traj, n_time), dtype=state.dtype)
        cum_rate2 = np.zeros((self.n_models, n_traj, n_time), dtype=state.dtype)
        t_arr = np.arange(n_time, dtype=state.dtype)

        for m_idx in range(self.n_models):
            tau1 = np.array(self.param_ds.tau1[m_idx].values, dtype=state.dtype)
            tau2 = np.array(self.param_ds.tau2[m_idx].values, dtype=state.dtype)
            gamma = np.array(self.param_ds.gamma[m_idx].values, dtype=state.dtype)

            # Forcing base is model-specific due to gamma. Shape: (n_traj, n_time)
            forcing_base = np.sign(tas) * (np.abs(tas) ** gamma)

            # Decay factors broadcasted to 2D: (1, n_time)
            df1 = ((t_arr * dt / tau1**2) * np.exp(-t_arr * dt / tau1) * dt)[
                np.newaxis, :
            ]
            df2 = ((t_arr * dt / tau2**2) * np.exp(-t_arr * dt / tau2) * dt)[
                np.newaxis, :
            ]

            # Vectorized convolution across all trajectories (axes=1)
            rate_delayed1 = fftconvolve(forcing_base, df1, mode="full", axes=1)[
                :, :n_time
            ]
            cum_rate1[m_idx] = np.cumsum(rate_delayed1, axis=1) * dt

            rate_delayed2 = fftconvolve(forcing_base, df2, mode="full", axes=1)[
                :, :n_time
            ]
            cum_rate2[m_idx] = np.cumsum(rate_delayed2, axis=1) * dt

        return cum_rate1, cum_rate2

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """
        Projects AIS response using empirical additive bootstrapping aligned
        to the ClimateState ensemble size.
        """
        tas = np.atleast_2d(np.squeeze(state.T_ens))
        n_traj, n_time = tas.shape
        dt = 1.0
        nm = state.nt * state.num_members

        # 1. Precompute expensive convolutions for unique (Model x Trajectory) combos
        cum_rate1, cum_rate2 = self._precompute_delayed_rates(tas, dt, state)
        tas_int = np.cumsum(tas, axis=1) * dt

        # 2. Generate random model/residual assignments for all members
        model_indices = rng.integers(0, self.n_models, size=nm)
        all_residuals = np.asarray(
            self.param_ds.param_residuals.values, dtype=state.dtype
        )
        n_train_scenarios = all_residuals.shape[1]
        residual_indices = rng.integers(0, n_train_scenarios, size=nm)

        # 3. Create flat mapping array to match the (Trajectory x Member) layout
        # This groups by trajectory: [Traj0_Mem0...Traj0_Mem999, Traj1_Mem0...]
        t_indices = np.repeat(np.arange(n_traj, dtype=np.int32), state.num_members)

        # NOTE: If your required grouping is interleaved [Traj0_Mem0, Traj1_Mem0...]
        # uncomment the line below instead:
        # t_indices = np.tile(np.arange(n_traj, dtype=state.dtype), state.num_members)

        # 4. Vectorized Parameter Extraction
        general_p = np.asarray(
            self.param_ds.general_params.values[model_indices], dtype=state.dtype
        )
        sampled_residuals = all_residuals[model_indices, residual_indices, :]
        total_params = general_p + sampled_residuals

        # Slice with [:, 0:1] to maintain a 2D shape (nm, 1) for broadcasting against time
        alpha1 = total_params[:, 0:1]
        alpha2 = total_params[:, 1:2]
        beta = total_params[:, 2:3]

        # 5. Final Vectorized Assembly
        term_slow = (
            alpha1 * cum_rate1[model_indices, t_indices]
            + alpha2 * cum_rate2[model_indices, t_indices]
        )
        term_fast = beta * tas_int[t_indices]

        return term_slow + term_fast


class AntarcticaDynAR5(Component):
    """
    AR5 Antarctic ice-dynamics response to warming, as a function of
    cumulative emissions or scenario. Following the implementation
    as given by Jonathan Gregory's ar5gmslr (https://github.com/JonathanGregory/ar5gmslr).

    Requires cumulative emissions total to be specified for a given scenario,
    or will default to scenario-based regression, based on rcp scenarios.
    """

    def __init__(
        self, d_ant: float = (2.37 + 0.13) * 1e-3, cum_emissions_total: float = None
    ):
        self.d_ant = d_ant
        self.cum_emissions_total = cum_emissions_total

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
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
        if self.cum_emissions_total is not None:
            upper = (0.000110 * self.cum_emissions_total) + 0.375  # in metres
            lower = (1.363e-05 * self.cum_emissions_total) + 0.0392  # in metres
            final = [lower, upper]
        else:
            lcoeff = dict(
                rcp26=[-2.881, 0.923, 0.000],
                rcp45=[-2.676, 0.850, 0.000],
                rcp60=[-2.660, 0.870, 0.000],
                rcp85=[-2.399, 0.860, 0.000],
            )
            lcoeff = lcoeff[state.scenario]

            ascale = norm.ppf(state.fraction).astype(state.dtype)
            final = np.exp(lcoeff[2] * ascale**2 + lcoeff[1] * ascale + lcoeff[0])
            final = final.reshape(state.num_members, state.nt)
        return (
            time_projection(
                state,
                0.41,
                0.20,
                final,
                rng,
                fraction=state.fraction,
            )
            + self.d_ant
        )


class AntarcticaSMBAR5(Component):
    """
    AR5 Antarctic SMB contribution to GMSLR, as a function of global
    mean surface temperature change.

    Following the implementation as given by Jonathan Gregory's ar5gmslr
    (https://github.com/JonathanGregory/ar5gmslr).
    """

    def __init__(self):
        pass

    def project(
        self,
        state: ClimateState,
        rng: np.random.Generator,
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
        # Conversion factor for Gt to m SLE
        mSLEoGt = 1e12 / 3.61e14 * 1e-3
        # The following are [mean,SD]
        pcoK = [5.1, 1.5]  # % change in Ant SMB per K of warming from G&H06
        KoKg = [1.1, 0.2]  # ratio of Antarctic warming to global warming from G&H06

        # Generate a distribution of products of the above two factors
        pcoKg = (
            pcoK[0]
            + rng.standard_normal([state.num_members, state.nt], dtype=state.dtype)
            * pcoK[1]
        ) * (
            KoKg[0]
            + rng.standard_normal([state.num_members, state.nt], dtype=state.dtype)
            * KoKg[1]
        )
        meansmb = 1923  # model-mean time-mean 1979-2010 Gt yr-1 from 13.3.3.2
        moaoKg = (
            -pcoKg * 1e-2 * meansmb * mSLEoGt
        )  # m yr-1 of SLE per K of global warming

        if state.fraction is None:
            fraction = rng.random((state.num_members, state.nt), dtype=state.dtype)
        elif state.fraction.size != state.num_members * state.nt:
            raise ValueError("fraction is the wrong size")
        else:
            fraction = state.fraction.reshape((state.num_members, state.nt))

        smax = 0.35  # max value of S in 13.SM.1.5
        ainterfactor = 1 - fraction * smax

        z = moaoKg * ainterfactor
        z = z[:, :, np.newaxis]
        antsmb = z * state.T_int_ens
        antsmb = antsmb.reshape(antsmb.shape[0] * antsmb.shape[1], antsmb.shape[2])
        return antsmb
