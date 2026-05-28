from pathlib import Path
from rich.progress import track
import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import norm
import xarray as xr

from profsea.components.core.base import Component
from profsea.components.core.global_model import ClimateState
from profsea.components.core.time_projection import time_projection


class AntarcticaISMIP6:
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

    def __init__(self, params_path: Path | str):
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
    ) -> np.ndarray:
        """Computes the slow response term for a single trajectory."""
        n_time = tas_1d.shape[0]
        alpha1, alpha2, _ = params_1d

        forcing_base = np.sign(tas_1d) * (np.abs(tas_1d) ** gamma)

        # decay_factors1 = np.exp(-np.arange(n_time) * dt / tau1) * (dt / tau1)
        t_arr = np.arange(n_time)
        decay_factors1 = (t_arr * dt / tau1**2) * np.exp(-t_arr*dt/tau1) * dt
        rate_delayed1 = fftconvolve(forcing_base, decay_factors1, mode="full")[:n_time]
        term_slow1 = alpha1 * (np.cumsum(rate_delayed1) * dt)

        # decay_factors2 = np.exp(-np.arange(n_time) * dt / tau2) * (dt / tau2)
        decay_factors2 = (t_arr * dt / tau2**2) * np.exp(-t_arr*dt/tau2) * dt
        rate_delayed2 = fftconvolve(forcing_base, decay_factors2, mode="full")[:n_time]
        term_slow2 = alpha2 * (np.cumsum(rate_delayed2) * dt)

        return term_slow1 + term_slow2

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """
        Projects AIS response using empirical additive bootstrapping aligned
        to the ClimateState ensemble size.
        """
        tas = state.T_ens
        if tas.ndim > 2:
            tas = np.squeeze(tas)
        if tas.ndim == 1:
            tas = np.expand_dims(tas, axis=0)

        dt = 1.0
        nm = tas.shape[0]
        n_time = tas.shape[1]

        preds = np.zeros((nm, n_time))
        tas_int = np.cumsum(tas, axis=1) * dt
        physical_time = np.arange(n_time) * dt

        # Randomly assign an ISMIP6 model and residual draw to each ensemble member
        model_indices = rng.integers(0, self.n_models, size=nm)
        all_residuals = self.param_ds.param_residuals.values
        n_train_scenarios = all_residuals.shape[1]
        residual_indices = rng.integers(0, n_train_scenarios, size=nm)

        for i in range(nm):
            m_idx = model_indices[i]
            r_idx = residual_indices[i]

            # Extract assigned model parameters
            tau1 = float(self.param_ds.tau1[m_idx].values)
            tau2 = float(self.param_ds.tau2[m_idx].values)
            gamma = float(self.param_ds.gamma[m_idx].values)

            general_p = self.param_ds.general_params[m_idx].values
            sampled_residuals = all_residuals[m_idx, r_idx, :]
            total_params = general_p + sampled_residuals

            # Slow response
            term_slow = self._impulse_response_term(
                tas[i], tau1, tau2, gamma, total_params, dt
            )

            # Fast response
            beta = total_params[2]
            term_fast = beta * tas_int[i]

            # Drift term
            # drift_coeff = total_params[3]
            # term_drift = drift_coeff * physical_time

            # Combine
            preds[i, :] = term_fast + term_slow

        return preds


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
        if self.cum_emissions_total:
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

            ascale = norm.ppf(state.fraction)
            final = np.exp(lcoeff[2] * ascale**2 + lcoeff[1] * ascale + lcoeff[0])
            final = final.reshape(state.num_members, state.nt)
        return (
            time_projection(state, 0.41, 0.20, final, rng, fraction=state.fraction)
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
        # Conversion factor for Gt to m SLE
        self.mSLEoGt = 1e12 / 3.61e14 * 1e-3

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
        # The following are [mean,SD]
        pcoK = [5.1, 1.5]  # % change in Ant SMB per K of warming from G&H06
        KoKg = [1.1, 0.2]  # ratio of Antarctic warming to global warming from G&H06

        # Generate a distribution of products of the above two factors
        pcoKg = (pcoK[0] + rng.standard_normal([state.num_members, state.nt]) * pcoK[1]) * (
            KoKg[0] + rng.standard_normal([state.num_members, state.nt]) * KoKg[1]
        )
        meansmb = 1923  # model-mean time-mean 1979-2010 Gt yr-1 from 13.3.3.2
        moaoKg = (
            -pcoKg * 1e-2 * meansmb * self.mSLEoGt
        )  # m yr-1 of SLE per K of global warming

        if state.fraction is None:
            fraction = rng.random((state.num_members, state.nt))
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
